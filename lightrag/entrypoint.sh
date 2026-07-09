#!/bin/bash
# ============================================
# LightRAG Entrypoint
# Sources .env config, applies LM Studio patch, starts server
# ============================================
set -e

CONFIG_FILE="/lightrag-config.env"

# Load config from mounted .env file (overrides compose env_file)
if [ -f "$CONFIG_FILE" ]; then
    echo "[entrypoint] Loading config from $CONFIG_FILE"
    set -a
    source "$CONFIG_FILE"
    set +a
fi

# Ensure language is set
export SUMMARY_LANGUAGE="${SUMMARY_LANGUAGE:-German}"
echo "[entrypoint] Language: $SUMMARY_LANGUAGE"

# Disable Qdrant — use file-based storage only
export QDRANT_URL=""

# LM Studio compatibility patch (only for openai binding)
if [ "$LLM_BINDING" = "openai" ]; then
    echo "[entrypoint] Applying LM Studio compatibility patch..."
    cat > /app/sitecustomize.py << 'PATCHEOF'
import httpx
import json as _json

_original_send = httpx.AsyncClient.send

async def _patched_send(self, request, **kwargs):
    try:
        url_str = str(request.url)
        
        # Fix embedding endpoint: /api/embed → /embeddings (LM Studio compat)
        if "/api/embed" in url_str and request.method == "POST":
            request._url = httpx.URL(url_str.replace("/api/embed", "/embeddings"))
        
        if request.method == "POST" and "/chat/completions" in url_str:
            body = _json.loads(request.content)
            changed = False

            # Convert json_object → json_schema (LM Studio doesn't support json_object)
            rf = body.get("response_format", {})
            if isinstance(rf, dict) and rf.get("type") == "json_object":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": {"type": "object", "additionalProperties": True},
                        "strict": False
                    }
                }
                changed = True

            # Inject enable_thinking=false (prevents empty responses from reasoning models)
            extra = body.get("extra_body", {})
            tmpl = extra.get("chat_template_kwargs", {})
            if "enable_thinking" not in tmpl:
                tmpl["enable_thinking"] = False
                extra["chat_template_kwargs"] = tmpl
                body["extra_body"] = extra
                changed = True

            # Ensure max_tokens is high enough
            if "max_tokens" not in body or body.get("max_tokens", 0) < 1024:
                body["max_tokens"] = 8192
                changed = True

            if changed:
                new_body = _json.dumps(body).encode()
                request._content = new_body
                request.headers["content-length"] = str(len(new_body))
    except Exception:
        pass
    return await _original_send(self, request, **kwargs)

httpx.AsyncClient.send = _patched_send
PATCHEOF
    # Patch openai.py directly: fix embedding URL + response_format
    OPENAI_PY="/app/lightrag/llm/openai.py"
    if [ -f "$OPENAI_PY" ]; then
        echo "[entrypoint] Patching $OPENAI_PY..."
        
        # 1. Fix embedding endpoint: LightRAG sends to /api/embed, LM Studio wants /embeddings
        sed -i 's|/api/embed|/embeddings|g' "$OPENAI_PY"
        
        # 2. Strip response_format json_object (LM Studio doesn't support it)
        sed -i 's|kwargs\["response_format"\] = {"type": "json_object"}|kwargs["response_format"] = None|g' "$OPENAI_PY"
        
        # 3. Add enable_thinking=False and max_tokens before the API call
        if ! grep -q "enable_thinking" "$OPENAI_PY"; then
            sed -i '/openai_async_client\.chat\.completions\.create/i\        kwargs.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = False\n        kwargs.setdefault("max_tokens", 8192)' "$OPENAI_PY"
        fi
        
        echo "[entrypoint] openai.py patched (embedding URL + response_format + thinking)"
    fi
fi

echo "[entrypoint] Starting LightRAG server..."
exec lightrag-server --host 0.0.0.0 --port 9621
