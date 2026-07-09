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
        if request.method == "POST" and "/chat/completions" in str(request.url):
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
    export PYTHONPATH="/app:${PYTHONPATH:-}"
    echo "[entrypoint] LM Studio patch installed"
fi

echo "[entrypoint] Starting LightRAG server..."
exec lightrag-server --host 0.0.0.0 --port 9621
