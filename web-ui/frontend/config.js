document.addEventListener("DOMContentLoaded", () => {
    // Hilfsfunktion für API-Fehler (z.B. abgelaufene Sitzung)
    function handleApiError(response) {
        if (response.status === 401) {
            alert("Ihre Sitzung ist abgelaufen. Sie werden zum Login weitergeleitet.");
            window.location.href = '/login';
            return true; // Signalisiert, dass ein Fehler aufgetreten ist
        }
        return false;
    }

    const form = document.getElementById("config-form");
    const saveResult = document.getElementById("save-result");
    const zammadTestResult = document.getElementById("zammad-test-result");
    let config = {};

    async function loadConfig() {
        try {
            const res = await fetch("/api/config");
            if (handleApiError(res)) return;
            config = await res.json();
            
            // Vollständige Ollama URL verwenden
            document.getElementById("zammad-url").value = config.ZAMMAD_URL || "";
            document.getElementById("zammad-token").value = config.ZAMMAD_TOKEN || "";
            document.getElementById("ollama-url").value = config.OLLAMA_URL || "";
            document.getElementById("qdrant-collection").value = config.COLLECTION_NAME || "";
            document.getElementById("qdrant-api-key").value = config.QDRANT_API_KEY || "";
            document.getElementById("min-closed-days").value = config.MIN_CLOSED_DAYS || "";
            document.getElementById("min-date").value = config.MIN_TICKET_DATE || "";

            // Neue KI-Flags mit Defaults (ASKKI=false, RAG=true)
            const askki = String(config.ENABLE_ASKKI || "").toLowerCase();
            const rag = String(config.ENABLE_RAG_NOTE || "").toLowerCase();
            document.getElementById("enable-askki").checked = askki === "true" ? true : false; // Default false
            document.getElementById("enable-rag-note").checked = rag === "false" ? false : true; // Default true
            
            await loadOllamaModels();
        } catch (error) {
            console.error("Fehler beim Laden der Konfiguration:", error);
            saveResult.textContent = "❌ Fehler beim Laden der Konfiguration.";
        }
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        saveResult.textContent = "Speichern...";
        
        const selectedModel = document.getElementById("model-selection").value;
        
        const data = {
            ZAMMAD_URL: document.getElementById("zammad-url").value,
            ZAMMAD_TOKEN: document.getElementById("zammad-token").value,
            OLLAMA_URL: document.getElementById("ollama-url").value,
            OLLAMA_MODEL: selectedModel, // Ausgewähltes Modell verwenden
            COLLECTION_NAME: document.getElementById("qdrant-collection").value,
            QDRANT_API_KEY: document.getElementById("qdrant-api-key").value,
            MIN_CLOSED_DAYS: document.getElementById("min-closed-days").value,
            MIN_TICKET_DATE: document.getElementById("min-date").value,
            ENABLE_ASKKI: document.getElementById("enable-askki").checked ? "true" : "false",
            ENABLE_RAG_NOTE: document.getElementById("enable-rag-note").checked ? "true" : "false",
        };

        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            if (handleApiError(res)) return;

            if (res.ok) {
                saveResult.textContent = "✅ Konfiguration erfolgreich gespeichert.";
            } else {
                saveResult.textContent = `❌ Fehler beim Speichern (${res.status}).`;
            }
        } catch (error) {
            console.error("Fehler beim Speichern:", error);
            saveResult.textContent = "❌ Fehler beim Speichern.";
        }
    });

    async function loadOllamaModels() {
        try {
            const res = await fetch("/api/ollama/models");
            if (handleApiError(res)) return;
            const data = await res.json();
            
            const modelSelect = document.getElementById("model-selection");
            modelSelect.innerHTML = '<option value="">Modell auswählen...</option>';
            
            if (data.success && data.models.length > 0) {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
                
                // Aktuelles Modell aus der Konfiguration auswählen
                if (config.OLLAMA_MODEL) {
                    modelSelect.value = config.OLLAMA_MODEL;
                }
            } else if (!data.success) {
                showModelResult(`Fehler beim Laden der Modelle: ${data.error}`, "error");
            }
        } catch (error) {
            console.error("Fehler beim Laden der Modelle:", error);
            showModelResult("Fehler beim Laden der Modelle", "error");
        }
    }

    async function pullModel() {
        const modelName = document.getElementById("new-model-input").value.trim();
        if (!modelName) {
            showModelResult("Bitte geben Sie einen Modellnamen ein", "error");
            return;
        }

        const pullBtn = document.getElementById("pull-model-btn");
        const originalText = pullBtn.textContent;
        pullBtn.disabled = true;
        pullBtn.textContent = "⏳ Lade herunter...";

        try {
            const res = await fetch("/api/ollama/pull", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: modelName })
            });

            if (handleApiError(res)) return;

            const result = await res.json();
            if (result.success) {
                showModelResult(`Modell ${modelName} erfolgreich heruntergeladen`, "success");
                document.getElementById("new-model-input").value = "";
                await loadOllamaModels();
            } else {
                showModelResult(`Fehler: ${result.error}`, "error");
            }
        } catch (error) {
            showModelResult(`Netzwerkfehler: ${error.message}`, "error");
        } finally {
            pullBtn.disabled = false;
            pullBtn.textContent = originalText;
        }
    }

    async function deleteModel() {
        const selectedModel = document.getElementById("model-selection").value;
        if (!selectedModel) {
            showModelResult("Bitte wählen Sie ein Modell zum Löschen aus", "error");
            return;
        }

        if (!confirm(`Sind Sie sicher, dass Sie das Modell "${selectedModel}" löschen möchten?`)) return;

        const deleteBtn = document.getElementById("delete-model-btn");
        const originalText = deleteBtn.textContent;
        deleteBtn.disabled = true;
        deleteBtn.textContent = "⏳ Lösche...";

        try {
            const res = await fetch(`/api/ollama/models/${encodeURIComponent(selectedModel)}`, {
                method: "DELETE"
            });

            if (handleApiError(res)) return;

            const result = await res.json();
            if (result.success) {
                showModelResult(`Modell ${selectedModel} erfolgreich gelöscht`, "success");
                await loadOllamaModels();
            } else {
                showModelResult(`Fehler: ${result.error}`, "error");
            }
        } catch (error) {
            showModelResult(`Netzwerkfehler: ${error.message}`, "error");
        } finally {
            deleteBtn.disabled = false;
            deleteBtn.textContent = originalText;
        }
    }

    function showModelResult(message, type) {
        const resultDiv = document.getElementById("model-result");
        resultDiv.textContent = message;
        resultDiv.className = type;
        resultDiv.style.display = "block";
        setTimeout(() => resultDiv.style.display = "none", 5000);
    }

    function addCopyToClipboard(buttonId, inputId) {
        const copyButton = document.getElementById(buttonId);
        const inputElement = document.getElementById(inputId);

        if (copyButton && inputElement) {
            copyButton.addEventListener("click", () => {
                if (inputElement.value) {
                    navigator.clipboard.writeText(inputElement.value).then(() => {
                        const originalTitle = copyButton.title;
                        copyButton.title = "Kopiert!";
                        setTimeout(() => copyButton.title = originalTitle, 2000);
                    }).catch(err => console.error("Fehler beim Kopieren: ", err));
                }
            });
        }
    }

    document.getElementById("test-zammad-btn").addEventListener("click", async () => {
        // Hinweis: Direkte Fetches zum Zammad-Server von der Config-Seite schlagen
        // im Browser oft wegen CORS/Netzwerk (Mixed-Content, andere Origin) fehl.
        // Deshalb testen wir die Verbindung serverseitig über unseren Backend-Proxy.
        zammadTestResult.textContent = "Teste Verbindung...";
        const url = document.getElementById("zammad-url").value.trim();
        const token = document.getElementById("zammad-token").value.trim();

        try {
            const res = await fetch("/api/zammad/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url, token })
            });

            if (handleApiError(res)) return;

            const result = await res.json();
            if (res.ok && result.success) {
                zammadTestResult.textContent = `✅ Verbindung erfolgreich: Nutzer ${result.user}`;
            } else {
                zammadTestResult.textContent = `❌ Fehler: ${result.error || "Unbekannter Fehler"}`;
            }
        } catch (error) {
            zammadTestResult.textContent = `❌ Verbindungsfehler: ${error.message}`;
        }
    });
    
    // Qdrant Verbindung testen (serverseitig über Backend)
    const testQdrantBtn = document.getElementById("test-qdrant-btn");
    if (testQdrantBtn) {
        testQdrantBtn.addEventListener("click", async () => {
            const resultEl = document.getElementById("qdrant-test-result");
            resultEl.textContent = "Teste Qdrant Verbindung...";
            const collection = document.getElementById("qdrant-collection").value.trim();
            const apiKey = document.getElementById("qdrant-api-key").value.trim();
            try {
                const res = await fetch("/api/qdrant/test", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ collection, api_key: apiKey })
                });
                if (handleApiError(res)) return;
                const data = await res.json();
                if (res.ok && data.success) {
                    resultEl.textContent = `✅ Verbindung erfolgreich: Punkte in "${data.collection}": ${data.points_count}`;
                } else {
                    resultEl.textContent = `❌ Fehler: ${data.error || "Unbekannter Fehler"}`;
                }
            } catch (error) {
                resultEl.textContent = `❌ Verbindungsfehler: ${error.message}`;
            }
        });
    }
    
    addCopyToClipboard("copy-zammad-token-btn", "zammad-token");
    addCopyToClipboard("copy-qdrant-key-btn", "qdrant-api-key");
    
    document.getElementById("refresh-models-btn").addEventListener("click", loadOllamaModels);
    document.getElementById("pull-model-btn").addEventListener("click", pullModel);
    document.getElementById("delete-model-btn").addEventListener("click", deleteModel);
    
    document.getElementById("model-selection").addEventListener("change", (e) => {
        document.getElementById("delete-model-btn").disabled = !e.target.value;
        
        // Direkt in die Konfiguration schreiben, wenn ein Modell ausgewählt wird
        if (e.target.value) {
            config.OLLAMA_MODEL = e.target.value;
            saveResult.textContent = "⚠️ Modell ausgewählt - bitte 'Speichern' klicken um zu übernehmen.";
        }
    });
    
    loadConfig();
});
