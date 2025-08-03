// Hilfsfunktion für API-Fehler (z.B. abgelaufene Sitzung)
function handleApiError(response) {
    if (response.status === 401) {
        alert("Ihre Sitzung ist abgelaufen. Sie werden zum Login weitergeleitet.");
        window.location.href = '/login';
        return true; // Signalisiert, dass ein Fehler aufgetreten ist
    }
    return false;
}

class Dashboard {
    constructor() {
        this.updateInterval = null;
        this.init();
    }

    init() {
        // Die Prüfung auf ein Token entfällt. Der Server leitet bei Bedarf um.
        this.addLogoutButton();
        this.updateDashboard();
        this.startAutoUpdate();
    }

    addLogoutButton() {
        const header = document.querySelector('header');
        const actions = header?.querySelector('.header-actions') || header;
        if (actions && !document.getElementById('logout-btn')) {
            const logoutBtn = document.createElement('button');
            logoutBtn.id = 'logout-btn';
            logoutBtn.textContent = 'Abmelden';
            logoutBtn.className = 'btn-config';
            logoutBtn.onclick = this.logout;
            actions.appendChild(logoutBtn);
        }
    }

    async logout() {
        try {
            await fetch('/api/logout', { method: 'POST' });
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            // Die Seite wird einfach neu geladen und der Server leitet dann zum Login um.
            window.location.href = '/login';
        }
    }

    async updateDashboard() {
        try {
            await Promise.all([
                this.updateSystemStatus(),
                this.updateActivities(),
                this.updateControlStatus(),
                this.updateIngestStatus()
            ]);
            document.getElementById('lastUpdate').textContent = `Letztes Update: ${new Date().toLocaleString('de-DE')}`;
        } catch (error) {
            console.error('Dashboard Update Error:', error);
        }
    }

    async updateSystemStatus() {
        try {
            const response = await fetch('/api/status');
            if (handleApiError(response)) return;
            const data = await response.json();

            // Qdrant
            const qdrantCard = document.getElementById('qdrant-status');
            const qdrantDetails = document.getElementById('qdrant-details');
            const qStat = data?.services?.qdrant?.status || 'unknown';
            qdrantCard.className = 'status-indicator ' + (qStat === 'ok' ? 'online' : (qStat === 'unknown' ? '' : 'offline'));
            if (qStat === 'ok') {
                qdrantDetails.innerHTML = `<strong>Status:</strong> OK`;
            } else if (qStat === 'unknown') {
                qdrantDetails.innerHTML = `<strong>Hinweis:</strong> QDRANT_URL nicht gesetzt`;
            } else {
                const qErr = data?.services?.qdrant?.detail || data?.services?.qdrant?.error || 'unbekannt';
                qdrantDetails.innerHTML = `<strong>Fehler:</strong> ${qErr}`;
            }

            // Ollama
            const ollamaCard = document.getElementById('ollama-status');
            const ollamaDetails = document.getElementById('ollama-details');
            const oStat = data?.services?.ollama?.status || 'unknown';
            ollamaCard.className = 'status-indicator ' + (oStat === 'ok' ? 'online' : (oStat === 'unknown' ? '' : 'offline'));
            if (oStat === 'ok') {
                ollamaDetails.innerHTML = `<strong>Status:</strong> OK`;
            } else if (oStat === 'unknown') {
                ollamaDetails.innerHTML = `<strong>Hinweis:</strong> OLLAMA_URL nicht gesetzt`;
            } else {
                const oErr = data?.services?.ollama?.detail || data?.services?.ollama?.error || 'unbekannt';
                ollamaDetails.innerHTML = `<strong>Fehler:</strong> ${oErr}`;
            }

            // Zammad
            const zammadCard = document.getElementById('zammad-status');
            const zammadDetails = document.getElementById('zammad-details');
            const zStat = data?.services?.zammad?.status || 'unknown';
            zammadCard.className = 'status-indicator ' + (zStat === 'ok' ? 'online' : (zStat === 'unknown' ? '' : 'offline'));
            if (zStat === 'ok') {
                zammadDetails.innerHTML = `<strong>Status:</strong> OK`;
            } else if (zStat === 'unknown') {
                zammadDetails.innerHTML = `<strong>Hinweis:</strong> ZAMMAD_URL oder ZAMMAD_TOKEN fehlen`;
            } else {
                const zErr = data?.services?.zammad?.detail || data?.services?.zammad?.error || 'unbekannt';
                zammadDetails.innerHTML = `<strong>Fehler:</strong> ${zErr}`;
            }

            // System Stats (Backend liefert: { cpu, mem, disk })
            if (data.system) {
                const cpu = data.system.cpu ?? 'undefined';
                const mem = data.system.mem ?? 'undefined';
                const disk = data.system.disk ?? 'undefined';
                document.getElementById('cpu-usage').textContent = cpu !== null ? `${cpu}%` : 'undefined%';
                document.getElementById('memory-usage').textContent = mem !== null ? `${mem}%` : 'undefined%';
                document.getElementById('disk-usage').textContent = disk !== null ? `${disk}%` : 'undefined%';
            }
        } catch (error) {
            console.error('Status Update Error:', error);
        }
    }

    async updateActivities() {
        try {
            const response = await fetch('/api/activities');
            if (handleApiError(response)) return;
            let activities = await response.json();
            const activityList = document.getElementById('activity-list');

            if (activities.error) {
                activityList.innerHTML = `<div class="activity-item">Fehler: ${activities.error}</div>`;
                return;
            }

            // Sortierung: primär nach Ticket-ID (numerisch) absteigend, sekundär nach processed_at absteigend.
            if (Array.isArray(activities)) {
                // Normalisieren: processed_at und ticket_id numerisch extrahieren
                activities = activities.map(a => {
                    // processed_at -> number | null
                    let ts = a && a.processed_at;
                    if (typeof ts !== 'number') {
                        const n = Number(ts);
                        ts = Number.isFinite(n) ? n : null;
                    }
                    // ticket_id -> number | null
                    let tidNum = null;
                    const rawTid = a && a.ticket_id;
                    if (rawTid !== undefined && rawTid !== null && rawTid !== '') {
                        const t = Number(rawTid);
                        tidNum = Number.isFinite(t) ? t : null;
                    }
                    return { ...a, processed_at: ts, _tidNum: tidNum };
                });

                // Sortierfunktion:
                // 1) beide mit Ticket-ID -> absteigend nach _tidNum
                // 2) nur einer mit Ticket-ID -> dieser zuerst
                // 3) keiner mit Ticket-ID -> absteigend nach processed_at
                activities.sort((a, b) => {
                    const aHasTid = Number.isFinite(a._tidNum);
                    const bHasTid = Number.isFinite(b._tidNum);
                    if (aHasTid && bHasTid) {
                        if (b._tidNum !== a._tidNum) return b._tidNum - a._tidNum;
                        // sekundär: Zeit
                        const at = Number.isFinite(a.processed_at) ? a.processed_at : -Infinity;
                        const bt = Number.isFinite(b.processed_at) ? b.processed_at : -Infinity;
                        return bt - at;
                    }
                    if (aHasTid && !bHasTid) return -1;
                    if (!aHasTid && bHasTid) return 1;
                    // beide ohne Ticket-ID: nach Zeit
                    const at = Number.isFinite(a.processed_at) ? a.processed_at : -Infinity;
                    const bt = Number.isFinite(b.processed_at) ? b.processed_at : -Infinity;
                    return bt - at;
                });

                // Nur 5 oberste Einträge anzeigen
                activities = activities.slice(0, 5);
            }

            if (Array.isArray(activities) && activities.length > 0) {
                const html = activities.map(a => {
                    const tsDisplay = (typeof a.processed_at === 'number' && Number.isFinite(a.processed_at))
                        ? new Date(a.processed_at * 1000).toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                        : '';
                    const kind = a.kind || 'activity';
                    const ticketId = (a.ticket_id !== undefined && a.ticket_id !== null && a.ticket_id !== '')
                        ? `#${a.ticket_id}`
                        : '';
                    const titleText = (a.title || a.message || '').toString().trim();

                    // Einheitliches Layout: Ticket-ID und Titel links, Zeit ganz rechts
                    const line = (() => {
                        if (kind === 'ingest' || kind === 'qdrant_point') {
                            return `${ticketId} ${titleText}`.trim();
                        }
                        if (kind === 'ingest_skip') {
                            const reason = a.reason || 'skip';
                            return `${ticketId} übersprungen – ${reason}`.trim();
                        }
                        if (kind === 'ingest_summary') {
                            const count = a.count ?? '';
                            return `Ingest abgeschlossen (${count} Tickets)`;
                        }
                        if (kind === 'ingest_batch') {
                            const msg = a.message || 'Batch-Import';
                            const dur = (a.duration_s !== undefined) ? ` (${a.duration_s}s)` : '';
                            return `${msg}${dur}`;
                        }
                        return `${ticketId} ${titleText || JSON.stringify(a)}`.trim();
                    })();

                    return `
                        <div class="activity-item" style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
                            <span class="activity-title" style="flex:1 1 auto; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${line}</span>
                            <span class="activity-time" style="flex:0 0 auto; color:#a1a5ad;">${tsDisplay}</span>
                        </div>`;
                }).join('');
                activityList.innerHTML = html;
            } else {
                activityList.innerHTML = '<div class="activity-item">Keine aktuellen Aktivitäten verfügbar.</div>';
            }
        } catch (error) {
            console.error('Activities Update Error:', error);
            document.getElementById('activity-list').innerHTML = '<div class="activity-item">Fehler beim Laden der Aktivitäten.</div>';
        }
    }

    async updateControlStatus() {
        try {
            const response = await fetch('/api/services/zammad_rag_poller/status');
            if (handleApiError(response)) return;
            const data = await response.json();
            const statusIndicator = document.getElementById('poller-status');
            const statusText = statusIndicator.querySelector('.status-text');
            statusIndicator.className = 'service-status-indicator'; // Reset

            // Backend liefert { status: "running" | "stopped", detail: {...} }
            const pStat = (data && data.status) || 'unknown';
            if (pStat === 'running') {
                statusIndicator.classList.add('online');
                statusText.textContent = 'Läuft';
            } else if (pStat === 'stopped') {
                statusIndicator.classList.add('offline');
                statusText.textContent = 'Gestoppt';
            } else {
                statusText.textContent = 'Unbekannt';
            }
        } catch (error) {
            console.error('Control Status Update Error:', error);
            document.querySelector('#poller-status .status-text').textContent = 'Fehler';
        }
    }

    async updateIngestStatus() {
        try {
            const res = await fetch('/api/ingest/status');
            if (handleApiError(res)) return;
            const data = await res.json();

            const statusIndicator = document.getElementById('ingest-status');
            const statusText = statusIndicator?.querySelector('.status-text');
            const lastMsg = document.getElementById('ingest-lastmsg');

            if (!statusIndicator || !statusText || !lastMsg) return;

            statusIndicator.className = 'service-status-indicator';
            if (data.running) {
                statusIndicator.classList.add('online');
                statusText.textContent = 'Läuft';
            } else if (data.error) {
                statusIndicator.classList.add('offline');
                statusText.textContent = 'Fehler';
            } else {
                statusText.textContent = 'Gestoppt';
            }

            const started = data.started_at ? new Date(data.started_at * 1000).toLocaleString('de-DE') : '-';
            const finished = data.finished_at ? new Date(data.finished_at * 1000).toLocaleString('de-DE') : '-';
            const lastSucc = data.last_success ? new Date(data.last_success * 1000).toLocaleString('de-DE') : '-';
            const msg = data.last_message || '';
            const err = data.error ? `Fehler: ${data.error}` : '';
            lastMsg.innerHTML = `
                <div><strong>Letzte Meldung:</strong> ${msg || '–'}</div>
                <div><strong>Gestartet:</strong> ${started}</div>
                <div><strong>Beendet:</strong> ${finished}</div>
                <div><strong>Letzter Erfolg:</strong> ${lastSucc}</div>
                ${err ? `<div style="color:#f88;"><strong>${err}</strong></div>` : ''}
            `;
        } catch (e) {
            console.error('Ingest Status Update Error:', e);
        }
    }

    startAutoUpdate() {
        this.updateInterval = setInterval(() => this.updateDashboard(), 30000);
    }

    stopAutoUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

async function controlService(serviceName, action) {
    const button = event.target;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '⏳ Verarbeite...';
    try {
        const response = await fetch(`/api/services/${serviceName}/control?action=${action}`, { method: 'POST' });
        if (handleApiError(response)) return;
        const result = await response.json();
        if (response.ok && result.success) {
            alert(`✅ Aktion '${action}' erfolgreich.\nNachricht: ${result.message || ''}`);
        } else {
            alert(`❌ Fehler bei Aktion '${action}'.\nGrund: ${result.detail || result.message || ''}`);
        }
    } catch (error) {
        alert(`❌ Netzwerkfehler: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = originalText;
        setTimeout(() => dashboard.updateDashboard(), 1500);
    }
}

// Ingest-Start Button verdrahten
document.addEventListener('DOMContentLoaded', () => {
    const ingestBtn = document.getElementById('ingest-start-btn');
    if (ingestBtn) {
        ingestBtn.addEventListener('click', async () => {
            const original = ingestBtn.textContent;
            ingestBtn.disabled = true;
            ingestBtn.textContent = '⏳ Starte...';
            try {
                const res = await fetch('/api/ingest/start', { method: 'POST' });
                if (handleApiError(res)) return;
                const data = await res.json();
                if (res.ok && data.success) {
                    alert('✅ Batch-Import gestartet.');
                } else {
                    alert('❌ Start fehlgeschlagen: ' + (data.detail || ''));
                }
            } catch (e) {
                alert('❌ Netzwerkfehler: ' + e.message);
            } finally {
                ingestBtn.disabled = false;
                ingestBtn.textContent = original;
                setTimeout(() => dashboard.updateIngestStatus(), 1500);
            }
        });
    }
});

// Initialisiert das Dashboard, wenn das DOM geladen ist.
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
