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
                this.updateControlStatus()
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
            if (data.services.qdrant.status === 'online') {
                qdrantCard.className = 'status-indicator online';
                qdrantDetails.innerHTML = `<strong>URL:</strong> ${data.services.qdrant.url}<br><strong>Tickets:</strong> ${data.services.qdrant.points_count || 0}`;
            } else {
                qdrantCard.className = 'status-indicator offline';
                qdrantDetails.innerHTML = `<strong>Fehler:</strong> ${data.services.qdrant.error}`;
            }

            // Ollama
            const ollamaCard = document.getElementById('ollama-status');
            const ollamaDetails = document.getElementById('ollama-details');
            if (data.services.ollama.status === 'online') {
                ollamaCard.className = 'status-indicator online';
                ollamaDetails.innerHTML = `<strong>Model:</strong> ${data.services.ollama.current_model}<br><strong>Verfügbar:</strong> ${data.services.ollama.available_models?.length || 0} Modelle`;
            } else {
                ollamaCard.className = 'status-indicator offline';
                ollamaDetails.innerHTML = `<strong>Fehler:</strong> ${data.services.ollama.error}`;
            }

            // Zammad
            const zammadCard = document.getElementById('zammad-status');
            const zammadDetails = document.getElementById('zammad-details');
            if (data.services.zammad.status === 'online') {
                zammadCard.className = 'status-indicator online';
                zammadDetails.innerHTML = `<strong>URL:</strong> ${data.services.zammad.url}<br><strong>User:</strong> ${data.services.zammad.user}`;
            } else {
                zammadCard.className = 'status-indicator offline';
                zammadDetails.innerHTML = `<strong>Fehler:</strong> ${data.services.zammad.error}`;
            }

            // System Stats
            if (data.system && !data.system.error) {
                document.getElementById('cpu-usage').textContent = `${data.system.cpu_percent}%`;
                document.getElementById('memory-usage').textContent = `${data.system.memory_percent}%`;
                document.getElementById('disk-usage').textContent = `${data.system.disk_percent}%`;
            }
        } catch (error) {
            console.error('Status Update Error:', error);
        }
    }

    async updateActivities() {
        try {
            const response = await fetch('/api/activities');
            if (handleApiError(response)) return;
            const activities = await response.json();
            const activityList = document.getElementById('activity-list');
            
            if (activities.error) {
                activityList.innerHTML = `<div class="activity-item">Fehler: ${activities.error}</div>`;
                return;
            }
            if (Array.isArray(activities) && activities.length > 0) {
                activityList.innerHTML = activities.map(activity => `
                    <div class="activity-item">
                        <span class="activity-title">Ticket #${activity.ticket_id}: ${activity.title}</span>
                        <span class="activity-time">${new Date(activity.processed_at).toLocaleString('de-DE')}</span>
                    </div>`).join('');
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
            statusIndicator.classList.add(data.status); // Add new status class

            if (data.status === 'online') {
                statusText.textContent = 'Läuft';
            } else if (data.status === 'offline') {
                statusText.textContent = 'Gestoppt';
            } else {
                statusText.textContent = 'Unbekannt';
            }
        } catch (error) {
            console.error('Control Status Update Error:', error);
            document.querySelector('#poller-status .status-text').textContent = 'Fehler';
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
            alert(`✅ Aktion '${action}' erfolgreich.\nNachricht: ${result.message}`);
        } else {
            alert(`❌ Fehler bei Aktion '${action}'.\nGrund: ${result.detail || result.message}`);
        }
    } catch (error) {
        alert(`❌ Netzwerkfehler: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = originalText;
        // Kurze Verzögerung, um dem Service Zeit zu geben, seinen Status zu aktualisieren
        setTimeout(() => dashboard.updateDashboard(), 1500);
    }
}

// Initialisiert das Dashboard, wenn das DOM geladen ist.
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
