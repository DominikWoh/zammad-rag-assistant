// Lightweight UI language handler (DE/EN) persisted via localStorage
(function () {
  let currentLang = 'DE';
  const de2en = new Map([
    ['Einstellungen', 'Settings'],
    ['KI-Features', 'AI features'],
    ['Automatische Ticket-Verarbeitung mit Ollama und RAG', 'Automatic ticket processing with Ollama and RAG'],
    ['KI-Einstellungen', 'AI settings'],
    ['Zammad Konfiguration', 'Zammad configuration'],
    ['API-Verbindung zu Zammad', 'API connection to Zammad'],
    ['Zammad API Token', 'Zammad API token'],
    ['Status wird geprüft...', 'Checking status...'],
    ['Qdrant Konfiguration', 'Qdrant configuration'],
    ['Vector Database Verbindung', 'Vector database connection'],
    ['Qdrant API Token', 'Qdrant API token'],
    ['Wartung & Administration', 'Maintenance & administration'],
    ['System-Wartung und Datenverwaltung', 'System maintenance and data management'],
    ['BM25 Cache löschen', 'Clear BM25 cache'],
    ['Entfernt alle BM25-Cache-Dateien und zwingt zur Neuberechnung', 'Removes all BM25 cache files and forces recomputation'],
    ['Qdrant Collection zurücksetzen', 'Reset Qdrant collection'],
    ['Einstellungen speichern', 'Save settings'],
    ['Speichern', 'Save'],
    ['Verbunden', 'Connected'],
    ['Alle Konfigurationen persistent speichern', 'Persist all configurations'],
    ['Collection zurücksetzen', 'Reset collection'],
    ['Löscht alle Daten und legt eine neue Collection an', 'Deletes all data and creates a new collection'],

    // Dashboard
    ['Transfer Status', 'Transfer status'],
    ['Bereit', 'Ready'],
    ['Bereit für Transfer', 'Ready for transfer'],
    ['BM25 Vocabulary', 'BM25 vocabulary'],
    ['BM25 Cache verwenden', 'Use BM25 cache'],
    ['Aktiviert schnellen Modus mit vorhandenem BM25-Index', 'Enables fast mode with existing BM25 index'],
    ['Mindestalter für geschlossene Tickets (Tage)', 'Minimum age for closed tickets (days)'],
    ['Tickets indexieren ab Datum', 'Index tickets from date'],
    ['Hybrid Search', 'Hybrid search'],
    ['Top Tickets', 'Top tickets'],
    ['Hybrid Search ausführen', 'Run hybrid search'],
    ['Suchergebnisse', 'Search results'],
    ['Keine Suchergebnisse', 'No search results'],
    ['Gestoppt', 'Stopped'],
    ['● Gestoppt', '● Stopped'],
    ['Läuft', 'Running'],
    ['● Läuft', '● Running'],
    ['Server starten', 'Start server'],
    ['Server stoppen', 'Stop server'],
    ['Service starten', 'Start service'],
    ['Service stoppen', 'Stop service'],
    ['Aktivieren', 'Enable'],
    ['Deaktivieren', 'Disable'],
    ['Aktiviert', 'Enabled'],
    ['Deaktiviert', 'Disabled'],
    ['Stündlich', 'Hourly'],
    ['Täglich', 'Daily'],
    ['Wöchentlich', 'Weekly'],
    ['BM25 Statistiken', 'BM25 statistics'],
    ['Server Fehler', 'Server error'],
    ['Unbekannt', 'Unknown'],
    ['Fehler beim Laden der Logs', 'Error loading logs'],
    ['Fehler beim Starten des Transfers', 'Error starting transfer'],
    ['Fehler beim Stoppen des Transfers', 'Error stopping transfer'],
    ['Aktualisieren', 'Refresh'],
    ['Fehler bei der Suche', 'Error during search'],
    ['Zeitgesteuerte Transfers', 'Scheduled transfers'],
    ['Neuen Zeitplan erstellen', 'Create new schedule'],
    ['Zeitplan erstellen', 'Create schedule'],
    ['Aktive Zeitpläne', 'Active schedules'],
    ['Keine Zeitpläne erstellt', 'No schedules created'],
    ['Uhrzeit (HH:MM)', 'Time (HH:MM)'],
    ['Bitte geben Sie eine Uhrzeit ein.', 'Please enter a time.'],
    ['Bitte geben Sie einen Namen für den Zeitplan ein.', 'Please enter a name for the schedule.'],
    ['Nächster Lauf:', 'Next run:'],
    ['Letzter Lauf:', 'Last run:'],

    // AI Settings
    ['Status unbekannt', 'Status unknown'],
    ['Ollama-Konfiguration und Automatisierung', 'Ollama configuration and automation'],
    ['Ollama-Server', 'Ollama server'],
    ['Ollama URL', 'Ollama URL'],
    ['Ollama Model', 'Ollama model'],
    ['Verfügbare Model von Ollama laden', 'Load available models from Ollama'],
    ['Verbindung testen', 'Test connection'],
    ['Ermöglicht automatische Antworten auf neue Tickets ohne Antworten', 'Enables automatic replies to new tickets without answers'],
    ['KI-Service Status', 'AI service status'],
    ['Ticket-Check Intervall (Sekunden)', 'Ticket check interval (seconds)'],
    ['Wie oft soll nach neuen Tickets gesucht werden?', 'How often to look for new tickets?'],
    ['Maximales Ticket-Alter (Tage)', 'Maximum ticket age (days)'],
    ['Nur Tickets die jünger als X Tage sind werden verarbeitet', 'Only tickets younger than X days are processed'],
    ['Top K (Vektor-Suchergebnisse)', 'Top K (vector results)'],
    ['Anzahl der Vektor-Suchergebnisse für die RAG-Suche', 'Number of vector results for the RAG search'],
    ['Top Tickets (relevante Tickets)', 'Top tickets (most relevant)'],
    ['Anzahl der relevantesten Tickets für die Antwortgenerierung', 'Number of most relevant tickets for response generation'],
    ['Verarbeitete Tickets:', 'Processed tickets:'],
    ['Aktuelles Intervall:', 'Current interval:'],
    ['KI-Einstellungen speichern', 'Save AI settings'],
    ['Fehler beim Laden der KI-Einstellungen', 'Error loading AI settings'],
    ['Keine Modelle verfügbar', 'No models available'],
    ['Verbindung wird getestet...', 'Testing connection...'],
    ['Verbindungsfehler:', 'Connection error:'],
    ['KI aktiviert', 'AI enabled'],
    ['KI deaktiviert', 'AI disabled'],
    ['KI-Einstellungen erfolgreich gespeichert!', 'AI settings saved successfully!'],
    ['Starte KI-Service...', 'Starting AI service...'],
    ['Stoppe KI-Service...', 'Stopping AI service...'],
    ['Fehler beim Speichern:', 'Error saving:'],
    ['Unbekannter Fehler', 'Unknown error'],
    ['KI-Service erfolgreich gestartet!', 'AI service started successfully!'],
    ['Fehler beim Starten des KI-Services:', 'Error starting AI service:'],
    ['KI-Service erfolgreich gestoppt!', 'AI service stopped successfully!'],
    ['Fehler beim Stoppen des KI-Services:', 'Error stopping AI service:'],

    // Prompts
    ['Prompt-Konfiguration', 'Prompt configuration'],
    ['RAG-Suchbegriff-Prompt', 'RAG search term prompt'],
    ['Zammad-Notiz-Prompt', 'Zammad note prompt'],
    ['Verfügbare Variablen:', 'Available variables:'],
    ['Automatische Ticket-Bearbeitung aktivieren', 'Enable automatic ticket processing'],
  
    // Common
    ['Einstellungen - Zammad Qdrant Interface', 'Settings - Zammad Qdrant Interface'],
    ['KI-Einstellungen - Zammad Qdrant Interface', 'AI Settings - Zammad Qdrant Interface']
  ]);

  // Reverse map for EN -> DE when toggling back
  const en2de = new Map(Array.from(de2en.entries()).map(([de, en]) => [en, de]));

  const placeholderDe2En = new Map([
    ['Ihr API-Token', 'Your API token'],
    ['Ihr API-Token (optional)', 'Your API token (optional)'],
    ['Erstelle einen prägnanten Suchbegriff für die RAG-Suche basierend auf folgendem Ticket: {ticket_content}',
     'Create a concise search term for the RAG search based on the following ticket: {ticket_content}'],
    ['Erstelle eine hilfreiche und professionelle Antwort für folgendes Zammad-Ticket basierend auf den verfügbaren Informationen: {ticket_content}\n\nRelevante Informationen:\n{search_results}',
     'Create a helpful and professional reply for the following Zammad ticket based on the available information: {ticket_content}\n\nRelevant information:\n{search_results}']
  ]);
  const placeholderEn2De = new Map(Array.from(placeholderDe2En.entries()).map(([de, en]) => [en, de]));
  
  function fetchLanguage() {
    try {
      // Use localStorage for demo
      const stored = localStorage.getItem('ui-language');
      return (stored || 'DE').toUpperCase();
    } catch (e) {
      console.warn('UI language fetch failed, defaulting to DE', e);
      return 'DE';
    }
  }

  function saveLanguage(lang) {
    try {
      // Use localStorage for demo
      localStorage.setItem('ui-language', lang);
    } catch (e) {
      console.error('Failed to save language', e);
    }
  }

  function applyLanguage(lang) {
    try {
      currentLang = lang;
      const html = document.documentElement;
      html.setAttribute('lang', lang === 'DE' ? 'de' : 'en');
      const select = document.getElementById('ui-language-select');
      if (select && select.value !== lang) {
        select.value = lang;
      }
      
      // Translate document title
      const title = document.title;
      if (lang === 'EN' && de2en.has(title)) {
        document.title = de2en.get(title);
      } else if (lang === 'DE' && en2de.has(title)) {
        document.title = en2de.get(title);
      }
      
      // Translate text content
      const textNodesSelector = 'h1,h2,h3,h4,h5,h6,p,span,button,label,a,option,div';
      const nodes = document.querySelectorAll(textNodesSelector);

      function translateChunk(text) {
        const original = (text || '').trim();
        if (!original) return text;
        let replaced = original;
        if (lang === 'EN') {
          if (de2en.has(original)) {
            replaced = de2en.get(original);
          } else {
            replaced = replaced
              .replace(/^Keine Suchergebnisse für \"(.+)\" gefunden$/i, 'No search results for "$1" found')
              .replace(/^Täglich um (.+)$/i, 'Daily at $1')
              .replace(/^Wöchentlich \((.*)\) um (.+)$/i, 'Weekly ($1) at $2')
              .replace(/^Stündlich$/i, 'Hourly')
              .replace(/^Nächster Lauf: (.+)$/i, 'Next run: $1')
              .replace(/^Letzter Lauf: (.+)$/i, 'Last run: $1')
              .replace(/^Fehler beim Starten des MCP Servers: /i, 'Error starting the MCP server: ')
              .replace(/^Fehler beim Stoppen des MCP Servers: /i, 'Error stopping the MCP server: ')
              .replace(/^Fehler beim Starten des Schedulers: /i, 'Error starting the scheduler: ')
              .replace(/^Fehler beim Stoppen des Schedulers: /i, 'Error stopping the scheduler: ')
              .replace(/^Fehler beim Erstellen des Zeitplans: /i, 'Error creating the schedule: ')
              .replace(/^Fehler beim Umschalten des Zeitplans: /i, 'Error toggling the schedule: ')
              .replace(/^Fehler beim Löschen des Zeitplans: /i, 'Error deleting the schedule: ')
              .replace(/^● Server Fehler$/i, '● Server error');
          }
        } else {
          if (en2de.has(original)) {
            replaced = en2de.get(original);
          } else {
            replaced = replaced
              .replace(/^No search results for \"(.+)\" found$/i, 'Keine Suchergebnisse für "$1" gefunden')
              .replace(/^Daily at (.+)$/i, 'Täglich um $1')
              .replace(/^Weekly \((.*)\) at (.+)$/i, 'Wöchentlich ($1) um $2')
              .replace(/^Hourly$/i, 'Stündlich')
              .replace(/^Next run: (.+)$/i, 'Nächster Lauf: $1')
              .replace(/^Last run: (.+)$/i, 'Letzter Lauf: $1')
              .replace(/^Error starting the MCP server: /i, 'Fehler beim Starten des MCP Servers: ')
              .replace(/^Error stopping the MCP server: /i, 'Fehler beim Stoppen des MCP Servers: ')
              .replace(/^Error starting the scheduler: /i, 'Fehler beim Starten des Schedulers: ')
              .replace(/^Error stopping the scheduler: /i, 'Fehler beim Stoppen des Schedulers: ')
              .replace(/^Error creating the schedule: /i, 'Fehler beim Erstellen des Zeitplans: ')
              .replace(/^Error toggling the schedule: /i, 'Fehler beim Umschalten des Zeitplans: ')
              .replace(/^Error deleting the schedule: /i, 'Fehler beim Löschen des Zeitplans: ')
              .replace(/^● Server error$/i, '● Server Fehler');
          }
        }
        if (replaced === original) return text; // keep original including spacing
        
        // In EN, translate German day abbreviations inside schedule strings
        if (lang === 'EN') {
          replaced = replaced
            .replace(/\bMo\b/g, 'Mon')
            .replace(/\bDi\b/g, 'Tue')
            .replace(/\bMi\b/g, 'Wed')
            .replace(/\bDo\b/g, 'Thu')
            .replace(/\bFr\b/g, 'Fri')
            .replace(/\bSa\b/g, 'Sat')
            .replace(/\bSo\b/g, 'Sun');
        }
        // restore leading/trailing spaces
        const leading = text.startsWith(' ') ? ' ' : '';
        const trailing = text.endsWith(' ') ? ' ' : '';
        return leading + replaced + trailing;
      }

      nodes.forEach((el) => {
        if (el.childNodes && el.childNodes.length) {
          el.childNodes.forEach((node) => {
            if (node.nodeType === Node.TEXT_NODE) {
              const newText = translateChunk(node.nodeValue);
              if (newText !== node.nodeValue) node.nodeValue = newText;
            }
          });
        } else {
          const newText = translateChunk(el.textContent || '');
          if (newText !== (el.textContent || '')) el.textContent = newText;
        }
      });

      // Translate placeholders
      const inputs = document.querySelectorAll('input[placeholder], textarea[placeholder]');
      inputs.forEach((el) => {
        const ph = el.getAttribute('placeholder');
        if (!ph) return;
        if (lang === 'EN' && placeholderDe2En.has(ph)) {
          el.setAttribute('placeholder', placeholderDe2En.get(ph));
        } else if (lang === 'DE' && placeholderEn2De.has(ph)) {
          el.setAttribute('placeholder', placeholderEn2De.get(ph));
        }
      });
    } catch (e) {
      console.warn('Apply language failed', e);
    }
  }

  async function initLanguageDropdown() {
    const select = document.getElementById('ui-language-select');
    if (!select) return;
    const current = fetchLanguage();
    applyLanguage(current);
    select.addEventListener('change', async (e) => {
      const value = String(e.target.value || 'DE').toUpperCase();
      applyLanguage(value);
      saveLanguage(value);
      // Keep icons fresh if using Lucide
      if (window.lucide && typeof window.lucide.createIcons === 'function') {
        try { window.lucide.createIcons(); } catch (_) {}
      }
    });
  }

  // Expose init to be called after DOMContentLoaded from pages
  window.initLanguageDropdown = initLanguageDropdown;

  // Re-apply translations when DOM updates (for dynamic status text)
  const observer = new MutationObserver((mutations) => {
    // debounce via microtask
    if (observer._scheduled) return;
    observer._scheduled = true;
    Promise.resolve().then(() => {
      observer._scheduled = false;
      try { applyLanguage(currentLang); } catch (_) {}
    });
  });
  try {
    observer.observe(document.documentElement, { subtree: true, childList: true, characterData: true });
  } catch (_) {}
})();