# 🎯 Zammad Qdrant Interface - Live Demo

A **1:1 identical** interactive demo of the Zammad Qdrant RAG system for GitHub Pages.

## ✨ Features

This demo shows **all functions** of the original system in functional form:

### 🏠 **Dashboard (index.html)**
- **Live Status Cards**: Transfer status, BM25 vocabulary, Qdrant collections, search performance
- **Transfer Control Panel**: 
  - BM25 cache on/off
  - Configure minimum age
  - Set start date
  - Live transfer simulation with progress bar
- **Hybrid Search Interface**:
  - Intelligent search with demo results
  - Configurable Top K Chunks/Tickets
  - Realistic search results with popup details
- **Scheduled Transfers**:
  - Create schedules (hourly/daily/weekly)
  - Start/stop scheduler
  - Manage active schedules
- **BM25 Statistics**: Live updates with demo data
- **MCP Server Control**: Server status and management

### ⚙️ **Settings (settings.html)**
- **Zammad Configuration**: Demo URLs and tokens
- **Qdrant Configuration**: Connection status simulation
- **System Parameters**: BM25 cache, default intervals
- **Maintenance & Administration**:
  - Clear BM25 cache (with animation)
  - Reset Qdrant collection
- **Live Connection Status**: Green/circle indicators

### 🧠 **AI Settings (ai-settings.html)**
- **Ollama Server Configuration**:
  - URL and model selection
  - Connection test with demo response
  - Model refresh function
- **AI Features Enable**:
  - Automatic ticket processing
  - Service status with live indicators
  - Check interval (30s-1h)
  - Max. ticket age configurable
  - Top K/Tickets parameters
- **Service Control**:
  - Start/Stop buttons with status updates
  - Processed tickets counter
  - Thread status display
- **Prompt Configuration**:
  - RAG search term prompt
  - Zammad note prompt
  - Variable support

## 🎨 UI/UX Features

### Multi-language
- **German/English** toggle
- Persistent language selection (localStorage)
- All texts fully translated
- Automatic weekday translation

### Responsive Design
- **Mobile-first** design
- Fully responsive on all devices
- Touch-optimized controls
- Adaptive layouts

### Animations & Interactivity
- **Live status updates** every 3 seconds
- **Realistic progress bars**
- **Smooth transitions** for all UI changes
- **Loading states** with spinner animations
- **Success/Error notifications**

### Demo Interactions
- **Simulated API responses** for realistic experience
- **Live counter updates** (processed tickets)
- **Randomized demo data** for authentic representation
- **Popup modals** for ticket details
- **Form validation** with demo feedback

## 🚀 GitHub Pages Deployment

### 1. Repository Setup
```bash
# In your GitHub repository
git add demo/
git commit -m "Add complete Zammad Qdrant demo"
git push origin main
```

### 2. Enable GitHub Pages
1. **Repository Settings** open
2. **Pages** select
3. **Source**: Deploy from branch
4. **Branch**: main → / (root)
5. **Folder**: / (root) or /demo

### 3. Demo Access
After activation, the demo is available at:
```
https://username.github.io/repository-name/
```

For demo folder:
```
https://username.github.io/repository-name/demo/
```

## 📂 File Structure

```
demo/
├── index.html          # Main dashboard
├── settings.html       # System settings
├── ai-settings.html    # AI configuration
├── style.css          # Additional styles
├── lang.js            # Multi-language handler
└── README-Demo.md     # This documentation
```

## 🔧 Technical Details

### Client-side Simulation
- **No backend dependencies**
- **localStorage** for persistent data
- **setInterval** for live updates
- **CSS/JavaScript** for all interactions

### Framework Integration
- **TailwindCSS** for responsive layouts
- **Lucide Icons** for consistent icons
- **Alpine.js** for reactive components
- **HTMX-ready** for backend integration

### Browser Compatibility
- **Modern browsers** (Chrome, Firefox, Safari, Edge)
- **ES6+** JavaScript
- **CSS Grid** & **Flexbox**
- **Progressive Enhancement**

## 🎯 Authentic Demo Experience

### Realistic Data
- **BM25 Vocabulary**: 1,247 terms
- **Documents**: 8,432 tickets
- **Collections**: 2 active
- **Search Results**: Contextual demo tickets

### Interactive Simulations
- **Transfer Progress**: 0-100% with animation
- **Live Logs**: Timestamp + Level + Message
- **Connection Status**: Real-time status updates
- **Service Statistics**: Live counter updates

### Responsive Functionality
- **Cross-page navigation**: Fully linked
- **Form state**: Persisted between pages
- **Error handling**: Authentic error messages
- **Success feedback**: Realistic success states

## 📱 Mobile Experience

### Touch Optimized
- **Large touch targets** (min. 44px)
- **Swipe gestures** for navigation
- **Responsive tables** for demo results
- **Mobile-first** layout approach

### Performance
- **Optimized assets** for fast loading
- **Lazy loading** for large content
- **Efficient DOM** updates
- **Minimal JavaScript** for better performance

---

**🎉 This demo is a complete, functional 1:1 copy of the original system and is perfect for presentations, demos and GitHub Pages hosting!**