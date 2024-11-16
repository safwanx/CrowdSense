// Dashboard.js - CrowdSense Dashboard Functionality

document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startRealTimeUpdates();
});

// Dashboard Initialization
function initializeDashboard() {
    initializeCharts();
    loadInitialData();
    setupWebSocket();
}

// Charts Initialization
function initializeCharts() {
    // Density Trend Chart
    const densityCtx = document.getElementById('densityTrend')?.getContext('2d');
    if (densityCtx) {
        new Chart(densityCtx, {
            type: 'line',
            data: {
                labels: ['3 PM', '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM'],
                datasets: [{
                    label: 'Today',
                    data: [16, 5, 13, 28, 16, 10, 8],
                    borderColor: '#4F46E5',
                    tension: 0.4
                }, {
                    label: 'Average',
                    data: [8, 10, 7, 12, 8, 9, 8],
                    borderColor: '#06B6D4',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// WebSocket Connection
function setupWebSocket() {
    const ws = new WebSocket('wss://your-websocket-server');
    
    ws.onopen = () => {
        console.log('Connected to real-time updates');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showNotification('Connection error', 'error');
    };
}

// Real-time Updates
function startRealTimeUpdates() {
    // Simulate real-time updates for demo
    setInterval(() => {
        updateStats({
            totalCount: Math.floor(12000 + Math.random() * 1000),
            activeIncidents: Math.floor(10 + Math.random() * 10),
            currentAlerts: Math.floor(3 + Math.random() * 5),
            averageDensity: (3 + Math.random()).toFixed(1)
        });
    }, 5000);
}

// Update Dashboard Elements
function updateDashboard(data) {
    updateStats(data.stats);
    updateDensityMap(data.densityData);
    updateAlerts(data.alerts);
    updateRecommendations(data.recommendations);
}

function updateStats(stats) {
    const elements = {
        totalCount: document.querySelector('[data-stat="total-count"]'),
        activeIncidents: document.querySelector('[data-stat="active-incidents"]'),
        currentAlerts: document.querySelector('[data-stat="current-alerts"]'),
        averageDensity: document.querySelector('[data-stat="average-density"]')
    };

    for (const [key, element] of Object.entries(elements)) {
        if (element && stats[key] !== undefined) {
            element.textContent = stats[key];
            updateTrend(element.parentElement, stats[key + 'Trend']);
        }
    }
}

function updateDensityMap(densityData) {
    // Update density visualization
    const densityMap = document.querySelector('.density-visualization');
    if (densityMap && densityData) {
        // Update map visualization based on data
        // This would integrate with your chosen visualization library
    }
}

function updateAlerts(alerts) {
    const alertsPanel = document.querySelector('.alerts-panel');
    if (alertsPanel && alerts) {
        alerts.forEach(alert => {
            const alertElement = createAlertElement(alert);
            alertsPanel.insertBefore(alertElement, alertsPanel.firstChild);
        });

        // Remove old alerts
        while (alertsPanel.children.length > 10) {
            alertsPanel.removeChild(alertsPanel.lastChild);
        }
    }
}

function createAlertElement(alert) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert-item ${alert.priority}`;
    alertDiv.innerHTML = `
        <div class="alert-icon ${alert.type}">
            <i class="fas fa-${alert.icon}"></i>
        </div>
        <div class="alert-content">
            <h4>${alert.title}</h4>
            <p>${alert.message}</p>
            <small>${alert.time}</small>
        </div>
    `;
    return alertDiv;
}

// Event Listeners
function setupEventListeners() {
    // Sidebar Toggle
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    const sidebar = document.querySelector('.sidebar');
    
    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
    }

    // View Controls
    const viewControls = document.querySelectorAll('.view-controls button');
    viewControls.forEach(button => {
        button.addEventListener('click', function() {
            viewControls.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            updateVisualizationType(this.textContent.toLowerCase());
        });
    });

    // Action Buttons
    document.querySelectorAll('.action-item button').forEach(button => {
        button.addEventListener('click', function() {
            const action = this.closest('.action-item');
            handleActionResponse(action.dataset.actionId);
        });
    });
}

// Utility Functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 3000);
}

function updateTrend(element, trend) {
    const trendElement = element.querySelector('.trend');
    if (trendElement && trend) {
        const isPositive = trend > 0;
        trendElement.className = `trend ${isPositive ? 'positive' : 'negative'}`;
        trendElement.innerHTML = `
            <i class="fas fa-arrow-${isPositive ? 'up' : 'down'}"></i>
            ${Math.abs(trend)}%
        `;
    }
}

function handleActionResponse(actionId) {
    // Handle action button clicks
    console.log(`Action ${actionId} triggered`);
    showNotification('Action recorded and being processed', 'success');
}

function updateVisualizationType(type) {
    const visualizationElement = document.querySelector('.density-visualization');
    if (visualizationElement) {
        visualizationElement.dataset.viewType = type;
        // Update visualization based on type
    }
}

// Initial Data Load
function loadInitialData() {
    // Simulate initial data load
    setTimeout(() => {
        updateDashboard({
            stats: {
                totalCount: 12628,
                activeIncidents: 14,
                currentAlerts: 5,
                averageDensity: 3.5
            },
            alerts: [],
            recommendations: []
        });
    }, 1000);
}

// Add to dashboard.js

// Video View Control
document.addEventListener('DOMContentLoaded', function() {
    const views = {
        normal: document.getElementById('normalView'),
        density: document.getElementById('densityView'),
        overlay: document.getElementById('overlayView')
    };
    
    // Initialize all videos
    Object.values(views).forEach(video => {
        if (video) {
            video.loop = true;
            video.muted = true;
            video.playsinline = true;
            video.play().catch(error => console.log("Video autoplay failed:", error));
        }
    });

    // View switching functionality
    const viewButtons = document.querySelectorAll('.view-controls button');
    viewButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Update button states
            viewButtons.forEach(btn => {
                btn.classList.remove('active');
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-outline-primary');
            });
            this.classList.add('active');
            this.classList.add('btn-primary');
            this.classList.remove('btn-outline-primary');

            // Update video visibility
            const selectedView = this.dataset.view;
            Object.entries(views).forEach(([key, video]) => {
                if (key === selectedView) {
                    video.classList.remove('hidden-view');
                    video.classList.add('active-view');
                    video.play();
                } else {
                    video.classList.add('hidden-view');
                    video.classList.remove('active-view');
                    video.pause();
                }
            });
        });
    });

    // Live count update
    function updateLiveCount() {
        const liveCountElement = document.getElementById('liveCount');
        const densityLevelElement = document.getElementById('densityLevel');
        const flowRateElement = document.getElementById('flowRate');

        if (liveCountElement && densityLevelElement && flowRateElement) {
            // Generate random values for demonstration
            const count = Math.floor(300 + Math.random() * 100);
            const density = Math.floor(40 + Math.random() * 30);
            const flowRate = Math.floor(30 + Math.random() * 20);

            // Update DOM
            liveCountElement.textContent = count;
            densityLevelElement.style.width = `${density}%`;
            densityLevelElement.textContent = `${density}%`;
            flowRateElement.textContent = `${flowRate} people/min`;

            // Update density level color based on value
            if (density > 80) {
                densityLevelElement.className = 'progress-bar bg-danger';
            } else if (density > 60) {
                densityLevelElement.className = 'progress-bar bg-warning';
            } else {
                densityLevelElement.className = 'progress-bar bg-success';
            }
        }
    }

    // Update live count every second
    setInterval(updateLiveCount, 1000);
});