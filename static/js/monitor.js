class MonitorManager {
    constructor() {
        this.cameras = [];
        this.alerts = [];
        this.systemStatus = {
            activeDogs: [],
            camerasOnline: 0,
            camerasTotal: 0,
            detectionFPS: 0,
            gpuUsage: 0,
            memoryUsage: 0,
            uptime: 0
        };
        this.gridLayout = '3x4';
        this.isFullscreen = false;
        this.selectedCamera = null;
        this.updateInterval = null;
        
        this.initializeEventListeners();
        this.startMonitoring();
    }
    
    initializeEventListeners() {
        // Grid layout selector
        const gridSelector = document.getElementById('gridLayout');
        if (gridSelector) {
            gridSelector.addEventListener('change', (e) => {
                this.gridLayout = e.target.value;
                this.updateCameraGrid();
            });
        }
        
        // Fullscreen button
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        if (fullscreenBtn) {
            fullscreenBtn.addEventListener('click', () => {
                this.toggleFullscreen();
            });
        }
        
        // Exit fullscreen button
        const exitFullscreenBtn = document.getElementById('exitFullscreenBtn');
        if (exitFullscreenBtn) {
            exitFullscreenBtn.addEventListener('click', () => {
                this.exitFullscreen();
            });
        }
        
        // Clear alerts button
        const clearAlertsBtn = document.getElementById('clearAlertsBtn');
        if (clearAlertsBtn) {
            clearAlertsBtn.addEventListener('click', () => {
                this.clearAlerts();
            });
        }
        
        // Mute alerts button
        const muteAlertsBtn = document.getElementById('muteAlertsBtn');
        if (muteAlertsBtn) {
            muteAlertsBtn.addEventListener('click', () => {
                this.toggleAlertMute();
            });
        }
        
        // Refresh status button
        const refreshStatusBtn = document.getElementById('refreshStatusBtn');
        if (refreshStatusBtn) {
            refreshStatusBtn.addEventListener('click', () => {
                this.updateSystemStatus();
            });
        }
        
        // Start cameras button
        const startCamerasBtn = document.getElementById('startCamerasBtn');
        if (startCamerasBtn) {
            startCamerasBtn.addEventListener('click', async () => {
                await this.startAllCameras();
            });
        }
        
        // Stop cameras button
        const stopCamerasBtn = document.getElementById('stopCamerasBtn');
        if (stopCamerasBtn) {
            stopCamerasBtn.addEventListener('click', async () => {
                await this.stopAllCameras();
            });
        }
        
        // Camera cell click handlers
        document.addEventListener('click', (e) => {
            const cameraCell = e.target.closest('.camera-cell');
            if (cameraCell && !e.target.closest('.camera-controls')) {
                const cameraId = cameraCell.dataset.camera;
                this.showCameraFullscreen(cameraId);
            }
        });
    }
    
    async startMonitoring() {
        // Initial load
        await this.loadCameras();
        await this.updateSystemStatus();
        await this.loadRecentAlerts();
        
        // Set up periodic updates
        this.updateInterval = setInterval(() => {
            this.updateSystemStatus();
            this.checkForNewAlerts();
        }, 5000); // Update every 5 seconds
        
        // Update camera feeds
        this.updateCameraFeeds();
    }
    
    async loadCameras() {
        try {
            const response = await fetch('/api/cameras');
            if (response.ok) {
                const data = await response.json();
                this.cameras = data.cameras || [];
                this.updateCameraGrid();
            }
        } catch (error) {
            console.error('Error loading cameras:', error);
            // If no camera endpoint exists yet, create empty grid
            this.updateCameraGrid();
        }
    }
    
    updateCameraGrid() {
        const gridContainer = document.getElementById('cameraGrid');
        if (!gridContainer) return;
        
        // Clear existing grid
        gridContainer.innerHTML = '';
        
        console.log('Updating camera grid with cameras:', this.cameras);
        
        // If no cameras configured, show message
        if (!this.cameras || this.cameras.length === 0) {
            gridContainer.innerHTML = `
                <div class="no-cameras-message">
                    <h3>No Cameras Configured</h3>
                    <p>Go to Settings to add cameras, then click "Start All Cameras"</p>
                </div>
            `;
            return;
        }
        
        // Determine grid size based on number of cameras (not fixed layout)
        const numCameras = this.cameras.length;
        let cols = Math.ceil(Math.sqrt(numCameras));
        let rows = Math.ceil(numCameras / cols);
        
        // Apply grid CSS
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        gridContainer.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
        
        console.log(`Creating grid: ${rows}x${cols} for ${numCameras} cameras`);
        
        // Create camera cells ONLY for configured cameras
        this.cameras.forEach(camera => {
            const cell = document.createElement('div');
            cell.className = 'camera-cell';
            cell.dataset.camera = camera.id;
            
            const isOnline = camera.status === 'online';
            
            console.log(`Creating cell for camera ${camera.id}: ${camera.name} (${camera.status})`);
            
            cell.innerHTML = `
                <div class="camera-frame">
                    <img src="/video_feed?camera=${camera.id}" 
                         alt="${camera.name}" 
                         class="camera-feed"
                         onerror="this.src='/static/images/no-signal.jpg'; this.onerror=null;">
                    <div class="camera-overlay">
                        <div class="camera-label">${camera.name}</div>
                        <div class="camera-status ${isOnline ? 'online' : 'offline'}">
                            ${isOnline ? 'ONLINE' : 'OFFLINE'}
                        </div>
                        <div class="detection-overlay" id="detections-${camera.id}"></div>
                    </div>
                </div>
            `;
            
            gridContainer.appendChild(cell);
        });
    }
    
    updateCameraFeeds() {
        // Update detection overlays via WebSocket or polling
        // For now, we'll use polling
        setInterval(() => {
            this.cameras.forEach(camera => {
                if (camera.status === 'online') {
                    this.updateDetectionOverlay(camera.id);
                }
            });
        }, 5000); // Update every 5 seconds
    }
    
    async updateDetectionOverlay(cameraId) {
        try {
            const response = await fetch(`/api/detections?camera=${cameraId}`);
            if (response.ok) {
                const data = await response.json();
                const overlay = document.getElementById(`detections-${cameraId}`);
                if (overlay && data.detections) {
                    this.renderDetections(overlay, data.detections);
                }
            }
        } catch (error) {
            // Silently fail - detection endpoint might not exist yet
        }
    }
    
    renderDetections(overlay, detections) {
        overlay.innerHTML = '';
        
        detections.forEach(detection => {
            const box = document.createElement('div');
            box.className = `detection-box ${detection.type}`;
            box.style.left = `${detection.x * 100}%`;
            box.style.top = `${detection.y * 100}%`;
            box.style.width = `${detection.width * 100}%`;
            box.style.height = `${detection.height * 100}%`;
            
            if (detection.label) {
                const label = document.createElement('div');
                label.className = 'detection-label';
                label.textContent = `${detection.label} ${Math.round(detection.confidence * 100)}%`;
                box.appendChild(label);
            }
            
            overlay.appendChild(box);
        });
    }
    
    async updateSystemStatus() {
        try {
            const response = await fetch('/api/system/status');
            if (response.ok) {
                const data = await response.json();
                this.systemStatus = data;
                this.renderSystemStatus();
            }
        } catch (error) {
            // Use default values if endpoint doesn't exist
            this.renderSystemStatus();
        }
    }
    
    renderSystemStatus() {
        // Update active dogs
        const activeDogsEl = document.getElementById('activeDogs');
        if (activeDogsEl) {
            activeDogsEl.textContent = this.systemStatus.activeDogs.length;
            const detailEl = activeDogsEl.nextElementSibling;
            if (detailEl) {
                detailEl.textContent = this.systemStatus.activeDogs.join(', ') || 'None detected';
            }
        }
        
        // Update cameras online
        const camerasOnlineEl = document.getElementById('camerasOnline');
        if (camerasOnlineEl) {
            const onlineCount = this.cameras.filter(c => c.status === 'online').length;
            const totalCount = this.cameras.length || 12;
            camerasOnlineEl.textContent = `${onlineCount}/${totalCount}`;
            const detailEl = camerasOnlineEl.nextElementSibling;
            if (detailEl) {
                const percentage = totalCount > 0 ? Math.round((onlineCount / totalCount) * 100) : 0;
                detailEl.textContent = `${percentage}% operational`;
            }
        }
        
        // Update detection rate
        const detectionRateEl = document.getElementById('detectionRate');
        if (detectionRateEl) {
            detectionRateEl.textContent = this.systemStatus.detectionFPS.toFixed(1);
        }
        
        // Update GPU usage
        const gpuUsageEl = document.getElementById('gpuUsage');
        if (gpuUsageEl) {
            gpuUsageEl.textContent = `${this.systemStatus.gpuUsage}%`;
        }
        
        // Update memory usage
        const memoryUsageEl = document.getElementById('memoryUsage');
        if (memoryUsageEl) {
            memoryUsageEl.textContent = this.formatBytes(this.systemStatus.memoryUsage);
        }
        
        // Update uptime
        const uptimeEl = document.getElementById('systemUptime');
        if (uptimeEl) {
            uptimeEl.textContent = this.formatUptime(this.systemStatus.uptime);
        }
    }
    
    async loadRecentAlerts() {
        try {
            const response = await fetch('/api/alerts/recent');
            if (response.ok) {
                const data = await response.json();
                this.alerts = data.alerts || [];
                this.renderAlerts();
            }
        } catch (error) {
            // Clear alerts if endpoint doesn't exist
            this.renderAlerts();
        }
    }
    
    async checkForNewAlerts() {
        try {
            const response = await fetch('/api/alerts/new');
            if (response.ok) {
                const data = await response.json();
                if (data.alerts && data.alerts.length > 0) {
                    data.alerts.forEach(alert => {
                        this.addAlert(alert);
                    });
                }
            }
        } catch (error) {
            // Silently fail
        }
    }
    
    addAlert(alert) {
        this.alerts.unshift(alert);
        if (this.alerts.length > 50) {
            this.alerts = this.alerts.slice(0, 50);
        }
        this.renderAlerts();
        
        // Show notification
        this.showNotification(alert.message, alert.priority);
    }
    
    renderAlerts() {
        const alertList = document.getElementById('alertList');
        if (!alertList) return;
        
        alertList.innerHTML = '';
        
        if (this.alerts.length === 0) {
            alertList.innerHTML = '<div class="no-alerts">No recent alerts</div>';
            return;
        }
        
        this.alerts.forEach(alert => {
            const alertEl = document.createElement('div');
            alertEl.className = `alert-item ${alert.priority || 'info'}`;
            
            const time = new Date(alert.timestamp);
            const timeStr = time.toLocaleTimeString();
            
            alertEl.innerHTML = `
                <div class="alert-time">${timeStr}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-source">${alert.source || 'System'}</div>
            `;
            
            alertList.appendChild(alertEl);
        });
    }
    
    clearAlerts() {
        this.alerts = [];
        this.renderAlerts();
        this.showNotification('All alerts cleared', 'info');
    }
    
    toggleAlertMute() {
        const muteBtn = document.getElementById('muteAlertsBtn');
        if (muteBtn) {
            const isMuted = muteBtn.classList.toggle('muted');
            muteBtn.textContent = isMuted ? 'Unmute' : 'Mute';
            this.showNotification(`Alerts ${isMuted ? 'muted' : 'unmuted'}`, 'info');
        }
    }
    
    showCameraFullscreen(cameraId) {
        const modal = document.getElementById('fullscreenModal');
        const image = document.getElementById('fullscreenImage');
        const title = document.getElementById('fullscreenTitle');
        
        if (modal && image && title) {
            const camera = this.cameras.find(c => c.id === parseInt(cameraId));
            title.textContent = camera ? camera.name : `Camera ${cameraId}`;
            image.src = `/video_feed?camera=${cameraId}`;
            modal.style.display = 'flex';
            this.selectedCamera = cameraId;
        }
    }
    
    exitFullscreen() {
        const modal = document.getElementById('fullscreenModal');
        if (modal) {
            modal.style.display = 'none';
            this.selectedCamera = null;
        }
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
    
    showNotification(message, type = 'info') {
        const container = document.getElementById('statusMessages');
        if (!container) return;
        
        const notification = document.createElement('div');
        notification.className = `status-message ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    formatUptime(seconds) {
        if (seconds === 0) return '0s';
        
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        
        const parts = [];
        if (days > 0) parts.push(`${days}d`);
        if (hours > 0) parts.push(`${hours}h`);
        if (minutes > 0 && days === 0) parts.push(`${minutes}m`);
        
        return parts.join(' ') || '< 1m';
    }
    
    async startAllCameras() {
        try {
            const startBtn = document.getElementById('startCamerasBtn');
            if (startBtn) {
                startBtn.disabled = true;
                startBtn.textContent = 'Starting...';
            }
            
            const response = await fetch('/api/cameras/start_all', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message, 'success');
                
                // Start polling for camera status updates
                this.pollCameraStatus();
            } else {
                const error = await response.json();
                this.showNotification(error.message || 'Failed to start cameras', 'error');
            }
        } catch (error) {
            console.error('Error starting cameras:', error);
            this.showNotification('Failed to start cameras', 'error');
        } finally {
            const startBtn = document.getElementById('startCamerasBtn');
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.textContent = 'Start All Cameras';
            }
        }
    }
    
    async stopAllCameras() {
        try {
            const stopBtn = document.getElementById('stopCamerasBtn');
            if (stopBtn) {
                stopBtn.disabled = true;
                stopBtn.textContent = 'Stopping...';
            }
            
            const response = await fetch('/api/cameras/stop_all', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            });
            
            if (response.ok) {
                const result = await response.json();
                this.showNotification(result.message, 'success');
                
                // Reload cameras after a short delay
                setTimeout(() => {
                    this.loadCameras();
                    this.updateSystemStatus();
                }, 1000);
            } else {
                const error = await response.json();
                this.showNotification(error.message || 'Failed to stop cameras', 'error');
            }
        } catch (error) {
            console.error('Error stopping cameras:', error);
            this.showNotification('Failed to stop cameras', 'error');
        } finally {
            const stopBtn = document.getElementById('stopCamerasBtn');
            if (stopBtn) {
                stopBtn.disabled = false;
                stopBtn.textContent = 'Stop All Cameras';
            }
        }
    }
    
    pollCameraStatus() {
        // Poll camera status until all cameras are connected or failed
        let pollCount = 0;
        const maxPolls = 30; // 30 seconds max
        
        const pollInterval = setInterval(async () => {
            pollCount++;
            
            try {
                await this.loadCameras();
                await this.updateSystemStatus();
                
                // Check if all cameras are done connecting (either online or error)
                const onlineCount = this.cameras.filter(c => c.status === 'online').length;
                const errorCount = this.cameras.filter(c => c.status === 'error').length;
                const connectingCount = this.cameras.filter(c => c.status === 'connecting').length;
                
                if (connectingCount === 0 || pollCount >= maxPolls) {
                    clearInterval(pollInterval);
                    
                    if (onlineCount > 0) {
                        this.showNotification(`${onlineCount} camera(s) connected successfully`, 'success');
                    }
                    if (errorCount > 0) {
                        this.showNotification(`${errorCount} camera(s) failed to connect`, 'error');
                    }
                }
            } catch (error) {
                console.error('Error polling camera status:', error);
                if (pollCount >= maxPolls) {
                    clearInterval(pollInterval);
                }
            }
        }, 1000); // Poll every second
    }
    
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Initialize monitor manager when page loads
let monitorManager;
document.addEventListener('DOMContentLoaded', () => {
    monitorManager = new MonitorManager();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (monitorManager) {
        monitorManager.destroy();
    }
});