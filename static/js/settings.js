// Settings page functionality for Dog Tracking System

class SettingsManager {
    constructor() {
        this.cameraUrls = [];
        this.currentSettings = {};
        this.init();
    }

    init() {
        this.loadSettings();
        this.setupEventListeners();
        this.setupRangeSliders();
        this.setupColorPicker();
    }

    setupEventListeners() {
        // Auto-save on input changes (with debounce)
        const inputs = document.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('change', () => this.debouncedAutoSave());
            if (input.type === 'range') {
                input.addEventListener('input', () => this.updateRangeDisplay(input));
            }
        });
    }

    setupRangeSliders() {
        const ranges = document.querySelectorAll('input[type="range"]');
        ranges.forEach(range => {
            this.updateRangeDisplay(range);
        });
    }

    setupColorPicker() {
        const colorPicker = document.getElementById('boundary-color');
        const colorPreview = document.querySelector('.color-preview');
        const colorName = document.querySelector('.color-name');

        colorPicker.addEventListener('change', (e) => {
            const color = e.target.value;
            colorPreview.style.backgroundColor = color;
            colorName.textContent = this.getColorName(color);
        });

        // Initial setup
        colorPreview.style.backgroundColor = colorPicker.value;
        colorName.textContent = this.getColorName(colorPicker.value);
    }

    updateRangeDisplay(rangeInput) {
        const valueDisplay = rangeInput.parentNode.querySelector('.range-value');
        if (valueDisplay) {
            const value = rangeInput.value;
            const id = rangeInput.id;
            
            // Format display based on input type
            if (id.includes('confidence') || id.includes('opacity')) {
                // Show as percentage
                valueDisplay.textContent = Math.round(value * 100) + '%';
            } else if (id === 'boundary-thickness') {
                valueDisplay.textContent = value + 'px';
            } else if (id === 'reference-tolerance') {
                valueDisplay.textContent = value + 'px';
            } else if (id === 'alert-cooldown') {
                valueDisplay.textContent = value + ' seconds';
            } else if (id === 'processing-fps') {
                valueDisplay.textContent = value + ' FPS';
            } else if (id === 'frame-skip') {
                valueDisplay.textContent = `Process every ${value}${value === '1' ? 'st' : value === '2' ? 'nd' : value === '3' ? 'rd' : 'th'} frame`;
            } else if (id === 'gpu-memory-limit') {
                valueDisplay.textContent = value + ' GB';
            } else {
                valueDisplay.textContent = value;
            }
        }
    }

    getColorName(hex) {
        const colors = {
            '#ff0000': 'Red',
            '#00ff00': 'Green', 
            '#0000ff': 'Blue',
            '#ffff00': 'Yellow',
            '#ff00ff': 'Magenta',
            '#00ffff': 'Cyan',
            '#ffffff': 'White',
            '#000000': 'Black'
        };
        return colors[hex.toLowerCase()] || 'Custom';
    }

    async loadSettings() {
        try {
            const response = await fetch('/api/settings');
            if (response.ok) {
                this.currentSettings = await response.json();
                this.populateForm();
                this.showStatus('Settings loaded successfully', 'success');
            } else {
                throw new Error('Failed to load settings');
            }
        } catch (error) {
            console.error('Error loading settings:', error);
            this.showStatus('Failed to load settings', 'error');
            // Load defaults if server settings fail
            this.loadDefaults();
        }
    }

    populateForm() {
        const settings = this.currentSettings;

        // Camera URLs
        this.cameraUrls = settings.camera_urls || [];
        this.renderCameraList();
        this.updateActiveCameraList();

        // Camera Configuration
        this.setIfExists('active-camera', settings.active_camera || 0);
        this.setIfExists('camera-layout', settings.camera_layout || '3x4');

        // Detection Configuration
        this.setIfExists('animal-confidence', settings.animal_confidence || 0.7);
        this.setIfExists('person-confidence', settings.person_confidence || 0.8);
        this.setIfExists('vehicle-confidence', settings.vehicle_confidence || 0.6);
        this.setIfExists('dog-id-confidence', settings.dog_id_confidence || 0.8);
        this.setIfExists('show-confidence', settings.show_confidence || false);
        this.setIfExists('show-bounding-boxes', settings.show_bounding_boxes !== false);
        this.setIfExists('enable-cross-camera-tracking', settings.enable_cross_camera_tracking !== false);

        // Boundary & Reference Points
        this.setIfExists('boundary-color', settings.boundary_color || '#ff0000');
        this.setIfExists('boundary-thickness', settings.boundary_thickness || 3);
        this.setIfExists('reference-tolerance', settings.reference_tolerance || 15);
        this.setIfExists('boundary-opacity', settings.boundary_opacity || 0.7);
        this.setIfExists('enable-dynamic-boundaries', settings.enable_dynamic_boundaries !== false);
        this.setIfExists('auto-reference-calibration', settings.auto_reference_calibration !== false);
        this.setIfExists('show-reference-points', settings.show_reference_points || false);

        // Actions & Alerts
        this.setIfExists('action-email', settings.action_email || false);
        this.setIfExists('action-sound', settings.action_sound !== false);
        this.setIfExists('action-log', settings.action_log !== false);
        this.setIfExists('save-snapshots', settings.save_snapshots !== false);
        this.setIfExists('email-address', settings.email_address || '');
        this.setIfExists('alert-cooldown', settings.alert_cooldown || 30);

        // Performance & System
        this.setIfExists('processing-fps', settings.processing_fps || 15);
        this.setIfExists('frame-skip', settings.frame_skip || 2);
        this.setIfExists('gpu-memory-limit', settings.gpu_memory_limit || 6);
        this.setIfExists('storage-days', settings.storage_days || 30);
        this.setIfExists('enable-tensorrt', settings.enable_tensorrt || false);
        this.setIfExists('debug-mode', settings.debug_mode || false);
        this.setIfExists('auto-restart', settings.auto_restart !== false);
        this.setIfExists('monitor-system-health', settings.monitor_system_health !== false);

        // Update range displays
        document.querySelectorAll('input[type="range"]').forEach(range => {
            this.updateRangeDisplay(range);
        });

        // Update color picker
        this.setupColorPicker();
    }

    setIfExists(id, value) {
        const element = document.getElementById(id);
        if (element) {
            if (element.type === 'checkbox') {
                element.checked = value;
            } else {
                element.value = value;
            }
        }
    }

    renderCameraList() {
        const container = document.getElementById('camera-list');
        container.innerHTML = '';

        if (this.cameraUrls.length === 0) {
            const placeholder = document.createElement('div');
            placeholder.className = 'camera-placeholder';
            placeholder.innerHTML = `
                <div class="placeholder-content">
                    <span class="placeholder-icon">📹</span>
                    <p>No cameras configured</p>
                    <p class="placeholder-text">Click "Add Camera" to get started</p>
                </div>
            `;
            container.appendChild(placeholder);
            return;
        }

        this.cameraUrls.forEach((url, index) => {
            const cameraItem = document.createElement('div');
            cameraItem.className = 'camera-item';
            cameraItem.innerHTML = `
                <div class="camera-input-group">
                    <label class="camera-label">Camera ${index + 1}</label>
                    <input type="text" 
                           value="${url}" 
                           onchange="settingsManager.updateCamera(${index}, this.value)"
                           placeholder="rtsp://username:password@camera-ip:554/stream"
                           class="camera-url-input">
                    <button type="button" 
                            onclick="settingsManager.testCamera(${index})" 
                            class="btn btn-outline btn-small">
                        Test
                    </button>
                    <button type="button" 
                            onclick="settingsManager.removeCamera(${index})" 
                            class="btn btn-danger btn-small">
                        Remove
                    </button>
                </div>
                <div class="camera-status" id="camera-status-${index}">
                    <span class="status-dot status-unknown"></span>
                    <span class="status-text">Not tested</span>
                </div>
            `;
            container.appendChild(cameraItem);
        });
    }

    updateActiveCameraList() {
        const select = document.getElementById('active-camera');
        select.innerHTML = '';

        if (this.cameraUrls.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No cameras available';
            option.disabled = true;
            select.appendChild(option);
        } else {
            this.cameraUrls.forEach((url, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `Camera ${index + 1}${url ? ` (${this.getUrlHost(url)})` : ''}`;
                select.appendChild(option);
            });
        }
    }

    getUrlHost(url) {
        try {
            const urlObj = new URL(url);
            return urlObj.hostname;
        } catch {
            return 'Invalid URL';
        }
    }

    addCamera() {
        this.cameraUrls.push('');
        this.renderCameraList();
        this.updateActiveCameraList();
        this.debouncedAutoSave();
        
        // Focus on the new camera input
        setTimeout(() => {
            const inputs = document.querySelectorAll('.camera-url-input');
            if (inputs.length > 0) {
                inputs[inputs.length - 1].focus();
            }
        }, 100);
    }

    removeCamera(index) {
        if (confirm(`Remove Camera ${index + 1}?`)) {
            this.cameraUrls.splice(index, 1);
            this.renderCameraList();
            this.updateActiveCameraList();
            this.debouncedAutoSave();
            this.showStatus(`Camera ${index + 1} removed`, 'success');
        }
    }

    updateCamera(index, value) {
        this.cameraUrls[index] = value;
        this.updateActiveCameraList();
        this.debouncedAutoSave();
    }

    async testCamera(index) {
        const url = this.cameraUrls[index];
        if (!url) {
            this.showStatus('Enter a camera URL first', 'error');
            return;
        }

        const statusElement = document.getElementById(`camera-status-${index}`);
        this.updateCameraStatus(statusElement, 'testing', 'Testing connection...');

        try {
            const response = await fetch('/api/camera/test', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({camera_url: url})
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.updateCameraStatus(statusElement, 'success', 'Connected');
                this.showStatus(`Camera ${index + 1} connected successfully`, 'success');
            } else {
                this.updateCameraStatus(statusElement, 'error', result.message || 'Connection failed');
                this.showStatus(`Camera ${index + 1} test failed: ${result.message}`, 'error');
            }
        } catch (error) {
            this.updateCameraStatus(statusElement, 'error', 'Test failed');
            this.showStatus(`Camera ${index + 1} test error: ${error.message}`, 'error');
        }
    }

    updateCameraStatus(statusElement, status, message) {
        const dot = statusElement.querySelector('.status-dot');
        const text = statusElement.querySelector('.status-text');
        
        dot.className = `status-dot status-${status}`;
        text.textContent = message;
    }

    async testCameraConnections() {
        if (this.cameraUrls.length === 0) {
            this.showStatus('No cameras configured to test', 'error');
            return;
        }

        this.showStatus('Testing all camera connections...', 'info');
        
        const filteredUrls = this.cameraUrls.filter(url => url.trim() !== '');
        const promises = filteredUrls.map((_, index) => this.testCamera(index));
        
        await Promise.all(promises);
        this.showStatus('Camera testing completed', 'success');
    }

    async saveSettings() {
        try {
            const settings = this.gatherFormData();
            
            console.log('Saving settings:', settings);
            
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            });

            const result = await response.json();
            
            if (response.ok) {
                this.showStatus('Settings saved successfully', 'success');
                document.getElementById('saveStatus').textContent = 'Saved';
                
                // Update current settings with saved values
                this.currentSettings = settings;
            } else {
                const errorMsg = result.message || 'Save request failed';
                throw new Error(errorMsg);
            }
        } catch (error) {
            console.error('Error saving settings:', error);
            
            // Provide more specific error messages
            let message = 'Failed to save settings';
            if (error.message) {
                message += ': ' + error.message;
            }
            if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
                message = 'Cannot connect to server. Please check if the server is running.';
            }
            
            this.showStatus(message, 'error', 10000);
            document.getElementById('saveStatus').textContent = 'Error';
        }
    }

    gatherFormData() {
        return {
            // Camera Configuration
            camera_urls: this.cameraUrls.filter(url => url.trim() !== ''),
            active_camera: parseInt(document.getElementById('active-camera')?.value) || 0,
            camera_layout: document.getElementById('camera-layout')?.value || '3x4',
            
            // Detection Configuration
            animal_confidence: parseFloat(document.getElementById('animal-confidence')?.value) || 0.7,
            person_confidence: parseFloat(document.getElementById('person-confidence')?.value) || 0.8,
            vehicle_confidence: parseFloat(document.getElementById('vehicle-confidence')?.value) || 0.6,
            dog_id_confidence: parseFloat(document.getElementById('dog-id-confidence')?.value) || 0.8,
            show_confidence: document.getElementById('show-confidence')?.checked || false,
            show_bounding_boxes: document.getElementById('show-bounding-boxes')?.checked || true,
            enable_cross_camera_tracking: document.getElementById('enable-cross-camera-tracking')?.checked || true,
            
            // Boundary & Reference Points
            boundary_color: document.getElementById('boundary-color')?.value || '#ff0000',
            boundary_thickness: parseInt(document.getElementById('boundary-thickness')?.value) || 3,
            reference_tolerance: parseInt(document.getElementById('reference-tolerance')?.value) || 15,
            boundary_opacity: parseFloat(document.getElementById('boundary-opacity')?.value) || 0.7,
            enable_dynamic_boundaries: document.getElementById('enable-dynamic-boundaries')?.checked || true,
            auto_reference_calibration: document.getElementById('auto-reference-calibration')?.checked || true,
            show_reference_points: document.getElementById('show-reference-points')?.checked || false,
            
            // Actions & Alerts
            action_email: document.getElementById('action-email')?.checked || false,
            action_sound: document.getElementById('action-sound')?.checked || true,
            action_log: document.getElementById('action-log')?.checked || true,
            save_snapshots: document.getElementById('save-snapshots')?.checked || true,
            email_address: document.getElementById('email-address')?.value || '',
            alert_cooldown: parseInt(document.getElementById('alert-cooldown')?.value) || 30,
            
            // Performance & System
            processing_fps: parseInt(document.getElementById('processing-fps')?.value) || 15,
            frame_skip: parseInt(document.getElementById('frame-skip')?.value) || 2,
            gpu_memory_limit: parseInt(document.getElementById('gpu-memory-limit')?.value) || 6,
            storage_days: parseInt(document.getElementById('storage-days')?.value) || 30,
            enable_tensorrt: document.getElementById('enable-tensorrt')?.checked || false,
            debug_mode: document.getElementById('debug-mode')?.checked || false,
            auto_restart: document.getElementById('auto-restart')?.checked || true,
            monitor_system_health: document.getElementById('monitor-system-health')?.checked || true
        };
    }

    loadDefaults() {
        const defaults = {
            // Camera Configuration
            camera_urls: [],
            active_camera: 0,
            camera_layout: '3x4',
            
            // Detection Configuration
            animal_confidence: 0.7,
            person_confidence: 0.8,
            vehicle_confidence: 0.6,
            dog_id_confidence: 0.8,
            show_confidence: false,
            show_bounding_boxes: true,
            enable_cross_camera_tracking: true,
            
            // Boundary & Reference Points
            boundary_color: '#ff0000',
            boundary_thickness: 3,
            reference_tolerance: 15,
            boundary_opacity: 0.7,
            enable_dynamic_boundaries: true,
            auto_reference_calibration: true,
            show_reference_points: false,
            
            // Actions & Alerts
            action_email: false,
            action_sound: true,
            action_log: true,
            save_snapshots: true,
            email_address: '',
            alert_cooldown: 30,
            
            // Performance & System
            processing_fps: 15,
            frame_skip: 2,
            gpu_memory_limit: 6,
            storage_days: 30,
            enable_tensorrt: false,
            debug_mode: false,
            auto_restart: true,
            monitor_system_health: true
        };

        this.currentSettings = defaults;
        this.populateForm();
        this.showStatus('Default settings loaded', 'info');
    }

    resetToDefaults() {
        if (confirm('Reset all settings to defaults? This cannot be undone.')) {
            this.loadDefaults();
            this.debouncedAutoSave();
        }
    }

    // Debounced auto-save to prevent excessive saves
    debouncedAutoSave = this.debounce(() => {
        document.getElementById('saveStatus').textContent = 'Saving...';
        setTimeout(() => {
            document.getElementById('saveStatus').textContent = 'Auto-saved';
        }, 500);
    }, 1000);

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    showStatus(message, type = 'info', duration = 5000) {
        const container = document.getElementById('statusMessages');
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message ${type}`;
        statusDiv.textContent = message;
        
        container.appendChild(statusDiv);
        
        setTimeout(() => {
            statusDiv.remove();
        }, duration);
    }
}

// Global functions for HTML onclick handlers
let settingsManager;

function addCamera() {
    settingsManager.addCamera();
}

function testCameraConnections() {
    settingsManager.testCameraConnections();
}

function resetToDefaults() {
    settingsManager.resetToDefaults();
}

function saveSettings() {
    settingsManager.saveSettings();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    settingsManager = new SettingsManager();
});