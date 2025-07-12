class BoundaryDrawer {
    constructor() {
        console.log('Initializing BoundaryDrawer...');
        
        this.canvas = document.getElementById('boundaryCanvas');
        this.videoFeed = document.getElementById('boundaryVideoFeed');
        
        if (!this.canvas) {
            console.error('Canvas element not found!');
            return;
        }
        if (!this.videoFeed) {
            console.error('Video feed element not found!');
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        this.isDrawing = false;
        this.currentBoundary = [];
        this.boundaries = [];
        this.selectedBoundary = null;
        this.showCompletedBoundaries = false; // Let server handle permanent display
        
        // Manual reference point mode
        this.manualReferenceMode = false;
        
        console.log('Elements found:', {
            canvas: !!this.canvas,
            videoFeed: !!this.videoFeed,
            ctx: !!this.ctx
        });
        
        this.setupCanvas();
        this.setupEventListeners();
        this.updateUI();
        
        // Load existing boundaries
        this.loadBoundaries();
        
        console.log('BoundaryDrawer initialized successfully');
    }
    
    setupCanvas() {
        // Make canvas overlay the video
        this.setupCanvasPosition();
        
        // Resize canvas when video loads
        this.videoFeed.onload = () => {
            setTimeout(() => this.setupCanvasPosition(), 100);
        };
        
        // Handle window resize
        window.addEventListener('resize', () => {
            setTimeout(() => this.setupCanvasPosition(), 100);
        });
        
        // Set canvas style for proper overlay
        this.canvas.style.position = 'absolute';
        this.canvas.style.zIndex = '10';
        this.canvas.style.pointerEvents = 'auto';
        this.canvas.style.cursor = 'crosshair';
        
        console.log('Canvas setup complete');
    }
    
    setupCanvasPosition() {
        try {
            // Wait for video to load and get its actual displayed size
            const videoRect = this.videoFeed.getBoundingClientRect();
            
            // Set canvas dimensions to match displayed video exactly
            this.canvas.width = this.videoFeed.offsetWidth;
            this.canvas.height = this.videoFeed.offsetHeight;
            
            // Position canvas exactly over video
            this.canvas.style.position = 'absolute';
            this.canvas.style.left = '0px';
            this.canvas.style.top = '0px';
            this.canvas.style.width = this.videoFeed.offsetWidth + 'px';
            this.canvas.style.height = this.videoFeed.offsetHeight + 'px';
            this.canvas.style.zIndex = '10';
            this.canvas.style.pointerEvents = 'auto';
            
            console.log('Canvas positioned over video:', {
                videoDisplayWidth: this.videoFeed.offsetWidth,
                videoDisplayHeight: this.videoFeed.offsetHeight,
                canvasWidth: this.canvas.width,
                canvasHeight: this.canvas.height,
                videoRect: videoRect
            });
            
            this.redrawBoundaries();
        } catch (error) {
            console.error('Error positioning canvas:', error);
        }
    }
    
    resizeCanvas() {
        this.setupCanvasPosition();
    }
    
    setupEventListeners() {
        // Canvas click events
        this.canvas.addEventListener('click', (e) => {
            // Handle manual reference point placement
            if (this.manualReferenceMode) {
                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // Validate coordinates are within displayed canvas bounds
                if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) {
                    console.log('Manual reference point click outside canvas bounds, ignoring');
                    return;
                }
                
                // Convert to normalized coordinates using DISPLAYED canvas dimensions
                const normalizedX = x / rect.width;
                const normalizedY = y / rect.height;
                
                console.log('📍 Manual reference point click:', {
                    clientX: e.clientX,
                    clientY: e.clientY,
                    canvasDisplayX: x,
                    canvasDisplayY: y,
                    normalizedX: normalizedX,
                    normalizedY: normalizedY,
                    displayedCanvasWidth: rect.width,
                    displayedCanvasHeight: rect.height
                });
                
                this.addManualReferencePoint(normalizedX, normalizedY);
                return;
            }
            
            // Handle boundary drawing
            if (!this.isDrawing) return;
            
            // Get the actual displayed canvas rectangle
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Validate coordinates are within displayed canvas bounds
            if (x < 0 || y < 0 || x >= rect.width || y >= rect.height) {
                console.log('Click outside displayed canvas bounds, ignoring');
                return;
            }
            
            // Convert to normalized coordinates using DISPLAYED canvas dimensions
            const normalizedX = x / rect.width;
            const normalizedY = y / rect.height;
            
            // Additional validation for normalized coordinates
            if (normalizedX < 0 || normalizedX > 1 || normalizedY < 0 || normalizedY > 1) {
                console.log('Invalid normalized coordinates, ignoring:', normalizedX, normalizedY);
                return;
            }
            
            console.log('FIXED click detected:', {
                clientX: e.clientX,
                clientY: e.clientY,
                canvasDisplayX: x,
                canvasDisplayY: y,
                normalizedX: normalizedX,
                normalizedY: normalizedY,
                displayedCanvasWidth: rect.width,
                displayedCanvasHeight: rect.height,
                internalCanvasWidth: this.canvas.width,
                internalCanvasHeight: this.canvas.height
            });
            
            this.addPoint(normalizedX, normalizedY);
        });
        
        // Mouse move for preview
        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.isDrawing) return;
            
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Only draw preview if mouse is within displayed canvas bounds
            if (x >= 0 && y >= 0 && x < rect.width && y < rect.height) {
                this.drawPreview(x, y);
            } else {
                // Clear preview when mouse leaves canvas
                this.redrawBoundaries();
            }
        });
        
        // Clear preview when mouse leaves canvas
        this.canvas.addEventListener('mouseleave', () => {
            if (this.isDrawing) {
                this.redrawBoundaries();
            }
        });
        
        // Button events
        document.getElementById('startDrawingBtn').addEventListener('click', () => {
            this.startDrawing();
        });
        
        document.getElementById('finishBoundaryBtn').addEventListener('click', () => {
            this.finishBoundary();
        });
        
        document.getElementById('cancelDrawingBtn').addEventListener('click', () => {
            this.cancelDrawing();
        });
        
        document.getElementById('clearAllBtn').addEventListener('click', () => {
            this.clearAllBoundaries();
        });
        
        document.getElementById('saveConfigBtn').addEventListener('click', () => {
            this.saveConfiguration();
        });
        
        document.getElementById('loadBoundariesBtn').addEventListener('click', () => {
            this.loadSavedBoundaries();
        });
        
        // Additional boundary tools
        document.getElementById('testCameraBtn').addEventListener('click', () => {
            this.testCamera();
        });
        
        document.getElementById('refreshFeedBtn').addEventListener('click', () => {
            this.refreshFeed();
        });
        
        document.getElementById('zoomInBtn').addEventListener('click', () => {
            this.zoomIn();
        });
        
        document.getElementById('zoomOutBtn').addEventListener('click', () => {
            this.zoomOut();
        });
        
        document.getElementById('resetZoomBtn').addEventListener('click', () => {
            this.resetZoom();
        });
        
        document.getElementById('saveBoundaryBtn').addEventListener('click', () => {
            this.saveBoundary();
        });
        
        document.getElementById('deleteBoundaryBtn').addEventListener('click', () => {
            this.deleteBoundary();
        });
        
        document.getElementById('editBoundaryBtn').addEventListener('click', () => {
            this.editBoundary();
        });
        
        document.getElementById('refreshListBtn').addEventListener('click', () => {
            this.refreshBoundaryList();
        });
        
        document.getElementById('exportBoundariesBtn').addEventListener('click', () => {
            this.exportBoundaries();
        });
        
        document.getElementById('resetConfigBtn').addEventListener('click', () => {
            this.resetConfiguration();
        });
        
        // Camera selection
        document.getElementById('cameraSelect').addEventListener('change', (e) => {
            this.switchCamera(e.target.value);
        });
        
        // Configuration controls
        document.getElementById('boundaryOpacity').addEventListener('input', (e) => {
            this.updateOpacity(e.target.value);
        });
        
        document.getElementById('referencePointTolerance').addEventListener('input', (e) => {
            this.updateTolerance(e.target.value);
        });
        
        // Reference Point System Event Handlers
        document.getElementById('showReferenceBtn').addEventListener('click', () => {
            this.toggleReferencePoints();
        });
        
        document.getElementById('calibrateRefBtn').addEventListener('click', () => {
            this.calibrateReferences();
        });
        
        document.getElementById('addReferenceBtn').addEventListener('click', () => {
            this.addReferencePoint();
        });
    }
    
    startDrawing() {
        console.log('Starting boundary drawing...');
        this.isDrawing = true;
        this.currentBoundary = [];
        this.updateDrawingButtons(true);
        
        // Make canvas visible and interactive
        this.canvas.style.cursor = 'crosshair';
        this.canvas.style.border = '2px dashed #ff0000';
        
        // Update drawing status
        const statusEl = document.getElementById('drawingStatus');
        if (statusEl) {
            statusEl.textContent = 'Drawing Active - Click to add points';
        }
        
        this.showStatus('Drawing mode active! Click on the video to place boundary points', 'info');
        console.log('Drawing mode activated');
    }
    
    addPoint(x, y) {
        console.log('Adding point:', x, y);
        this.currentBoundary.push([x, y]);
        
        // Update status display
        const statusEl = document.getElementById('drawingStatus');
        if (statusEl) {
            statusEl.textContent = `Drawing - ${this.currentBoundary.length} points placed`;
        }
        
        // Update UI displays
        this.updateUI();
        this.redrawBoundaries();
        
        this.showStatus(`Point ${this.currentBoundary.length} added`, 'success');
        console.log('Current boundary points:', this.currentBoundary.length);
    }
    
    finishBoundary() {
        if (this.currentBoundary.length < 3) {
            this.showStatus('Need at least 3 points to create a boundary', 'error');
            return;
        }
        
        console.log('🎯 ===== FINISH BOUNDARY DEBUG START =====');
        console.log('📍 Raw normalized coordinates from user clicks:', this.currentBoundary);
        
        // Get current canvas display info for debugging
        const rect = this.canvas.getBoundingClientRect();
        console.log('🖼️ Canvas display info:', {
            displayWidth: rect.width,
            displayHeight: rect.height,
            internalWidth: this.canvas.width,
            internalHeight: this.canvas.height
        });
        
        console.log('🔍 Getting actual video dimensions from server...');
        
        // Get actual video dimensions from server before conversion
        fetch('/api/video_dimensions')
        .then(response => response.json())
        .then(dimensionData => {
            console.log('📺 Server video dimensions response:', dimensionData);
            
            const actualVideoWidth = dimensionData.width;
            const actualVideoHeight = dimensionData.height;
            
            console.log('🖥️ Using ACTUAL server video dimensions:', { actualVideoWidth, actualVideoHeight });
            
            // Convert normalized coordinates (0-1) to actual server pixel coordinates
            console.log('🔄 Converting each point using ACTUAL dimensions:');
            const pixelBoundary = this.currentBoundary.map((point, index) => {
                const pixelX = Math.round(point[0] * actualVideoWidth);
                const pixelY = Math.round(point[1] * actualVideoHeight);
                console.log(`   Point ${index + 1}: normalized(${point[0].toFixed(3)}, ${point[1].toFixed(3)}) → pixels(${pixelX}, ${pixelY})`);
                return [pixelX, pixelY];
            });
            
            console.log('📦 Final pixel boundary for server (using actual dimensions):', pixelBoundary);
            console.log('📡 Sending to /save_boundary endpoint...');
            
            this.sendBoundaryToServer(pixelBoundary);
        })
        .catch(error => {
            console.error('❌ Error getting video dimensions, using fallback:', error);
            // Fallback to hardcoded dimensions if API fails
            const fallbackWidth = 640;
            const fallbackHeight = 480;
            console.log('🔄 Using fallback dimensions:', { fallbackWidth, fallbackHeight });
            
            const pixelBoundary = this.currentBoundary.map((point, index) => {
                const pixelX = Math.round(point[0] * fallbackWidth);
                const pixelY = Math.round(point[1] * fallbackHeight);
                console.log(`   Point ${index + 1}: normalized(${point[0].toFixed(3)}, ${point[1].toFixed(3)}) → pixels(${pixelX}, ${pixelY})`);
                return [pixelX, pixelY];
            });
            
            this.sendBoundaryToServer(pixelBoundary);
        });
    }
    
    sendBoundaryToServer(pixelBoundary) {
        // Send to server
        fetch('/save_boundary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                boundary: pixelBoundary
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('📥 Server response received:', data);
            
            if (data.status === 'success') {
                console.log('✅ Server successfully saved boundary');
                console.log('💾 Adding to local boundaries array:', this.currentBoundary);
                
                this.boundaries.push(this.currentBoundary);
                this.currentBoundary = [];
                this.isDrawing = false;
                
                // Reset canvas appearance
                this.canvas.style.cursor = 'default';
                this.canvas.style.border = 'none';
                
                // Update drawing status
                const statusEl = document.getElementById('drawingStatus');
                if (statusEl) {
                    statusEl.textContent = 'Ready';
                }
                
                console.log('🎨 Redrawing boundaries on client canvas...');
                console.log('📊 Total boundaries now:', this.boundaries.length);
                
                // Complete cleanup - clear canvas and redraw only saved boundaries
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.updateDrawingButtons(false);
                this.updateUI();
                this.redrawBoundaries();
                
                this.showStatus('Boundary saved successfully', 'success');
                console.log('🎯 ===== FINISH BOUNDARY DEBUG END =====');
            } else {
                console.log('❌ Server error saving boundary:', data.message);
                this.showStatus('Error saving boundary: ' + data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Error saving boundary:', error);
            this.showStatus('Error saving boundary', 'error');
        });
    }
    
    cancelDrawing() {
        console.log('Cancelling drawing...');
        this.currentBoundary = [];
        this.isDrawing = false;
        
        // Reset canvas appearance
        this.canvas.style.cursor = 'default';
        this.canvas.style.border = 'none';
        
        // Update drawing status
        const statusEl = document.getElementById('drawingStatus');
        if (statusEl) {
            statusEl.textContent = 'Ready';
        }
        
        // Complete cleanup - clear canvas and redraw only saved boundaries
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.updateDrawingButtons(false);
        this.updateUI();
        this.redrawBoundaries();
        
        this.showStatus('Drawing cancelled', 'info');
        console.log('Drawing cancelled and cleaned up');
    }
    
    clearAllBoundaries() {
        if (confirm('Are you sure you want to clear all boundaries?')) {
            console.log('Clearing all boundaries...');
            
            // Clear locally first
            this.boundaries = [];
            this.currentBoundary = [];
            this.isDrawing = false;
            
            // Reset canvas appearance
            this.canvas.style.cursor = 'default';
            this.canvas.style.border = 'none';
            
            // Update drawing status
            const statusEl = document.getElementById('drawingStatus');
            if (statusEl) {
                statusEl.textContent = 'Ready';
            }
            
            // Complete canvas cleanup - clear everything including any stray preview elements
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Force a fresh start
            this.updateDrawingButtons(false);
            this.updateUI();
            
            // Clear on server
            fetch('/clear_boundaries', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    this.showStatus('All boundaries cleared', 'success');
                    console.log('Server boundaries cleared');
                } else {
                    this.showStatus('Error clearing server boundaries: ' + data.message, 'error');
                }
            })
            .catch(error => {
                console.error('Error clearing boundaries:', error);
                this.showStatus('Error clearing boundaries', 'error');
            });
        }
    }
    
    drawPreview(mouseX, mouseY) {
        this.redrawBoundaries();
        
        if (this.currentBoundary.length > 0) {
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 5]);
            
            // Get the last point and convert from normalized coordinates to canvas pixels
            const lastPoint = this.currentBoundary[this.currentBoundary.length - 1];
            const lastX = lastPoint[0] * this.canvas.width;
            const lastY = lastPoint[1] * this.canvas.height;
            
            this.ctx.beginPath();
            this.ctx.moveTo(lastX, lastY);
            this.ctx.lineTo(mouseX, mouseY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
            
            // Draw preview point (green with white border)
            this.ctx.fillStyle = '#00ff00';
            this.ctx.beginPath();
            this.ctx.arc(mouseX, mouseY, 4, 0, 2 * Math.PI);
            this.ctx.fill();
            
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        }
    }
    
    redrawBoundaries() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Use canvas dimensions directly since canvas matches video exactly
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        // Draw completed boundaries (GREEN - safe areas)
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([]);
        
        this.boundaries.forEach(boundary => {
            if (boundary.length > 2) {
                this.ctx.beginPath();
                boundary.forEach((point, index) => {
                    // Convert normalized coordinates (0-1) to canvas pixels
                    const x = point[0] * canvasWidth;
                    const y = point[1] * canvasHeight;
                    if (index === 0) {
                        this.ctx.moveTo(x, y);
                    } else {
                        this.ctx.lineTo(x, y);
                    }
                });
                this.ctx.closePath();
                this.ctx.stroke();
                this.ctx.fill();
            }
        });
        
        // Draw current boundary being drawn (green - safe area being defined)
        if (this.currentBoundary.length > 0) {
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
            this.ctx.lineWidth = 3;
            
            // Draw the boundary line
            if (this.currentBoundary.length > 1) {
                this.ctx.beginPath();
                this.currentBoundary.forEach((point, index) => {
                    const x = point[0] * canvasWidth;
                    const y = point[1] * canvasHeight;
                    if (index === 0) {
                        this.ctx.moveTo(x, y);
                    } else {
                        this.ctx.lineTo(x, y);
                    }
                });
                
                // Close path if we have enough points for a polygon
                if (this.currentBoundary.length > 2) {
                    this.ctx.closePath();
                    this.ctx.fill();
                }
                this.ctx.stroke();
            }
            
            // Draw point circles for each point (green dots)
            this.ctx.fillStyle = '#00ff00';
            this.currentBoundary.forEach((point) => {
                const x = point[0] * canvasWidth;
                const y = point[1] * canvasHeight;
                this.ctx.beginPath();
                this.ctx.arc(x, y, 4, 0, 2 * Math.PI);
                this.ctx.fill();
                
                // Add white border for better visibility
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            });
        }
    }
    
    updateDrawingButtons(drawing) {
        document.getElementById('startDrawingBtn').disabled = drawing;
        document.getElementById('finishBoundaryBtn').disabled = !drawing || this.currentBoundary.length < 3;
        document.getElementById('cancelDrawingBtn').disabled = !drawing;
        document.getElementById('saveBoundaryBtn').disabled = this.boundaries.length === 0;
        document.getElementById('deleteBoundaryBtn').disabled = !this.selectedBoundary;
        document.getElementById('editBoundaryBtn').disabled = !this.selectedBoundary;
    }
    
    updateUI() {
        // Update boundary count if element exists
        const boundaryCountEl = document.getElementById('boundaryCount');
        if (boundaryCountEl) {
            boundaryCountEl.textContent = `Boundaries: ${this.boundaries.length}`;
        }
        
        // Update current points if element exists
        const currentPointsEl = document.getElementById('currentPoints');
        if (currentPointsEl) {
            currentPointsEl.textContent = `Current Points: ${this.currentBoundary.length}`;
        }
        
        // Update drawing status in the panel
        const drawingStatusEl = document.getElementById('drawingStatus');
        if (drawingStatusEl && !this.isDrawing) {
            drawingStatusEl.textContent = 'Ready';
        }
        
        this.updateDrawingButtons(this.isDrawing);
        console.log('UI updated - Boundaries:', this.boundaries.length, 'Current points:', this.currentBoundary.length);
    }
    
    loadBoundaries() {
        // Get actual video dimensions first, then load boundaries
        fetch('/api/video_dimensions')
        .then(response => response.json())
        .then(dimensionData => {
            const actualVideoWidth = dimensionData.width;
            const actualVideoHeight = dimensionData.height;
            
            console.log('Loading boundaries using actual server dimensions:', { actualVideoWidth, actualVideoHeight });
            
            // Now load boundaries with correct dimensions
            return fetch('/get_boundaries')
            .then(response => response.json())
            .then(data => {
                console.log('Raw boundary data from server:', data.boundaries);
                
                // Convert server pixel coordinates back to normalized coordinates (0-1)
                this.boundaries = data.boundaries.map(boundary => 
                    boundary.map(point => [point[0] / actualVideoWidth, point[1] / actualVideoHeight])
                );
                
                console.log('Converted normalized boundaries:', this.boundaries);
                this.redrawBoundaries();
                this.updateUI();
            });
        })
        .catch(error => {
            console.error('Error loading boundaries:', error);
            // Fallback to hardcoded dimensions if API fails
            const fallbackWidth = 640;
            const fallbackHeight = 480;
            console.log('Using fallback dimensions for loading:', { fallbackWidth, fallbackHeight });
            
            fetch('/get_boundaries')
            .then(response => response.json())
            .then(data => {
                this.boundaries = data.boundaries.map(boundary => 
                    boundary.map(point => [point[0] / fallbackWidth, point[1] / fallbackHeight])
                );
                this.redrawBoundaries();
                this.updateUI();
            });
        });
        
        // Also load reference point status on startup
        this.loadReferencePointStatus();
        
        // Debug: Log initialization state
        console.log('🔧 BoundaryDrawer initialization complete:', {
            manualReferenceMode: this.manualReferenceMode,
            isDrawing: this.isDrawing,
            canvasConfigured: !!this.canvas && !!this.ctx
        });
        
        // Load and display model status immediately on page load
        setTimeout(() => {
            console.log('🔍 Checking EfficientDet model status on page load...');
            this.loadReferencePointStatus();
        }, 1000); // Wait 1 second for server to be ready
    }
    
    loadReferencePointStatus() {
        // Load reference point status to update UI
        console.log('🔍 Loading reference point system status...');
        
        fetch('/api/reference_points/health')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                const health = data.health;
                const statusEl = document.getElementById('calibrationStatus');
                
                // Enhanced logging with model status
                console.log('🔧 ===== REFERENCE POINT SYSTEM STATUS =====');
                console.log('📍 Reference point system health:', health);
                console.log('🤖 Model Status:');
                console.log(`   - Dependencies Available: ${health.grounding_dino_available}`);
                console.log(`   - Model Loaded: ${health.model_loaded}`);
                if (health.model_status) {
                    console.log(`   - Model Type: ${health.model_status.model_type || 'None'}`);
                    console.log(`   - Model Instance Created: ${health.model_status.model_instance_created}`);
                    console.log(`   - Static Objects Only: ${health.model_status.static_objects_only}`);
                    console.log(`   - Supported Prompts: ${health.model_status.supported_prompts?.join(', ') || 'None'}`);
                    console.log(`   - Last Detection Frame: ${health.model_status.last_detection_attempt || 'Never'}`);
                }
                console.log(`📊 System Status:`);
                console.log(`   - Baseline Established: ${health.baseline_established}`);
                console.log(`   - Reference Points: ${health.reference_points_count}`);
                console.log(`   - Detection Errors: ${health.detection_errors}/${health.max_errors}`);
                console.log(`   - System Healthy: ${health.system_healthy}`);
                console.log('🔧 ===== STATUS CHECK COMPLETE =====');
                
                if (statusEl) {
                    if (health.system_healthy) {
                        statusEl.textContent = `Calibrated (${health.reference_points_count} points)`;
                        statusEl.className = 'status calibrated';
                    } else if (health.baseline_established) {
                        statusEl.textContent = `Degraded (${health.detection_errors}/${health.max_errors} errors)`;
                        statusEl.className = 'status degraded';
                    } else {
                        statusEl.textContent = 'Not Calibrated';
                        statusEl.className = 'status not-calibrated';
                    }
                    
                    // Add model status indicator
                    if (!health.grounding_dino_available) {
                        statusEl.textContent += ' (No Grounding DINO)';
                    } else if (!health.model_loaded) {
                        statusEl.textContent += ' (Model Failed)';
                    } else if (health.model_status?.static_objects_only) {
                        statusEl.textContent += ' (Static Objects)';
                    }
                }
                
                // Show model status in UI if there's a dedicated element
                const modelStatusEl = document.getElementById('modelStatus');
                if (modelStatusEl) {
                    if (health.model_loaded) {
                        modelStatusEl.textContent = 'Grounding DINO: Loaded (Static Objects)';
                        modelStatusEl.className = 'status model-loaded';
                    } else if (health.grounding_dino_available) {
                        modelStatusEl.textContent = 'Grounding DINO: Failed to Load';
                        modelStatusEl.className = 'status model-failed';
                    } else {
                        modelStatusEl.textContent = 'Grounding DINO: Dependencies Missing';
                        modelStatusEl.className = 'status model-missing';
                    }
                }
            }
        })
        .catch(error => {
            console.error('❌ Error loading reference point status:', error);
            console.log('   - This could indicate server connection issues');
            console.log('   - Or the reference point system is not running');
        });
    }
    
    // Add periodic health monitoring
    startHealthMonitoring() {
        console.log('🔄 Starting periodic reference point system monitoring (every 30 seconds)');
        
        // Check health every 30 seconds
        setInterval(() => {
            console.log('⏰ Periodic reference point system health check...');
            this.loadReferencePointStatus();
        }, 30000);
    }
    
    saveConfiguration() {
        fetch('/save_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            this.showStatus(data.message, data.status === 'success' ? 'success' : 'error');
        });
    }
    
    loadSavedBoundaries() {
        console.log('📂 ===== LOAD BOUNDARY COORDINATE DEBUG =====');
        
        // Get actual video dimensions first, then load saved boundaries
        fetch('/api/video_dimensions')
        .then(response => response.json())
        .then(dimensionData => {
            const actualVideoWidth = dimensionData.width;
            const actualVideoHeight = dimensionData.height;
            
            console.log('📺 Server video dimensions during load:', { actualVideoWidth, actualVideoHeight });
            
            // Get canvas dimensions for comparison
            const canvasWidth = this.canvas.width;
            const canvasHeight = this.canvas.height;
            const rect = this.canvas.getBoundingClientRect();
            
            console.log('🖼️ Canvas dimensions during load:', {
                internalWidth: canvasWidth,
                internalHeight: canvasHeight,
                displayWidth: rect.width,
                displayHeight: rect.height
            });
            
            // Load saved boundaries from file
            return fetch('/load_config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    console.log('📥 Raw saved boundary data from server:');
                    data.boundaries.forEach((boundary, idx) => {
                        console.log(`   Boundary ${idx + 1}: ${boundary.length} points`);
                        boundary.forEach((point, pointIdx) => {
                            console.log(`      Point ${pointIdx + 1}: pixel(${point[0]}, ${point[1]})`);
                        });
                    });
                    
                    console.log('🔄 Converting pixel coordinates to normalized coordinates:');
                    // Convert server pixel coordinates back to normalized coordinates (0-1)
                    this.boundaries = data.boundaries.map((boundary, idx) => {
                        console.log(`   Converting boundary ${idx + 1}:`);
                        return boundary.map((point, pointIdx) => {
                            const normalizedX = point[0] / actualVideoWidth;
                            const normalizedY = point[1] / actualVideoHeight;
                            console.log(`      Point ${pointIdx + 1}: pixel(${point[0]}, ${point[1]}) → normalized(${normalizedX.toFixed(3)}, ${normalizedY.toFixed(3)})`);
                            return [normalizedX, normalizedY];
                        });
                    });
                    
                    console.log('🎨 Final normalized boundaries for client display:');
                    this.boundaries.forEach((boundary, idx) => {
                        console.log(`   Boundary ${idx + 1}: ${boundary.length} normalized points`);
                    });
                    
                    this.redrawBoundaries();
                    this.updateUI();
                    this.showStatus(`Loaded ${this.boundaries.length} saved boundaries`, 'success');
                    console.log('📂 ===== LOAD BOUNDARY DEBUG END =====');
                } else {
                    this.showStatus('No saved boundaries found', 'info');
                }
            });
        })
        .catch(error => {
            console.error('Error loading saved boundaries:', error);
            this.showStatus('Error loading saved boundaries', 'error');
        });
    }
    
    loadConfiguration() {
        // This is the same as loadSavedBoundaries - they both load from the same file
        this.loadSavedBoundaries();
    }
    
    startCamera() {
        fetch('/start_camera')
        .then(response => response.json())
        .then(data => {
            this.showStatus(data.message, data.status === 'success' ? 'success' : 'error');
            if (data.status === 'success') {
                document.getElementById('cameraStatus').textContent = 'Camera Running';
                document.getElementById('cameraStatus').className = 'status running';
            }
        });
    }
    
    stopCamera() {
        fetch('/stop_camera')
        .then(response => response.json())
        .then(data => {
            this.showStatus(data.message, data.status === 'success' ? 'success' : 'error');
            if (data.status === 'success') {
                document.getElementById('cameraStatus').textContent = 'Camera Stopped';
                document.getElementById('cameraStatus').className = 'status stopped';
            }
        });
    }
    
    showStatus(message, type) {
        const statusDiv = document.getElementById('statusMessages');
        const messageElement = document.createElement('div');
        messageElement.className = `status-message ${type}`;
        messageElement.textContent = message;
        
        statusDiv.appendChild(messageElement);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (messageElement.parentNode) {
                messageElement.parentNode.removeChild(messageElement);
            }
        }, 5000);
    }
    
    // Additional boundary management methods
    testCamera() {
        const selectedCamera = document.getElementById('cameraSelect').value;
        this.showStatus(`Testing camera ${selectedCamera}...`, 'info');
        
        fetch('/api/camera/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera_id: selectedCamera })
        })
        .then(response => response.json())
        .then(data => {
            this.showStatus(data.message, data.status === 'success' ? 'success' : 'error');
        })
        .catch(error => {
            this.showStatus('Camera test failed', 'error');
        });
    }
    
    refreshFeed() {
        const selectedCamera = document.getElementById('cameraSelect').value;
        this.videoFeed.src = `/video_feed?camera=${selectedCamera}&t=${Date.now()}`;
        this.showStatus('Camera feed refreshed', 'info');
    }
    
    zoomIn() {
        const currentScale = parseFloat(this.videoFeed.style.transform.replace('scale(', '').replace(')', '') || '1');
        const newScale = Math.min(currentScale * 1.2, 3);
        this.videoFeed.style.transform = `scale(${newScale})`;
        this.showStatus(`Zoomed to ${Math.round(newScale * 100)}%`, 'info');
    }
    
    zoomOut() {
        const currentScale = parseFloat(this.videoFeed.style.transform.replace('scale(', '').replace(')', '') || '1');
        const newScale = Math.max(currentScale / 1.2, 0.5);
        this.videoFeed.style.transform = `scale(${newScale})`;
        this.showStatus(`Zoomed to ${Math.round(newScale * 100)}%`, 'info');
    }
    
    resetZoom() {
        this.videoFeed.style.transform = 'scale(1)';
        this.showStatus('Zoom reset to 100%', 'info');
    }
    
    saveBoundary() {
        if (this.boundaries.length === 0) {
            this.showStatus('No boundaries to save', 'warning');
            return;
        }
        
        console.log('💾 Saving boundaries to persistent storage...');
        
        // Save all boundaries to file
        fetch('/save_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('✅ Boundaries saved to boundary_config.json');
                
                // Clear server memory to prevent memory vs file conflicts
                return fetch('/clear_memory_boundaries', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
            } else {
                throw new Error(data.message);
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('✅ Server memory cleared after file save');
                
                // Clear client-side boundaries to match server state
                this.boundaries = [];
                this.redrawBoundaries();
                this.updateUI();
                
                this.showStatus('Boundaries saved to file and memory cleared', 'success');
            } else {
                console.warn('⚠️ File saved but memory clear failed:', data.message);
                this.showStatus('Boundaries saved (memory clear warning)', 'warning');
            }
        })
        .catch(error => {
            console.error('Error in save process:', error);
            this.showStatus('Error saving boundaries: ' + error.message, 'error');
        });
    }
    
    deleteBoundary() {
        if (this.selectedBoundary !== null) {
            if (confirm('Are you sure you want to delete this boundary?')) {
                this.boundaries.splice(this.selectedBoundary, 1);
                this.selectedBoundary = null;
                this.redrawBoundaries();
                this.updateUI();
                this.showStatus('Boundary deleted', 'success');
            }
        } else {
            this.showStatus('Select a boundary to delete', 'warning');
        }
    }
    
    editBoundary() {
        if (this.selectedBoundary !== null) {
            this.currentBoundary = [...this.boundaries[this.selectedBoundary]];
            this.boundaries.splice(this.selectedBoundary, 1);
            this.selectedBoundary = null;
            this.isDrawing = true;
            this.updateDrawingButtons(true);
            this.redrawBoundaries();
            this.updateUI();
            this.showStatus('Editing boundary - click to add/modify points', 'info');
        } else {
            this.showStatus('Select a boundary to edit', 'warning');
        }
    }
    
    refreshBoundaryList() {
        this.loadBoundaries();
        this.showStatus('Boundary list refreshed', 'info');
    }
    
    exportBoundaries() {
        const data = {
            boundaries: this.boundaries,
            timestamp: new Date().toISOString(),
            camera: document.getElementById('cameraSelect').value
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `boundaries_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
        this.showStatus('Boundaries exported', 'success');
    }
    
    resetConfiguration() {
        if (confirm('Reset all configuration to defaults?')) {
            document.getElementById('boundaryOpacity').value = '0.7';
            document.getElementById('referencePointTolerance').value = '10';
            document.getElementById('enableDynamicBoundaries').checked = true;
            document.getElementById('showBoundariesOnMonitor').checked = true;
            document.getElementById('autoRecalibrate').checked = true;
            
            // Also clear reference points if requested
            if (confirm('Also clear reference points?')) {
                this.clearReferencePoints();
            }
            
            this.showStatus('Configuration reset to defaults', 'success');
        }
    }
    
    clearReferencePoints() {
        console.log('🗑️ Clearing reference points');
        
        fetch('/api/reference_points/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                this.showStatus(data.message, 'success');
                
                // Update UI to show cleared status
                const statusEl = document.getElementById('calibrationStatus');
                if (statusEl) {
                    statusEl.textContent = 'Not Calibrated';
                    statusEl.className = 'status not-calibrated';
                }
            } else {
                this.showStatus('Error clearing reference points', 'error');
            }
        })
        .catch(error => {
            console.error('Error clearing reference points:', error);
            this.showStatus('Error communicating with server', 'error');
        });
    }
    
    switchCamera(cameraId) {
        this.videoFeed.src = `/video_feed?camera=${cameraId}`;
        this.showStatus(`Switched to camera ${cameraId}`, 'info');
        this.resizeCanvas();
    }
    
    updateOpacity(value) {
        const percentage = Math.round(value * 100);
        document.querySelector('#boundaryOpacity + .range-value').textContent = `${percentage}%`;
        // Apply opacity to boundary overlays
        this.redrawBoundaries();
    }
    
    updateTolerance(value) {
        document.querySelector('#referencePointTolerance + .range-value').textContent = `${value}px`;
        this.showStatus(`Movement tolerance set to ${value}px`, 'info');
    }
    
    // Reference Point System Methods
    toggleReferencePoints() {
        console.log('🎯 Toggle reference points visibility');
        
        // Check current health status first
        fetch('/api/reference_points/health')
        .then(response => response.json())
        .then(healthData => {
            if (healthData.status === 'success') {
                const health = healthData.health;
                
                if (!health.model_loaded) {
                    this.showStatus('Grounding DINO model not loaded - check installation', 'error');
                    return;
                }
                
                if (!health.baseline_established) {
                    this.showStatus('No baseline established - calibrate first', 'warning');
                    return;
                }
                
                if (!health.system_healthy) {
                    this.showStatus(`System unhealthy: ${health.detection_errors}/${health.max_errors} errors`, 'warning');
                }
                
                // Load reference points
                return fetch('/api/reference_points');
            } else {
                throw new Error('Health check failed');
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                if (data.reference_points.length > 0) {
                    this.showStatus(`${data.count} reference points active`, 'success');
                    this.displayReferencePointsInfo(data.reference_points);
                } else {
                    this.showStatus('No reference points established - calibrate first', 'warning');
                }
            } else {
                this.showStatus('Error loading reference points', 'error');
            }
        })
        .catch(error => {
            console.error('Error toggling reference points:', error);
            this.showStatus('Error communicating with server', 'error');
        });
    }
    
    displayReferencePointsInfo(referencePoints) {
        console.log('📍 Reference Points Details:');
        referencePoints.forEach((point, index) => {
            console.log(`   Point ${point.id}: ${point.type} at (${point.baseline_position[0]}, ${point.baseline_position[1]}) - confidence: ${point.confidence.toFixed(2)}`);
        });
        
        // Create summary for display
        const summary = referencePoints.reduce((acc, point) => {
            acc[point.type] = (acc[point.type] || 0) + 1;
            return acc;
        }, {});
        
        const summaryText = Object.entries(summary)
            .map(([type, count]) => `${count} ${type}${count > 1 ? 's' : ''}`)
            .join(', ');
        
        this.showStatus(`Reference points: ${summaryText}`, 'info');
    }
    
    calibrateReferences() {
        console.log('🎯 Starting reference point calibration');
        this.showStatus('Calibrating reference points...', 'info');
        
        // Disable calibration button during process
        const calibrateBtn = document.getElementById('calibrateRefBtn');
        if (calibrateBtn) {
            calibrateBtn.disabled = true;
            calibrateBtn.textContent = 'Calibrating...';
        }
        
        // Set timeout for calibration to prevent hanging
        const calibrationTimeout = setTimeout(() => {
            this.showStatus('Calibration timeout - check server', 'error');
            this.resetCalibrateButton();
        }, 30000); // 30 second timeout
        
        fetch('/api/reference_points/calibrate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            clearTimeout(calibrationTimeout);
            this.resetCalibrateButton();
            
            if (data.status === 'success') {
                console.log('✅ Calibration successful:', data);
                this.showStatus(data.message, 'success');
                
                // Show detailed results
                if (data.detection_details) {
                    const details = data.detection_details;
                    console.log(`📊 Detection Details: ${details.total_objects_detected} objects found, ${details.high_confidence_points} high-confidence points`);
                }
                
                // Update UI to show calibration status
                if (data.baseline_established) {
                    const statusEl = document.getElementById('calibrationStatus');
                    if (statusEl) {
                        statusEl.textContent = 'Calibrated';
                        statusEl.className = 'status calibrated';
                    }
                }
                
                // Display reference point information
                if (data.reference_points && data.reference_points.length > 0) {
                    this.displayReferencePointsInfo(data.reference_points);
                }
                
            } else if (data.status === 'warning') {
                console.warn('⚠️ Calibration warning:', data.message);
                this.showStatus(data.message, 'warning');
            } else {
                console.error('❌ Calibration failed:', data.message);
                this.showStatus('Calibration failed: ' + data.message, 'error');
            }
        })
        .catch(error => {
            clearTimeout(calibrationTimeout);
            this.resetCalibrateButton();
            console.error('Error during calibration:', error);
            this.showStatus('Calibration error - check connection', 'error');
        });
    }
    
    resetCalibrateButton() {
        const calibrateBtn = document.getElementById('calibrateRefBtn');
        if (calibrateBtn) {
            calibrateBtn.disabled = false;
            calibrateBtn.textContent = 'Calibrate References';
        }
    }
    
    addReferencePoint() {
        console.log('🎯 Toggle manual reference point mode');
        
        if (this.manualReferenceMode) {
            // Exit manual mode
            this.manualReferenceMode = false;
            this.canvas.style.cursor = 'default';
            this.canvas.style.border = 'none';
            this.showStatus('Manual reference point mode disabled', 'info');
            
            // Update button text
            const addRefBtn = document.getElementById('addReferenceBtn');
            if (addRefBtn) {
                addRefBtn.textContent = 'Add Reference Point';
                addRefBtn.classList.remove('active');
            }
        } else {
            // Enter manual mode
            this.manualReferenceMode = true;
            this.canvas.style.cursor = 'crosshair';
            this.canvas.style.border = '2px dashed #0066ff';
            this.showStatus('Manual reference point mode active - Click on video to place reference point', 'info');
            
            // Update button text and style
            const addRefBtn = document.getElementById('addReferenceBtn');
            if (addRefBtn) {
                addRefBtn.textContent = 'Exit Manual Mode';
                addRefBtn.classList.add('active');
            }
        }
        
        console.log('Manual reference mode:', this.manualReferenceMode);
    }
    
    addManualReferencePoint(normalizedX, normalizedY) {
        console.log('📍 Adding manual reference point at normalized coordinates:', { normalizedX, normalizedY });
        
        // Send to server
        fetch('/api/reference_points/add_manual', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                x: normalizedX,
                y: normalizedY
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('📍 Server response for manual reference point:', data);
            
            if (data.status === 'success') {
                this.showStatus(data.message, 'success');
                
                // Exit manual mode after successful placement
                this.manualReferenceMode = false;
                this.canvas.style.cursor = 'default';
                this.canvas.style.border = 'none';
                
                // Update button
                const addRefBtn = document.getElementById('addReferenceBtn');
                if (addRefBtn) {
                    addRefBtn.textContent = 'Add Reference Point';
                    addRefBtn.classList.remove('active');
                }
                
                // Update calibration status if baseline was established
                if (data.baseline_established) {
                    const statusEl = document.getElementById('calibrationStatus');
                    if (statusEl) {
                        statusEl.textContent = `Calibrated (${data.total_points} points)`;
                        statusEl.className = 'status calibrated';
                    }
                }
                
                console.log(`📍 Manual reference point added successfully. Total points: ${data.total_points}`);
                
            } else {
                this.showStatus('Error adding manual reference point: ' + data.message, 'error');
                console.error('❌ Failed to add manual reference point:', data.message);
            }
        })
        .catch(error => {
            console.error('❌ Error communicating with server for manual reference point:', error);
            this.showStatus('Error communicating with server', 'error');
        });
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing BoundaryDrawer with reference point support');
    const boundaryDrawer = new BoundaryDrawer();
    
    // Start health monitoring
    boundaryDrawer.startHealthMonitoring();
});