class DogManager {
    constructor() {
        this.dogs = [];
        this.selectedDog = null;
        this.uploadedImages = [];
        this.isTraining = false;
        this.trainingProgress = 0;
        this.captureStream = null;
        
        this.initializeEventListeners();
        this.loadDogs();
    }
    
    initializeEventListeners() {
        // Add new dog button
        const addDogBtn = document.getElementById('addDogBtn');
        if (addDogBtn) {
            addDogBtn.addEventListener('click', () => {
                this.showEnrollmentForm();
            });
        }
        
        // Add new dog card
        const addNewCard = document.getElementById('addNewDogCard');
        if (addNewCard) {
            addNewCard.addEventListener('click', () => {
                this.showEnrollmentForm();
            });
        }
        
        // Refresh dogs button
        const refreshBtn = document.getElementById('refreshDogsBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadDogs();
            });
        }
        
        // Image upload
        const uploadZone = document.getElementById('imageUploadZone');
        const imageInput = document.getElementById('imageInput');
        
        if (uploadZone && imageInput) {
            uploadZone.addEventListener('click', () => {
                imageInput.click();
            });
            
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('drag-over');
            });
            
            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('drag-over');
            });
            
            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('drag-over');
                this.handleImageFiles(e.dataTransfer.files);
            });
            
            imageInput.addEventListener('change', (e) => {
                this.handleImageFiles(e.target.files);
            });
        }
        
        // Upload action buttons
        const selectFolderBtn = document.getElementById('selectFolderBtn');
        if (selectFolderBtn) {
            selectFolderBtn.addEventListener('click', () => {
                this.selectFolder();
            });
        }
        
        const clearImagesBtn = document.getElementById('clearImagesBtn');
        if (clearImagesBtn) {
            clearImagesBtn.addEventListener('click', () => {
                this.clearAllImages();
            });
        }
        
        // Training controls
        const startTrainingBtn = document.getElementById('startTrainingBtn');
        if (startTrainingBtn) {
            startTrainingBtn.addEventListener('click', () => {
                this.startTraining();
            });
        }
        
        const pauseTrainingBtn = document.getElementById('pauseTrainingBtn');
        if (pauseTrainingBtn) {
            pauseTrainingBtn.addEventListener('click', () => {
                this.pauseTraining();
            });
        }
        
        const cancelTrainingBtn = document.getElementById('cancelTrainingBtn');
        if (cancelTrainingBtn) {
            cancelTrainingBtn.addEventListener('click', () => {
                this.cancelTraining();
            });
        }
        
        // Form controls
        const clearFormBtn = document.getElementById('clearFormBtn');
        if (clearFormBtn) {
            clearFormBtn.addEventListener('click', () => {
                this.clearEnrollmentForm();
            });
        }
        
        const saveAsDraftBtn = document.getElementById('saveAsDraftBtn');
        if (saveAsDraftBtn) {
            saveAsDraftBtn.addEventListener('click', () => {
                this.saveDraft();
            });
        }
        
        // Dog configuration
        const dogSelect = document.getElementById('configDogSelect');
        if (dogSelect) {
            dogSelect.addEventListener('change', (e) => {
                this.loadDogConfiguration(e.target.value);
            });
        }
        
        const saveConfigBtn = document.getElementById('saveConfigBtn');
        if (saveConfigBtn) {
            saveConfigBtn.addEventListener('click', () => {
                this.saveDogConfiguration();
            });
        }
        
        // Form validation
        const dogName = document.getElementById('dogName');
        if (dogName) {
            dogName.addEventListener('input', () => {
                this.validateForm();
            });
        }
        
        // Alert cooldown range
        const cooldownRange = document.getElementById('alertCooldown');
        if (cooldownRange) {
            cooldownRange.addEventListener('input', (e) => {
                const value = e.target.value;
                const label = e.target.nextElementSibling;
                if (label) {
                    label.textContent = `${value} seconds`;
                }
            });
        }
    }
    
    async loadDogs() {
        try {
            // Show loading state
            const statusIndicator = document.querySelector('.status-indicator');
            if (statusIndicator) {
                statusIndicator.textContent = 'Loading dogs...';
                statusIndicator.className = 'status-indicator loading';
            }
            
            const response = await fetch('/api/dogs');
            if (response.ok) {
                const data = await response.json();
                this.dogs = data.dogs || [];
                this.renderDogs();
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Error loading dogs:', error);
            this.showNotification(`Failed to load dogs: ${error.message}`, 'error');
            // Show empty state
            this.dogs = [];
            this.renderDogs();
        }
    }
    
    renderDogs() {
        const dogsGrid = document.querySelector('.dogs-grid');
        if (!dogsGrid) return;
        
        // Clear existing cards except the add new card
        const existingCards = dogsGrid.querySelectorAll('.dog-card:not(.add-new)');
        existingCards.forEach(card => card.remove());
        
        // Update dog count
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
            statusIndicator.textContent = `${this.dogs.length} Dogs Enrolled`;
        }
        
        // Add dog cards
        this.dogs.forEach(dog => {
            const card = this.createDogCard(dog);
            dogsGrid.insertBefore(card, dogsGrid.querySelector('.add-new'));
        });
        
        // Update dog selector in config
        this.updateDogSelector();
    }
    
    createDogCard(dog) {
        const card = document.createElement('div');
        card.className = `dog-card ${dog.active ? 'active' : 'inactive'}`;
        card.dataset.dog = dog.id;
        
        const lastSeen = dog.lastSeen ? this.formatLastSeen(dog.lastSeen) : 'Never';
        const accuracy = dog.accuracy || 0;
        
        card.innerHTML = `
            <div class="dog-image">
                <img src="${dog.image || '/static/images/dog-placeholder.svg'}" 
                     alt="${dog.name}" 
                     onerror="this.src='/static/images/dog-placeholder.svg'"
                     loading="lazy">
                <div class="dog-status ${dog.active ? 'online' : 'offline'}">
                    ${dog.active ? 'Active' : 'Inactive'}
                </div>
            </div>
            <div class="dog-info">
                <h4 class="dog-name">${dog.name}</h4>
                <div class="dog-details">
                    <div class="dog-detail">
                        <span class="detail-label">Breed:</span>
                        <span class="detail-value">${dog.breed || 'Unknown'}</span>
                    </div>
                    <div class="dog-detail">
                        <span class="detail-label">Last Seen:</span>
                        <span class="detail-value">${lastSeen}</span>
                    </div>
                    <div class="dog-detail">
                        <span class="detail-label">Training Status:</span>
                        <span class="detail-value ${accuracy > 80 ? 'status-good' : 'status-warning'}">
                            ${accuracy}% Accuracy
                        </span>
                    </div>
                </div>
                <div class="dog-actions">
                    <button class="btn btn-secondary btn-small" onclick="dogManager.editDog('${dog.id}')">Edit</button>
                    <button class="btn btn-secondary btn-small" onclick="dogManager.retrainDog('${dog.id}')">Retrain</button>
                    <button class="btn btn-secondary btn-small" onclick="dogManager.viewHistory('${dog.id}')">View History</button>
                </div>
            </div>
        `;
        
        return card;
    }
    
    updateDogSelector() {
        const selector = document.getElementById('configDogSelect');
        if (!selector) return;
        
        selector.innerHTML = '<option value="">Select a dog</option>';
        
        this.dogs.forEach(dog => {
            const option = document.createElement('option');
            option.value = dog.id;
            option.textContent = dog.name;
            selector.appendChild(option);
        });
    }
    
    showEnrollmentForm() {
        // Scroll to enrollment panel
        const enrollmentPanel = document.querySelector('.enrollment-panel');
        if (enrollmentPanel) {
            enrollmentPanel.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Clear form
        this.clearEnrollmentForm();
    }
    
    clearEnrollmentForm() {
        // Clear text inputs
        document.getElementById('dogName').value = '';
        document.getElementById('dogBreed').value = '';
        document.getElementById('dogAge').value = '';
        document.getElementById('dogDescription').value = '';
        
        // Clear uploaded images
        this.uploadedImages = [];
        this.updateImageDisplay();
        
        // Reset any UI state
        
        // Validate form
        this.validateForm();
    }
    
    handleImageFiles(files) {
        const validImages = [];
        const errors = [];
        const maxFileSize = 10 * 1024 * 1024; // 10MB
        const maxImages = 100;
        
        for (let file of files) {
            if (!file.type.startsWith('image/')) {
                errors.push(`${file.name}: Not an image file`);
                continue;
            }
            
            if (file.size > maxFileSize) {
                errors.push(`${file.name}: File too large (max 10MB)`);
                continue;
            }
            
            if (this.uploadedImages.length + validImages.length >= maxImages) {
                errors.push(`Maximum ${maxImages} images allowed`);
                break;
            }
            
            validImages.push(file);
        }
        
        if (errors.length > 0) {
            this.showNotification(`Upload errors: ${errors.join(', ')}`, 'warning');
        }
        
        if (validImages.length === 0) {
            return;
        }
        
        // Add to uploaded images
        this.uploadedImages.push(...validImages);
        this.updateImageDisplay();
        this.validateForm();
        
        this.showNotification(`${validImages.length} images added successfully`, 'success');
    }
    
    updateImageDisplay() {
        const uploadedSection = document.getElementById('uploadedImages');
        const imageGrid = document.getElementById('imageGrid');
        const imageCount = document.getElementById('imageCount');
        
        if (!uploadedSection || !imageGrid || !imageCount) return;
        
        if (this.uploadedImages.length === 0) {
            uploadedSection.style.display = 'none';
            return;
        }
        
        uploadedSection.style.display = 'block';
        imageCount.textContent = this.uploadedImages.length;
        
        // Clear grid
        imageGrid.innerHTML = '';
        
        // Display images
        this.uploadedImages.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const imageDiv = document.createElement('div');
                imageDiv.className = 'uploaded-image';
                imageDiv.innerHTML = `
                    <img src="${e.target.result}" alt="Dog image ${index + 1}">
                    <button class="remove-image" onclick="dogManager.removeImage(${index})">×</button>
                `;
                imageGrid.appendChild(imageDiv);
            };
            reader.readAsDataURL(file);
        });
    }
    
    removeImage(index) {
        this.uploadedImages.splice(index, 1);
        this.updateImageDisplay();
        this.validateForm();
    }
    
    selectFolder() {
        // Create a new input element to support folder selection
        const folderInput = document.createElement('input');
        folderInput.type = 'file';
        folderInput.webkitdirectory = true;
        folderInput.multiple = true;
        folderInput.accept = 'image/jpeg,image/jpg,image/png,image/webp';
        
        folderInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageFiles(e.target.files);
                this.showNotification(`Selected ${e.target.files.length} files from folder`, 'info');
            }
        });
        
        folderInput.click();
    }
    
    clearAllImages() {
        if (this.uploadedImages.length === 0) {
            this.showNotification('No images to clear', 'info');
            return;
        }
        
        const count = this.uploadedImages.length;
        this.uploadedImages = [];
        this.updateImageDisplay();
        this.validateForm();
        this.showNotification(`Cleared ${count} images`, 'success');
    }
    
    validateForm() {
        const startBtn = document.getElementById('startTrainingBtn');
        if (!startBtn) return;
        
        const nameInput = document.getElementById('dogName');
        const hasName = nameInput && nameInput.value.trim().length > 0;
        const hasImages = this.uploadedImages.length >= 5; // Minimum 5 images
        const minImages = 5;
        const recommendedImages = 20;
        
        // Update button state
        startBtn.disabled = !hasName || !hasImages || this.isTraining;
        
        // Update validation messages
        const hint = document.querySelector('.upload-hint');
        if (hint) {
            if (this.uploadedImages.length === 0) {
                hint.textContent = `Upload ${minImages}-${recommendedImages}+ images for best results`;
                hint.style.color = '#666';
            } else if (this.uploadedImages.length < minImages) {
                hint.textContent = `Upload ${minImages - this.uploadedImages.length} more images (minimum ${minImages})`;
                hint.style.color = '#ff6b6b';
            } else if (this.uploadedImages.length < recommendedImages) {
                hint.textContent = `${this.uploadedImages.length} images uploaded. Consider adding more for better accuracy.`;
                hint.style.color = '#ff9500';
            } else {
                hint.textContent = `${this.uploadedImages.length} images uploaded - excellent!`;
                hint.style.color = '#28a745';
            }
        }
        
        // Update name field validation
        if (nameInput) {
            if (hasName) {
                nameInput.classList.remove('invalid');
                nameInput.classList.add('valid');
            } else {
                nameInput.classList.remove('valid');
                if (nameInput.value.length > 0) {
                    nameInput.classList.add('invalid');
                }
            }
        }
    }
    
    async startTraining() {
        if (this.isTraining) return;
        
        const dogName = document.getElementById('dogName').value.trim();
        const dogBreed = document.getElementById('dogBreed').value.trim();
        const dogAge = document.getElementById('dogAge').value;
        const dogDescription = document.getElementById('dogDescription').value.trim();
        
        if (!dogName || this.uploadedImages.length < 5) {
            this.showNotification('Please provide a name and at least 5 images', 'error');
            return;
        }
        
        // Show training panel
        const trainingPanel = document.getElementById('trainingPanel');
        if (trainingPanel) {
            trainingPanel.style.display = 'block';
        }
        
        // Update training dog name
        const trainingDogName = document.getElementById('trainingDogName');
        if (trainingDogName) {
            trainingDogName.textContent = `Training: ${dogName}`;
        }
        
        // Prepare form data
        const formData = new FormData();
        formData.append('name', dogName);
        formData.append('breed', dogBreed);
        formData.append('age', dogAge);
        formData.append('description', dogDescription);
        
        this.uploadedImages.forEach((file, index) => {
            formData.append('images', file);
        });
        
        this.isTraining = true;
        this.trainingProgress = 0;
        
        // Show loading state
        const startBtn = document.getElementById('startTrainingBtn');
        if (startBtn) {
            startBtn.disabled = true;
            startBtn.textContent = 'Training...';
        }
        
        try {
            // Start training
            const response = await fetch('/api/dogs/train', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                this.monitorTraining(data.trainingId);
            } else {
                throw new Error('Failed to start training');
            }
        } catch (error) {
            console.error('Training error:', error);
            this.showNotification('Failed to start training', 'error');
            
            // Reset button state
            if (startBtn) {
                startBtn.disabled = false;
                startBtn.textContent = 'Start Training';
            }
            
            this.cancelTraining();
        }
    }
    
    async monitorTraining(trainingId) {
        const progressInterval = setInterval(async () => {
            if (!this.isTraining) {
                clearInterval(progressInterval);
                return;
            }
            
            try {
                const response = await fetch(`/api/dogs/training/${trainingId}`);
                if (response.ok) {
                    const data = await response.json();
                    this.updateTrainingProgress(data);
                    
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(progressInterval);
                        this.completeTraining(data);
                    }
                }
            } catch (error) {
                console.error('Error monitoring training:', error);
            }
        }, 1000); // Check every second
    }
    
    updateTrainingProgress(data) {
        const progressBar = document.getElementById('trainingProgress');
        const progressText = document.getElementById('trainingPercent');
        const stepText = document.getElementById('trainingStep');
        
        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
        }
        
        if (progressText) {
            progressText.textContent = `${data.progress}%`;
        }
        
        if (stepText) {
            stepText.textContent = data.currentStep || 'Processing...';
        }
        
        // Update stage status
        if (data.stage) {
            const stageEl = document.getElementById(`stage${data.stage}`);
            if (stageEl) {
                stageEl.classList.add('active');
                const statusEl = stageEl.querySelector('.stage-status');
                if (statusEl) {
                    statusEl.textContent = 'In Progress';
                }
            }
            
            // Mark previous stages as complete
            for (let i = 1; i < data.stage; i++) {
                const prevStage = document.getElementById(`stage${i}`);
                if (prevStage) {
                    prevStage.classList.add('complete');
                    const statusEl = prevStage.querySelector('.stage-status');
                    if (statusEl) {
                        statusEl.textContent = 'Complete';
                    }
                }
            }
        }
        
        // Update training log
        if (data.log) {
            const logContent = document.getElementById('trainingLog');
            if (logContent) {
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.textContent = data.log;
                logContent.appendChild(entry);
                logContent.scrollTop = logContent.scrollHeight;
            }
        }
    }
    
    completeTraining(data) {
        this.isTraining = false;
        
        if (data.status === 'completed') {
            this.showNotification('Training completed successfully!', 'success');
            
            // Reload dogs list
            this.loadDogs();
            
            // Clear form
            this.clearEnrollmentForm();
            
            // Hide training panel after delay
            setTimeout(() => {
                const trainingPanel = document.getElementById('trainingPanel');
                if (trainingPanel) {
                    trainingPanel.style.display = 'none';
                }
            }, 3000);
        } else {
            this.showNotification('Training failed. Please try again.', 'error');
        }
    }
    
    pauseTraining() {
        // Not implemented - would pause training
        this.showNotification('Training paused', 'info');
    }
    
    cancelTraining() {
        this.isTraining = false;
        
        const trainingPanel = document.getElementById('trainingPanel');
        if (trainingPanel) {
            trainingPanel.style.display = 'none';
        }
        
        this.showNotification('Training cancelled', 'warning');
    }
    
    async editDog(dogId) {
        const dog = this.dogs.find(d => d.id === dogId);
        if (!dog) return;
        
        // Populate form with dog data
        document.getElementById('dogName').value = dog.name;
        document.getElementById('dogBreed').value = dog.breed || '';
        document.getElementById('dogAge').value = dog.age || '';
        document.getElementById('dogDescription').value = dog.description || '';
        
        // Scroll to form
        this.showEnrollmentForm();
    }
    
    async retrainDog(dogId) {
        const dog = this.dogs.find(d => d.id === dogId);
        if (!dog) return;
        
        if (confirm(`Are you sure you want to retrain ${dog.name}? This will update their recognition model.`)) {
            try {
                const response = await fetch(`/api/dogs/${dogId}/retrain`, {
                    method: 'POST'
                });
                
                if (response.ok) {
                    this.showNotification(`Retraining ${dog.name}...`, 'info');
                    // Would monitor retraining progress
                } else {
                    this.showNotification('Failed to start retraining', 'error');
                }
            } catch (error) {
                console.error('Error retraining dog:', error);
                this.showNotification('Error starting retraining', 'error');
            }
        }
    }
    
    viewHistory(dogId) {
        // Navigate to history page with dog filter
        window.location.href = `/history?dog=${dogId}`;
    }
    
    async loadDogConfiguration(dogId) {
        if (!dogId) {
            document.getElementById('boundaryRules').style.display = 'none';
            document.getElementById('alertSettings').style.display = 'none';
            document.getElementById('saveConfigBtn').disabled = true;
            return;
        }
        
        this.selectedDog = dogId;
        
        try {
            const response = await fetch(`/api/dogs/${dogId}/config`);
            if (response.ok) {
                const config = await response.json();
                this.renderDogConfiguration(config);
            }
        } catch (error) {
            console.error('Error loading dog configuration:', error);
        }
        
        // Show configuration sections
        document.getElementById('boundaryRules').style.display = 'block';
        document.getElementById('alertSettings').style.display = 'block';
        document.getElementById('saveConfigBtn').disabled = false;
    }
    
    renderDogConfiguration(config) {
        // Update boundary rules
        // This would populate based on actual boundaries and config
        
        // Update alert settings
        if (config.alerts) {
            document.getElementById('emailAlerts').checked = config.alerts.email;
            document.getElementById('soundAlerts').checked = config.alerts.sound;
            document.getElementById('alertCooldown').value = config.alerts.cooldown || 30;
        }
    }
    
    async saveDogConfiguration() {
        if (!this.selectedDog) return;
        
        const config = {
            alerts: {
                email: document.getElementById('emailAlerts').checked,
                sound: document.getElementById('soundAlerts').checked,
                cooldown: parseInt(document.getElementById('alertCooldown').value)
            },
            boundaries: {} // Would collect boundary rules
        };
        
        try {
            const response = await fetch(`/api/dogs/${this.selectedDog}/config`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });
            
            if (response.ok) {
                this.showNotification('Configuration saved successfully', 'success');
            } else {
                this.showNotification('Failed to save configuration', 'error');
            }
        } catch (error) {
            console.error('Error saving configuration:', error);
            this.showNotification('Error saving configuration', 'error');
        }
    }
    
    async saveDraft() {
        const draft = {
            name: document.getElementById('dogName').value,
            breed: document.getElementById('dogBreed').value,
            age: document.getElementById('dogAge').value,
            description: document.getElementById('dogDescription').value,
            imageCount: this.uploadedImages.length
        };
        
        localStorage.setItem('dogEnrollmentDraft', JSON.stringify(draft));
        this.showNotification('Draft saved', 'success');
    }
    
    formatLastSeen(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) {
            return 'Just now';
        } else if (diff < 3600000) {
            const minutes = Math.floor(diff / 60000);
            return `${minutes} min ago`;
        } else if (diff < 86400000) {
            const hours = Math.floor(diff / 3600000);
            return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        } else {
            const days = Math.floor(diff / 86400000);
            return `${days} day${days > 1 ? 's' : ''} ago`;
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
}

// Initialize dog manager when page loads
let dogManager;
document.addEventListener('DOMContentLoaded', () => {
    dogManager = new DogManager();
});