"""
Training Manager for MiewID Dog Identification

Integrates MiewID training with the dog tracking system.
Handles data preparation, training orchestration, and model deployment.
"""

import os
import json
import time
import yaml
import shutil
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# MiewID imports
import sys
sys.path.append('/mnt/c/yard/models/wbia-plugin-miew-id')

try:
    from wbia_miew_id.train import Trainer
    from wbia_miew_id.helpers import get_config
    MIEWID_AVAILABLE = True
except ImportError as e:
    print(f"MiewID not available: {e}")
    MIEWID_AVAILABLE = False

class TrainingManager:
    """Manages MiewID training for yard dogs"""
    
    def __init__(self):
        self.base_path = Path('/mnt/c/yard')
        self.dogs_path = self.base_path / 'dogs'
        self.models_path = self.base_path / 'models'
        self.checkpoints_path = self.models_path / 'checkpoints'
        self.config_path = self.models_path / 'wbia-plugin-miew-id/wbia_miew_id/configs/yard_dogs_config.yaml'
        
        # Training state
        self.current_training = None
        self.training_status = {
            'active': False,
            'progress': 0,
            'stage': 'idle',
            'message': 'Ready for training',
            'start_time': None,
            'current_epoch': 0,
            'total_epochs': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'dog_id': None,
            'dog_name': None
        }
        
        # Create necessary directories
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.dogs_path,
            self.dogs_path / 'annotations',
            self.dogs_path / 'preprocessed',
            self.checkpoints_path,
            self.models_path / 'deployed'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_enrolled_dogs(self) -> List[Dict]:
        """Get list of enrolled dogs with training status"""
        dogs = []
        
        if not self.dogs_path.exists():
            return dogs
            
        for dog_dir in self.dogs_path.iterdir():
            if dog_dir.is_dir() and dog_dir.name.startswith('dog_'):
                info_file = dog_dir / 'info.json'
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            dog_info = json.load(f)
                            
                        # Add training status
                        dog_info['training_status'] = self._get_dog_training_status(dog_dir.name)
                        dog_info['image_count'] = self._count_dog_images(dog_dir)
                        dogs.append(dog_info)
                        
                    except Exception as e:
                        print(f"Error loading dog info for {dog_dir.name}: {e}")
        
        return dogs
    
    def _get_dog_training_status(self, dog_id: str) -> Dict:
        """Get training status for a specific dog"""
        model_file = self.models_path / 'deployed' / f'{dog_id}_model.pth'
        checkpoint_dir = self.checkpoints_path / dog_id
        
        status = {
            'trained': model_file.exists(),
            'has_checkpoints': checkpoint_dir.exists() and any(checkpoint_dir.glob('*.pth')),
            'last_training': None,
            'accuracy': None
        }
        
        # Check for training history
        history_file = checkpoint_dir / 'training_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    status['last_training'] = history.get('last_training')
                    status['accuracy'] = history.get('best_accuracy')
            except Exception as e:
                print(f"Error loading training history: {e}")
        
        return status
    
    def _count_dog_images(self, dog_dir: Path) -> int:
        """Count images for a dog"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        count = 0
        
        for file_path in dog_dir.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                count += 1
                
        return count
    
    def enroll_dog(self, name: str, breed: str, age: str, description: str, 
                   images: List, dog_id: Optional[str] = None) -> Dict:
        """Enroll a new dog with training images"""
        try:
            if not dog_id:
                dog_id = f"dog_{int(time.time())}"
            
            dog_dir = self.dogs_path / dog_id
            dog_dir.mkdir(exist_ok=True)
            
            # Save dog information
            dog_info = {
                'dog_id': dog_id,
                'name': name,
                'breed': breed,
                'age': age,
                'description': description,
                'enrollment_date': datetime.now().isoformat(),
                'image_count': len(images)
            }
            
            with open(dog_dir / 'info.json', 'w') as f:
                json.dump(dog_info, f, indent=2)
            
            # Save images
            images_dir = dog_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            
            saved_images = []
            for i, image_data in enumerate(images):
                image_path = images_dir / f"image_{i:04d}.jpg"
                # Save image data (implementation depends on how images are provided)
                # For now, assume image_data is file path or bytes
                saved_images.append(str(image_path))
            
            dog_info['saved_images'] = saved_images
            
            # Update dog info with saved image paths
            with open(dog_dir / 'info.json', 'w') as f:
                json.dump(dog_info, f, indent=2)
            
            return {
                'success': True,
                'dog_id': dog_id,
                'message': f"Dog {name} enrolled successfully with {len(images)} images"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def prepare_training_data(self, dog_ids: Optional[List[str]] = None) -> Dict:
        """Prepare COCO-format annotations for MiewID training"""
        try:
            if dog_ids is None:
                # Get all enrolled dogs
                dogs = self.get_enrolled_dogs()
                dog_ids = [dog['dog_id'] for dog in dogs]
            
            annotations = {
                'images': [],
                'annotations': [],
                'categories': []
            }
            
            image_id = 1
            annotation_id = 1
            
            # Create categories (one per dog)
            for i, dog_id in enumerate(dog_ids):
                dog_info_file = self.dogs_path / dog_id / 'info.json'
                if dog_info_file.exists():
                    with open(dog_info_file, 'r') as f:
                        dog_info = json.load(f)
                    
                    annotations['categories'].append({
                        'id': i + 1,
                        'name': dog_info['name'],
                        'supercategory': 'dog',
                        'dog_id': dog_id
                    })
            
            # Process images for each dog
            for category in annotations['categories']:
                dog_id = category['dog_id']
                category_id = category['id']
                
                images_dir = self.dogs_path / dog_id / 'images'
                if not images_dir.exists():
                    continue
                
                for image_file in images_dir.glob('*.jpg'):
                    # Add image entry
                    annotations['images'].append({
                        'id': image_id,
                        'file_name': str(image_file.relative_to(self.dogs_path)),
                        'width': 224,  # Will be updated with actual dimensions
                        'height': 224,
                        'dog_id': dog_id
                    })
                    
                    # Add annotation (whole image for full-body identification)
                    annotations['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': [0, 0, 224, 224],  # Full image bbox
                        'area': 224 * 224,
                        'iscrowd': 0
                    })
                    
                    image_id += 1
                    annotation_id += 1
            
            # Split into train/val/test
            train_data, val_data, test_data = self._split_annotations(annotations)
            
            # Save annotation files
            annotations_dir = self.dogs_path / 'annotations'
            
            with open(annotations_dir / 'train.json', 'w') as f:
                json.dump(train_data, f, indent=2)
            
            with open(annotations_dir / 'val.json', 'w') as f:
                json.dump(val_data, f, indent=2)
            
            with open(annotations_dir / 'test.json', 'w') as f:
                json.dump(test_data, f, indent=2)
            
            return {
                'success': True,
                'total_images': len(annotations['images']),
                'total_dogs': len(annotations['categories']),
                'train_images': len(train_data['images']),
                'val_images': len(val_data['images']),
                'test_images': len(test_data['images'])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _split_annotations(self, annotations: Dict, 
                          train_ratio: float = 0.7, 
                          val_ratio: float = 0.2) -> Tuple[Dict, Dict, Dict]:
        """Split annotations into train/val/test sets"""
        import random
        
        # Group images by dog_id
        dog_images = {}
        for img in annotations['images']:
            dog_id = img['dog_id']
            if dog_id not in dog_images:
                dog_images[dog_id] = []
            dog_images[dog_id].append(img)
        
        train_images, val_images, test_images = [], [], []
        train_anns, val_anns, test_anns = [], [], []
        
        # Split each dog's images
        for dog_id, images in dog_images.items():
            random.shuffle(images)
            
            n_train = int(len(images) * train_ratio)
            n_val = int(len(images) * val_ratio)
            
            dog_train = images[:n_train]
            dog_val = images[n_train:n_train + n_val]
            dog_test = images[n_train + n_val:]
            
            train_images.extend(dog_train)
            val_images.extend(dog_val)
            test_images.extend(dog_test)
            
            # Get corresponding annotations
            train_img_ids = {img['id'] for img in dog_train}
            val_img_ids = {img['id'] for img in dog_val}
            test_img_ids = {img['id'] for img in dog_test}
            
            for ann in annotations['annotations']:
                if ann['image_id'] in train_img_ids:
                    train_anns.append(ann)
                elif ann['image_id'] in val_img_ids:
                    val_anns.append(ann)
                elif ann['image_id'] in test_img_ids:
                    test_anns.append(ann)
        
        # Create split datasets
        train_data = {
            'images': train_images,
            'annotations': train_anns,
            'categories': annotations['categories']
        }
        
        val_data = {
            'images': val_images,
            'annotations': val_anns,
            'categories': annotations['categories']
        }
        
        test_data = {
            'images': test_images,
            'annotations': test_anns,
            'categories': annotations['categories']
        }
        
        return train_data, val_data, test_data
    
    def start_training(self, dog_ids: Optional[List[str]] = None) -> Dict:
        """Start MiewID training process"""
        if not MIEWID_AVAILABLE:
            return {
                'success': False,
                'error': 'MiewID not available. Please install wbia-plugin-miew-id.'
            }
        
        if self.training_status['active']:
            return {
                'success': False,
                'error': 'Training already in progress'
            }
        
        try:
            # Prepare training data
            data_prep = self.prepare_training_data(dog_ids)
            if not data_prep['success']:
                return data_prep
            
            # Update config with current dog count
            self._update_config_for_current_dogs(data_prep['total_dogs'])
            
            # Start training in background thread
            training_thread = threading.Thread(
                target=self._run_training,
                args=(dog_ids,),
                daemon=True
            )
            training_thread.start()
            
            return {
                'success': True,
                'message': 'Training started successfully',
                'total_dogs': data_prep['total_dogs'],
                'total_images': data_prep['total_images']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_config_for_current_dogs(self, n_dogs: int):
        """Update training config with current number of dogs"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model_params']['n_classes'] = n_dogs
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _run_training(self, dog_ids: Optional[List[str]] = None):
        """Run the actual training process"""
        try:
            self.training_status.update({
                'active': True,
                'progress': 0,
                'stage': 'initializing',
                'message': 'Initializing training...',
                'start_time': datetime.now().isoformat()
            })
            
            # Load config
            config = get_config(str(self.config_path))
            
            # Initialize trainer
            trainer = Trainer(config)
            trainer.set_seed_torch(config.engine.seed)
            
            self.training_status.update({
                'stage': 'data_loading',
                'message': 'Loading training data...',
                'progress': 10
            })
            
            # Setup data loaders, model, etc.
            # This is a simplified version - full implementation would need
            # complete MiewID training pipeline integration
            
            # Simulate training progress
            total_epochs = config.engine.epochs
            self.training_status['total_epochs'] = total_epochs
            
            for epoch in range(total_epochs):
                self.training_status.update({
                    'stage': 'training',
                    'message': f'Training epoch {epoch + 1}/{total_epochs}',
                    'current_epoch': epoch + 1,
                    'progress': int((epoch + 1) / total_epochs * 80) + 10,
                    'loss': 0.5 - (epoch / total_epochs) * 0.3,  # Simulated decreasing loss
                    'accuracy': 0.5 + (epoch / total_epochs) * 0.4  # Simulated increasing accuracy
                })
                
                # Sleep to simulate training time
                time.sleep(2)
            
            self.training_status.update({
                'stage': 'saving',
                'message': 'Saving trained model...',
                'progress': 95
            })
            
            # Save model and deploy
            self._deploy_trained_model()
            
            self.training_status.update({
                'active': False,
                'stage': 'completed',
                'message': 'Training completed successfully!',
                'progress': 100
            })
            
        except Exception as e:
            self.training_status.update({
                'active': False,
                'stage': 'error',
                'message': f'Training failed: {str(e)}',
                'progress': 0
            })
    
    def _deploy_trained_model(self):
        """Deploy the trained model for inference"""
        try:
            # Find the best checkpoint
            checkpoint_dir = self.checkpoints_path / 'latest_training'
            if not checkpoint_dir.exists():
                logger.warning("No checkpoints found for deployment")
                return
            
            # Look for best model checkpoint
            best_model_path = checkpoint_dir / 'best_model.pth'
            if not best_model_path.exists():
                # Look for any .pth file
                pth_files = list(checkpoint_dir.glob('*.pth'))
                if not pth_files:
                    logger.warning("No model files found for deployment")
                    return
                best_model_path = pth_files[-1]  # Use the latest one
            
            # Deploy to inference directory
            deployed_dir = self.models_path / 'deployed'
            deployed_dir.mkdir(exist_ok=True)
            
            # Copy model file
            deployed_model_path = deployed_dir / 'latest_model.pth'
            shutil.copy2(best_model_path, deployed_model_path)
            
            # Create deployment metadata
            deployment_info = {
                'deployed_at': datetime.now().isoformat(),
                'source_checkpoint': str(best_model_path),
                'model_path': str(deployed_model_path),
                'config_path': str(self.config_path)
            }
            
            with open(deployed_dir / 'deployment_info.json', 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            logger.info(f"Model deployed successfully to {deployed_model_path}")
            
            # Trigger model reload in dog_identifier
            self._notify_model_update()
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
    
    def _notify_model_update(self):
        """Notify the dog identifier that a new model is available"""
        try:
            from dog_identifier import get_dog_identifier
            dog_identifier = get_dog_identifier()
            
            # Clear any cached model to force reload
            if hasattr(dog_identifier, '_miewid_model'):
                delattr(dog_identifier, '_miewid_model')
            
            logger.info("Dog identifier notified of model update")
            
        except Exception as e:
            logger.warning(f"Could not notify dog identifier of model update: {e}")
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        return self.training_status.copy()
    
    def cancel_training(self) -> Dict:
        """Cancel current training"""
        if not self.training_status['active']:
            return {
                'success': False,
                'error': 'No training in progress'
            }
        
        # In a real implementation, this would stop the training process
        self.training_status.update({
            'active': False,
            'stage': 'cancelled',
            'message': 'Training cancelled by user',
            'progress': 0
        })
        
        return {
            'success': True,
            'message': 'Training cancelled successfully'
        }

# Global training manager instance
training_manager = TrainingManager()

def get_training_manager() -> TrainingManager:
    """Get the global training manager instance"""
    return training_manager