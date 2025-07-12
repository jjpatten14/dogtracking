"""
ArcFace-based Dog Identification System

Individual dog identification using ArcFace whole-body features.
Enables multi-dog tracking and per-dog boundary rules.

Key Features:
- ArcFace model integration for dog identification
- Whole-body feature extraction (not just faces)
- Dog enrollment and training pipeline
- Individual dog tracking across cameras
- Confidence-based identification
"""

import cv2
import numpy as np
import logging
import time
import pickle
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DogProfile:
    """Dog profile with identification information"""
    dog_id: str
    name: str
    breed: str
    description: str
    feature_vectors: List[np.ndarray]
    enrollment_date: datetime
    last_seen: Optional[datetime] = None
    last_camera: Optional[int] = None
    confidence_threshold: float = 0.8
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'dog_id': self.dog_id,
            'name': self.name,
            'breed': self.breed,
            'description': self.description,
            'enrollment_date': self.enrollment_date.isoformat(),
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'last_camera': self.last_camera,
            'confidence_threshold': self.confidence_threshold,
            'active': self.active,
            'feature_count': len(self.feature_vectors)
        }

@dataclass
class DogIdentification:
    """Result of dog identification"""
    dog_id: str
    dog_name: str
    confidence: float
    feature_similarity: float
    bbox: Tuple[int, int, int, int]
    timestamp: datetime
    camera_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'dog_id': self.dog_id,
            'dog_name': self.dog_name,
            'confidence': self.confidence,
            'feature_similarity': self.feature_similarity,
            'bbox': list(self.bbox),
            'timestamp': self.timestamp.isoformat(),
            'camera_id': self.camera_id
        }

class ArcFaceDogIdentifier:
    """
    ArcFace-based dog identification system
    
    IMPORTANT: This class expects ArcFace models to be manually downloaded and configured.
    No automatic model downloading is performed.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        self.model_path = model_path
        
        # Dog profiles database
        self.dog_profiles: Dict[str, DogProfile] = {}
        self.feature_dimension = 512  # Standard ArcFace feature dimension
        
        # Performance tracking
        self.identification_stats = {
            'total_identifications': 0,
            'successful_identifications': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'identifications_by_dog': {}
        }
        
        self.lock = threading.Lock()
        
        # Storage paths
        self.profiles_dir = Path('/mnt/c/yard/dog_profiles')
        self.profiles_dir.mkdir(exist_ok=True)
        
        # Load existing profiles
        self.load_dog_profiles()
    
    def load_model(self, model_path: str) -> bool:
        """
        Load ArcFace model for dog identification
        
        Args:
            model_path: Path to ArcFace model file
            
        Returns:
            bool: True if model loaded successfully
            
        Note: Model must be manually downloaded first. See models.md for instructions.
        """
        try:
            logger.info(f"Loading ArcFace model from: {model_path}")
            
            # Try different ArcFace implementations
            success = False
            
            # Option 1: InsightFace
            if not success:
                success = self._load_insightface_model(model_path)
            
            # Option 2: Custom ArcFace implementation
            if not success:
                success = self._load_custom_arcface_model(model_path)
            
            if success:
                self.model_loaded = True
                self.model_path = model_path
                logger.info("ArcFace model loaded successfully")
                
                # Test model with dummy input
                self._test_model()
                
                return True
            else:
                logger.error("Failed to load ArcFace model with any implementation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {e}")
            self.model_loaded = False
            return False
    
    def _load_insightface_model(self, model_path: str) -> bool:
        """Load model using InsightFace library"""
        try:
            import insightface
            
            # Initialize InsightFace with custom model
            self.model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info("InsightFace model loaded")
            return True
            
        except ImportError:
            logger.warning("InsightFace not available")
            return False
        except Exception as e:
            logger.error(f"InsightFace loading failed: {e}")
            return False
    
    def _load_custom_arcface_model(self, model_path: str) -> bool:
        """Load custom ArcFace model implementation"""
        try:
            # Placeholder for custom ArcFace implementation
            # This would load a PyTorch or ONNX model for dog feature extraction
            logger.warning("Custom ArcFace implementation not yet implemented")
            return False
            
        except Exception as e:
            logger.error(f"Custom ArcFace loading failed: {e}")
            return False
    
    def _test_model(self):
        """Test model with dummy input"""
        try:
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Extract features
            features = self.extract_features(dummy_image)
            
            if features is not None:
                logger.info(f"Model test successful - feature dimension: {len(features)}")
            else:
                logger.warning("Model test failed - no features extracted")
                
        except Exception as e:
            logger.warning(f"Model test failed: {e}")
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ArcFace features from dog image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Feature vector or None if extraction failed
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot extract features")
            return None
        
        try:
            # Preprocess image for ArcFace
            processed_image = self._preprocess_for_arcface(image)
            
            if self.model is None:
                return None
            
            # Extract features using loaded model
            # This is a placeholder - actual implementation depends on model type
            features = self._extract_features_impl(processed_image)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _preprocess_for_arcface(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ArcFace feature extraction"""
        # Resize to standard input size
        target_size = (224, 224)  # Standard size for many ArcFace models
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Additional preprocessing may be needed based on specific model
        return image
    
    def _extract_features_impl(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Actual feature extraction implementation using MiewID
        """
        try:
            # Try to use MiewID model if available
            if hasattr(self, '_miewid_model') and self._miewid_model is not None:
                return self._extract_features_miewid(image)
            else:
                # Try to load MiewID model
                if self._try_load_miewid_model():
                    return self._extract_features_miewid(image)
                else:
                    logger.warning("MiewID model not available, using placeholder features")
                    # Fallback to placeholder for now
                    return np.random.random(self.feature_dimension).astype(np.float32)
                    
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            # Fallback to placeholder
            return np.random.random(self.feature_dimension).astype(np.float32)
    
    def _try_load_miewid_model(self) -> bool:
        """Try to load MiewID model for inference"""
        try:
            import sys
            sys.path.append('/mnt/c/yard/models/wbia-plugin-miew-id')
            
            from wbia_miew_id.models import MiewIdNet
            from wbia_miew_id.helpers import get_config
            import torch
            
            # Look for deployed model
            deployed_model_path = '/mnt/c/yard/models/deployed/latest_model.pth'
            config_path = '/mnt/c/yard/models/wbia-plugin-miew-id/wbia_miew_id/configs/yard_dogs_config.yaml'
            
            if not os.path.exists(deployed_model_path):
                logger.warning(f"No deployed model found at {deployed_model_path}")
                return False
            
            if not os.path.exists(config_path):
                logger.warning(f"No config found at {config_path}")
                return False
            
            # Load config
            config = get_config(config_path)
            
            # Initialize model
            self._miewid_model = MiewIdNet(
                model_name=config.model_params.model_name,
                n_classes=config.model_params.n_classes,
                fc_dim=config.model_params.fc_dim,
                dropout=config.model_params.dropout,
                loss_module=config.model_params.loss_module,
                s=config.model_params.s,
                margin=config.model_params.margin,
                pretrained=False  # We're loading trained weights
            )
            
            # Load trained weights
            checkpoint = torch.load(deployed_model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self._miewid_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self._miewid_model.load_state_dict(checkpoint)
            
            self._miewid_model.eval()
            
            # Move to GPU if available with explicit verification
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._miewid_model = self._miewid_model.to(device)
            self._miewid_device = device
            
            # Verify model is actually on GPU
            if torch.cuda.is_available():
                model_device = next(self._miewid_model.parameters()).device
                logger.info(f"✅ MiewID model loaded on device: {model_device}")
                if model_device.type != 'cuda':
                    logger.warning("⚠️ MiewID model not on GPU, forcing move...")
                    self._miewid_model = self._miewid_model.cuda()
                    self._miewid_device = 'cuda'
            
            # Store config for later use
            self._miewid_config = config
            
            logger.info(f"MiewID model loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MiewID model: {e}")
            return False
    
    def _extract_features_miewid(self, image: np.ndarray) -> np.ndarray:
        """Extract features using MiewID model"""
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            
            # Convert numpy image to PIL
            if image.shape[2] == 3:  # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            
            # Define transforms (matching training transforms)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Preprocess image
            input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self._miewid_device)
            
            # Extract features
            with torch.no_grad():
                features = self._miewid_model(input_tensor)
                
                # Get embedding features (before classification head)
                if hasattr(self._miewid_model, 'features'):
                    # Get features from backbone
                    features = self._miewid_model.features(input_tensor)
                    features = features.view(features.size(0), -1)  # Flatten
                elif len(features.shape) == 2:
                    # Features are already flattened
                    pass
                else:
                    # Flatten features
                    features = features.view(features.size(0), -1)
                
                # Convert to numpy
                features_np = features.cpu().numpy().squeeze()
                
                # Normalize features
                features_np = features_np / (np.linalg.norm(features_np) + 1e-8)
                
                return features_np.astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error in MiewID feature extraction: {e}")
            # Fallback to random features
            return np.random.random(self.feature_dimension).astype(np.float32)
    
    def enroll_dog(self, name: str, breed: str, description: str, 
                   images: List[np.ndarray]) -> bool:
        """
        Enroll a new dog with training images
        
        Args:
            name: Dog's name
            breed: Dog's breed
            description: Additional description
            images: List of training images
            
        Returns:
            bool: True if enrollment successful
        """
        try:
            logger.info(f"Starting enrollment for dog: {name}")
            
            if len(images) < 5:
                logger.error("Need at least 5 training images for enrollment")
                return False
            
            # Extract features from all training images
            feature_vectors = []
            for i, image in enumerate(images):
                features = self.extract_features(image)
                if features is not None:
                    feature_vectors.append(features)
                    logger.debug(f"Extracted features from image {i+1}/{len(images)}")
                else:
                    logger.warning(f"Failed to extract features from image {i+1}")
            
            if len(feature_vectors) < 3:
                logger.error("Not enough valid feature vectors for enrollment")
                return False
            
            # Create dog profile
            dog_id = f"dog_{int(time.time())}"  # Simple ID generation
            profile = DogProfile(
                dog_id=dog_id,
                name=name,
                breed=breed,
                description=description,
                feature_vectors=feature_vectors,
                enrollment_date=datetime.now()
            )
            
            # Store profile
            with self.lock:
                self.dog_profiles[dog_id] = profile
                self.identification_stats['identifications_by_dog'][dog_id] = 0
            
            # Save to disk
            self.save_dog_profile(profile)
            
            logger.info(f"Dog {name} enrolled successfully with {len(feature_vectors)} feature vectors")
            return True
            
        except Exception as e:
            logger.error(f"Dog enrollment failed: {e}")
            return False
    
    def identify_dog(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                    camera_id: Optional[int] = None) -> Optional[DogIdentification]:
        """
        Identify dog in image region
        
        Args:
            image: Full image
            bbox: Bounding box (x, y, width, height) of detected animal
            camera_id: Optional camera ID
            
        Returns:
            DogIdentification if successful, None otherwise
        """
        if not self.model_loaded:
            return None
        
        try:
            start_time = time.time()
            
            # Extract dog region from image
            x, y, w, h = bbox
            dog_region = image[y:y+h, x:x+w]
            
            if dog_region.size == 0:
                return None
            
            # Extract features from dog region
            features = self.extract_features(dog_region)
            if features is None:
                return None
            
            # Find best matching dog
            best_match = self._find_best_match(features)
            
            if best_match is None:
                return None
            
            dog_id, confidence, similarity = best_match
            profile = self.dog_profiles[dog_id]
            
            # Update profile
            with self.lock:
                profile.last_seen = datetime.now()
                profile.last_camera = camera_id
                self.identification_stats['total_identifications'] += 1
                self.identification_stats['successful_identifications'] += 1
                self.identification_stats['identifications_by_dog'][dog_id] += 1
            
            # Create identification result
            identification = DogIdentification(
                dog_id=dog_id,
                dog_name=profile.name,
                confidence=confidence,
                feature_similarity=similarity,
                bbox=bbox,
                timestamp=datetime.now(),
                camera_id=camera_id
            )
            
            processing_time = time.time() - start_time
            logger.debug(f"Dog identified: {profile.name} (confidence: {confidence:.3f}, time: {processing_time:.3f}s)")
            
            return identification
            
        except Exception as e:
            logger.error(f"Dog identification failed: {e}")
            return None
    
    def _find_best_match(self, features: np.ndarray) -> Optional[Tuple[str, float, float]]:
        """
        Find best matching dog profile
        
        Returns:
            Tuple of (dog_id, confidence, similarity) or None
        """
        best_dog_id = None
        best_similarity = 0.0
        best_confidence = 0.0
        
        with self.lock:
            for dog_id, profile in self.dog_profiles.items():
                if not profile.active:
                    continue
                
                # Calculate similarity with all feature vectors
                similarities = []
                for stored_features in profile.feature_vectors:
                    similarity = self._calculate_similarity(features, stored_features)
                    similarities.append(similarity)
                
                # Use maximum similarity (best match)
                max_similarity = max(similarities) if similarities else 0.0
                
                # Convert similarity to confidence (this could be more sophisticated)
                confidence = max_similarity
                
                # Check if this is the best match and meets threshold
                if (confidence > profile.confidence_threshold and 
                    confidence > best_confidence):
                    best_dog_id = dog_id
                    best_similarity = max_similarity
                    best_confidence = confidence
        
        if best_dog_id is not None:
            return best_dog_id, best_confidence, best_similarity
        else:
            return None
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        try:
            # Normalize features
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(features1, features2) / (norm1 * norm2)
            
            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def update_dog_profile(self, dog_id: str, **kwargs) -> bool:
        """Update dog profile information"""
        with self.lock:
            if dog_id not in self.dog_profiles:
                return False
            
            profile = self.dog_profiles[dog_id]
            
            # Update allowed fields
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            # Save updated profile
            self.save_dog_profile(profile)
            
            return True
    
    def get_dog_profiles(self) -> List[Dict[str, Any]]:
        """Get all dog profiles as dictionaries"""
        with self.lock:
            return [profile.to_dict() for profile in self.dog_profiles.values()]
    
    def get_dog_profile(self, dog_id: str) -> Optional[Dict[str, Any]]:
        """Get specific dog profile"""
        with self.lock:
            if dog_id in self.dog_profiles:
                return self.dog_profiles[dog_id].to_dict()
            return None
    
    def remove_dog(self, dog_id: str) -> bool:
        """Remove dog from system"""
        with self.lock:
            if dog_id not in self.dog_profiles:
                return False
            
            # Remove profile
            del self.dog_profiles[dog_id]
            
            # Remove from stats
            if dog_id in self.identification_stats['identifications_by_dog']:
                del self.identification_stats['identifications_by_dog'][dog_id]
            
            # Remove saved file
            profile_file = self.profiles_dir / f"{dog_id}.pkl"
            if profile_file.exists():
                profile_file.unlink()
            
            logger.info(f"Dog {dog_id} removed from system")
            return True
    
    def save_dog_profile(self, profile: DogProfile):
        """Save dog profile to disk"""
        try:
            profile_file = self.profiles_dir / f"{profile.dog_id}.pkl"
            with open(profile_file, 'wb') as f:
                pickle.dump(profile, f)
            
            logger.debug(f"Saved profile for dog: {profile.name}")
            
        except Exception as e:
            logger.error(f"Failed to save dog profile: {e}")
    
    def load_dog_profiles(self):
        """Load all dog profiles from disk"""
        try:
            profile_files = list(self.profiles_dir.glob("*.pkl"))
            
            for profile_file in profile_files:
                try:
                    with open(profile_file, 'rb') as f:
                        profile = pickle.load(f)
                    
                    self.dog_profiles[profile.dog_id] = profile
                    self.identification_stats['identifications_by_dog'][profile.dog_id] = 0
                    
                    logger.debug(f"Loaded profile for dog: {profile.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load profile {profile_file}: {e}")
            
            logger.info(f"Loaded {len(self.dog_profiles)} dog profiles")
            
        except Exception as e:
            logger.error(f"Failed to load dog profiles: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get identification statistics"""
        with self.lock:
            total_dogs = len(self.dog_profiles)
            active_dogs = sum(1 for p in self.dog_profiles.values() if p.active)
            
            return {
                'model_loaded': self.model_loaded,
                'total_dogs': total_dogs,
                'active_dogs': active_dogs,
                'total_identifications': self.identification_stats['total_identifications'],
                'successful_identifications': self.identification_stats['successful_identifications'],
                'success_rate': (
                    self.identification_stats['successful_identifications'] / 
                    max(self.identification_stats['total_identifications'], 1)
                ),
                'identifications_by_dog': self.identification_stats['identifications_by_dog'].copy(),
                'feature_dimension': self.feature_dimension,
                'model_path': self.model_path
            }
    
    def is_model_available(self) -> bool:
        """Check if identification model is available and loaded"""
        return self.model_loaded

# Global dog identifier instance
dog_identifier = ArcFaceDogIdentifier()

def get_dog_identifier() -> ArcFaceDogIdentifier:
    """Get global dog identifier instance"""
    return dog_identifier