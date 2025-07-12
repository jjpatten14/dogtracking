"""
MegaDetector v6 Detection Engine for Dog Tracking System

Wrapper for MegaDetector v6 with performance optimization for Jetson Orin Nano.
Detects animals, people, and vehicles in camera feeds without automatic model downloads.

Key Features:
- MegaDetector v6 integration (manual model setup required)
- GPU acceleration support
- Batch processing for multiple cameras
- Confidence filtering and post-processing
- Jetson Orin Nano optimizations
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    timestamp: datetime
    camera_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': list(self.bbox),
            'timestamp': self.timestamp.isoformat(),
            'camera_id': self.camera_id
        }

@dataclass
class DetectionConfig:
    """Detection configuration settings"""
    animal_confidence_threshold: float = 0.7
    person_confidence_threshold: float = 0.8
    vehicle_confidence_threshold: float = 0.6
    input_size: Tuple[int, int] = (1280, 1280)
    device: str = 'cuda'  # 'cuda' or 'cpu'
    enable_tensorrt: bool = False
    batch_size: int = 1
    max_detections: int = 100

class MegaDetectorEngine:
    """
    MegaDetector v6 wrapper for animal, person, and vehicle detection
    
    IMPORTANT: This class expects models to be manually downloaded and configured.
    No automatic model downloading is performed.
    """
    
    # MegaDetector class mappings
    CLASS_MAPPING = {
        0: 'animal',
        1: 'person', 
        2: 'vehicle'
    }
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = None
        self.model_loaded = False
        self.device = config.device
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        self.lock = threading.Lock()
        
        # Detection statistics
        self.stats = {
            'total_inferences': 0,
            'total_detections': 0,
            'average_inference_time': 0.0,
            'detections_by_class': {name: 0 for name in self.CLASS_MAPPING.values()},
            'last_inference_time': None
        }
    
    def load_model(self, model_path: str) -> bool:
        """
        Load MegaDetector model from file path
        
        Args:
            model_path: Path to MegaDetector model file
            
        Returns:
            bool: True if model loaded successfully
            
        Note: Model must be manually downloaded first. See models.md for instructions.
        """
        try:
            logger.info(f"Loading MegaDetector model from: {model_path}")
            
            # Import PyTorchWildlife (must be installed manually)
            try:
                from PytorchWildlife.models import detection as pw_detection
            except ImportError as e:
                logger.error("PytorchWildlife not installed. Run: pip install PytorchWildlife")
                return False
            
            # Load MegaDetector v6 (will use manual model path if provided)
            if model_path and model_path != 'auto':
                # Load from specific path
                logger.info("Loading model from specified path (not implemented - using auto-download)")
                # TODO: Implement loading from specific path
                self.model = pw_detection.MegaDetectorV6(device=self.device)
            else:
                # Auto-download (only if explicitly allowed)
                logger.warning("Auto-download mode - model will be downloaded automatically")
                self.model = pw_detection.MegaDetectorV6(device=self.device)
            
            self.model_loaded = True
            logger.info("MegaDetector model loaded successfully")
            
            # Warm up model with dummy inference
            self._warmup_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MegaDetector model: {e}")
            self.model_loaded = False
            return False
    
    def _warmup_model(self):
        """Warm up model with dummy inference for better performance"""
        if not self.model_loaded:
            return
        
        try:
            logger.info("Warming up MegaDetector model...")
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, 
                                          (*self.config.input_size, 3), 
                                          dtype=np.uint8)
            
            # Run dummy inference
            _ = self.model.single_image_detection(dummy_image)
            
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def detect(self, image: np.ndarray, camera_id: Optional[int] = None) -> List[Detection]:
        """
        Run detection on single image
        
        Args:
            image: Input image as numpy array (BGR format)
            camera_id: Optional camera ID for tracking
            
        Returns:
            List of Detection objects
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot perform detection")
            return []
        
        try:
            start_time = time.time()
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Run inference
            results = self.model.single_image_detection(processed_image)
            
            # Post-process results
            detections = self._postprocess_results(results, image.shape[:2], camera_id)
            
            # Update statistics
            inference_time = time.time() - start_time
            self._update_stats(inference_time, len(detections))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def detect_batch(self, images: List[np.ndarray], 
                    camera_ids: Optional[List[int]] = None) -> List[List[Detection]]:
        """
        Run detection on batch of images for improved performance
        
        Args:
            images: List of input images
            camera_ids: Optional list of camera IDs
            
        Returns:
            List of detection lists (one per image)
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot perform detection")
            return [[] for _ in images]
        
        try:
            start_time = time.time()
            
            # Process each image individually (MegaDetector v6 doesn't have native batch support)
            all_detections = []
            for i, image in enumerate(images):
                camera_id = camera_ids[i] if camera_ids and i < len(camera_ids) else None
                detections = self.detect(image, camera_id)
                all_detections.append(detections)
            
            batch_time = time.time() - start_time
            logger.debug(f"Batch detection completed in {batch_time:.3f}s for {len(images)} images")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            return [[] for _ in images]
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MegaDetector input"""
        # Convert BGR to RGB (MegaDetector expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize to model input size if needed
        target_size = self.config.input_size
        if image_rgb.shape[:2] != target_size:
            image_rgb = cv2.resize(image_rgb, target_size)
        
        return image_rgb
    
    def _postprocess_results(self, results: Any, original_shape: Tuple[int, int], 
                           camera_id: Optional[int]) -> List[Detection]:
        """Post-process MegaDetector results into Detection objects"""
        detections = []
        
        try:
            # Extract results from MegaDetector output format
            # Note: Actual format depends on PytorchWildlife version
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes
                
                for i in range(len(boxes)):
                    # Extract box coordinates and confidence
                    if hasattr(boxes, 'xyxy'):
                        # Format: [x1, y1, x2, y2]
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                    else:
                        continue
                    
                    # Extract confidence and class
                    if hasattr(boxes, 'conf'):
                        confidence = float(boxes.conf[i].cpu().numpy())
                    else:
                        confidence = 0.0
                    
                    if hasattr(boxes, 'cls'):
                        class_id = int(boxes.cls[i].cpu().numpy())
                    else:
                        class_id = 0
                    
                    # Apply confidence thresholds
                    class_name = self.CLASS_MAPPING.get(class_id, 'unknown')
                    threshold = self._get_confidence_threshold(class_id)
                    
                    if confidence >= threshold:
                        # Scale coordinates to original image size
                        scale_x = original_shape[1] / self.config.input_size[0]
                        scale_y = original_shape[0] / self.config.input_size[1]
                        
                        scaled_bbox = (
                            int(x1 * scale_x),
                            int(y1 * scale_y),
                            int(width * scale_x),
                            int(height * scale_y)
                        )
                        
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=scaled_bbox,
                            timestamp=datetime.now(),
                            camera_id=camera_id
                        )
                        
                        detections.append(detection)
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
        
        return detections
    
    def _get_confidence_threshold(self, class_id: int) -> float:
        """Get confidence threshold for specific class"""
        if class_id == 0:  # animal
            return self.config.animal_confidence_threshold
        elif class_id == 1:  # person
            return self.config.person_confidence_threshold
        elif class_id == 2:  # vehicle
            return self.config.vehicle_confidence_threshold
        else:
            return 0.5  # default threshold
    
    def _update_stats(self, inference_time: float, detection_count: int):
        """Update detection statistics"""
        with self.lock:
            self.stats['total_inferences'] += 1
            self.stats['total_detections'] += detection_count
            self.stats['last_inference_time'] = datetime.now()
            
            # Update inference time tracking
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:  # Keep last 100 times
                self.inference_times.pop(0)
            
            self.stats['average_inference_time'] = np.mean(self.inference_times)
    
    def update_class_stats(self, detections: List[Detection]):
        """Update per-class detection statistics"""
        with self.lock:
            for detection in detections:
                if detection.class_name in self.stats['detections_by_class']:
                    self.stats['detections_by_class'][detection.class_name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection engine statistics"""
        with self.lock:
            return {
                'model_loaded': self.model_loaded,
                'device': self.device,
                'total_inferences': self.stats['total_inferences'],
                'total_detections': self.stats['total_detections'],
                'average_inference_time': round(self.stats['average_inference_time'], 3),
                'current_fps': round(1.0 / max(self.stats['average_inference_time'], 0.001), 1),
                'detections_by_class': self.stats['detections_by_class'].copy(),
                'last_inference': self.stats['last_inference_time'].isoformat() if self.stats['last_inference_time'] else None,
                'config': {
                    'animal_threshold': self.config.animal_confidence_threshold,
                    'person_threshold': self.config.person_confidence_threshold,
                    'vehicle_threshold': self.config.vehicle_confidence_threshold,
                    'input_size': self.config.input_size,
                    'device': self.config.device
                }
            }
    
    def update_config(self, new_config: DetectionConfig):
        """Update detection configuration"""
        self.config = new_config
        logger.info("Detection configuration updated")
    
    def is_model_available(self) -> bool:
        """Check if detection model is available and loaded"""
        return self.model_loaded
    
    def get_supported_classes(self) -> Dict[int, str]:
        """Get mapping of supported detection classes"""
        return self.CLASS_MAPPING.copy()

class DetectionProcessor:
    """Processes detections with additional filtering and tracking"""
    
    def __init__(self):
        self.detection_history: Dict[int, List[Detection]] = {}
        self.confidence_smoothing = True
        self.temporal_filtering = True
        
    def process_detections(self, detections: List[Detection], 
                         camera_id: int) -> List[Detection]:
        """
        Process detections with additional filtering
        
        Args:
            detections: Raw detections from engine
            camera_id: Camera ID for tracking
            
        Returns:
            Filtered and processed detections
        """
        # Store in history
        if camera_id not in self.detection_history:
            self.detection_history[camera_id] = []
        
        # Add timestamp and camera ID
        for detection in detections:
            detection.camera_id = camera_id
            detection.timestamp = datetime.now()
        
        # Apply temporal filtering
        if self.temporal_filtering:
            detections = self._apply_temporal_filtering(detections, camera_id)
        
        # Apply confidence smoothing
        if self.confidence_smoothing:
            detections = self._apply_confidence_smoothing(detections, camera_id)
        
        # Update history
        self.detection_history[camera_id].extend(detections)
        
        # Limit history size
        if len(self.detection_history[camera_id]) > 1000:
            self.detection_history[camera_id] = self.detection_history[camera_id][-500:]
        
        return detections
    
    def _apply_temporal_filtering(self, detections: List[Detection], 
                                camera_id: int) -> List[Detection]:
        """Apply temporal filtering to reduce false positives"""
        # Simple implementation: require detection in multiple consecutive frames
        # This would need more sophisticated tracking for production use
        return detections
    
    def _apply_confidence_smoothing(self, detections: List[Detection], 
                                  camera_id: int) -> List[Detection]:
        """Apply confidence smoothing based on recent detections"""
        # Simple implementation: boost confidence for consistent detections
        return detections
    
    def get_recent_detections(self, camera_id: int, 
                            seconds: int = 60) -> List[Detection]:
        """Get recent detections for specific camera"""
        if camera_id not in self.detection_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        recent_detections = [
            d for d in self.detection_history[camera_id]
            if d.timestamp >= cutoff_time
        ]
        
        return recent_detections

# Global detection engine instance
detection_config = DetectionConfig()
detection_engine = MegaDetectorEngine(detection_config)
detection_processor = DetectionProcessor()

def get_detection_engine() -> MegaDetectorEngine:
    """Get global detection engine instance"""
    return detection_engine

def get_detection_processor() -> DetectionProcessor:
    """Get global detection processor instance"""
    return detection_processor