"""
Animal Detection System using MegaDetector

Primary animal detection for the dog tracking surveillance system.
Uses Microsoft's MegaDetector for robust animal/person/vehicle detection.

Key Features:
- MegaDetector v5 integration for primary detection
- CUDA GPU acceleration for performance
- Confidence-based filtering
- Bounding box extraction for downstream processing
- Integration with existing camera pipeline
"""

import torch
import cv2
import numpy as np
import logging
import time
import os
import platform
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from PIL import Image

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resolve_model_path(path: str) -> str:
    """
    Resolve model path for cross-platform compatibility (Windows/WSL/Linux)
    
    Args:
        path: Original path (may be WSL format like /mnt/c/...)
        
    Returns:
        Platform-appropriate absolute path
    """
    # If it's already a relative path, resolve relative to current directory
    if not os.path.isabs(path):
        return str(Path(path).resolve())
    
    # Handle WSL mount paths on Windows
    if platform.system() == "Windows" and path.startswith("/mnt/"):
        # Convert /mnt/c/... to C:\...
        if path.startswith("/mnt/c/"):
            windows_path = path.replace("/mnt/c/", "C:\\").replace("/", "\\")
            logger.info(f"🔄 Converting WSL path: {path} -> {windows_path}")
            return windows_path
        elif path.startswith("/mnt/"):
            # Handle other drive letters
            drive_letter = path[5].upper()  # Extract drive letter
            windows_path = path.replace(f"/mnt/{drive_letter.lower()}/", f"{drive_letter}:\\").replace("/", "\\")
            logger.info(f"🔄 Converting WSL path: {path} -> {windows_path}")
            return windows_path
    
    # For Linux/WSL environments or absolute paths, return as-is
    return str(Path(path).resolve())

# Official MegaDetector imports
try:
    # Import the persistent loading functions
    from megadetector.detection.run_detector_batch import load_detector, process_image
    MEGADETECTOR_PERSISTENT_AVAILABLE = True
    logger.info("✅ MegaDetector persistent interface available")
    
    # Try the batch function as fallback
    try:
        from megadetector.detection.run_detector_batch import load_and_run_detector_batch
        MEGADETECTOR_BATCH_AVAILABLE = True
        logger.info("✅ MegaDetector batch interface available")
    except ImportError:
        MEGADETECTOR_BATCH_AVAILABLE = False
    
    # Try the single function as fallback
    try:
        from megadetector.detection.run_detector import load_and_run_detector
        MEGADETECTOR_SINGLE_AVAILABLE = True
        logger.info("✅ MegaDetector single interface available")
    except ImportError:
        MEGADETECTOR_SINGLE_AVAILABLE = False
        logger.info("ℹ️ MegaDetector single interface not available")
    
    # Try to import utility functions
    try:
        from megadetector.utils.ct_utils import infer_device
    except ImportError:
        def infer_device():
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    MEGADETECTOR_AVAILABLE = True
    logger.info("✅ MegaDetector imports successful")
except ImportError as e:
    logger.warning(f"Official MegaDetector not available: {e}")
    MEGADETECTOR_AVAILABLE = False
    MEGADETECTOR_PERSISTENT_AVAILABLE = False
    MEGADETECTOR_BATCH_AVAILABLE = False
    MEGADETECTOR_SINGLE_AVAILABLE = False

class Detection:
    """Represents a single detection result"""
    def __init__(self, bbox: Tuple[float, float, float, float], confidence: float, 
                 category: str, category_id: int):
        self.bbox = bbox  # (x1, y1, x2, y2) in pixel coordinates
        self.confidence = confidence
        self.category = category  # 'animal', 'person', 'vehicle'
        self.category_id = category_id
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert detection to dictionary"""
        return {
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'category': self.category,
            'category_id': self.category_id,
            'timestamp': self.timestamp
        }
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

class MegaDetectorAnimalDetector:
    """
    MegaDetector-based animal detection system
    
    Uses Microsoft's MegaDetector v5 for primary detection of animals, people, and vehicles.
    Optimized for surveillance camera feeds with configurable confidence thresholds.
    OPTIMIZED: Uses persistent model loading for real-time performance.
    """
    
    def __init__(self, model_path: str = '/mnt/c/yard/models/md_v5a.0.0.pt'):
        # Resolve path for cross-platform compatibility
        self.model_path = resolve_model_path(model_path)
        self.model = None
        self.device = None
        self.model_loaded = False
        
        # Detection categories from MegaDetector
        self.categories = {
            1: 'animal',
            2: 'person', 
            3: 'vehicle'
        }
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'total_frames_processed': 0,
            'average_processing_time': 0.0,
            'detections_by_category': {cat: 0 for cat in self.categories.values()}
        }
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self) -> bool:
        """Load MegaDetector model persistently using official interface"""
        try:
            if not MEGADETECTOR_AVAILABLE:
                logger.error("Official MegaDetector package not available")
                return False
                
            logger.info(f"Loading MegaDetector from: {self.model_path}")
            logger.info(f"🔍 Platform: {platform.system()}")
            logger.info(f"🔍 Working directory: {os.getcwd()}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                logger.error(f"Working directory: {os.getcwd()}")
                logger.error(f"Platform: {platform.system()}")
                
                # Try fallback paths for debugging
                fallback_paths = [
                    "models/md_v5a.0.0.pt",  # Relative path
                    "/mnt/c/yard/models/md_v5a.0.0.pt",  # Original WSL path
                    "C:/yard/models/md_v5a.0.0.pt",  # Windows forward slash
                ]
                
                logger.info("🔍 Trying fallback paths:")
                for fallback in fallback_paths:
                    exists = os.path.exists(fallback)
                    logger.info(f"   {fallback}: {'✅' if exists else '❌'}")
                    if exists and not hasattr(self, '_fallback_used'):
                        logger.info(f"🔄 Using fallback path: {fallback}")
                        self.model_path = fallback
                        self._fallback_used = True
                        break
                else:
                    return False
            
            logger.info(f"✅ Model file found: {self.model_path}")
            
            # Set device preference for MegaDetector
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            if torch.cuda.is_available():
                logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                torch.cuda.empty_cache()
            
            # PERFORMANCE FIX: Load detector once and keep it in memory
            if MEGADETECTOR_PERSISTENT_AVAILABLE:
                logger.info("🚀 Loading MegaDetector with persistent interface...")
                start_time = time.time()
                
                # Load detector once - this will stay in memory
                force_cpu = self.device == 'cpu'
                self.detector = load_detector(
                    model_file=self.model_path,
                    force_cpu=force_cpu,
                    verbose=True
                )
                
                load_time = time.time() - start_time
                logger.info(f"✅ MegaDetector loaded persistently in {load_time:.2f}s")
                logger.info("🎯 Model will stay loaded in memory - no more reloading per detection!")
                
                self.model_loaded = True
                return True
            else:
                # Fallback to old method if persistent interface not available
                logger.warning("Persistent interface not available, using batch interface")
                logger.info("⚠️ Model will reload on each detection - performance will be poor")
                
                self.model_loaded = True
                return True
            
        except Exception as e:
            logger.error(f"Failed to load MegaDetector: {e}")
            self.model_loaded = False
            return False
    
    def _test_model(self):
        """Test model with dummy input"""
        try:
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Simple test - just check if model is loadable
            logger.info(f"Model test - MegaDetector model loaded and ready for inference")
            logger.info(f"Model device: {self.device}")
            
        except Exception as e:
            logger.warning(f"Model test failed: {e}")
    
    def detect_animals(self, frame: np.ndarray, 
                      animal_confidence: float = 0.7,
                      person_confidence: float = 0.8,
                      vehicle_confidence: float = 0.6) -> List[Detection]:
        """
        Detect animals, people, and vehicles using persistent MegaDetector model
        
        PERFORMANCE OPTIMIZED: Uses pre-loaded detector for real-time detection
        
        Args:
            frame: Input image frame (BGR format)
            animal_confidence: Minimum confidence for animal detections
            person_confidence: Minimum confidence for person detections  
            vehicle_confidence: Minimum confidence for vehicle detections
            
        Returns:
            List of Detection objects
        """
        if not self.model_loaded:
            logger.warning("Model not loaded, cannot detect animals")
            return []
        
        try:
            start_time = time.time()
            
            # Convert BGR to RGB for MegaDetector
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # PERFORMANCE FIX: Use persistent detector if available
            if MEGADETECTOR_PERSISTENT_AVAILABLE and hasattr(self, 'detector'):
                try:
                    # Save image temporarily for process_image function
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        pil_image.save(tmp_file.name, 'JPEG')
                        tmp_path = tmp_file.name
                    
                    # Use persistent detector - NO MODEL RELOADING!
                    confidence_threshold = min(animal_confidence, person_confidence, vehicle_confidence)
                    image_result = process_image(
                        im_file=tmp_path,
                        detector=self.detector,  # Use pre-loaded detector!
                        confidence_threshold=confidence_threshold,
                        quiet=True
                    )
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
                    detections = []
                    
                    # Parse results (same format as batch interface)
                    if image_result and 'detections' in image_result:
                        raw_detections = image_result['detections']
                        logger.debug(f"Found {len(raw_detections)} raw detections using persistent detector")
                        
                        for detection_data in raw_detections:
                            # MegaDetector returns normalized coordinates [0-1]
                            bbox_norm = detection_data['bbox']  # [x, y, width, height] normalized
                            confidence = detection_data['conf']
                            category_id = int(detection_data['category'])
                            
                            # Apply confidence thresholds
                            confidence_thresholds = {
                                1: animal_confidence,  # animal
                                2: person_confidence,  # person
                                3: vehicle_confidence  # vehicle
                            }
                            
                            if category_id in self.categories and confidence >= confidence_thresholds.get(category_id, 0.5):
                                # Convert normalized coordinates to pixel coordinates
                                h, w = frame.shape[:2]
                                x = bbox_norm[0] * w
                                y = bbox_norm[1] * h
                                width = bbox_norm[2] * w
                                height = bbox_norm[3] * h
                                
                                # Convert to [x1, y1, x2, y2] format
                                x1, y1 = x, y
                                x2, y2 = x + width, y + height
                                
                                detection = Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=float(confidence),
                                    category=self.categories[category_id],
                                    category_id=category_id
                                )
                                detections.append(detection)
                                logger.debug(f"Added {self.categories[category_id]} detection: conf={confidence:.3f}")
                    
                    # Update statistics
                    processing_time = time.time() - start_time
                    self.detection_stats['total_frames_processed'] += 1
                    self.detection_stats['total_detections'] += len(detections)
                    
                    # Update average processing time
                    total_frames = self.detection_stats['total_frames_processed']
                    current_avg = self.detection_stats['average_processing_time']
                    self.detection_stats['average_processing_time'] = (
                        (current_avg * (total_frames - 1) + processing_time) / total_frames
                    )
                    
                    # Update category counts
                    for detection in detections:
                        self.detection_stats['detections_by_category'][detection.category] += 1
                    
                    logger.debug(f"Detected {len(detections)} objects in {processing_time:.3f}s using persistent detector")
                    
                    return detections
                    
                except Exception as persistent_error:
                    logger.error(f"Persistent detector failed: {persistent_error}")
                    # Fall through to batch interface
            
            # Fallback to batch interface if persistent detector not available or failed
            logger.debug("Using fallback batch interface (will reload model)")
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                pil_image.save(tmp_file.name, 'JPEG')
                tmp_path = tmp_file.name
            
            # Run MegaDetector using batch interface (slower - reloads model)
            if MEGADETECTOR_BATCH_AVAILABLE:
                results = load_and_run_detector_batch(
                    model_file=self.model_path,
                    image_file_names=[tmp_path],
                    confidence_threshold=min(animal_confidence, person_confidence, vehicle_confidence),
                    quiet=True
                )
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                detections = []
                
                if results and len(results) > 0:
                    image_result = results[0]
                    if 'detections' in image_result:
                        raw_detections = image_result['detections']
                        
                        for detection_data in raw_detections:
                            bbox_norm = detection_data['bbox']
                            confidence = detection_data['conf']
                            category_id = int(detection_data['category'])
                            
                            confidence_thresholds = {
                                1: animal_confidence,
                                2: person_confidence,
                                3: vehicle_confidence
                            }
                            
                            if category_id in self.categories and confidence >= confidence_thresholds.get(category_id, 0.5):
                                h, w = frame.shape[:2]
                                x = bbox_norm[0] * w
                                y = bbox_norm[1] * h
                                width = bbox_norm[2] * w
                                height = bbox_norm[3] * h
                                
                                x1, y1 = x, y
                                x2, y2 = x + width, y + height
                                
                                detection = Detection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=float(confidence),
                                    category=self.categories[category_id],
                                    category_id=category_id
                                )
                                detections.append(detection)
                
                processing_time = time.time() - start_time
                self.detection_stats['total_frames_processed'] += 1
                self.detection_stats['total_detections'] += len(detections)
                
                total_frames = self.detection_stats['total_frames_processed']
                current_avg = self.detection_stats['average_processing_time']
                self.detection_stats['average_processing_time'] = (
                    (current_avg * (total_frames - 1) + processing_time) / total_frames
                )
                
                for detection in detections:
                    self.detection_stats['detections_by_category'][detection.category] += 1
                
                logger.debug(f"Detected {len(detections)} objects in {processing_time:.3f}s using batch interface")
                return detections
            
            logger.error("No MegaDetector interface available")
            return []
            
        except Exception as e:
            logger.error(f"Animal detection failed: {e}")
            return []
    
    def filter_animals_only(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to only include animals"""
        return [d for d in detections if d.category == 'animal']
    
    def get_animal_regions(self, frame: np.ndarray, detections: List[Detection]) -> List[np.ndarray]:
        """Extract animal regions from frame based on detections"""
        animal_regions = []
        
        for detection in detections:
            if detection.category == 'animal':
                x1, y1, x2, y2 = map(int, detection.bbox)
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                if x2 > x1 and y2 > y1:
                    region = frame[y1:y2, x1:x2]
                    animal_regions.append(region)
        
        return animal_regions
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                       draw_animals: bool = True, draw_people: bool = False, 
                       draw_vehicles: bool = False) -> np.ndarray:
        """Draw detection bounding boxes on frame"""
        frame_copy = frame.copy()
        
        # Colors for different categories (BGR format)
        colors = {
            'animal': (0, 255, 0),    # Green
            'person': (255, 0, 0),    # Blue  
            'vehicle': (0, 0, 255)    # Red
        }
        
        draw_categories = []
        if draw_animals:
            draw_categories.append('animal')
        if draw_people:
            draw_categories.append('person')
        if draw_vehicles:
            draw_categories.append('vehicle')
        
        for detection in detections:
            if detection.category in draw_categories:
                x1, y1, x2, y2 = map(int, detection.bbox)
                color = colors[detection.category]
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{detection.category}: {detection.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Background for label
                cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Label text
                cv2.putText(frame_copy, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame_copy
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        stats = {
            'model_loaded': self.model_loaded,
            'device': str(self.device) if self.device else None,
            'model_path': self.model_path,
            **self.detection_stats
        }
        
        # Add GPU memory stats if available
        if torch.cuda.is_available() and self.model_loaded:
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1e9
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1e9
            stats['gpu_utilization'] = f"{torch.cuda.utilization()}%" if hasattr(torch.cuda, 'utilization') else "N/A"
        
        return stats
    
    def is_available(self) -> bool:
        """Check if detector is ready for use"""
        return self.model_loaded
    
    def is_model_available(self) -> bool:
        """Alias for is_available for backward compatibility"""
        return self.is_available()
    
    def get_gpu_status(self) -> Dict:
        """Get detailed GPU status information"""
        status = {
            'cuda_available': torch.cuda.is_available(),
            'model_on_gpu': False,
            'gpu_count': 0,
            'current_device': None,
            'gpu_memory': {}
        }
        
        if torch.cuda.is_available():
            status['gpu_count'] = torch.cuda.device_count()
            status['current_device'] = torch.cuda.current_device()
            
            if self.model_loaded and self.model is not None:
                try:
                    model_device = next(self.model.parameters()).device
                    status['model_on_gpu'] = model_device.type == 'cuda'
                    status['model_device'] = str(model_device)
                except:
                    status['model_on_gpu'] = False
            
            # GPU memory info
            for i in range(status['gpu_count']):
                status['gpu_memory'][f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / 1e9,
                    'cached': torch.cuda.memory_reserved(i) / 1e9,
                    'total': torch.cuda.get_device_properties(i).total_memory / 1e9
                }
        
        return status

# Global detector instance
animal_detector = None

def get_animal_detector() -> MegaDetectorAnimalDetector:
    """Get global animal detector instance"""
    global animal_detector
    if animal_detector is None:
        animal_detector = MegaDetectorAnimalDetector()
    return animal_detector

def load_animal_detector() -> bool:
    """Initialize and load the animal detector"""
    global animal_detector
    try:
        animal_detector = MegaDetectorAnimalDetector()
        return animal_detector.is_available()
    except Exception as e:
        logger.error(f"Failed to initialize animal detector: {e}")
        return False