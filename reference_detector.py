"""
Reference Point Detection for Dynamic Boundary System

Detects static landmarks (poles, corners, trees) for camera movement compensation.
Enables dynamic boundary adjustment when cameras move or vibrate.

Key Features:
- Edge detection for poles and linear features
- Template matching for known landmarks
- Corner detection for building features
- Reference point tracking and stability monitoring
- Camera movement compensation
"""

import cv2
import numpy as np
import logging
import time
import pickle
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReferencePoint:
    """A reference point in camera view"""
    ref_id: str
    name: str
    type: str  # 'pole', 'corner', 'tree', 'custom'
    position: Tuple[int, int]  # (x, y) in image coordinates
    template: Optional[np.ndarray]  # Template image for matching
    confidence: float
    last_seen: datetime
    camera_id: int
    stability_score: float = 1.0  # How stable this reference point is
    detection_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'ref_id': self.ref_id,
            'name': self.name,
            'type': self.type,
            'position': list(self.position),
            'confidence': self.confidence,
            'last_seen': self.last_seen.isoformat(),
            'camera_id': self.camera_id,
            'stability_score': self.stability_score,
            'detection_count': self.detection_count,
            'has_template': self.template is not None
        }

@dataclass
class CameraMovement:
    """Detected camera movement information"""
    camera_id: int
    movement_vector: Tuple[float, float]  # (dx, dy) movement in pixels
    rotation_angle: float  # Rotation in degrees
    scale_factor: float  # Scale change factor
    confidence: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'camera_id': self.camera_id,
            'movement_vector': list(self.movement_vector),
            'rotation_angle': self.rotation_angle,
            'scale_factor': self.scale_factor,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

class ReferencePointDetector:
    """Detects and tracks reference points for camera stability"""
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.reference_points: Dict[str, ReferencePoint] = {}
        self.templates_dir = Path(f'/mnt/c/yard/models/landmark_templates/camera_{camera_id}')
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Detection parameters
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150
        self.line_threshold = 100
        self.corner_quality = 0.01
        self.corner_min_distance = 10
        
        # Template matching parameters
        self.template_match_threshold = 0.7
        self.template_size = (64, 64)
        
        # Movement detection
        self.movement_history: List[CameraMovement] = []
        self.previous_frame: Optional[np.ndarray] = None
        self.previous_keypoints: Optional[List[cv2.KeyPoint]] = None
        self.previous_descriptors: Optional[np.ndarray] = None
        
        # Feature detector for movement tracking
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Load saved reference points
        self.load_reference_points()
        
        logger.info(f"Reference point detector initialized for camera {camera_id}")
    
    def detect_reference_points(self, frame: np.ndarray) -> List[ReferencePoint]:
        """
        Detect reference points in current frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detected reference points
        """
        detected_points = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect different types of reference points
        pole_points = self._detect_poles(gray, frame)
        corner_points = self._detect_corners(gray, frame)
        template_points = self._match_templates(gray, frame)
        
        detected_points.extend(pole_points)
        detected_points.extend(corner_points)
        detected_points.extend(template_points)
        
        # Update reference point tracking
        self._update_reference_tracking(detected_points)
        
        return detected_points
    
    def _detect_poles(self, gray: np.ndarray, frame: np.ndarray) -> List[ReferencePoint]:
        """Detect vertical poles using edge detection and Hough transform"""
        poles = []
        
        try:
            # Edge detection
            edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=self.line_threshold)
            
            if lines is not None:
                for i, line in enumerate(lines):
                    rho, theta = line[0]
                    
                    # Filter for vertical lines (poles)
                    angle_deg = np.degrees(theta)
                    if abs(angle_deg - 90) < 15 or abs(angle_deg - 270) < 15:  # Near vertical
                        # Calculate line endpoints
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        
                        # Use midpoint as reference
                        ref_x = (x1 + x2) // 2
                        ref_y = (y1 + y2) // 2
                        
                        # Ensure point is within frame
                        if 0 <= ref_x < frame.shape[1] and 0 <= ref_y < frame.shape[0]:
                            ref_id = f"pole_{self.camera_id}_{i}"
                            pole = ReferencePoint(
                                ref_id=ref_id,
                                name=f"Pole {i+1}",
                                type="pole",
                                position=(ref_x, ref_y),
                                template=None,
                                confidence=0.8,
                                last_seen=datetime.now(),
                                camera_id=self.camera_id
                            )
                            poles.append(pole)
        
        except Exception as e:
            logger.error(f"Pole detection failed: {e}")
        
        return poles
    
    def _detect_corners(self, gray: np.ndarray, frame: np.ndarray) -> List[ReferencePoint]:
        """Detect corners using goodFeaturesToTrack"""
        corners_detected = []
        
        try:
            # Detect corners
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=20,
                qualityLevel=self.corner_quality,
                minDistance=self.corner_min_distance
            )
            
            if corners is not None:
                corners = np.int0(corners)
                
                for i, corner in enumerate(corners):
                    x, y = corner.ravel()
                    
                    ref_id = f"corner_{self.camera_id}_{i}"
                    corner_point = ReferencePoint(
                        ref_id=ref_id,
                        name=f"Corner {i+1}",
                        type="corner",
                        position=(x, y),
                        template=None,
                        confidence=0.7,
                        last_seen=datetime.now(),
                        camera_id=self.camera_id
                    )
                    corners_detected.append(corner_point)
        
        except Exception as e:
            logger.error(f"Corner detection failed: {e}")
        
        return corners_detected
    
    def _match_templates(self, gray: np.ndarray, frame: np.ndarray) -> List[ReferencePoint]:
        """Match known templates against current frame"""
        template_matches = []
        
        try:
            # Load and match each saved template
            for template_file in self.templates_dir.glob("*.pkl"):
                try:
                    with open(template_file, 'rb') as f:
                        ref_point = pickle.load(f)
                    
                    if ref_point.template is not None:
                        # Perform template matching
                        result = cv2.matchTemplate(gray, ref_point.template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val >= self.template_match_threshold:
                            # Update position and confidence
                            ref_point.position = max_loc
                            ref_point.confidence = max_val
                            ref_point.last_seen = datetime.now()
                            ref_point.detection_count += 1
                            
                            template_matches.append(ref_point)
                
                except Exception as e:
                    logger.error(f"Template matching failed for {template_file}: {e}")
        
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
        
        return template_matches
    
    def _update_reference_tracking(self, detected_points: List[ReferencePoint]):
        """Update tracking of reference points"""
        current_time = datetime.now()
        
        # Update existing reference points or add new ones
        for point in detected_points:
            if point.ref_id in self.reference_points:
                # Update existing point
                existing = self.reference_points[point.ref_id]
                
                # Calculate stability score based on position consistency
                old_pos = existing.position
                new_pos = point.position
                distance = np.sqrt((old_pos[0] - new_pos[0])**2 + (old_pos[1] - new_pos[1])**2)
                
                # Update stability score (lower distance = higher stability)
                stability_update = max(0.1, 1.0 - distance / 100.0)
                existing.stability_score = 0.9 * existing.stability_score + 0.1 * stability_update
                
                # Update other fields
                existing.position = point.position
                existing.confidence = point.confidence
                existing.last_seen = current_time
                existing.detection_count += 1
            else:
                # Add new reference point
                self.reference_points[point.ref_id] = point
                logger.debug(f"Added new reference point: {point.name}")
        
        # Remove old reference points that haven't been seen recently
        cutoff_time = current_time - timedelta(minutes=5)
        to_remove = [
            ref_id for ref_id, point in self.reference_points.items()
            if point.last_seen < cutoff_time
        ]
        
        for ref_id in to_remove:
            del self.reference_points[ref_id]
            logger.debug(f"Removed old reference point: {ref_id}")
    
    def detect_camera_movement(self, frame: np.ndarray) -> Optional[CameraMovement]:
        """
        Detect camera movement by comparing with previous frame
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            CameraMovement object if movement detected, None otherwise
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and descriptors
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            if self.previous_descriptors is not None and descriptors is not None:
                # Match features between frames
                matches = self.matcher.match(self.previous_descriptors, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 10:  # Need sufficient matches
                    # Extract matched points
                    src_pts = np.float32([self.previous_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    # Find transformation matrix
                    transform_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    
                    if transform_matrix is not None:
                        # Extract movement parameters
                        dx = transform_matrix[0, 2]
                        dy = transform_matrix[1, 2]
                        
                        # Calculate rotation and scale
                        scale_x = np.sqrt(transform_matrix[0, 0]**2 + transform_matrix[0, 1]**2)
                        scale_y = np.sqrt(transform_matrix[1, 0]**2 + transform_matrix[1, 1]**2)
                        scale_factor = (scale_x + scale_y) / 2
                        
                        rotation = np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0]))
                        
                        # Calculate confidence based on number of inliers
                        inliers = np.sum(mask) if mask is not None else len(matches)
                        confidence = min(1.0, inliers / len(matches))
                        
                        # Only report significant movement
                        movement_magnitude = np.sqrt(dx**2 + dy**2)
                        if movement_magnitude > 2.0 or abs(rotation) > 1.0 or abs(scale_factor - 1.0) > 0.05:
                            movement = CameraMovement(
                                camera_id=self.camera_id,
                                movement_vector=(dx, dy),
                                rotation_angle=rotation,
                                scale_factor=scale_factor,
                                confidence=confidence,
                                timestamp=datetime.now()
                            )
                            
                            # Store in history
                            self.movement_history.append(movement)
                            if len(self.movement_history) > 100:
                                self.movement_history.pop(0)
                            
                            return movement
            
            # Store current frame data for next comparison
            self.previous_frame = gray.copy()
            self.previous_keypoints = keypoints
            self.previous_descriptors = descriptors
            
        except Exception as e:
            logger.error(f"Camera movement detection failed: {e}")
        
        return None
    
    def add_manual_reference_point(self, name: str, ref_type: str, position: Tuple[int, int], 
                                 frame: np.ndarray) -> str:
        """
        Manually add a reference point with template extraction
        
        Args:
            name: Reference point name
            ref_type: Type ('pole', 'corner', 'tree', 'custom')
            position: (x, y) position in frame
            frame: Current frame for template extraction
            
        Returns:
            Reference point ID
        """
        try:
            ref_id = f"manual_{self.camera_id}_{len(self.reference_points)}"
            
            # Extract template around position
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y = position
            
            # Define template region
            template_half_size = 32
            x1 = max(0, x - template_half_size)
            y1 = max(0, y - template_half_size)
            x2 = min(gray.shape[1], x + template_half_size)
            y2 = min(gray.shape[0], y + template_half_size)
            
            template = gray[y1:y2, x1:x2]
            
            # Create reference point
            ref_point = ReferencePoint(
                ref_id=ref_id,
                name=name,
                type=ref_type,
                position=position,
                template=template,
                confidence=1.0,
                last_seen=datetime.now(),
                camera_id=self.camera_id
            )
            
            # Store reference point
            self.reference_points[ref_id] = ref_point
            self.save_reference_point(ref_point)
            
            logger.info(f"Added manual reference point: {name} at {position}")
            return ref_id
            
        except Exception as e:
            logger.error(f"Failed to add manual reference point: {e}")
            return ""
    
    def remove_reference_point(self, ref_id: str) -> bool:
        """Remove reference point"""
        if ref_id in self.reference_points:
            del self.reference_points[ref_id]
            
            # Remove saved file
            template_file = self.templates_dir / f"{ref_id}.pkl"
            if template_file.exists():
                template_file.unlink()
            
            logger.info(f"Removed reference point: {ref_id}")
            return True
        return False
    
    def get_stable_reference_points(self, min_stability: float = 0.7) -> List[ReferencePoint]:
        """Get reference points with high stability scores"""
        return [
            point for point in self.reference_points.values()
            if point.stability_score >= min_stability
        ]
    
    def calibrate_references(self, frame: np.ndarray) -> bool:
        """
        Recalibrate all reference points using current frame
        
        Args:
            frame: Current frame for recalibration
            
        Returns:
            True if calibration successful
        """
        try:
            # Detect reference points in current frame
            detected_points = self.detect_reference_points(frame)
            
            logger.info(f"Recalibrated {len(detected_points)} reference points")
            return True
            
        except Exception as e:
            logger.error(f"Reference calibration failed: {e}")
            return False
    
    def save_reference_point(self, ref_point: ReferencePoint):
        """Save reference point to disk"""
        try:
            template_file = self.templates_dir / f"{ref_point.ref_id}.pkl"
            with open(template_file, 'wb') as f:
                pickle.dump(ref_point, f)
            
        except Exception as e:
            logger.error(f"Failed to save reference point: {e}")
    
    def load_reference_points(self):
        """Load all saved reference points"""
        try:
            for template_file in self.templates_dir.glob("*.pkl"):
                try:
                    with open(template_file, 'rb') as f:
                        ref_point = pickle.load(f)
                    
                    self.reference_points[ref_point.ref_id] = ref_point
                    
                except Exception as e:
                    logger.error(f"Failed to load reference point {template_file}: {e}")
            
            logger.info(f"Loaded {len(self.reference_points)} reference points for camera {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Failed to load reference points: {e}")
    
    def get_movement_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get recent camera movement history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_movements = [
            movement.to_dict() for movement in self.movement_history
            if movement.timestamp >= cutoff_time
        ]
        return recent_movements
    
    def get_reference_points_info(self) -> List[Dict[str, Any]]:
        """Get information about all reference points"""
        return [point.to_dict() for point in self.reference_points.values()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reference detector statistics"""
        stable_points = len(self.get_stable_reference_points())
        
        return {
            'camera_id': self.camera_id,
            'total_reference_points': len(self.reference_points),
            'stable_reference_points': stable_points,
            'recent_movements': len(self.movement_history),
            'templates_saved': len(list(self.templates_dir.glob("*.pkl"))),
            'average_stability': np.mean([p.stability_score for p in self.reference_points.values()]) if self.reference_points else 0.0
        }

class MultiCameraReferenceManager:
    """Manages reference point detection across multiple cameras"""
    
    def __init__(self):
        self.detectors: Dict[int, ReferencePointDetector] = {}
        self.global_movement_threshold = 5.0  # Global movement detection threshold
        
    def add_camera(self, camera_id: int) -> ReferencePointDetector:
        """Add camera for reference point detection"""
        if camera_id not in self.detectors:
            self.detectors[camera_id] = ReferencePointDetector(camera_id)
            logger.info(f"Added reference point detector for camera {camera_id}")
        
        return self.detectors[camera_id]
    
    def process_frame(self, camera_id: int, frame: np.ndarray) -> Tuple[List[ReferencePoint], Optional[CameraMovement]]:
        """Process frame for reference points and movement detection"""
        if camera_id not in self.detectors:
            self.add_camera(camera_id)
        
        detector = self.detectors[camera_id]
        
        # Detect reference points and camera movement
        ref_points = detector.detect_reference_points(frame)
        movement = detector.detect_camera_movement(frame)
        
        return ref_points, movement
    
    def get_all_reference_points(self) -> Dict[int, List[Dict[str, Any]]]:
        """Get reference points from all cameras"""
        all_points = {}
        for camera_id, detector in self.detectors.items():
            all_points[camera_id] = detector.get_reference_points_info()
        return all_points
    
    def get_system_stability(self) -> Dict[str, Any]:
        """Get overall system stability metrics"""
        total_stable = 0
        total_points = 0
        camera_stability = {}
        
        for camera_id, detector in self.detectors.items():
            stats = detector.get_stats()
            stable_points = stats['stable_reference_points']
            total_points_cam = stats['total_reference_points']
            
            total_stable += stable_points
            total_points += total_points_cam
            
            camera_stability[camera_id] = {
                'stable_points': stable_points,
                'total_points': total_points_cam,
                'stability_ratio': stable_points / max(total_points_cam, 1),
                'average_stability': stats['average_stability']
            }
        
        return {
            'overall_stability_ratio': total_stable / max(total_points, 1),
            'total_stable_points': total_stable,
            'total_reference_points': total_points,
            'camera_stability': camera_stability
        }

# Global reference manager
reference_manager = MultiCameraReferenceManager()

def get_reference_manager() -> MultiCameraReferenceManager:
    """Get global reference manager instance"""
    return reference_manager