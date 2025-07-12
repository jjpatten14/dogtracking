"""
Cross-Camera Dog Tracking System

Tracks individual dogs across multiple cameras with dead zone management.
Handles dog handoffs between cameras and maintains tracking continuity.

Key Features:
- Cross-camera dog tracking
- Dead zone and blind spot management
- Trajectory prediction
- Track association and handoff
- Movement pattern analysis
"""

import numpy as np
import cv2
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackState(Enum):
    """Track state enumeration"""
    ACTIVE = "active"
    LOST = "lost"
    DEAD_ZONE = "dead_zone"
    TRANSFERRED = "transferred"
    TERMINATED = "terminated"

@dataclass
class Position:
    """2D position with timestamp"""
    x: float
    y: float
    timestamp: datetime
    camera_id: int
    confidence: float = 1.0
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'x': self.x,
            'y': self.y,
            'timestamp': self.timestamp.isoformat(),
            'camera_id': self.camera_id,
            'confidence': self.confidence
        }

@dataclass
class Trajectory:
    """Movement trajectory for prediction"""
    positions: List[Position] = field(default_factory=list)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def add_position(self, position: Position):
        """Add new position and update trajectory"""
        self.positions.append(position)
        
        # Limit history size
        if len(self.positions) > 10:
            self.positions.pop(0)
        
        # Update velocity
        if len(self.positions) >= 2:
            self._update_velocity()
        
        self.last_update = position.timestamp
    
    def _update_velocity(self):
        """Update velocity based on recent positions"""
        if len(self.positions) < 2:
            return
        
        recent_positions = self.positions[-5:]  # Use last 5 positions
        if len(recent_positions) < 2:
            return
        
        # Calculate average velocity
        total_vx = 0.0
        total_vy = 0.0
        count = 0
        
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            
            time_diff = (curr_pos.timestamp - prev_pos.timestamp).total_seconds()
            if time_diff > 0:
                vx = (curr_pos.x - prev_pos.x) / time_diff
                vy = (curr_pos.y - prev_pos.y) / time_diff
                
                total_vx += vx
                total_vy += vy
                count += 1
        
        if count > 0:
            self.velocity_x = total_vx / count
            self.velocity_y = total_vy / count
    
    def predict_position(self, time_ahead_seconds: float) -> Position:
        """Predict future position based on trajectory"""
        if not self.positions:
            return Position(0, 0, datetime.now(), -1, 0.0)
        
        last_pos = self.positions[-1]
        
        # Simple linear prediction
        pred_x = last_pos.x + self.velocity_x * time_ahead_seconds
        pred_y = last_pos.y + self.velocity_y * time_ahead_seconds
        pred_time = last_pos.timestamp + timedelta(seconds=time_ahead_seconds)
        
        return Position(pred_x, pred_y, pred_time, last_pos.camera_id, 0.5)
    
    def get_direction(self) -> float:
        """Get movement direction in radians"""
        if abs(self.velocity_x) < 0.1 and abs(self.velocity_y) < 0.1:
            return 0.0  # No movement
        
        return math.atan2(self.velocity_y, self.velocity_x)
    
    def get_speed(self) -> float:
        """Get movement speed (pixels per second)"""
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)

@dataclass
class DogTrack:
    """Individual dog track across cameras"""
    track_id: str
    dog_id: Optional[str]
    dog_name: Optional[str]
    current_camera: Optional[int]
    state: TrackState
    trajectory: Trajectory = field(default_factory=Trajectory)
    created_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    last_detection_time: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    # Dead zone tracking
    entered_dead_zone_time: Optional[datetime] = None
    predicted_exit_camera: Optional[int] = None
    
    # Statistics
    total_detections: int = 0
    cameras_visited: List[int] = field(default_factory=list)
    
    def update_position(self, position: Position, dog_id: Optional[str] = None, 
                       dog_name: Optional[str] = None):
        """Update track with new position"""
        self.trajectory.add_position(position)
        self.current_camera = position.camera_id
        self.last_update = position.timestamp
        self.last_detection_time = position.timestamp
        self.total_detections += 1
        
        # Update dog identity if provided
        if dog_id:
            self.dog_id = dog_id
        if dog_name:
            self.dog_name = dog_name
        
        # Track cameras visited
        if position.camera_id not in self.cameras_visited:
            self.cameras_visited.append(position.camera_id)
        
        # Update state
        if self.state == TrackState.LOST or self.state == TrackState.DEAD_ZONE:
            self.state = TrackState.ACTIVE
            self.entered_dead_zone_time = None
    
    def enter_dead_zone(self, predicted_exit_camera: Optional[int] = None):
        """Mark track as entering dead zone"""
        self.state = TrackState.DEAD_ZONE
        self.entered_dead_zone_time = datetime.now()
        self.predicted_exit_camera = predicted_exit_camera
        
        logger.debug(f"Track {self.track_id} entered dead zone, predicted exit: Camera {predicted_exit_camera}")
    
    def is_lost(self, timeout_seconds: float = 30.0) -> bool:
        """Check if track should be considered lost"""
        time_since_detection = (datetime.now() - self.last_detection_time).total_seconds()
        return time_since_detection > timeout_seconds
    
    def is_in_dead_zone_too_long(self, max_dead_zone_seconds: float = 300.0) -> bool:
        """Check if track has been in dead zone too long"""
        if self.state != TrackState.DEAD_ZONE or not self.entered_dead_zone_time:
            return False
        
        time_in_dead_zone = (datetime.now() - self.entered_dead_zone_time).total_seconds()
        return time_in_dead_zone > max_dead_zone_seconds
    
    def get_age_seconds(self) -> float:
        """Get track age in seconds"""
        return (datetime.now() - self.created_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'track_id': self.track_id,
            'dog_id': self.dog_id,
            'dog_name': self.dog_name,
            'current_camera': self.current_camera,
            'state': self.state.value,
            'created_time': self.created_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'last_detection_time': self.last_detection_time.isoformat(),
            'confidence': self.confidence,
            'total_detections': self.total_detections,
            'cameras_visited': self.cameras_visited,
            'current_position': self.trajectory.positions[-1].to_dict() if self.trajectory.positions else None,
            'trajectory_length': len(self.trajectory.positions),
            'speed': self.trajectory.get_speed(),
            'direction': self.trajectory.get_direction(),
            'age_seconds': self.get_age_seconds(),
            'predicted_exit_camera': self.predicted_exit_camera
        }

class CameraZone:
    """Defines a camera's coverage zone and dead zones"""
    
    def __init__(self, camera_id: int, coverage_area: List[Tuple[int, int]], 
                 dead_zones: List[List[Tuple[int, int]]] = None):
        self.camera_id = camera_id
        self.coverage_area = coverage_area  # List of (x, y) points defining polygon
        self.dead_zones = dead_zones or []  # List of polygons defining dead zones
        
        # Adjacent cameras for handoff prediction
        self.adjacent_cameras: List[int] = []
        
        # Transition zones to adjacent cameras
        self.transition_zones: Dict[int, List[Tuple[int, int]]] = {}
    
    def add_adjacent_camera(self, camera_id: int, transition_zone: List[Tuple[int, int]] = None):
        """Add adjacent camera with optional transition zone"""
        if camera_id not in self.adjacent_cameras:
            self.adjacent_cameras.append(camera_id)
        
        if transition_zone:
            self.transition_zones[camera_id] = transition_zone
    
    def is_position_in_coverage(self, x: float, y: float) -> bool:
        """Check if position is within camera coverage"""
        return self._point_in_polygon(x, y, self.coverage_area)
    
    def is_position_in_dead_zone(self, x: float, y: float) -> bool:
        """Check if position is in any dead zone"""
        for dead_zone in self.dead_zones:
            if self._point_in_polygon(x, y, dead_zone):
                return True
        return False
    
    def get_likely_exit_camera(self, position: Position, trajectory: Trajectory) -> Optional[int]:
        """Predict which camera the dog will likely appear on next"""
        # Check if position is in any transition zone
        for camera_id, transition_zone in self.transition_zones.items():
            if self._point_in_polygon(position.x, position.y, transition_zone):
                return camera_id
        
        # Use trajectory to predict
        if trajectory.positions:
            pred_pos = trajectory.predict_position(5.0)  # 5 seconds ahead
            
            # Check which adjacent camera's coverage area the prediction falls into
            # This would require access to other cameras' coverage areas
            # For now, return the first adjacent camera
            if self.adjacent_cameras:
                return self.adjacent_cameras[0]
        
        return None
    
    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using ray casting algorithm"""
        if len(polygon) < 3:
            return False
        
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

class CrossCameraTracker:
    """Multi-camera dog tracking system"""
    
    def __init__(self):
        self.tracks: Dict[str, DogTrack] = {}
        self.camera_zones: Dict[int, CameraZone] = {}
        self.next_track_id = 1
        
        # Configuration
        self.max_track_age = 600.0  # 10 minutes
        self.max_dead_zone_time = 300.0  # 5 minutes
        self.association_distance_threshold = 100.0  # pixels
        self.association_time_threshold = 10.0  # seconds
        
        # Performance tracking
        self.stats = {
            'total_tracks_created': 0,
            'active_tracks': 0,
            'tracks_in_dead_zone': 0,
            'successful_handoffs': 0,
            'lost_tracks': 0
        }
        
        self.lock = threading.Lock()
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def add_camera_zone(self, camera_zone: CameraZone):
        """Add camera zone definition"""
        self.camera_zones[camera_zone.camera_id] = camera_zone
        logger.info(f"Added camera zone for camera {camera_zone.camera_id}")
    
    def update_tracks(self, detections: List[Any], camera_id: int) -> List[str]:
        """
        Update tracks with new detections from camera
        
        Args:
            detections: List of detection objects with bbox and dog_id
            camera_id: Camera ID
            
        Returns:
            List of track IDs that were updated
        """
        updated_tracks = []
        
        with self.lock:
            current_time = datetime.now()
            
            # Convert detections to positions
            detection_positions = []
            for detection in detections:
                if hasattr(detection, 'bbox'):
                    x, y, w, h = detection.bbox
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    position = Position(
                        x=center_x,
                        y=center_y,
                        timestamp=current_time,
                        camera_id=camera_id,
                        confidence=getattr(detection, 'confidence', 1.0)
                    )
                    
                    dog_id = getattr(detection, 'dog_id', None)
                    dog_name = getattr(detection, 'dog_name', None)
                    
                    detection_positions.append((position, dog_id, dog_name))
            
            # Associate detections with existing tracks
            associations = self._associate_detections(detection_positions, camera_id)
            
            # Update associated tracks
            for track_id, (position, dog_id, dog_name) in associations.items():
                if track_id in self.tracks:
                    self.tracks[track_id].update_position(position, dog_id, dog_name)
                    updated_tracks.append(track_id)
            
            # Create new tracks for unassociated detections
            unassociated = [
                det for i, det in enumerate(detection_positions)
                if not any(i in assoc for assoc in associations.values())
            ]
            
            for position, dog_id, dog_name in unassociated:
                track_id = self._create_new_track(position, dog_id, dog_name)
                updated_tracks.append(track_id)
            
            # Update tracks that might be in dead zones
            self._update_dead_zone_tracks(camera_id)
        
        return updated_tracks
    
    def _associate_detections(self, detection_positions: List[Tuple[Position, str, str]], 
                            camera_id: int) -> Dict[str, Tuple[Position, str, str]]:
        """Associate detections with existing tracks"""
        associations = {}
        
        # Get relevant tracks (active tracks and tracks that might exit to this camera)
        relevant_tracks = []
        for track in self.tracks.values():
            if (track.current_camera == camera_id or 
                track.predicted_exit_camera == camera_id or
                track.state == TrackState.DEAD_ZONE):
                relevant_tracks.append(track)
        
        # Calculate association costs
        cost_matrix = []
        for det_pos, dog_id, dog_name in detection_positions:
            row = []
            for track in relevant_tracks:
                cost = self._calculate_association_cost(det_pos, track, dog_id)
                row.append(cost)
            cost_matrix.append(row)
        
        # Simple greedy association (could be improved with Hungarian algorithm)
        used_tracks = set()
        for i, (det_pos, dog_id, dog_name) in enumerate(detection_positions):
            best_track_idx = None
            best_cost = float('inf')
            
            for j, track in enumerate(relevant_tracks):
                if track.track_id in used_tracks:
                    continue
                
                cost = cost_matrix[i][j]
                if cost < best_cost and cost < self.association_distance_threshold:
                    best_cost = cost
                    best_track_idx = j
            
            if best_track_idx is not None:
                track = relevant_tracks[best_track_idx]
                associations[track.track_id] = (det_pos, dog_id, dog_name)
                used_tracks.add(track.track_id)
        
        return associations
    
    def _calculate_association_cost(self, position: Position, track: DogTrack, 
                                  dog_id: Optional[str]) -> float:
        """Calculate cost of associating detection with track"""
        if not track.trajectory.positions:
            return float('inf')
        
        # Distance cost
        last_pos = track.trajectory.positions[-1]
        distance = position.distance_to(last_pos)
        
        # Time cost
        time_diff = (position.timestamp - last_pos.timestamp).total_seconds()
        if time_diff > self.association_time_threshold:
            return float('inf')
        
        # Predicted position cost (if track has trajectory)
        if len(track.trajectory.positions) >= 2:
            pred_pos = track.trajectory.predict_position(time_diff)
            pred_distance = position.distance_to(pred_pos)
            distance = min(distance, pred_distance)
        
        # Identity consistency cost
        identity_cost = 0.0
        if dog_id and track.dog_id and dog_id != track.dog_id:
            identity_cost = 50.0  # Penalty for identity mismatch
        
        # Camera transition cost
        camera_cost = 0.0
        if position.camera_id != track.current_camera:
            # Check if this is a valid camera transition
            if not self._is_valid_camera_transition(track, position.camera_id):
                camera_cost = 100.0
        
        return distance + identity_cost + camera_cost
    
    def _is_valid_camera_transition(self, track: DogTrack, new_camera_id: int) -> bool:
        """Check if camera transition is valid based on camera topology"""
        if not track.current_camera:
            return True
        
        current_camera_zone = self.camera_zones.get(track.current_camera)
        if not current_camera_zone:
            return True  # No topology info, allow transition
        
        return new_camera_id in current_camera_zone.adjacent_cameras
    
    def _create_new_track(self, position: Position, dog_id: Optional[str], 
                         dog_name: Optional[str]) -> str:
        """Create new track"""
        track_id = f"track_{self.next_track_id}"
        self.next_track_id += 1
        
        track = DogTrack(
            track_id=track_id,
            dog_id=dog_id,
            dog_name=dog_name,
            current_camera=position.camera_id,
            state=TrackState.ACTIVE
        )
        
        track.update_position(position, dog_id, dog_name)
        
        self.tracks[track_id] = track
        self.stats['total_tracks_created'] += 1
        
        logger.debug(f"Created new track {track_id} for {dog_name or 'unknown dog'} on camera {position.camera_id}")
        
        return track_id
    
    def _update_dead_zone_tracks(self, camera_id: int):
        """Update tracks that might be transitioning through dead zones"""
        current_time = datetime.now()
        
        for track in self.tracks.values():
            if track.state == TrackState.ACTIVE and track.current_camera != camera_id:
                # Check if track should enter dead zone
                if track.is_lost(30.0):  # 30 seconds without detection
                    camera_zone = self.camera_zones.get(track.current_camera)
                    if camera_zone and track.trajectory.positions:
                        last_pos = track.trajectory.positions[-1]
                        predicted_exit = camera_zone.get_likely_exit_camera(last_pos, track.trajectory)
                        track.enter_dead_zone(predicted_exit)
            
            elif track.state == TrackState.DEAD_ZONE:
                # Check if track has been in dead zone too long
                if track.is_in_dead_zone_too_long(self.max_dead_zone_time):
                    track.state = TrackState.LOST
                    logger.debug(f"Track {track.track_id} lost after {self.max_dead_zone_time}s in dead zone")
    
    def _cleanup_loop(self):
        """Background cleanup of old tracks"""
        while True:
            try:
                time.sleep(30)  # Run cleanup every 30 seconds
                
                with self.lock:
                    tracks_to_remove = []
                    
                    for track_id, track in self.tracks.items():
                        # Remove very old tracks
                        if track.get_age_seconds() > self.max_track_age:
                            tracks_to_remove.append(track_id)
                        
                        # Remove lost tracks
                        elif track.state == TrackState.LOST and track.is_lost(120.0):
                            tracks_to_remove.append(track_id)
                    
                    # Remove tracks
                    for track_id in tracks_to_remove:
                        del self.tracks[track_id]
                        logger.debug(f"Removed old track: {track_id}")
                    
                    # Update statistics
                    self._update_stats()
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def _update_stats(self):
        """Update tracking statistics"""
        self.stats['active_tracks'] = sum(
            1 for t in self.tracks.values() if t.state == TrackState.ACTIVE
        )
        self.stats['tracks_in_dead_zone'] = sum(
            1 for t in self.tracks.values() if t.state == TrackState.DEAD_ZONE
        )
        self.stats['lost_tracks'] = sum(
            1 for t in self.tracks.values() if t.state == TrackState.LOST
        )
    
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get all active tracks"""
        with self.lock:
            return [
                track.to_dict() for track in self.tracks.values()
                if track.state in [TrackState.ACTIVE, TrackState.DEAD_ZONE]
            ]
    
    def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get specific track information"""
        with self.lock:
            if track_id in self.tracks:
                return self.tracks[track_id].to_dict()
            return None
    
    def get_tracks_for_camera(self, camera_id: int) -> List[Dict[str, Any]]:
        """Get tracks currently on specific camera"""
        with self.lock:
            return [
                track.to_dict() for track in self.tracks.values()
                if track.current_camera == camera_id and track.state == TrackState.ACTIVE
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        with self.lock:
            self._update_stats()
            return self.stats.copy()
    
    def force_track_termination(self, track_id: str) -> bool:
        """Manually terminate a track"""
        with self.lock:
            if track_id in self.tracks:
                self.tracks[track_id].state = TrackState.TERMINATED
                logger.info(f"Track {track_id} manually terminated")
                return True
            return False

# Global tracker instance
cross_camera_tracker = CrossCameraTracker()

def get_cross_camera_tracker() -> CrossCameraTracker:
    """Get global cross-camera tracker instance"""
    return cross_camera_tracker