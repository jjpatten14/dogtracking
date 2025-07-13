"""
Boundary Manager for Dog Detection System

Handles boundary violation detection using point-in-polygon algorithms.
Tracks dog entry/exit events and triggers appropriate alerts.

Key Features:
- Point-in-polygon detection for boundary violations
- Entry/exit state tracking for dogs
- Integration with alert system for notifications
- Support for multiple boundaries per camera
"""

import json
import time
import logging
import os
import platform
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from project_paths import get_path, PROJECT_ROOT

# Configure logging
logger = logging.getLogger(__name__)

# Legacy function - now replaced by project_paths.py
# Keeping for backwards compatibility during transition
def resolve_config_path(path: str) -> str:
    """
    DEPRECATED: Use project_paths.get_path() instead.
    Legacy config path resolution for backwards compatibility.
    """
    logger.warning("resolve_config_path() is deprecated. Use project_paths.get_path() instead.")
    
    # If it's a relative path, resolve relative to project root
    if not os.path.isabs(path):
        return str(PROJECT_ROOT / path)
    
    # For absolute paths, try to convert them to portable paths
    path_obj = Path(path)
    if path_obj.name == 'boundary_config.json':
        return str(get_path('boundary_config'))
    
    # Return as-is for other absolute paths
    return str(Path(path).resolve())

@dataclass
class BoundaryZone:
    """Represents a boundary zone with detection properties"""
    name: str
    points: List[Tuple[int, int]]  # Polygon points [(x1, y1), (x2, y2), ...]
    camera_id: int
    zone_type: str = "restricted"  # "restricted", "safe", "monitoring"
    
class BoundaryViolation:
    """Represents a boundary violation event"""
    def __init__(self, dog_id: str, dog_name: str, zone_name: str, 
                 camera_id: int, event_type: str, position: Tuple[float, float]):
        self.dog_id = dog_id
        self.dog_name = dog_name
        self.zone_name = zone_name
        self.camera_id = camera_id
        self.event_type = event_type  # "entry", "exit"
        self.position = position
        self.timestamp = time.time()

class BoundaryManager:
    """
    Manages boundary zones and detects violations
    
    Features:
    - Point-in-polygon detection for multiple zones
    - Entry/exit event tracking per dog
    - Alert integration for boundary violations
    """
    
    def __init__(self, config_file: str = None):
        # Use portable path system
        if config_file is None:
            self.config_file = str(get_path('boundary_config'))
        else:
            # For custom paths, resolve relative to project root
            if not os.path.isabs(config_file):
                self.config_file = str(PROJECT_ROOT / config_file)
            else:
                self.config_file = resolve_config_path(config_file)  # Legacy support
        self.zones: Dict[str, BoundaryZone] = {}
        self.dog_states: Dict[str, Dict[str, bool]] = {}  # {dog_id: {zone_name: is_inside}}
        self.last_violations: Dict[str, float] = {}  # Rate limiting for alerts
        self.alert_cooldown = 5.0  # Seconds between repeat alerts for same dog/zone
        
        # Load boundary zones from config
        self.load_zones()
        
        logger.info(f"BoundaryManager initialized with {len(self.zones)} zones")
    
    def load_zones(self):
        """Load boundary zones from configuration file"""
        try:
            logger.info(f"🔍 Loading boundary config from: {self.config_file}")
            
            if not Path(self.config_file).exists():
                logger.warning(f"❌ No boundary config found at {self.config_file}")
                logger.info(f"💡 Create boundaries in the web interface first!")
                return
            
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            boundaries = config.get('boundaries', [])
            logger.info(f"📋 Found {len(boundaries)} boundary definitions in config")
            
            # For now, create a single zone per boundary
            # Later this can be enhanced with named zones
            for i, boundary_points in enumerate(boundaries):
                if len(boundary_points) >= 3:  # Need at least 3 points for a polygon
                    zone_name = f"Zone_{i+1}"
                    zone = BoundaryZone(
                        name=zone_name,
                        points=[(int(p[0]), int(p[1])) for p in boundary_points],
                        camera_id=1,  # Default to camera 1 for now
                        zone_type="restricted"
                    )
                    self.zones[zone_name] = zone
                    logger.info(f"✅ Loaded zone: {zone_name} with {len(boundary_points)} points")
                    logger.debug(f"   Zone points: {boundary_points}")
                else:
                    logger.warning(f"⚠️ Skipping boundary {i+1}: needs at least 3 points, got {len(boundary_points)}")
            
            logger.info(f"🎯 Successfully loaded {len(self.zones)} boundary zones from config")
            
        except Exception as e:
            logger.error(f"❌ Failed to load boundary zones: {e}")
    
    def point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm
        
        Args:
            point: (x, y) coordinates to test
            polygon: List of (x, y) polygon vertices
            
        Returns:
            True if point is inside polygon, False otherwise
        """
        x, y = point
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
    
    def check_position_violations(self, dog_id: str, dog_name: str, camera_id: int, 
                                position: Tuple[float, float]) -> List[BoundaryViolation]:
        """
        Check if a dog's position violates any boundary zones
        
        Args:
            dog_id: Unique identifier for the dog
            dog_name: Display name for the dog
            camera_id: Camera where detection occurred
            position: (x, y) position of the dog
            
        Returns:
            List of BoundaryViolation events (entry/exit)
        """
        violations = []
        
        # Debug: Show dog detection
        logger.debug(f"🐕 Checking position for {dog_name} at ({position[0]:.1f}, {position[1]:.1f}) on camera {camera_id}")
        
        # Check if we have any zones for this camera
        camera_zones = [z for z in self.zones.values() if z.camera_id == camera_id]
        if not camera_zones:
            logger.debug(f"⚠️ No zones configured for camera {camera_id}")
            return violations
        
        # Initialize dog state if not exists
        if dog_id not in self.dog_states:
            self.dog_states[dog_id] = {}
            logger.debug(f"📝 Initializing tracking for dog: {dog_name}")
        
        # Check each zone for this camera
        for zone_name, zone in self.zones.items():
            if zone.camera_id != camera_id:
                continue
            
            # Check if dog is currently inside this zone
            is_inside = self.point_in_polygon(position, zone.points)
            
            # Get previous state for this dog/zone
            was_inside = self.dog_states[dog_id].get(zone_name, False)
            
            # Debug: Show zone check result
            logger.debug(f"🔍 {dog_name} in {zone_name}: {is_inside} (was: {was_inside})")
            
            # Detect state changes (entry/exit events)
            if is_inside and not was_inside:
                # Dog entered the zone
                violation = BoundaryViolation(
                    dog_id=dog_id,
                    dog_name=dog_name,
                    zone_name=zone_name,
                    camera_id=camera_id,
                    event_type="entry",
                    position=position
                )
                violations.append(violation)
                logger.info(f"🚨 Dog {dog_name} ENTERED zone {zone_name} at position ({position[0]:.1f}, {position[1]:.1f})")
                
            elif not is_inside and was_inside:
                # Dog exited the zone
                violation = BoundaryViolation(
                    dog_id=dog_id,
                    dog_name=dog_name,
                    zone_name=zone_name,
                    camera_id=camera_id,
                    event_type="exit",
                    position=position
                )
                violations.append(violation)
                logger.info(f"✅ Dog {dog_name} EXITED zone {zone_name} at position ({position[0]:.1f}, {position[1]:.1f})")
            
            # Update state
            self.dog_states[dog_id][zone_name] = is_inside
        
        if violations:
            logger.info(f"🎯 Found {len(violations)} boundary violations for {dog_name}")
        
        return violations
    
    def should_send_alert(self, dog_id: str, zone_name: str, event_type: str) -> bool:
        """Check if we should send an alert based on rate limiting"""
        alert_key = f"{dog_id}_{zone_name}_{event_type}"
        current_time = time.time()
        
        if alert_key in self.last_violations:
            time_since_last = current_time - self.last_violations[alert_key]
            if time_since_last < self.alert_cooldown:
                return False
        
        self.last_violations[alert_key] = current_time
        return True
    
    def get_dogs_in_zones(self) -> Dict[str, List[str]]:
        """Get current status of which dogs are in which zones"""
        dogs_in_zones = {}
        
        for dog_id, dog_zones in self.dog_states.items():
            for zone_name, is_inside in dog_zones.items():
                if is_inside:
                    if zone_name not in dogs_in_zones:
                        dogs_in_zones[zone_name] = []
                    dogs_in_zones[zone_name].append(dog_id)
        
        return dogs_in_zones
    
    def get_stats(self) -> Dict:
        """Get boundary manager statistics"""
        total_dogs_tracked = len(self.dog_states)
        dogs_in_zones = self.get_dogs_in_zones()
        total_dogs_in_zones = sum(len(dogs) for dogs in dogs_in_zones.values())
        
        return {
            'total_zones': len(self.zones),
            'total_dogs_tracked': total_dogs_tracked,
            'dogs_currently_in_zones': total_dogs_in_zones,
            'zones_with_dogs': len(dogs_in_zones),
            'zone_status': dogs_in_zones
        }

# Global boundary manager instance
boundary_manager = None

def get_boundary_manager() -> BoundaryManager:
    """Get global boundary manager instance"""
    global boundary_manager
    if boundary_manager is None:
        boundary_manager = BoundaryManager()
    return boundary_manager