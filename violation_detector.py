"""
Boundary Violation Detection System

Real-time detection of boundary violations with cross-camera awareness.
Integrates with tracking system to provide comprehensive violation monitoring.

Key Features:
- Real-time violation detection
- Cross-camera violation tracking
- Confidence-based filtering
- Temporal violation analysis
- Integration with tracking system
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViolationSeverity(Enum):
    """Violation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ViolationEvent:
    """Single violation event with context"""
    event_id: str
    dog_id: str
    dog_name: str
    track_id: str
    zone_id: str
    zone_name: str
    camera_id: int
    position: Tuple[float, float]
    violation_type: str
    severity: ViolationSeverity
    confidence: float
    timestamp: datetime
    duration_seconds: float = 0.0
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'dog_id': self.dog_id,
            'dog_name': self.dog_name,
            'track_id': self.track_id,
            'zone_id': self.zone_id,
            'zone_name': self.zone_name,
            'camera_id': self.camera_id,
            'position': list(self.position),
            'violation_type': self.violation_type,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }

@dataclass
class ViolationContext:
    """Context information for violation analysis"""
    previous_positions: List[Tuple[float, float, datetime]] = field(default_factory=list)
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    movement_pattern: str = "unknown"  # "entering", "exiting", "stationary", "passing_through"
    cross_camera_context: Dict[int, List[Tuple[float, float, datetime]]] = field(default_factory=dict)

@dataclass
class ViolationRule:
    """Configurable violation detection rule"""
    rule_id: str
    name: str
    zone_id: str
    dog_id: Optional[str] = None  # None means applies to all dogs
    violation_type: str = "forbidden_entry"
    min_confidence: float = 0.7
    min_duration_seconds: float = 2.0
    severity: ViolationSeverity = ViolationSeverity.MEDIUM
    active: bool = True
    time_restrictions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'zone_id': self.zone_id,
            'dog_id': self.dog_id,
            'violation_type': self.violation_type,
            'min_confidence': self.min_confidence,
            'min_duration_seconds': self.min_duration_seconds,
            'severity': self.severity.value,
            'active': self.active,
            'time_restrictions': self.time_restrictions
        }

class ViolationDetector:
    """Detects and tracks boundary violations with cross-camera awareness"""
    
    def __init__(self):
        self.active_violations: Dict[str, ViolationEvent] = {}
        self.violation_history: List[ViolationEvent] = []
        self.violation_contexts: Dict[str, ViolationContext] = {}
        self.violation_rules: Dict[str, ViolationRule] = {}
        
        # Detection parameters
        self.position_buffer_size = 10
        self.violation_timeout_seconds = 30.0
        self.confidence_decay_rate = 0.1  # Per second
        
        # Performance tracking
        self.stats = {
            'total_violations_detected': 0,
            'active_violations': 0,
            'violations_resolved': 0,
            'false_positives_filtered': 0,
            'average_violation_duration': 0.0,
            'violations_by_severity': {s.value: 0 for s in ViolationSeverity},
            'violations_by_type': {},
            'violations_by_dog': {},
            'violations_by_camera': {}
        }
        
        self.lock = threading.Lock()
        
        # Dependencies
        self.boundary_manager = None
        self.cross_camera_tracker = None
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def set_dependencies(self, boundary_manager, cross_camera_tracker):
        """Set required dependencies"""
        self.boundary_manager = boundary_manager
        self.cross_camera_tracker = cross_camera_tracker
        logger.info("Violation detector dependencies set")
    
    def process_detection(self, dog_id: str, dog_name: str, track_id: str,
                         camera_id: int, position: Tuple[float, float],
                         confidence: float) -> List[ViolationEvent]:
        """
        Process dog detection for violation analysis
        
        Args:
            dog_id: Dog identifier
            dog_name: Dog name
            track_id: Track identifier
            camera_id: Camera ID
            position: Dog position (x, y)
            confidence: Detection confidence
            
        Returns:
            List of new violation events
        """
        if not self.boundary_manager:
            return []
        
        new_violations = []
        current_time = datetime.now()
        
        # Update position context
        self._update_position_context(track_id, camera_id, position, current_time)
        
        # Check for boundary violations
        boundary_violations = self.boundary_manager.check_position_violations(
            dog_id, dog_name, camera_id, position
        )
        
        # Process each boundary violation
        for boundary_violation in boundary_violations:
            violation_key = f"{track_id}_{boundary_violation.zone_id}"
            
            # Check if this is a new violation or continuation
            if violation_key in self.active_violations:
                # Update existing violation
                self._update_existing_violation(violation_key, position, confidence, current_time)
            else:
                # Create new violation event
                violation_event = self._create_violation_event(
                    boundary_violation, track_id, position, confidence, current_time
                )
                
                if violation_event and self._validate_violation(violation_event):
                    new_violations.append(violation_event)
                    self._add_active_violation(violation_key, violation_event)
        
        # Check for violation resolutions
        self._check_violation_resolutions(track_id, camera_id, position)
        
        return new_violations
    
    def _update_position_context(self, track_id: str, camera_id: int, 
                               position: Tuple[float, float], timestamp: datetime):
        """Update position context for track"""
        if track_id not in self.violation_contexts:
            self.violation_contexts[track_id] = ViolationContext()
        
        context = self.violation_contexts[track_id]
        
        # Add to position history
        position_entry = (position[0], position[1], timestamp)
        context.previous_positions.append(position_entry)
        
        # Limit buffer size
        if len(context.previous_positions) > self.position_buffer_size:
            context.previous_positions.pop(0)
        
        # Add to cross-camera context
        if camera_id not in context.cross_camera_context:
            context.cross_camera_context[camera_id] = []
        
        context.cross_camera_context[camera_id].append(position_entry)
        
        # Limit cross-camera buffer
        if len(context.cross_camera_context[camera_id]) > self.position_buffer_size:
            context.cross_camera_context[camera_id].pop(0)
        
        # Analyze movement pattern
        context.movement_pattern = self._analyze_movement_pattern(context.previous_positions)
    
    def _analyze_movement_pattern(self, positions: List[Tuple[float, float, datetime]]) -> str:
        """Analyze movement pattern from position history"""
        if len(positions) < 3:
            return "unknown"
        
        # Calculate movement vectors
        movements = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = (positions[i][2] - positions[i-1][2]).total_seconds()
            
            if dt > 0:
                speed = np.sqrt(dx*dx + dy*dy) / dt
                movements.append((dx, dy, speed))
        
        if not movements:
            return "stationary"
        
        # Analyze movement characteristics
        speeds = [m[2] for m in movements]
        avg_speed = np.mean(speeds)
        
        if avg_speed < 5.0:  # pixels per second
            return "stationary"
        
        # Analyze direction consistency
        directions = [np.arctan2(m[1], m[0]) for m in movements]
        direction_changes = 0
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i-1])
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            if angle_diff > np.pi/4:  # 45 degrees
                direction_changes += 1
        
        consistency = 1.0 - (direction_changes / max(len(directions) - 1, 1))
        
        if consistency > 0.8:
            return "passing_through"
        else:
            return "exploring"
    
    def _create_violation_event(self, boundary_violation, track_id: str,
                              position: Tuple[float, float], confidence: float,
                              timestamp: datetime) -> Optional[ViolationEvent]:
        """Create new violation event"""
        try:
            event_id = f"violation_{int(timestamp.timestamp())}_{track_id}"
            
            # Determine severity based on zone type and rules
            severity = self._determine_violation_severity(
                boundary_violation.zone_id, 
                boundary_violation.dog_id,
                boundary_violation.violation_type
            )
            
            violation_event = ViolationEvent(
                event_id=event_id,
                dog_id=boundary_violation.dog_id,
                dog_name=boundary_violation.dog_name,
                track_id=track_id,
                zone_id=boundary_violation.zone_id,
                zone_name=boundary_violation.zone_name,
                camera_id=boundary_violation.camera_id,
                position=position,
                violation_type=boundary_violation.violation_type,
                severity=severity,
                confidence=confidence,
                timestamp=timestamp
            )
            
            return violation_event
            
        except Exception as e:
            logger.error(f"Failed to create violation event: {e}")
            return None
    
    def _determine_violation_severity(self, zone_id: str, dog_id: str, 
                                    violation_type: str) -> ViolationSeverity:
        """Determine violation severity based on rules"""
        # Check for specific rules
        for rule in self.violation_rules.values():
            if (rule.zone_id == zone_id and 
                (rule.dog_id is None or rule.dog_id == dog_id) and
                rule.violation_type == violation_type and
                rule.active):
                return rule.severity
        
        # Default severity based on violation type
        if violation_type == "forbidden_entry":
            return ViolationSeverity.HIGH
        elif violation_type == "allowed_exit":
            return ViolationSeverity.MEDIUM
        elif violation_type == "time_restriction":
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    
    def _validate_violation(self, violation_event: ViolationEvent) -> bool:
        """Validate violation against rules and filters"""
        # Check confidence threshold
        min_confidence = 0.7  # Default
        
        # Check for specific rule
        for rule in self.violation_rules.values():
            if (rule.zone_id == violation_event.zone_id and
                (rule.dog_id is None or rule.dog_id == violation_event.dog_id) and
                rule.violation_type == violation_event.violation_type and
                rule.active):
                min_confidence = rule.min_confidence
                break
        
        if violation_event.confidence < min_confidence:
            self.stats['false_positives_filtered'] += 1
            return False
        
        # Check temporal filters (could add more sophisticated filtering here)
        return True
    
    def _add_active_violation(self, violation_key: str, violation_event: ViolationEvent):
        """Add violation to active violations"""
        with self.lock:
            self.active_violations[violation_key] = violation_event
            self.violation_history.append(violation_event)
            
            # Update statistics
            self.stats['total_violations_detected'] += 1
            self.stats['active_violations'] += 1
            self.stats['violations_by_severity'][violation_event.severity.value] += 1
            
            if violation_event.violation_type not in self.stats['violations_by_type']:
                self.stats['violations_by_type'][violation_event.violation_type] = 0
            self.stats['violations_by_type'][violation_event.violation_type] += 1
            
            if violation_event.dog_id not in self.stats['violations_by_dog']:
                self.stats['violations_by_dog'][violation_event.dog_id] = 0
            self.stats['violations_by_dog'][violation_event.dog_id] += 1
            
            if violation_event.camera_id not in self.stats['violations_by_camera']:
                self.stats['violations_by_camera'][violation_event.camera_id] = 0
            self.stats['violations_by_camera'][violation_event.camera_id] += 1
        
        logger.warning(f"New violation detected: {violation_event.dog_name} in {violation_event.zone_name}")
    
    def _update_existing_violation(self, violation_key: str, position: Tuple[float, float],
                                 confidence: float, timestamp: datetime):
        """Update existing violation with new information"""
        if violation_key not in self.active_violations:
            return
        
        violation = self.active_violations[violation_key]
        
        # Update position and confidence
        violation.position = position
        violation.confidence = max(violation.confidence * 0.9, confidence)  # Confidence decay with new evidence
        
        # Update duration
        violation.duration_seconds = (timestamp - violation.timestamp).total_seconds()
    
    def _check_violation_resolutions(self, track_id: str, camera_id: int, position: Tuple[float, float]):
        """Check if any violations should be resolved"""
        current_time = datetime.now()
        
        # Find violations to resolve
        violations_to_resolve = []
        
        with self.lock:
            for violation_key, violation in self.active_violations.items():
                if violation.track_id != track_id:
                    continue
                
                # Check if dog has left the violation zone
                if self.boundary_manager:
                    current_violations = self.boundary_manager.check_position_violations(
                        violation.dog_id, violation.dog_name, camera_id, position
                    )
                    
                    # If no current violations in this zone, resolve the violation
                    zone_still_violated = any(
                        v.zone_id == violation.zone_id for v in current_violations
                    )
                    
                    if not zone_still_violated:
                        violations_to_resolve.append(violation_key)
        
        # Resolve violations
        for violation_key in violations_to_resolve:
            self._resolve_violation(violation_key, current_time)
    
    def _resolve_violation(self, violation_key: str, resolution_time: datetime):
        """Resolve active violation"""
        if violation_key not in self.active_violations:
            return
        
        with self.lock:
            violation = self.active_violations[violation_key]
            violation.resolved = True
            violation.resolution_time = resolution_time
            violation.duration_seconds = (resolution_time - violation.timestamp).total_seconds()
            
            # Remove from active violations
            del self.active_violations[violation_key]
            
            # Update statistics
            self.stats['active_violations'] -= 1
            self.stats['violations_resolved'] += 1
            
            # Update average duration
            total_duration = sum(v.duration_seconds for v in self.violation_history if v.resolved)
            resolved_count = sum(1 for v in self.violation_history if v.resolved)
            self.stats['average_violation_duration'] = total_duration / max(resolved_count, 1)
        
        logger.info(f"Violation resolved: {violation.dog_name} left {violation.zone_name} after {violation.duration_seconds:.1f}s")
    
    def _cleanup_loop(self):
        """Background cleanup of old violations and contexts"""
        while True:
            try:
                time.sleep(60)  # Run cleanup every minute
                current_time = datetime.now()
                
                # Clean up old violation contexts
                with self.lock:
                    contexts_to_remove = []
                    for track_id, context in self.violation_contexts.items():
                        if context.previous_positions:
                            last_update = context.previous_positions[-1][2]
                            if (current_time - last_update).total_seconds() > 600:  # 10 minutes
                                contexts_to_remove.append(track_id)
                    
                    for track_id in contexts_to_remove:
                        del self.violation_contexts[track_id]
                
                # Clean up old violation history (keep last 1000)
                with self.lock:
                    if len(self.violation_history) > 1000:
                        self.violation_history = self.violation_history[-500:]
                
                # Timeout old active violations
                violations_to_timeout = []
                with self.lock:
                    for violation_key, violation in self.active_violations.items():
                        age = (current_time - violation.timestamp).total_seconds()
                        if age > self.violation_timeout_seconds:
                            violations_to_timeout.append(violation_key)
                
                for violation_key in violations_to_timeout:
                    self._resolve_violation(violation_key, current_time)
                    logger.warning(f"Violation timed out: {violation_key}")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def add_violation_rule(self, rule: ViolationRule):
        """Add custom violation rule"""
        self.violation_rules[rule.rule_id] = rule
        logger.info(f"Added violation rule: {rule.name}")
    
    def remove_violation_rule(self, rule_id: str) -> bool:
        """Remove violation rule"""
        if rule_id in self.violation_rules:
            del self.violation_rules[rule_id]
            logger.info(f"Removed violation rule: {rule_id}")
            return True
        return False
    
    def get_active_violations(self) -> List[Dict[str, Any]]:
        """Get all active violations"""
        with self.lock:
            return [violation.to_dict() for violation in self.active_violations.values()]
    
    def get_violation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get violation history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_violations = [
                violation.to_dict() for violation in self.violation_history
                if violation.timestamp >= cutoff_time
            ]
        
        return recent_violations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get violation detection statistics"""
        with self.lock:
            return self.stats.copy()
    
    def force_resolve_violation(self, violation_key: str) -> bool:
        """Manually resolve violation"""
        if violation_key in self.active_violations:
            self._resolve_violation(violation_key, datetime.now())
            return True
        return False

# Global violation detector instance
violation_detector = ViolationDetector()

def get_violation_detector() -> ViolationDetector:
    """Get global violation detector instance"""
    return violation_detector