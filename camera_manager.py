"""
Multi-Camera Stream Manager for Dog Tracking System

Handles simultaneous streaming from up to 12 IP cameras with performance optimization
for Jetson Orin Nano. Supports RTSP, HTTP, and ONVIF camera protocols.

Key Features:
- Concurrent camera stream management
- Automatic reconnection on connection loss
- Frame rate management for performance
- GPU memory optimization
- Cross-camera synchronization
"""

import cv2
import threading
import time
import queue
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
from project_paths import get_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraConfig:
    """Configuration for individual camera"""
    id: int
    name: str
    url: str
    enabled: bool = True
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 15
    codec: str = 'auto'
    location: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'enabled': self.enabled,
            'resolution': list(self.resolution),
            'fps': self.fps,
            'codec': self.codec,
            'location': self.location
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraConfig':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            name=data['name'],
            url=data['url'],
            enabled=data.get('enabled', True),
            resolution=tuple(data.get('resolution', [1280, 720])),
            fps=data.get('fps', 15),
            codec=data.get('codec', 'auto'),
            location=data.get('location', '')
        )

@dataclass
class CameraStatus:
    """Current status of a camera"""
    id: int
    online: bool
    fps_actual: float
    frame_count: int
    last_frame_time: datetime
    error_message: str = ''
    connection_attempts: int = 0
    
class CameraStream:
    """Individual camera stream handler"""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = queue.Queue(maxsize=3)  # Small buffer to prevent memory issues
        self.current_frame: Optional[Any] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.status = CameraStatus(
            id=config.id,
            online=False,
            fps_actual=0.0,
            frame_count=0,
            last_frame_time=datetime.now()
        )
        self._frame_times = []
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """Start camera stream"""
        if self.running:
            return True
            
        try:
            logger.info(f"Starting camera {self.config.id}: {self.config.name}")
            
            # Initialize camera capture
            if self.config.url.startswith('rtsp://') or self.config.url.startswith('http://'):
                # IP camera
                self.cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
            else:
                # Local camera (for testing)
                self.cap = cv2.VideoCapture(int(self.config.url))
            
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera stream: {self.config.url}")
            
            # Configure camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            
            # Test frame capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Failed to capture test frame")
            
            # Start capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            self.status.online = True
            self.status.error_message = ''
            logger.info(f"Camera {self.config.id} started successfully")
            return True
            
        except Exception as e:
            self.status.online = False
            self.status.error_message = str(e)
            self.status.connection_attempts += 1
            logger.error(f"Failed to start camera {self.config.id}: {e}")
            
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def stop(self):
        """Stop camera stream"""
        logger.info(f"Stopping camera {self.config.id}")
        
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.status.online = False
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        frame_interval = 1.0 / self.config.fps
        last_capture_time = time.time()
        
        while self.running and self.cap:
            try:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_capture_time < frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning(f"Camera {self.config.id}: Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Update frame in queue (drop old frames if queue is full)
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()  # Remove oldest frame
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip this frame if queue is still full
                
                # Update current frame with thread safety
                with self._lock:
                    self.current_frame = frame.copy()
                    self.status.frame_count += 1
                    self.status.last_frame_time = datetime.now()
                
                # Calculate actual FPS
                self._frame_times.append(current_time)
                if len(self._frame_times) > 30:  # Keep last 30 frame times
                    self._frame_times.pop(0)
                
                if len(self._frame_times) > 1:
                    time_diff = self._frame_times[-1] - self._frame_times[0]
                    self.status.fps_actual = len(self._frame_times) / time_diff
                
                last_capture_time = current_time
                
            except Exception as e:
                logger.error(f"Camera {self.config.id} capture error: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[Any]:
        """Get latest frame (thread-safe)"""
        with self._lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_frame_queue(self) -> Optional[Any]:
        """Get frame from queue (non-blocking)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_alive(self) -> bool:
        """Check if camera is alive and capturing"""
        if not self.status.online:
            return False
        
        # Check if we've received frames recently
        time_since_last_frame = datetime.now() - self.status.last_frame_time
        return time_since_last_frame.total_seconds() < 5.0  # 5 second timeout
    
    def get_info(self) -> Dict[str, Any]:
        """Get camera information and statistics"""
        return {
            'id': self.config.id,
            'name': self.config.name,
            'url': self.config.url,
            'online': self.status.online,
            'fps_actual': round(self.status.fps_actual, 1),
            'fps_target': self.config.fps,
            'frame_count': self.status.frame_count,
            'last_frame': self.status.last_frame_time.isoformat(),
            'error': self.status.error_message,
            'connection_attempts': self.status.connection_attempts,
            'resolution': self.config.resolution
        }

class MultiCameraManager:
    """Manages multiple camera streams with performance optimization"""
    
    def __init__(self, max_cameras: int = 12):
        self.max_cameras = max_cameras
        self.cameras: Dict[int, CameraStream] = {}
        self.camera_configs: Dict[int, CameraConfig] = {}
        self.running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Performance settings for Jetson Orin Nano
        self.performance_mode = 'balanced'  # 'performance', 'balanced', 'power_save'
        self.max_concurrent_streams = 8  # Limit concurrent processing
        
        # Load saved configurations
        self.load_configurations()
    
    def add_camera(self, config: CameraConfig) -> bool:
        """Add a new camera configuration"""
        if len(self.camera_configs) >= self.max_cameras:
            logger.error(f"Maximum camera limit ({self.max_cameras}) reached")
            return False
        
        if config.id in self.camera_configs:
            logger.warning(f"Camera {config.id} already exists, updating configuration")
        
        self.camera_configs[config.id] = config
        self.save_configurations()
        
        # Start camera if manager is running and camera is enabled
        if self.running and config.enabled:
            return self.start_camera(config.id)
        
        return True
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove camera configuration"""
        if camera_id not in self.camera_configs:
            logger.error(f"Camera {camera_id} not found")
            return False
        
        # Stop camera if running
        if camera_id in self.cameras:
            self.stop_camera(camera_id)
        
        # Remove configuration
        del self.camera_configs[camera_id]
        self.save_configurations()
        
        return True
    
    def start_camera(self, camera_id: int) -> bool:
        """Start specific camera"""
        if camera_id not in self.camera_configs:
            logger.error(f"Camera {camera_id} configuration not found")
            return False
        
        config = self.camera_configs[camera_id]
        if not config.enabled:
            logger.info(f"Camera {camera_id} is disabled")
            return False
        
        # Stop existing camera if running
        if camera_id in self.cameras:
            self.stop_camera(camera_id)
        
        # Create and start new camera stream
        camera_stream = CameraStream(config)
        if camera_stream.start():
            self.cameras[camera_id] = camera_stream
            logger.info(f"Camera {camera_id} started successfully")
            return True
        else:
            logger.error(f"Failed to start camera {camera_id}")
            return False
    
    def stop_camera(self, camera_id: int) -> bool:
        """Stop specific camera"""
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not running")
            return False
        
        self.cameras[camera_id].stop()
        del self.cameras[camera_id]
        logger.info(f"Camera {camera_id} stopped")
        return True
    
    def start_all(self) -> bool:
        """Start all enabled cameras"""
        self.running = True
        
        # Start enabled cameras
        success_count = 0
        for camera_id, config in self.camera_configs.items():
            if config.enabled:
                if self.start_camera(camera_id):
                    success_count += 1
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Started {success_count}/{len(self.camera_configs)} cameras")
        return success_count > 0
    
    def stop_all(self):
        """Stop all cameras"""
        self.running = False
        
        # Stop all cameras
        camera_ids = list(self.cameras.keys())
        for camera_id in camera_ids:
            self.stop_camera(camera_id)
        
        # Stop monitoring thread
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2)
        
        logger.info("All cameras stopped")
    
    def _monitoring_loop(self):
        """Monitor camera health and attempt reconnections"""
        while self.running:
            try:
                # Check each camera's health
                for camera_id, camera_stream in list(self.cameras.items()):
                    if not camera_stream.is_alive():
                        logger.warning(f"Camera {camera_id} appears offline, attempting restart")
                        
                        # Attempt restart
                        camera_stream.stop()
                        time.sleep(1)
                        
                        if not camera_stream.start():
                            logger.error(f"Failed to restart camera {camera_id}")
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def get_frame(self, camera_id: int) -> Optional[Any]:
        """Get latest frame from specific camera"""
        if camera_id not in self.cameras:
            return None
        
        return self.cameras[camera_id].get_frame()
    
    def get_all_frames(self) -> Dict[int, Any]:
        """Get latest frames from all cameras"""
        frames = {}
        for camera_id, camera_stream in self.cameras.items():
            frame = camera_stream.get_frame()
            if frame is not None:
                frames[camera_id] = frame
        return frames
    
    def get_camera_info(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """Get information about specific camera"""
        if camera_id not in self.cameras:
            # Return config info even if camera is not running
            if camera_id in self.camera_configs:
                config = self.camera_configs[camera_id]
                return {
                    'id': config.id,
                    'name': config.name,
                    'url': config.url,
                    'online': False,
                    'enabled': config.enabled,
                    'error': 'Camera not started'
                }
            return None
        
        return self.cameras[camera_id].get_info()
    
    def get_all_camera_info(self) -> List[Dict[str, Any]]:
        """Get information about all cameras"""
        info_list = []
        for camera_id in self.camera_configs:
            info = self.get_camera_info(camera_id)
            if info:
                info_list.append(info)
        return info_list
    
    def test_camera_connection(self, url: str) -> Tuple[bool, str]:
        """Test camera connection without adding to manager"""
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                return False, "Failed to open camera stream"
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False, "Failed to capture test frame"
            
            height, width = frame.shape[:2]
            return True, f"Connection successful - Resolution: {width}x{height}"
            
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def save_configurations(self):
        """Save camera configurations to file"""
        try:
            config_data = {
                'cameras': [config.to_dict() for config in self.camera_configs.values()],
                'settings': {
                    'max_cameras': self.max_cameras,
                    'performance_mode': self.performance_mode,
                    'max_concurrent_streams': self.max_concurrent_streams
                }
            }
            
            with open(str(get_path('camera_config')), 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info("Camera configurations saved")
            
        except Exception as e:
            logger.error(f"Failed to save configurations: {e}")
    
    def load_configurations(self):
        """Load camera configurations from file"""
        try:
            with open(str(get_path('camera_config')), 'r') as f:
                config_data = json.load(f)
            
            # Load camera configurations
            for camera_data in config_data.get('cameras', []):
                config = CameraConfig.from_dict(camera_data)
                self.camera_configs[config.id] = config
            
            # Load settings
            settings = config_data.get('settings', {})
            self.max_cameras = settings.get('max_cameras', 12)
            self.performance_mode = settings.get('performance_mode', 'balanced')
            self.max_concurrent_streams = settings.get('max_concurrent_streams', 8)
            
            logger.info(f"Loaded {len(self.camera_configs)} camera configurations")
            
        except FileNotFoundError:
            logger.info("No camera configuration file found, starting with empty configuration")
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
    
    def set_performance_mode(self, mode: str):
        """Set performance mode for Jetson optimization"""
        valid_modes = ['performance', 'balanced', 'power_save']
        if mode not in valid_modes:
            logger.error(f"Invalid performance mode: {mode}")
            return False
        
        self.performance_mode = mode
        
        # Adjust settings based on performance mode
        if mode == 'performance':
            self.max_concurrent_streams = 12
            # Higher quality settings
        elif mode == 'balanced':
            self.max_concurrent_streams = 8
            # Balanced settings
        elif mode == 'power_save':
            self.max_concurrent_streams = 6
            # Lower quality settings for power saving
        
        self.save_configurations()
        logger.info(f"Performance mode set to: {mode}")
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        online_cameras = sum(1 for cam in self.cameras.values() if cam.status.online)
        total_fps = sum(cam.status.fps_actual for cam in self.cameras.values())
        
        return {
            'total_cameras': len(self.camera_configs),
            'online_cameras': online_cameras,
            'running_cameras': len(self.cameras),
            'total_fps': round(total_fps, 1),
            'average_fps': round(total_fps / max(len(self.cameras), 1), 1),
            'performance_mode': self.performance_mode,
            'max_concurrent_streams': self.max_concurrent_streams,
            'uptime_seconds': (datetime.now() - datetime.now()).total_seconds() if self.running else 0
        }

# Global camera manager instance
camera_manager = MultiCameraManager()

def get_camera_manager() -> MultiCameraManager:
    """Get global camera manager instance"""
    return camera_manager