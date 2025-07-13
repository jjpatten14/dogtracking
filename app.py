from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import json
import numpy as np
from threading import Thread
import time
import os
import logging
from datetime import datetime
from project_paths import get_path, get_snapshot_path, PROJECT_ROOT

# Configure logging
logger = logging.getLogger(__name__)

def save_boundary_snapshot(frame, violation):
    """
    Save a snapshot of the current frame when a boundary violation occurs
    
    Args:
        frame: Current video frame (numpy array)
        violation: BoundaryViolation object with details
    """
    try:
        # Create timestamp for filename
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H%M%S")
        
        # Create date-based subdirectory using portable path system
        snapshots_dir = get_snapshot_path(date_str, create_parents=True)
        
        # Create descriptive filename
        filename = f"boundary_alert_CAMERA{violation.camera_id}_{violation.event_type.upper()}_{violation.zone_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        file_path = snapshots_dir / filename
        
        # Save the frame
        success = cv2.imwrite(str(file_path), frame)
        
        if success:
            print(f" Snapshot saved: {file_path}")
            return file_path
        else:
            print(f" Failed to save snapshot: {file_path}")
            return None
            
    except Exception as e:
        print(f" Error saving snapshot: {e}")
        return None

# GPU video acceleration
try:
    from gpu_video import get_gpu_processor, encode_frame_fast, is_gpu_available
    GPU_VIDEO_AVAILABLE = True
    print("GPU video acceleration available")
except ImportError as e:
    GPU_VIDEO_AVAILABLE = False
    print(f"GPU video acceleration not available: {e}")

# Grounding DINO disabled for now - focusing on dog detection
GROUNDING_DINO_AVAILABLE = False
print("Grounding DINO disabled - focusing on dog detection performance")

# Training manager imports
try:
    from training_manager import get_training_manager
    TRAINING_MANAGER_AVAILABLE = True
    print(" Training manager loaded successfully")
except ImportError as e:
    TRAINING_MANAGER_AVAILABLE = False
    print(f"  Training manager not available: {e}")

# Animal detection and boundary system imports
try:
    from animal_detector import get_animal_detector, load_animal_detector
    from dog_identifier import get_dog_identifier
    from alert_system import get_alert_system, AlertType, AlertSeverity
    from boundary_manager import get_boundary_manager
    SURVEILLANCE_SYSTEMS_AVAILABLE = True
    print(" Surveillance systems loaded successfully")
except ImportError as e:
    SURVEILLANCE_SYSTEMS_AVAILABLE = False
    print(f"  Surveillance systems not available: {e}")

# TTS system imports
try:
    from tts.tts_service import TTSService
    import soundfile as sf
    import tempfile
    TTS_AVAILABLE = True
    print(" TTS system loaded successfully")
except ImportError as e:
    TTS_AVAILABLE = False
    print(f"  TTS system not available: {e}")

app = Flask(__name__)

# Global variables for camera and boundaries
camera = None
boundary_list = []  # Renamed to avoid conflict with boundaries() route function
current_boundary = []
drawing = False

# Global surveillance system instances
animal_detector = None
dog_identifier = None
alert_system = None
boundary_manager = None
tts_service = None

# Global variables for reference points
reference_points = []  # Storage for reference point data
reference_model = None  # EfficientDet model for reference detection
REFERENCE_CONFIG_FILE = 'reference_points.json'
baseline_established = False
frame_counter = 0  # Counter for reference point detection frequency
last_movement_check = time.time()  # Timestamp of last movement check
reference_detection_errors = 0  # Error counter for recovery
max_detection_errors = 5  # Max errors before disabling detection
current_movement_offset = None  # Track cumulative camera movement for reference point rendering

# Global variable to store last detections for reuse
last_detections = {}

# Settings file path
SETTINGS_FILE = 'camera_settings.json'

# Default settings
DEFAULT_SETTINGS = {
    # Camera Configuration
    "camera_urls": [],
    "active_camera": 0,
    "camera_layout": "3x4",
    
    # Detection Configuration
    "animal_confidence": 0.7,
    "person_confidence": 0.8,
    "vehicle_confidence": 0.6,
    "dog_id_confidence": 0.8,
    "show_confidence": False,
    "show_bounding_boxes": True,
    "enable_cross_camera_tracking": True,
    
    # Boundary & Reference Points
    "boundary_color": "#ff0000",
    "boundary_thickness": 3,
    "reference_tolerance": 50,
    "boundary_opacity": 0.7,
    "enable_dynamic_boundaries": True,
    "auto_reference_calibration": True,
    "show_reference_points": True,
    
    # Actions & Alerts
    "action_email": False,
    "action_sound": True,
    "action_log": True,
    "save_snapshots": True,
    "email_address": "",
    "alert_cooldown": 30,
    
    # Performance & System
    "processing_fps": 15,
    "frame_skip": 2,
    "gpu_memory_limit": 6,
    "storage_days": 30,
    "enable_tensorrt": False,
    "debug_mode": False,
    "auto_restart": True,
    "monitor_system_health": True
}

def load_settings():
    """Load settings from file or return defaults"""
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged_settings = DEFAULT_SETTINGS.copy()
                merged_settings.update(settings)
                return merged_settings
        return DEFAULT_SETTINGS.copy()
    except Exception as e:
        print(f"Error loading settings: {e}")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

class CameraStream:
    def __init__(self, camera_source=0, camera_id=None):
        self.camera_source = camera_source
        self.camera_id = camera_id
        self.cap = None
        self.latest_frame = None  # Use latest_frame like the working system
        self.running = False
        self.thread = None
        self.last_error = None
        
    def start(self):
        """Start camera in a separate thread to avoid blocking"""
        self.last_error = None
        # Start connection in background thread
        connection_thread = Thread(target=self._connect_camera, daemon=True)
        connection_thread.start()
        return True  # Return immediately, actual status will be updated asynchronously
    
    def _connect_camera(self):
        """Internal method to connect to camera with timeout"""
        try:
            print(f" Camera {self.camera_id}: Attempting connection to {self.camera_source}")
            
            # Support both IP cameras and local cameras
            if isinstance(self.camera_source, str):
                print(f" Camera {self.camera_id}: Using RTSP/IP camera mode")
                # IP camera URL
                self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                # Set timeouts for RTSP streams (from working system)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                print(f" Camera {self.camera_id}: Set 5-second timeouts")
            else:
                print(f" Camera {self.camera_id}: Using local camera mode")
                # Local camera index
                self.cap = cv2.VideoCapture(self.camera_source)
            
            print(f" Camera {self.camera_id}: Setting video properties...")
            # Set properties (use camera's native resolution for best performance)
            self.cap.set(cv2.CAP_PROP_FPS, 15)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print(f" Camera {self.camera_id}: Testing connection...")
            # Test camera connection
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Failed to read test frame from camera")
            
            print(f" Camera {self.camera_id}: Connected! Frame size: {test_frame.shape}")
            self.latest_frame = test_frame.copy()
            
            self.running = True
            self.thread = Thread(target=self._capture_frames, daemon=True)
            self.thread.start()
            print(f" Camera {self.camera_id}: Frame capture thread started successfully")
            
        except Exception as e:
            print(f" Camera {self.camera_id}: Connection failed - {e}")
            self.last_error = str(e)
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def _capture_frames(self):
        """Continuous frame capture thread (from working system)"""
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Camera {self.camera_id}: Could not read frame after {frame_count} frames")
                self.last_error = "Failed to read frame"
                break
            
            # Update latest_frame immediately for smooth video streaming
            self.latest_frame = frame.copy()
            frame_count += 1
            
            # Debug output every 100 frames
            if frame_count % 100 == 0:
                print(f"Camera {self.camera_id}: {frame_count} frames captured")
            
            time.sleep(0.033)  # ~30 fps capture rate
            
    def get_frame(self):
        return self.latest_frame
    
    def set_camera_source(self, source):
        """Change camera source (URL or index)"""
        was_running = self.running
        if was_running:
            self.stop()
        
        self.camera_source = source
        
        if was_running:
            return self.start()
        return True
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap:
            self.cap.release()
            self.cap = None

class MultiCameraManager:
    def __init__(self):
        self.cameras = {}
        self.max_cameras = 12
        
    def add_camera(self, camera_id, camera_url):
        """Add a camera to the manager"""
        print(f" Adding camera {camera_id} with URL: {camera_url}")
        if camera_id not in self.cameras:
            self.cameras[camera_id] = CameraStream(camera_url, camera_id)
            result = self.cameras[camera_id].start()
            print(f" Camera {camera_id} start result: {result}")
            return result
        else:
            print(f"  Camera {camera_id} already exists")
        return False
        
    def remove_camera(self, camera_id):
        """Remove a camera from the manager"""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            
    def get_camera(self, camera_id):
        """Get a specific camera stream"""
        return self.cameras.get(camera_id)
        
    def get_frame(self, camera_id):
        """Get frame from specific camera"""
        camera = self.cameras.get(camera_id)
        if camera and camera.running:
            return camera.get_frame()
        return None
        
    def update_cameras_from_settings(self):
        """Update cameras based on settings"""
        print(" Loading settings for camera update...")
        settings = load_settings()
        camera_urls = settings.get('camera_urls', [])
        
        print(f" Processing {len(camera_urls)} camera URLs from settings")
        
        # Stop and remove cameras that are no longer in settings
        current_ids = list(self.cameras.keys())
        for cam_id in current_ids:
            if cam_id > len(camera_urls):
                print(f"  Removing camera {cam_id} (no longer in settings)")
                self.remove_camera(cam_id)
        
        # Add or update cameras from settings
        for i, url in enumerate(camera_urls):
            camera_id = i + 1
            if url:  # Only add if URL is not empty
                print(f" Processing camera {camera_id}: {url}")
                if camera_id in self.cameras:
                    # Update existing camera if URL changed
                    if self.cameras[camera_id].camera_source != url:
                        print(f" Updating camera {camera_id} URL")
                        self.cameras[camera_id].set_camera_source(url)
                    else:
                        print(f" Camera {camera_id} already configured with correct URL")
                else:
                    # Add new camera
                    print(f" Adding new camera {camera_id}")
                    self.add_camera(camera_id, url)
            else:
                print(f"  Skipping camera {camera_id} - empty URL")
                    
    def get_all_statuses(self):
        """Get status of all cameras"""
        statuses = {}
        for cam_id, camera in self.cameras.items():
            # Determine status
            if camera.running:
                status = 'online'
            elif camera.last_error:
                status = 'error'
            else:
                status = 'connecting'
                
            statuses[cam_id] = {
                'running': camera.running,
                'source': camera.camera_source,
                'error': camera.last_error,
                'status': status
            }
        return statuses
        
    def stop_all(self):
        """Stop all cameras"""
        for camera in self.cameras.values():
            camera.stop()

# Initialize camera manager
camera_manager = MultiCameraManager()

# Keep single camera stream for backward compatibility
camera_stream = CameraStream()

def load_reference_model():
    """Load Grounding DINO model for reference point detection"""
    global reference_model
    
    print(" ===== GROUNDING DINO MODEL LOADING DEBUG =====")
    print(f" GROUNDING_DINO_AVAILABLE status: {GROUNDING_DINO_AVAILABLE}")
    
    if not GROUNDING_DINO_AVAILABLE:
        print(" Grounding DINO dependencies not available:")
        print("   - Missing packages: transformers, torch, or PIL")
        print("   - Install with: pip install transformers torch pillow")
        print("   - Reference point detection disabled")
        return False
    
    print(" Grounding DINO dependencies are available")
    print(" Available imports:")
    print(f"   - torch: {torch.__version__}")
    print(f"   - transformers: loaded successfully")
    
    try:
        print(" Attempting to load Grounding DINO model...")
        print("   - Model: IDEA-Research/grounding-dino-tiny")
        print("   - Using Hugging Face Transformers")
        print("   - This may take a while on first run (downloading model)...")
        
        # Load processor and model with GPU acceleration
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f" Model loaded on device: {device}")
        
        if device == "cuda":
            print(f" GPU: {torch.cuda.get_device_name(0)}")
            print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print(" Setting model to evaluation mode...")
        model.eval()
        print(" Evaluation mode set")
        
        # Store both processor and model
        reference_model = {'model': model, 'processor': processor}
        
        print(" Grounding DINO loaded successfully for reference point detection")
        print(" Model ready for natural language object detection")
        print(" Supported prompts: 'tree trunk', 'building corner', 'fence post', etc.")
        print(" ===== MODEL LOADING COMPLETE =====")
        return True
        
    except ImportError as e:
        print(f" Import error during model loading: {e}")
        print("   - Check if transformers is properly installed")
        reference_model = None
        return False
    except Exception as e:
        print(f" Failed to load Grounding DINO: {e}")
        print(f"   - Error type: {type(e).__name__}")
        print(f"   - Error details: {str(e)}")
        import traceback
        print("   - Full traceback:")
        traceback.print_exc()
        reference_model = None
        return False

def detect_reference_objects(frame):
    """Detect reference objects in frame using Grounding DINO"""
    global reference_model
    
    if reference_model is None or not GROUNDING_DINO_AVAILABLE:
        print("  Grounding DINO model not available")
        return []
    
    try:
        print(f"  ===== GROUNDING DINO DETECTION DEBUG =====")
        print(f" Original frame shape: {frame.shape}")
        
        # Convert OpenCV BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        print(f"  PIL Image size: {pil_image.size}")
        
        # Define static reference point prompts (NO MOVING OBJECTS)
        static_prompts = [
            "tree trunk",
            "large tree", 
            "building corner",
            "house corner",
            "fence post",
            "utility pole",
            "telephone pole",
            "roof edge",
            "chimney"
        ]
        
        print(f" Detection prompts: {static_prompts}")
        print(f" Focus: STATIC objects only (no cars, people, or moving objects)")
        
        # Prepare inputs for Grounding DINO
        model = reference_model['model']
        processor = reference_model['processor']
        
        # Join prompts with periods as required by Grounding DINO
        text_prompt = ". ".join(static_prompts) + "."
        print(f" Final text prompt: {text_prompt}")
        
        # Process image and text
        inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt")
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        print(f" Running Grounding DINO inference on {device}...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract results
        results = processor.post_process_grounded_object_detection(
            outputs, 
            inputs["input_ids"], 
            box_threshold=0.2,  # Lower threshold for static objects
            text_threshold=0.2,
            target_sizes=[pil_image.size[::-1]]  # (height, width)
        )[0]
        
        detected_objects = []
        
        print(f" ===== GROUNDING DINO RESULTS =====")
        print(f" Raw detections found: {len(results['boxes'])}")
        
        if len(results['boxes']) > 0:
            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy() 
            labels = results['labels']
            
            print(f"    Total detections: {len(boxes)}")
            print(f"    Score range: {scores.min():.3f} - {scores.max():.3f}")
            
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                print(f"   Detection {i}: '{label}' score={score:.3f} center=({center_x}, {center_y})")
                
                # All detected objects are static by design of our prompts
                detected_objects.append({
                    'id': i,
                    'type': label,  # Natural language label from Grounding DINO
                    'position': [center_x, center_y],
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(score),
                    'class': label,
                    'priority': 1.0,  # All static objects have equal priority
                    'raw_score': float(score)
                })
        
        # Sort by confidence (highest first)
        detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
        max_reference_points = 10
        detected_objects = detected_objects[:max_reference_points]
        
        print(f" ===== FINAL STATIC REFERENCE POINTS =====")
        print(f" Detected {len(detected_objects)} static reference objects")
        if detected_objects:
            print(f" QUALIFIED STATIC REFERENCE POINTS:")
            for obj in detected_objects:
                print(f"    {obj['type']} at ({obj['position'][0]}, {obj['position'][1]}) - confidence: {obj['confidence']:.2f}")
        else:
            print(f" NO static objects detected")
            print(f" Try manual reference points or adjust detection thresholds")
        print(f" ===== DETECTION COMPLETE =====")
        
        return detected_objects
        
    except Exception as e:
        print(f" Error in Grounding DINO detection: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_reference_points():
    """Save reference points to JSON file"""
    global reference_points, baseline_established
    try:
        config = {
            "reference_points": reference_points,
            "baseline_established": baseline_established,
            "baseline_timestamp": time.time(),
            "camera_id": 1
        }
        with open(REFERENCE_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f" Reference points saved to {REFERENCE_CONFIG_FILE}")
        return True
    except Exception as e:
        print(f" Error saving reference points: {e}")
        return False

def load_reference_points():
    """Load reference points from JSON file"""
    global reference_points, baseline_established
    try:
        if os.path.exists(REFERENCE_CONFIG_FILE):
            with open(REFERENCE_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            reference_points = config.get('reference_points', [])
            baseline_established = config.get('baseline_established', False)
            print(f" Loaded {len(reference_points)} reference points from file")
            return True
        else:
            print(f"  No reference points file found at {REFERENCE_CONFIG_FILE}")
            return False
    except Exception as e:
        print(f" Error loading reference points: {e}")
        return False

def calculate_camera_movement(baseline_points, current_points, tolerance=10):
    """Calculate camera movement by comparing baseline and current reference points"""
    if not baseline_points or not current_points:
        return None
    
    try:
        settings = load_settings()
        max_distance = settings.get('reference_tolerance', 100)  # Use configurable threshold
        
        print(f" Matching {len(baseline_points)} baseline points with {len(current_points)} current detections")
        print(f" Using max distance threshold: {max_distance}px")
        
        # Match points between baseline and current based on proximity and object type
        matched_pairs = []
        
        for i, baseline_point in enumerate(baseline_points):
            baseline_pos = baseline_point['baseline_position']
            baseline_type = baseline_point.get('type', 'unknown')
            best_match = None
            best_distance = float('inf')
            
            print(f"   Baseline {i}: {baseline_type} at ({baseline_pos[0]}, {baseline_pos[1]})")
            
            for j, current_point in enumerate(current_points):
                current_pos = current_point['position']
                current_type = current_point.get('type', 'unknown')
                
                # Calculate distance between points
                distance = np.sqrt(
                    (baseline_pos[0] - current_pos[0])**2 + 
                    (baseline_pos[1] - current_pos[1])**2
                )
                
                # Check if object types are compatible (very permissive matching)
                baseline_words = set(baseline_type.lower().split())
                current_words = set(current_type.lower().split())
                
                type_compatible = (
                    baseline_type.lower() == current_type.lower() or  # Exact match
                    len(baseline_words & current_words) > 0 or  # Any common word
                    'pole' in baseline_type.lower() and 'pole' in current_type.lower() or  # Both have "pole"
                    'tree' in baseline_type.lower() and 'tree' in current_type.lower() or  # Both have "tree"
                    'chimney' in baseline_type.lower() and 'chimney' in current_type.lower() or  # Both have "chimney"
                    'corner' in baseline_type.lower() and 'corner' in current_type.lower() or  # Both have "corner"
                    'edge' in baseline_type.lower() and 'edge' in current_type.lower()  # Both have "edge"
                )
                
                # Debug: Show detailed comparison
                distance_ok = distance < max_distance
                print(f"     Current {j}: {current_type} at ({current_pos[0]}, {current_pos[1]})")
                print(f"        Distance: {distance:.1f}px (limit: {max_distance}px) - {'' if distance_ok else ''}")
                print(f"        Type match: {type_compatible} - {'' if type_compatible else ''}")
                print(f"        Overall valid: {distance_ok and type_compatible}")
                
                # Only consider matches within distance and with compatible types
                if distance < best_distance and distance < max_distance and type_compatible:
                    best_distance = distance
                    best_match = current_point
                    print(f"       NEW BEST MATCH! Distance: {distance:.1f}px")
                elif not distance_ok:
                    print(f"       Rejected: distance {distance:.1f}px > {max_distance}px limit")
                elif not type_compatible:
                    print(f"       Rejected: type mismatch '{baseline_type}' vs '{current_type}'")
                else:
                    print(f"       Rejected: distance {distance:.1f}px > current best {best_distance:.1f}px")
            
            if best_match:
                matched_pairs.append((baseline_point, best_match))
                print(f"     Matched: {baseline_type} -> {best_match.get('type', 'unknown')} (distance: {best_distance:.1f}px)")
            else:
                print(f"     No match found for {baseline_type}")
        
        if len(matched_pairs) < 2:
            print(f"  Not enough matched reference points ({len(matched_pairs)}) for movement calculation")
            return None
        
        # Calculate average movement
        total_dx = 0
        total_dy = 0
        
        for baseline_point, current_point in matched_pairs:
            dx = current_point['position'][0] - baseline_point['baseline_position'][0]
            dy = current_point['position'][1] - baseline_point['baseline_position'][1]
            total_dx += dx
            total_dy += dy
        
        avg_dx = total_dx / len(matched_pairs)
        avg_dy = total_dy / len(matched_pairs)
        
        # Only return movement if it's above tolerance
        movement_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
        if movement_magnitude > tolerance:
            print(f" Camera movement detected: dx={avg_dx:.1f}, dy={avg_dy:.1f} (magnitude: {movement_magnitude:.1f}px)")
            return {
                'dx': avg_dx,
                'dy': avg_dy,
                'magnitude': movement_magnitude,
                'matched_points': len(matched_pairs)
            }
        else:
            print(f" Camera movement within tolerance: {movement_magnitude:.1f}px <= {tolerance}px")
            return None
            
    except Exception as e:
        print(f" Error calculating camera movement: {e}")
        return None

def adjust_boundaries_for_movement(movement_offset):
    """Adjust boundary coordinates based on camera movement"""
    global boundary_list
    
    if not movement_offset or not boundary_list:
        return False
    
    try:
        dx = movement_offset['dx']
        dy = movement_offset['dy']
        
        # Sanity check: don't apply massive movements (likely detection errors)
        if abs(dx) > 200 or abs(dy) > 200:
            print(f" Rejecting large movement adjustment: dx={dx:.1f}, dy={dy:.1f} (likely detection error)")
            return False
        
        print(f" Adjusting {len(boundary_list)} boundaries for camera movement: dx={dx:.1f}, dy={dy:.1f}")
        
        # Apply offset to all boundaries
        for i, boundary in enumerate(boundary_list):
            for j, point in enumerate(boundary):
                # Ensure boundaries don't go outside frame bounds
                new_x = max(0, min(point[0] + dx, 9999))  # Reasonable upper bound
                new_y = max(0, min(point[1] + dy, 9999))
                boundary_list[i][j] = [new_x, new_y]
        
        print(f" Boundaries adjusted for camera movement")
        return True
        
    except Exception as e:
        print(f" Error adjusting boundaries: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/boundaries')
def boundaries():
    return render_template('boundaries.html')

@app.route('/dogs')
def dogs():
    return render_template('dogs.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/tts')
def tts():
    return render_template('tts.html')

@app.route('/api/tts/synthesize', methods=['POST'])
def tts_synthesize():
    """Generate TTS audio from text"""
    global tts_service
    
    if not TTS_AVAILABLE:
        return jsonify({'error': 'TTS system not available'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice = data.get('voice', 'af_heart')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate TTS (service already loaded at startup)
        audio, sr, gen_time = tts_service.speak(text, voice=voice)
        
        # Save to persistent temporary file
        import uuid
        temp_filename = f"tts_audio_{uuid.uuid4().hex}_{int(time.time())}.wav"
        temp_filepath = os.path.join('.', temp_filename)
        sf.write(temp_filepath, audio, sr)
        
        # Auto-play audio through system (no GUI)
        def play_audio_async(file_path):
            try:
                if os.name == 'nt':  # Windows - use pygame for direct audio
                    try:
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load(file_path)
                        pygame.mixer.music.play()
                        # Wait for playback to finish
                        while pygame.mixer.music.get_busy():
                            import time
                            time.sleep(0.1)
                        pygame.mixer.quit()
                        
                        # Clean up temp file after playback
                        os.remove(file_path)
                        
                    except ImportError:
                        # Fallback to PowerShell if pygame not available
                        os.system(f'powershell -c "(New-Object Media.SoundPlayer \\"{file_path}\\").PlaySync()"')
                        os.remove(file_path)
                else:  # Linux/Jetson
                    os.system(f'aplay "{file_path}"')
                    os.remove(file_path)
            except Exception as e:
                # Try to clean up even on error
                try:
                    os.remove(file_path)
                except:
                    pass
                logger.warning(f"Audio playback error: {e}")
        
        try:
            from threading import Thread
            Thread(target=play_audio_async, args=(temp_filepath,), daemon=True).start()
            audio_played = True
        except Exception as play_error:
            logger.warning(f"Could not auto-play audio: {play_error}")
            audio_played = False
        
        return jsonify({
            'success': True,
            'audio_file': temp_filename,
            'generation_time': round(gen_time, 3),
            'duration': round(len(audio) / sr, 3),
            'auto_played': audio_played
        })
        
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve TTS audio files"""
    try:
        return send_file(filename, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    settings = load_settings()
    return jsonify(settings)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    try:
        new_settings = request.get_json()
        if save_settings(new_settings):
            
            # Update active camera if changed
            settings = load_settings()
            camera_urls = settings.get('camera_urls', [])
            active_camera_index = settings.get('active_camera', 0)
            
            if camera_urls and active_camera_index < len(camera_urls):
                camera_url = camera_urls[active_camera_index]
                camera_stream.set_camera_source(camera_url)
            
            return jsonify({"status": "success", "message": "Settings saved successfully"})
        else:
            return jsonify({"status": "error", "message": "Failed to save settings"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/camera/test', methods=['POST'])
def test_camera():
    """Test camera connection"""
    try:
        data = request.get_json()
        camera_url = data.get('camera_url', '')
        
        if not camera_url:
            return jsonify({"status": "error", "message": "No camera URL provided"})
        
        # Test the camera connection
        test_cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
        ret, frame = test_cap.read()
        test_cap.release()
        
        if ret:
            return jsonify({
                "status": "success", 
                "message": "Camera connected successfully",
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "Unknown"
            })
        else:
            return jsonify({"status": "error", "message": "Failed to connect to camera"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def generate_frames():
    global boundary_list
    
    while True:
        frame = camera_stream.get_frame()
        if frame is None:
            # Generate a blank frame if no camera
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Connected", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Draw boundaries on frame
            for boundary in boundary_list:
                if len(boundary) > 2:
                    pts = np.array(boundary, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_frames_multi(camera_id):
    """Generate MJPEG frames for a specific camera with animal detection and reference point detection"""
    global boundary_list, reference_points, baseline_established, frame_counter, last_movement_check
    global animal_detector, dog_identifier, alert_system, boundary_manager, last_detections, current_movement_offset
    
    print(f"Starting video stream for camera {camera_id}")
    
    while True:
        try:
            frame = None
            frame_counter += 1
            
            # Get frame directly from camera manager
            camera = camera_manager.get_camera(camera_id)
            if camera and hasattr(camera, 'latest_frame') and camera.latest_frame is not None:
                frame = camera.latest_frame.copy()
            
            if frame is None:
                # Generate error frame with clear message
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (40, 40, 40)  # Dark gray background
                
                # Add camera status text
                cv2.putText(frame, f"Camera {camera_id}", (220, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "No Signal", (250, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
                cv2.putText(frame, "Check connection", (200, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            else:
                # Process animal detection every 10th frame for performance
                if animal_detector and animal_detector.is_available() and frame_counter % 10 == 0:
                    try:
                        detections = animal_detector.detect_animals(frame)
                        # Store detections for this camera to reuse for next 9 frames
                        last_detections[camera_id] = detections
                        
                        # If we have animal detections, try to identify dogs
                        for detection in detections:
                            if detection.category == 'animal' and dog_identifier:
                                try:
                                    # Convert bbox format (x1, y1, x2, y2) to (x, y, w, h)
                                    x1, y1, x2, y2 = detection.bbox
                                    bbox_xywh = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                                    
                                    # Try to identify the dog
                                    dog_id_result = dog_identifier.identify_dog(frame, bbox_xywh, camera_id)
                                    if dog_id_result:
                                        detection.dog_id = dog_id_result.dog_id
                                        detection.dog_name = dog_id_result.dog_name
                                        detection.dog_confidence = dog_id_result.confidence
                                    else:
                                        detection.dog_id = 'unknown'
                                        detection.dog_name = 'Unknown'
                                        detection.dog_confidence = 0.0
                                except Exception as dog_id_error:
                                    print(f"Dog identification error: {dog_id_error}")
                                    detection.dog_id = 'unknown'
                                    detection.dog_name = 'Unknown'
                                    detection.dog_confidence = 0.0
                        
                        # Check for boundary violations
                        if boundary_manager and alert_system and detections:
                            try:
                                for detection in detections:
                                    if detection.category == 'animal':
                                        # Get dog position (center of bounding box)
                                        dog_position = detection.get_center()
                                        
                                        # Get dog identity (use "Unknown Dog" if not identified)
                                        dog_id = getattr(detection, 'dog_id', 'unknown')
                                        dog_name = getattr(detection, 'dog_name', 'Unknown Dog')
                                        
                                        # Debug: Show animal detection
                                        print(f" Animal detected: {dog_name} at position ({dog_position[0]:.1f}, {dog_position[1]:.1f}) on camera {camera_id}")
                                        
                                        # Check for boundary violations
                                        violations = boundary_manager.check_position_violations(
                                            dog_id, dog_name, camera_id, dog_position
                                        )
                                        
                                        # Send alerts for violations
                                        for violation in violations:
                                            if boundary_manager.should_send_alert(violation.dog_id, violation.zone_name, violation.event_type):
                                                if violation.event_type == "entry":
                                                    alert_message = f"{violation.dog_name} entered boundary {violation.zone_name}"
                                                    print(f" BOUNDARY ALERT: {alert_message}")
                                                    
                                                    # Take snapshot of boundary violation
                                                    save_boundary_snapshot(frame, violation)
                                                    
                                                    alert_system.send_alert(
                                                        AlertType.BOUNDARY_VIOLATION,
                                                        AlertSeverity.WARNING,
                                                        f"Dog Entered Zone: {violation.dog_name}",
                                                        alert_message,
                                                        {
                                                            'dog_id': violation.dog_id,
                                                            'dog_name': violation.dog_name,
                                                            'zone_name': violation.zone_name,
                                                            'camera_id': violation.camera_id,
                                                            'event_type': violation.event_type,
                                                            'position': violation.position
                                                        }
                                                    )
                                                elif violation.event_type == "exit":
                                                    alert_message = f"{violation.dog_name} left boundary {violation.zone_name}"
                                                    print(f" BOUNDARY ALERT: {alert_message}")
                                                    
                                                    # Take snapshot of boundary violation
                                                    save_boundary_snapshot(frame, violation)
                                                    
                                                    alert_system.send_alert(
                                                        AlertType.BOUNDARY_VIOLATION,
                                                        AlertSeverity.INFO,
                                                        f"Dog Left Zone: {violation.dog_name}",
                                                        alert_message,
                                                        {
                                                            'dog_id': violation.dog_id,
                                                            'dog_name': violation.dog_name,
                                                            'zone_name': violation.zone_name,
                                                            'camera_id': violation.camera_id,
                                                            'event_type': violation.event_type,
                                                            'position': violation.position
                                                        }
                                                    )
                            except Exception as boundary_error:
                                print(f"Boundary violation check error: {boundary_error}")
                    
                    except Exception as detection_error:
                        print(f"Animal detection error: {detection_error}")
                else:
                    # Reuse last detections for frames 1-9
                    detections = last_detections.get(camera_id, [])
                
                # Draw boundaries on frame using boundary_list (EVERY FRAME)
                try:
                    settings = load_settings()
                    if camera_id == settings.get('active_camera', 0) + 1:
                        # Draw boundaries from the list
                        for i, boundary in enumerate(boundary_list):
                            if len(boundary) > 2:
                                pts = np.array(boundary, np.int32)
                                pts = pts.reshape((-1, 1, 2))
                                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                        
                        # Reference points disabled - skip drawing
                        show_ref_points = False
                        if show_ref_points and reference_points:
                            try:
                                # Draw visible reference points with enhanced styling
                                for i, ref_point in enumerate(reference_points):
                                    if ref_point.get('baseline_position'):
                                        # Apply camera movement compensation
                                        pos = ref_point['baseline_position']
                                        if current_movement_offset:
                                            pos = [pos[0] - current_movement_offset[0], pos[1] - current_movement_offset[1]]
                                        
                                        # Color coding based on reference point type
                                        if ref_point.get('type') == 'manual':
                                            # Bright cyan for manual reference points
                                            color = (255, 255, 0)  # BGR format: bright cyan
                                            label = f"M{ref_point['id']}"
                                        else:
                                            # Bright magenta for auto-detected reference points
                                            color = (255, 0, 255)  # BGR format: bright magenta
                                            label = f"A{ref_point['id']}"
                                        
                                        # Draw larger, more visible circle and label
                                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 12, color, 3)  # Larger circle
                                        cv2.circle(frame, (int(pos[0]), int(pos[1])), 6, (255, 255, 255), -1)  # White center
                                        cv2.putText(frame, label, 
                                                  (int(pos[0]) + 15, int(pos[1]) - 15), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # Larger text
                                        
                                        # Add object type label
                                        type_label = ref_point.get('class', ref_point.get('type', 'ref'))[:10]  # Truncate long names
                                        cv2.putText(frame, type_label, 
                                                  (int(pos[0]) + 15, int(pos[1]) + 5), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                        
                            except Exception as boundary_error:
                                print(f" Error drawing boundaries for camera {camera_id}: {boundary_error}")
                        
                except Exception as boundary_error:
                    print(f" Error drawing boundaries for camera {camera_id}: {boundary_error}")
                
                # Draw detection results on frame
                if detections and animal_detector:
                    try:
                        frame = animal_detector.draw_detections(frame, detections, 
                                                              draw_animals=True, 
                                                              draw_people=True, 
                                                              draw_vehicles=False)
                        
                        # Add dog identification labels for animals
                        for detection in detections:
                            if detection.category == 'animal' and hasattr(detection, 'dog_id'):
                                x1, y1, x2, y2 = map(int, detection.bbox)
                                
                                # Draw dog ID label below the bounding box
                                if detection.dog_id != 'unknown':
                                    dog_label = f"{detection.dog_name} ({detection.dog_confidence:.2f})"
                                    label_color = (0, 255, 255)  # Yellow for known dogs
                                else:
                                    dog_label = "Unknown Dog"
                                    label_color = (0, 165, 255)  # Orange for unknown dogs
                                
                                # Background for dog label
                                label_size = cv2.getTextSize(dog_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                cv2.rectangle(frame, (x1, y2), (x1 + label_size[0], y2 + label_size[1] + 5), 
                                            label_color, -1)
                                
                                # Dog ID text
                                cv2.putText(frame, dog_label, (x1, y2 + label_size[1]), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    except Exception as draw_error:
                        print(f"Error drawing detections: {draw_error}")
                
                # Reference point detection disabled - focusing on dog detection
                should_check_movement = False
                
                if should_check_movement:
                    try:
                        # Detect current reference points
                        current_objects = detect_reference_objects(frame)
                        
                        if current_objects:
                            # Calculate camera movement
                            movement_offset = calculate_camera_movement(
                                reference_points, current_objects, tolerance=10
                            )
                            
                            if movement_offset:
                                # Adjust boundaries for camera movement
                                adjust_boundaries_for_movement(movement_offset)
                                
                                # Accumulate movement offset for reference point rendering
                                if current_movement_offset is None:
                                    current_movement_offset = {'dx': 0, 'dy': 0}
                                current_movement_offset['dx'] += movement_offset['dx']
                                current_movement_offset['dy'] += movement_offset['dy']
                                
                                print(f" Boundaries adjusted for camera movement")
                                print(f" Cumulative movement: dx={current_movement_offset['dx']:.1f}, dy={current_movement_offset['dy']:.1f}")
                            
                            # Reset error counter on successful detection
                            reference_detection_errors = 0
                        else:
                            # Increment error counter if no objects detected
                            reference_detection_errors += 1
                            print(f" No reference objects detected ({reference_detection_errors}/{max_detection_errors})")
                            
                            # More graceful degradation - only disable after persistent failures
                            if reference_detection_errors >= max_detection_errors:
                                print(f" Many detection failures - attempting recalibration instead of shutdown")
                                # Try auto-recalibration instead of complete shutdown
                                try:
                                    recalibration_objects = detect_reference_objects(frame)
                                    if recalibration_objects and len(recalibration_objects) >= 3:
                                        print(f" Auto-recalibration successful with {len(recalibration_objects)} objects")
                                        reference_points.clear()
                                        reference_points.extend(recalibration_objects)
                                        save_reference_points()
                                        reference_detection_errors = 0  # Reset error counter
                                    else:
                                        print(f" Auto-recalibration failed - temporarily disabling reference tracking")
                                        baseline_established = False
                                except Exception as recal_error:
                                    print(f" Recalibration error: {recal_error} - disabling reference tracking")
                                    baseline_established = False
                            
                        last_movement_check = current_time
                        
                    except Exception as ref_error:
                        reference_detection_errors += 1
                        print(f" Error in reference point processing ({reference_detection_errors}/{max_detection_errors}): {ref_error}")
                        
                        # Disable reference point detection if too many errors
                        if reference_detection_errors >= max_detection_errors:
                            print(f" Disabling reference point detection due to repeated errors")
                            baseline_established = False
                
                # Reference point detection and boundary rendering moved above to run every frame
            
            # GPU-accelerated frame encoding for reduced latency
            if GPU_VIDEO_AVAILABLE:
                try:
                    # Use GPU acceleration for frame processing and encoding
                    # Keep original resolution
                    encoded_bytes = encode_frame_fast(
                        frame, 
                        resolution=(frame.shape[1], frame.shape[0]),  # Keep original size
                        quality=85
                    )
                    
                    if encoded_bytes:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + encoded_bytes + b'\r\n')
                    else:
                        # GPU encoding failed, fallback to CPU
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        else:
                            print(f"Failed to encode frame for camera {camera_id}")
                        
                except Exception as gpu_error:
                    # GPU processing failed, fallback to CPU
                    logger.debug(f"GPU encoding failed, using CPU: {gpu_error}")
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        print(f"Failed to encode frame for camera {camera_id}")
            else:
                # CPU-only encoding (fallback)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    print(f"Failed to encode frame for camera {camera_id}")
                
        except Exception as e:
            print(f"Error in video stream for camera {camera_id}: {e}")
            # Generate error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Stream Error", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Adaptive frame rate: faster when GPU available, slower when CPU-only
        if GPU_VIDEO_AVAILABLE:
            time.sleep(0.020)  # ~50 FPS with GPU acceleration
        else:
            time.sleep(0.040)  # ~25 FPS with CPU-only (conservative)

@app.route('/start_camera')
def start_camera():
    try:
        # Load settings to get the active camera
        settings = load_settings()
        camera_urls = settings.get('camera_urls', [])
        active_camera_index = settings.get('active_camera', 0)
        
        if camera_urls and active_camera_index < len(camera_urls):
            camera_url = camera_urls[active_camera_index]
            camera_stream.set_camera_source(camera_url)
        else:
            # Fallback to local camera
            camera_stream.set_camera_source(0)
        
        if camera_stream.start():
            return jsonify({"status": "success", "message": "Camera started"})
        else:
            return jsonify({"status": "error", "message": "Failed to start camera"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/cameras/start_all', methods=['POST'])
def start_all_cameras():
    """Start all configured cameras"""
    try:
        print("=" * 60)
        print(" START ALL CAMERAS REQUEST RECEIVED")
        print("=" * 60)
        
        # Load and display settings
        settings = load_settings()
        camera_urls = settings.get('camera_urls', [])
        
        print(f" Found {len(camera_urls)} camera(s) in settings:")
        for i, url in enumerate(camera_urls):
            print(f"   Camera {i+1}: {url}")
        
        if not camera_urls:
            print(" No cameras configured in settings!")
            return jsonify({"status": "error", "message": "No cameras configured in settings"})
        
        print("\n Updating camera manager...")
        camera_manager.update_cameras_from_settings()
        
        # Show current camera manager status
        print(f" Camera manager status:")
        print(f"   Active cameras: {len(camera_manager.cameras)}")
        for cam_id, camera in camera_manager.cameras.items():
            print(f"   Camera {cam_id}: {camera.camera_source} - Running: {camera.running}")
        
        return jsonify({
            "status": "success", 
            "message": f"Starting {len(camera_urls)} cameras... Check console for progress",
            "cameras_initiated": len(camera_urls)
        })
    except Exception as e:
        print(f" ERROR starting cameras: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/cameras/stop_all', methods=['POST'])
def stop_all_cameras():
    """Stop all cameras"""
    try:
        camera_manager.stop_all()
        return jsonify({"status": "success", "message": "All cameras stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_camera')
def stop_camera():
    try:
        camera_stream.stop()
        return jsonify({"status": "success", "message": "Camera stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/save_boundary', methods=['POST'])
def save_boundary():
    global boundary_list
    try:
        print(" ===== SAVE BOUNDARY COORDINATE DEBUG =====")
        data = request.get_json()
        boundary = data.get('boundary', [])
        
        # Get current video dimensions for comparison
        frame = None
        camera = camera_manager.get_camera(1)
        if camera and hasattr(camera, 'latest_frame') and camera.latest_frame is not None:
            frame = camera.latest_frame
        
        if frame is not None:
            frame_height, frame_width = frame.shape[:2]
            print(f" Server video dimensions during save: {frame_width}x{frame_height}")
        else:
            print(f"  No frame available during save")
            frame_width, frame_height = 640, 480
        
        print(f" Received {len(boundary)} boundary points from client:")
        for i, point in enumerate(boundary):
            print(f"   Point {i+1}: pixel({point[0]}, {point[1]})")
            # Check if within current frame bounds
            if point[0] < 0 or point[0] >= frame_width or point[1] < 0 or point[1] >= frame_height:
                print(f"     Point {i+1} outside frame bounds ({frame_width}x{frame_height})")
        
        if len(boundary) >= 3:
            boundary_list.append(boundary)
            print(f" Boundary saved to server (total: {len(boundary_list)})")
            print(" ===== SAVE BOUNDARY DEBUG END =====")
            return jsonify({"status": "success", "message": "Boundary saved"})
        else:
            return jsonify({"status": "error", "message": "Need at least 3 points for boundary"})
    except Exception as e:
        print(f" Error saving boundary: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/clear_boundaries', methods=['POST'])
def clear_boundaries():
    global boundary_list
    print(f"  Clearing {len(boundary_list)} boundaries...")
    boundary_list = []
    print(" All boundaries cleared from server")
    return jsonify({"status": "success", "message": "Boundaries cleared"})

@app.route('/clear_memory_boundaries', methods=['POST'])
def clear_memory_boundaries():
    global boundary_list
    print(f" Clearing {len(boundary_list)} boundaries from memory (keeping file intact)...")
    boundary_list = []
    print(" Memory boundaries cleared, file boundaries preserved")
    return jsonify({"status": "success", "message": "Memory boundaries cleared"})

@app.route('/get_boundaries')
def get_boundaries():
    print(f" Sending {len(boundary_list)} boundaries to client")
    for i, boundary in enumerate(boundary_list):
        print(f"   Boundary {i+1}: {len(boundary)} points")
    return jsonify({"boundaries": boundary_list})

@app.route('/api/video_dimensions')
def get_video_dimensions():
    """Get the actual video frame dimensions from the camera"""
    try:
        # Get frame from camera to determine actual dimensions
        frame = None
        camera = camera_manager.get_camera(1)  # Get camera 1
        if camera and hasattr(camera, 'latest_frame') and camera.latest_frame is not None:
            frame = camera.latest_frame
        
        if frame is not None:
            height, width = frame.shape[:2]
            print(f" Actual video dimensions: {width}x{height}")
            return jsonify({
                "status": "success",
                "width": width,
                "height": height,
                "message": f"Video dimensions: {width}x{height}"
            })
        else:
            # Return default if no frame available
            print(f"  No frame available, returning default dimensions")
            return jsonify({
                "status": "success", 
                "width": 640,
                "height": 480,
                "message": "Default dimensions (no frame available)"
            })
    except Exception as e:
        print(f" Error getting video dimensions: {e}")
        return jsonify({
            "status": "error",
            "width": 640,
            "height": 480,
            "message": str(e)
        })

# Reference Point Management Endpoints
@app.route('/api/reference_points', methods=['GET'])
def get_reference_points():
    """Get current reference points"""
    global reference_points, baseline_established
    return jsonify({
        "status": "success",
        "reference_points": reference_points,
        "baseline_established": baseline_established,
        "count": len(reference_points)
    })

@app.route('/api/reference_points/calibrate', methods=['POST'])
def calibrate_reference_points():
    """Establish baseline reference points by detecting objects in current frame"""
    global reference_points, baseline_established, reference_model, reference_detection_errors
    
    try:
        # Reset error counter on new calibration attempt
        reference_detection_errors = 0
        
        if reference_model is None:
            if not load_reference_model():
                return jsonify({
                    "status": "error",
                    "message": "EfficientDet model not available - check installation"
                })
        
        # Get current frame from camera
        camera = camera_manager.get_camera(1)
        if not camera or not hasattr(camera, 'latest_frame') or camera.latest_frame is None:
            return jsonify({
                "status": "error",
                "message": "No camera frame available for calibration - check camera connection"
            })
        
        frame = camera.latest_frame.copy()
        print(f" Starting reference point calibration on frame: {frame.shape}")
        
        # Detect reference objects using EfficientDet
        detected_objects = detect_reference_objects(frame)
        
        if not detected_objects:
            return jsonify({
                "status": "warning",
                "message": "No reference objects detected - try adjusting camera view or lighting",
                "reference_points": [],
                "baseline_established": False
            })
        
        # Convert detected objects to reference points
        detected_points = []
        for obj in detected_objects:
            # Only keep high-confidence detections (lowered for Grounding DINO static objects)
            if obj['confidence'] > 0.25:
                detected_points.append({
                    'id': len(detected_points) + 1,
                    'type': obj['type'],
                    'baseline_position': obj['position'],
                    'confidence': obj['confidence'],
                    'timestamp': time.time(),
                    'class': obj['class']
                })
        
        if not detected_points:
            return jsonify({
                "status": "warning",
                "message": f"Found {len(detected_objects)} objects but none with sufficient confidence (>0.25)",
                "reference_points": [],
                "baseline_established": False
            })
        
        print(f" Converted {len(detected_points)} objects to reference points")
        
        # Log details about detected reference points
        for point in detected_points:
            print(f"   Reference Point {point['id']}: {point['type']} at ({point['baseline_position'][0]}, {point['baseline_position'][1]}) - confidence: {point['confidence']:.2f}")
        
        # Clear existing reference points and establish new baseline
        reference_points = detected_points
        baseline_established = True
        
        # Save to file
        if not save_reference_points():
            return jsonify({
                "status": "error",
                "message": "Calibration successful but failed to save to file"
            })
        
        return jsonify({
            "status": "success",
            "message": f"Calibration complete - detected {len(detected_points)} reference points",
            "reference_points": reference_points,
            "baseline_established": baseline_established,
            "detection_details": {
                "total_objects_detected": len(detected_objects),
                "high_confidence_points": len(detected_points)
            }
        })
        
    except Exception as e:
        print(f" Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Calibration failed: {str(e)}"
        })

@app.route('/api/reference_points/clear', methods=['POST'])
def clear_reference_points():
    """Clear all reference points and reset baseline"""
    global reference_points, baseline_established, reference_detection_errors
    
    try:
        print(f"  Clearing {len(reference_points)} reference points...")
        reference_points = []
        baseline_established = False
        reference_detection_errors = 0  # Reset error counter
        
        # Save cleared state to file
        if not save_reference_points():
            return jsonify({
                "status": "warning",
                "message": "Reference points cleared but failed to save to file"
            })
        
        print(" All reference points cleared")
        return jsonify({
            "status": "success",
            "message": "All reference points cleared"
        })
        
    except Exception as e:
        print(f" Error clearing reference points: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# New endpoint for reference point health check
@app.route('/api/reference_points/health', methods=['GET'])
def reference_points_health():
    """Get health status of reference point system"""
    global reference_points, baseline_established, reference_model, reference_detection_errors
    
    try:
        health_status = {
            "model_loaded": reference_model is not None,
            "grounding_dino_available": GROUNDING_DINO_AVAILABLE,
            "baseline_established": baseline_established,
            "reference_points_count": len(reference_points),
            "detection_errors": reference_detection_errors,
            "max_errors": max_detection_errors,
            "system_healthy": (
                reference_model is not None and 
                baseline_established and 
                reference_detection_errors < max_detection_errors
            ),
            "model_status": {
                "dependencies_available": GROUNDING_DINO_AVAILABLE,
                "model_instance_created": reference_model is not None,
                "model_type": "Grounding-DINO" if reference_model is not None else None,
                "last_detection_attempt": frame_counter if reference_model is not None else None,
                "static_objects_only": True,
                "supported_prompts": ["tree trunk", "building corner", "fence post", "utility pole"]
            }
        }
        
        return jsonify({
            "status": "success",
            "health": health_status
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/reference_points/add_manual', methods=['POST'])
def add_manual_reference_point():
    """Add a manual reference point from client click coordinates"""
    global reference_points, baseline_established
    
    try:
        data = request.json
        if not data or 'x' not in data or 'y' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing x,y coordinates in request"
            })
        
        # Get normalized coordinates from client (0-1 range)
        normalized_x = float(data['x'])
        normalized_y = float(data['y'])
        
        print(f" Manual reference point request: normalized({normalized_x:.3f}, {normalized_y:.3f})")
        
        # Validate normalized coordinates
        if not (0 <= normalized_x <= 1 and 0 <= normalized_y <= 1):
            return jsonify({
                "status": "error",
                "message": f"Invalid coordinates: x={normalized_x}, y={normalized_y} (must be 0-1 range)"
            })
        
        # Get current frame to determine actual video dimensions
        camera = camera_manager.get_camera(1)
        if not camera or not hasattr(camera, 'latest_frame') or camera.latest_frame is None:
            return jsonify({
                "status": "error",
                "message": "No camera frame available - check camera connection"
            })
        
        frame = camera.latest_frame
        frame_height, frame_width = frame.shape[:2]
        
        # Convert normalized coordinates to actual pixel coordinates
        pixel_x = int(normalized_x * frame_width)
        pixel_y = int(normalized_y * frame_height)
        
        print(f" Converted to pixels: ({pixel_x}, {pixel_y}) on frame {frame_width}x{frame_height}")
        
        # Create manual reference point
        manual_point = {
            'id': len(reference_points) + 1,
            'type': 'manual',
            'baseline_position': [pixel_x, pixel_y],
            'confidence': 1.0,  # Manual points have full confidence
            'timestamp': time.time(),
            'class': 'manual_reference'
        }
        
        # Add to reference points list
        reference_points.append(manual_point)
        
        print(f" Added manual reference point {manual_point['id']} at pixel coordinates ({pixel_x}, {pixel_y})")
        print(f" Total reference points after addition: {len(reference_points)}")
        print(f" Baseline established: {baseline_established}")
        print(f" Reference points list: {[f'{p['type']}_{p['id']}' for p in reference_points]}")
        
        # If this is the first reference point, establish baseline
        if not baseline_established:
            baseline_established = True
            print(" Baseline established with manual reference point")
        
        # Save reference points to file
        if not save_reference_points():
            return jsonify({
                "status": "warning",
                "message": "Manual point added but failed to save to file"
            })
        
        return jsonify({
            "status": "success",
            "message": f"Manual reference point added at ({pixel_x}, {pixel_y})",
            "reference_point": manual_point,
            "total_points": len(reference_points),
            "baseline_established": baseline_established
        })
        
    except Exception as e:
        print(f" Error adding manual reference point: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Failed to add manual reference point: {str(e)}"
        })

@app.route('/save_config', methods=['POST'])
def save_config():
    global boundary_list
    try:
        config = {
            "boundaries": boundary_list,
            "timestamp": time.time()
        }
        with open('boundary_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({"status": "success", "message": "Configuration saved"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/load_config', methods=['POST'])
def load_config():
    global boundary_list
    try:
        with open('boundary_config.json', 'r') as f:
            config = json.load(f)
        boundary_list = config.get('boundaries', [])
        
        return jsonify({"status": "success", "message": "Configuration loaded", "boundaries": boundary_list})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Camera management endpoints
@app.route('/api/cameras')
def get_cameras():
    """Get list of configured cameras"""
    settings = load_settings()
    camera_urls = settings.get('camera_urls', [])
    cameras = []
    
    # Get statuses from multi-camera manager
    camera_statuses = camera_manager.get_all_statuses()
    
    for i, url in enumerate(camera_urls):
        camera_id = i + 1
        status_info = camera_statuses.get(camera_id, {})
        
        cameras.append({
            'id': camera_id,
            'name': f'Camera {camera_id}',
            'location': f'Location {camera_id}',
            'url': url,
            'status': 'online' if status_info.get('running', False) else 'offline',
            'error': status_info.get('error')
        })
    
    return jsonify({'cameras': cameras})

@app.route('/video_feed')
def video_feed():
    """Video feed with camera parameter support"""
    camera_id = int(request.args.get('camera', '1'))
    print(f" Video feed requested for camera {camera_id}")
    return Response(generate_frames_multi(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# Dog management endpoints
@app.route('/api/dogs')
def get_dogs():
    """Get list of enrolled dogs"""
    if TRAINING_MANAGER_AVAILABLE:
        try:
            training_manager = get_training_manager()
            dogs = training_manager.get_enrolled_dogs()
            return jsonify({'dogs': dogs})
        except Exception as e:
            print(f"Error loading dogs from training manager: {e}")
    
    # Fallback to file-based approach
    dogs_file = 'dogs.json'
    try:
        if os.path.exists(dogs_file):
            with open(dogs_file, 'r') as f:
                dogs_data = json.load(f)
                return jsonify({'dogs': dogs_data})
    except Exception as e:
        print(f"Error loading dogs: {e}")
    
    return jsonify({'dogs': []})

@app.route('/api/dogs/train', methods=['POST'])
def train_dog():
    """Start training a new dog"""
    if not TRAINING_MANAGER_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'Training manager not available'
        }), 500
    
    try:
        name = request.form.get('name')
        breed = request.form.get('breed', '')
        age = request.form.get('age', '')
        description = request.form.get('description', '')
        
        # Handle image uploads
        uploaded_files = request.files.getlist('images')
        
        # Save uploaded images temporarily and get file paths
        images = []
        temp_dir = f"temp_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file and uploaded_file.filename:
                    temp_path = os.path.join(temp_dir, f'image_{i}.jpg')
                    uploaded_file.save(temp_path)
                    images.append(temp_path)
            
            # Use training manager to enroll dog
            training_manager = get_training_manager()
            enrollment_result = training_manager.enroll_dog(
                name=name,
                breed=breed,
                age=age,
                description=description,
                images=images
            )
            
            if not enrollment_result['success']:
                return jsonify({
                    'status': 'error',
                    'message': enrollment_result['error']
                }), 400
            
            dog_id = enrollment_result['dog_id']
            
            # Start training
            training_result = training_manager.start_training([dog_id])
            
            if not training_result['success']:
                return jsonify({
                    'status': 'error',
                    'message': training_result['error']
                }), 400
            
            return jsonify({
                'status': 'success',
                'trainingId': dog_id,
                'message': f'Training started for {name} with {len(images)} images',
                'total_dogs': training_result.get('total_dogs', 1),
                'total_images': training_result.get('total_images', len(images))
            })
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/training/status')
def get_training_status():
    """Get current training status"""
    if not TRAINING_MANAGER_AVAILABLE:
        return jsonify({
            'active': False,
            'message': 'Training manager not available'
        })
    
    try:
        training_manager = get_training_manager()
        status = training_manager.get_training_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'active': False,
            'error': str(e)
        }), 500

@app.route('/api/training/cancel', methods=['POST'])
def cancel_training():
    """Cancel current training"""
    if not TRAINING_MANAGER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Training manager not available'
        }), 500
    
    try:
        training_manager = get_training_manager()
        result = training_manager.cancel_training()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dogs/enroll', methods=['POST'])
def enroll_dog():
    """Enroll a new dog without starting training"""
    if not TRAINING_MANAGER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Training manager not available'
        }), 500
    
    try:
        name = request.form.get('name')
        breed = request.form.get('breed', '')
        age = request.form.get('age', '')
        description = request.form.get('description', '')
        
        # Handle image uploads
        uploaded_files = request.files.getlist('images')
        
        # Save uploaded images temporarily and get file paths
        images = []
        temp_dir = f"temp_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                if uploaded_file and uploaded_file.filename:
                    temp_path = os.path.join(temp_dir, f'image_{i}.jpg')
                    uploaded_file.save(temp_path)
                    images.append(temp_path)
            
            # Use training manager to enroll dog
            training_manager = get_training_manager()
            result = training_manager.enroll_dog(
                name=name,
                breed=breed,
                age=age,
                description=description,
                images=images
            )
            
            return jsonify(result)
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dogs/<dog_id>/config', methods=['GET', 'PUT'])
def dog_config(dog_id):
    """Get or update dog configuration"""
    config_file = f'dogs/{dog_id}/config.json'
    
    if request.method == 'GET':
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return jsonify(config)
            else:
                # Return default config
                return jsonify({
                    'alerts': {
                        'email': True,
                        'sound': True,
                        'cooldown': 30
                    },
                    'boundaries': {}
                })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    else:  # PUT
        try:
            config = request.get_json()
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# System status endpoints
@app.route('/api/system/status')
def get_system_status():
    """Get system status including camera counts"""
    settings = load_settings()
    camera_urls = settings.get('camera_urls', [])
    camera_statuses = camera_manager.get_all_statuses()
    
    online_count = sum(1 for status in camera_statuses.values() if status.get('running', False))
    
    # Calculate uptime
    uptime = int(time.time() - app.config.get('start_time', time.time()))
    
    # Get GPU video stats if available
    gpu_stats = {}
    if GPU_VIDEO_AVAILABLE:
        try:
            gpu_processor = get_gpu_processor()
            gpu_stats = gpu_processor.get_stats()
        except Exception as e:
            gpu_stats = {'error': str(e)}
    
    return jsonify({
        'activeDogs': [],  # Would be populated by detection system
        'camerasOnline': online_count,
        'camerasTotal': len(camera_urls),
        'detectionFPS': 0.0,  # Would be calculated from detection engine
        'gpuUsage': gpu_stats.get('gpu_usage_percent', 0),  # From GPU video processor
        'memoryUsage': 0,  # Would be from system monitoring
        'uptime': uptime,
        'gpuVideo': gpu_stats  # Detailed GPU video stats
    })

@app.route('/api/gpu/stats')
def get_gpu_video_stats():
    """Get detailed GPU video acceleration statistics"""
    if not GPU_VIDEO_AVAILABLE:
        return jsonify({
            'available': False,
            'reason': 'GPU video acceleration not installed'
        })
    
    try:
        gpu_processor = get_gpu_processor()
        stats = gpu_processor.get_stats()
        
        return jsonify({
            'available': True,
            'stats': stats,
            'performance': {
                'gpu_speedup': round(stats.get('avg_cpu_time', 0) / max(stats.get('avg_gpu_time', 0.001), 0.001), 2),
                'frames_processed': stats.get('total_frames_processed', 0),
                'gpu_efficiency': f"{stats.get('gpu_usage_percent', 0):.1f}%"
            }
        })
        
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        })

# Alert endpoints
@app.route('/api/alerts/recent')
def get_recent_alerts():
    """Get recent alerts"""
    # In a real system, this would query a database
    return jsonify({'alerts': []})

@app.route('/api/alerts/new')
def get_new_alerts():
    """Get new alerts since last check"""
    return jsonify({'alerts': []})


# Detection endpoint (placeholder)
@app.route('/api/detections')
def get_detections():
    """Get current detections for a camera"""
    camera_id = request.args.get('camera', '1')
    # In a real system, this would return actual detections
    return jsonify({'detections': []})

# Capture endpoint
@app.route('/api/capture')
def capture_frame():
    """Capture current frame from camera"""
    camera_id = request.args.get('camera', '1')
    
    frame = camera_stream.get_frame()
    if frame is not None:
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    
    return jsonify({'error': 'No frame available'}), 404

# Initialize app start time
app.config['start_time'] = time.time()

# Initialize reference point system on startup
def initialize_reference_system():
    """Initialize reference point system"""
    print(" ===== REFERENCE POINT SYSTEM STARTUP DEBUG =====")
    print(" Initializing reference point system...")
    
    # Debug dependency status
    print(f" Checking Grounding DINO dependencies...")
    print(f"   - GROUNDING_DINO_AVAILABLE: {GROUNDING_DINO_AVAILABLE}")
    
    if not GROUNDING_DINO_AVAILABLE:
        print(" Grounding DINO dependencies missing:")
        print("   - Required: transformers, torch, PIL")
        print("   - Install command: pip install transformers torch pillow")
    else:
        print(" Grounding DINO dependencies are available")
    
    # Load existing reference points if available
    print(" Loading existing reference points from file...")
    load_reference_points()
    print(f" Loaded {len(reference_points)} existing reference points")
    
    # Attempt to load Grounding DINO model
    print(" Attempting to load Grounding DINO model...")
    if GROUNDING_DINO_AVAILABLE:
        model_loaded = load_reference_model()
        if model_loaded:
            print(" Grounding DINO model loaded successfully - static object detection available")
        else:
            print(" Grounding DINO model failed to load - only manual reference points available")
    else:
        print("  Grounding DINO not available - reference point detection disabled")
        print("    Manual reference point placement will still work")
    
    print(" Reference point system initialization complete")
    print(" ===== STARTUP DEBUG COMPLETE =====")

def initialize_surveillance_systems():
    """Initialize animal detection, boundary system, and dog identification"""
    global animal_detector, dog_identifier, alert_system, boundary_manager
    
    print(" ===== SURVEILLANCE SYSTEMS STARTUP =====")
    print(" Initializing surveillance systems...")
    
    if not SURVEILLANCE_SYSTEMS_AVAILABLE:
        print(" Surveillance systems not available - modules not imported properly")
        return
    
    try:
        # Initialize animal detector (MegaDetector)
        print(" Loading animal detector (MegaDetector)...")
        animal_detector = get_animal_detector()
        if animal_detector and animal_detector.is_available():
            print(" MegaDetector loaded successfully")
            stats = animal_detector.get_stats()
            print(f"   - Model: {stats.get('model_path', 'unknown')}")
            print(f"   - Device: {stats.get('device', 'unknown')}")
        else:
            print(" Failed to load MegaDetector - animal detection disabled")
        
        
        # Initialize dog identifier  
        print("  Loading dog identifier (MiewID)...")
        dog_identifier = get_dog_identifier()
        if dog_identifier:
            stats = dog_identifier.get_stats()
            print(" Dog identifier loaded")
            print(f"   - Total dogs enrolled: {stats.get('total_dogs', 0)}")
            print(f"   - Active dogs: {stats.get('active_dogs', 0)}")
            if stats.get('model_loaded'):
                print("   - MiewID model: loaded")
            else:
                print("   - MiewID model: not loaded (will use placeholder)")
        else:
            print(" Failed to load dog identifier")
        
        # Initialize alert system
        print(" Loading alert system...")
        alert_system = get_alert_system()
        if alert_system:
            stats = alert_system.get_stats()
            print(" Alert system loaded")
            print(f"   - Alerts generated: {stats.get('total_alerts_generated', 0)}")
            print(f"   - Alerts sent: {stats.get('alerts_sent', 0)}")
        else:
            print(" Failed to load alert system")
        
        # Initialize boundary manager
        print(" Loading boundary manager...")
        boundary_manager = get_boundary_manager()
        if boundary_manager:
            stats = boundary_manager.get_stats()
            print(" Boundary manager loaded")
            print(f"   - Total zones: {stats.get('total_zones', 0)}")
            print(f"   - Dogs tracked: {stats.get('total_dogs_tracked', 0)}")
        else:
            print(" Failed to load boundary manager")
        
        # Initialize TTS system
        if TTS_AVAILABLE:
            print(" Loading TTS system...")
            global tts_service
            try:
                tts_service = TTSService()
                print(" TTS system loaded")
                print("   - Model: Kokoro-82M ready for voice alerts")
            except Exception as tts_error:
                print(f" Failed to load TTS system: {tts_error}")
                tts_service = None
        else:
            print("  TTS system not available")
            
    except Exception as e:
        print(f" Error initializing surveillance systems: {e}")
    
    print(" Surveillance systems initialization complete")
    print(" ===== SURVEILLANCE STARTUP COMPLETE =====")


if __name__ == '__main__':
    print("Starting Dog Tracking System...")
    print("Open your browser to http://localhost:5000")
    
    # Skip reference point system for now - focusing on dog detection
    print(" Reference point system disabled - focusing on dog detection")
    
    # Initialize surveillance systems
    initialize_surveillance_systems()
    
    app.run(debug=True, host='0.0.0.0', port=5000)