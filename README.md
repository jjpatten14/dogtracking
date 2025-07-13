# Dog Tracking System

A real-time surveillance system for monitoring dogs using computer vision, boundary detection, and text-to-speech alerts. Built with Flask, OpenCV, and MegaDetector for accurate animal detection.

## Features

### Advanced AI Detection
- **Multi-model animal detection** using PytorchWildlife (MegaDetector v5)
- **Individual dog identification** with MiewID re-identification system
- **Cross-camera tracking** for multi-camera deployments
- **Reference point detection** using EfficientDet for camera movement compensation
- **GPU-optimized model orchestration** for 10-12 concurrent camera feeds

### Professional Surveillance
- **Multi-camera support** for enterprise-grade monitoring (10-12 IP cameras)
- **Real-time boundary violation detection** with point-in-polygon algorithms
- **Automatic camera movement compensation** via reference point tracking
- **Advanced zone management** with entry/exit tracking
- **Live video streaming** with GPU acceleration support
- **Automatic snapshot capture** with intelligent timestamping
- **Cross-platform compatibility** (Windows/Linux/WSL/Jetson)

### Intelligent Alert System
- **Multi-level alert configurations** with customizable triggers
- **Automatic photo capture** with timestamped filenames and zone information
- **Text-to-Speech announcements** using offline Kokoro TTS
- **Web-based monitoring interface** with real-time status
- **MCP memory integration** for persistent alert history

### Advanced Text-to-Speech
- **Offline TTS** using Kokoro-82M model (327MB)
- **Multiple voice options** (af_heart, af_lollipop, af_joy, af_sarah, af_nicole)
- **System audio playback** without browser dependencies
- **GPU acceleration** for sub-second synthesis
- **Custom alert messages** with dynamic content

### Training and Model Management
- **Automated MiewID training pipeline** for custom dog identification
- **Model deployment automation** with version management
- **Training data preprocessing** with annotation tools
- **Performance monitoring** and model optimization

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Webcam or IP camera
- Windows 10/11 or Linux

### Dependencies

Install all dependencies using requirements.txt:
```bash
pip install -r requirements.txt
```

**Core ML/AI Framework:**
```bash
# PyTorch ecosystem (CUDA support recommended)
torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0
cupy>=12.0.0  # CUDA-accelerated numpy

# Computer Vision and Detection
opencv-python>=4.8.1 opencv-contrib-python>=4.8.0
pytorchwildlife>=1.1.0  # MegaDetector integration
ultralytics>=8.0.0  # YOLO detection framework
effdet>=0.3.0  # EfficientDet implementation
sahi>=0.11.0  # Slicing Aided Hyper Inference
albumentations>=1.3.0  # Image augmentations
```

**Individual Dog Identification:**
```bash
insightface>=0.7.3  # Face recognition and ArcFace
face-recognition>=1.3.0  # Face detection utilities
onnxruntime>=1.15.0  # ONNX model runtime
```

**Text-to-Speech System:**
```bash
kokoro-tts>=1.0.0  # Offline TTS with Kokoro
soundfile>=0.12.0  # Audio file I/O
pygame>=2.5.0  # Audio playback
espeakng-loader>=0.1.0  # eSpeak NG for phonemization
phonemize>=3.2.0  # Text-to-phoneme conversion
num2words>=0.5.12  # Number-to-words conversion
```

**Web Framework and APIs:**
```bash
flask>=2.3.3 flask-socketio>=5.3.4 flask-cors>=4.0.0
fastapi>=0.100.0  # Modern API framework
uvicorn>=0.22.0  # ASGI server
gradio>=3.35.0  # Web interface framework
eventlet>=0.33.3  # Async networking
```

**Data Processing and ML Utilities:**
```bash
numpy>=1.24.3 pandas>=2.0.0 scipy>=1.10.0
scikit-learn>=1.3.0 scikit-image>=0.21.0
pillow>=10.0.0 matplotlib>=3.7.0
tqdm>=4.65.0 joblib>=1.3.0
```

**Experiment Tracking and Monitoring:**
```bash
wandb>=0.15.0  # Experiment tracking and logging
psutil>=5.9.0  # System monitoring
memory-profiler>=0.61.0  # Memory usage tracking
loguru>=0.7.0  # Advanced logging
```

**Development and Annotation Tools:**
```bash
roboflow>=1.1.0  # Dataset management and annotation
transformers>=4.30.0  # Hugging Face transformers
spacy>=3.6.0  # NLP processing
pytest>=7.4.0  # Testing framework
ruff>=0.0.275  # Python linter and formatter
```

**Video Processing:**
```bash
av>=10.0.0  # PyAV for video processing
imageio>=2.31.0 imageio-ffmpeg>=0.4.8  # Video I/O
```

### Git LFS Setup (Required for Large Models)
```bash
# Install Git LFS
git lfs install

# Clone repository with LFS support
git clone https://github.com/jjpatten14/dogtracking.git
cd dogtracking

# Verify large model files downloaded
ls -la models/  # Should show 600MB+ of model files
```

### Model Setup
Large AI models are automatically downloaded via Git LFS:
- **MegaDetector v5** (280MB): `models/md_v5a.0.0.pt`
- **Kokoro TTS** (327MB): `models/kokoro-v1_0.pth`  
- **EfficientDet-D0** (52MB): `models/hub/checkpoints/efficientdet_d0-f3276ba8.pth`
- **MiewID Plugin**: `models/wbia-plugin-miew-id/`

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/jjpatten14/dogtracking.git
cd dogtracking
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure boundaries**
Edit `boundary_config.json` to define monitoring zones

4. **Run the application**
```bash
python app.py
```

5. **Access web interface**
Open http://localhost:5000 in your browser

## Configuration

## Advanced Configuration

### Multi-Camera Professional Setup
Configure `configs/camera_settings.json` for enterprise deployment:
```json
{
  "cameras": [
    {
      "id": "CAMERA1",
      "name": "Front Gate",
      "rtsp_url": "rtsp://admin:password@192.168.1.100/stream1",
      "resolution": [1920, 1080],
      "fps": 30,
      "enabled": true
    },
    {
      "id": "CAMERA2", 
      "name": "Back Yard",
      "rtsp_url": "rtsp://admin:password@192.168.1.101/stream1",
      "resolution": [1920, 1080],
      "fps": 30,
      "enabled": true
    }
  ],
  "processing": {
    "gpu_allocation": {
      "megadetector": 0.6,
      "miewid": 0.2,
      "efficientdet": 0.1,
      "tts": 0.1
    }
  }
}
```

### Advanced Boundary Configuration
Enhanced `configs/boundary_config.json` with zone management:
```json
{
  "zones": [
    {
      "name": "Zone_1",
      "polygon": [[100, 100], [500, 100], [500, 400], [100, 400]],
      "camera_id": "CAMERA1",
      "alert_type": "entry_exit",
      "reference_points": [[50, 50], [550, 50]],
      "movement_tolerance": 10
    }
  ],
  "global_settings": {
    "detection_confidence": 0.7,
    "tracking_enabled": true,
    "cross_camera_tracking": true
  }
}
```

### Alert System Configuration
Configure `configs/alert_config.json` for professional alerts:
```json
{
  "alert_levels": {
    "low": {
      "log_only": true,
      "tts_enabled": false
    },
    "medium": {
      "tts_enabled": true,
      "snapshot_capture": true,
      "voice": "af_sarah"
    },
    "high": {
      "tts_enabled": true,
      "snapshot_capture": true,
      "voice": "af_nicole",
      "immediate_notification": true
    }
  }
}
```

### MiewID Training Configuration
Set up `models/wbia-plugin-miew-id/configs/yard_dogs_config.yaml`:
```yaml
dataset:
  name: "yard_dogs"
  num_classes: 5  # Number of individual dogs
  image_size: 224
  
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  
deployment:
  confidence_threshold: 0.8
  auto_deploy: true
```

## Usage

### Web Interface
- **Live Monitor**: Real-time video feed with boundary overlays
- **Boundaries**: Configure and manage detection zones
- **Dogs**: Manage known dog profiles
- **Text-to-Speech**: Test TTS functionality
- **Settings**: System configuration
- **History**: View past alerts and snapshots

### Boundary Detection
The system automatically:
1. Detects animals in video feed
2. Checks if detected animals are in defined boundaries
3. Triggers alerts for boundary violations
4. Captures snapshots with timestamps
5. Logs events to console

### Snapshots
Photos are automatically saved to:
```
snapshots/
├── 2024-01-15/
│   ├── boundary_alert_CAMERA0_ENTRY_backyard_20240115_143022.jpg
│   └── boundary_alert_CAMERA0_EXIT_backyard_20240115_143045.jpg
```

### Text-to-Speech
- Access TTS interface at `/tts`
- Enter text and select voice
- Audio plays automatically through system speakers
- Works offline with local Kokoro model

## Advanced Architecture

### Multi-Model AI Pipeline
- **PytorchWildlife (MegaDetector)**: Primary animal detection (85% GPU allocation)
- **MiewID**: Individual dog re-identification and cross-camera tracking
- **EfficientDet-D0**: Reference point detection for camera movement compensation
- **InsightFace**: Facial recognition for enhanced dog identification
- **GPU Orchestration**: Intelligent model switching and resource allocation

### Core System Components
- **app.py**: Main Flask application and multi-camera web server
- **animal_detector.py**: Multi-model detection pipeline integration
- **boundary_manager.py**: Advanced boundary logic with movement compensation
- **alert_system.py**: Multi-level alert handling and notifications
- **dog_identifier.py**: MiewID-powered individual dog recognition
- **training_manager.py**: Automated MiewID training and deployment pipeline
- **reference_detector.py**: EfficientDet-based camera movement detection

### Professional Surveillance Features
- **camera_manager.py**: Multi-camera RTSP/HTTP stream management
- **violation_detector.py**: Advanced boundary violation detection
- **tracking_system.py**: Cross-camera animal tracking coordination

### Advanced TTS System
- **tts/kokoro_tts.py**: Kokoro TTS engine with GPU acceleration
- **tts/tts_service.py**: Multi-voice TTS service management
- **templates/tts.html**: Professional TTS interface

### Memory and Automation
- **MCP Memory Integration**: Persistent storage for alerts, tracking, and training data
- **scripts/**: Automation hooks for memory management and scope enforcement
- **configs/**: Professional configuration management system

### Enterprise Features
- **Cross-platform paths**: Automatic Windows/WSL/Linux/Jetson path resolution
- **GPU optimization**: CUDA support with intelligent resource allocation
- **Real-time processing**: Sub-100ms detection latency across 10+ cameras
- **Persistent storage**: Automated snapshot organization with metadata
- **Scalable deployment**: Jetson Orin Nano optimized for edge computing

## Troubleshooting

### Common Issues

**Boundary detection not working:**
- Check boundary_config.json path resolution
- Verify polygon coordinates are correct
- Enable debug logging in boundary_manager.py

**Camera not found:**
- Check camera ID in configuration
- Test camera access with other applications
- Try different camera indices (0, 1, 2...)

**TTS not working:**
- Verify Kokoro model is downloaded (327MB): `ls -la models/kokoro-v1_0.pth`
- Check audio system configuration and speakers
- Test with different voices via web interface
- Ensure GPU memory allocation for TTS (10% minimum)

**MiewID training issues:**
- Check training data in `dogs/annotations/` directory
- Verify YAML configuration in `configs/yard_dogs_config.yaml`
- Monitor training logs: `python training_manager.py --status`
- Ensure sufficient GPU memory for training

**Multi-camera connectivity:**
- Test RTSP streams individually: `ffplay rtsp://camera-ip/stream1`
- Verify network bandwidth (1Gbps+ recommended for 10+ cameras)
- Check camera authentication credentials
- Ensure proper camera configuration in `configs/camera_settings.json`

**Reference point detection not working:**
- Verify EfficientDet model: `ls -la models/hub/checkpoints/`
- Check reference points configuration in boundary settings
- Ensure stable objects are visible (poles, building corners)
- Monitor detection logs for reference point tracking

**Cross-camera tracking issues:**
- Verify MiewID model is trained for your specific dogs
- Check tracking configuration in boundary settings
- Ensure sufficient GPU memory allocation (20% for MiewID)
- Monitor tracking logs for individual dog identification

**Memory and automation:**
- Check MCP memory service: `cd mcp-memory && npm start`
- Verify scripts permissions: `ls -la scripts/`
- Monitor hook logs: `tail -f scripts/hook.log`
- Test scope enforcement: `python scripts/test_scope_enforcer.py`

### Debug Mode
Enable detailed logging by setting log level to DEBUG in app.py

## Development

### Advanced Project Structure
```
dogtracking/
├── app.py                          # Main Flask multi-camera application
├── animal_detector.py              # Multi-model detection pipeline
├── boundary_manager.py             # Advanced boundary logic with movement compensation
├── alert_system.py                 # Multi-level alert handling
├── dog_identifier.py               # MiewID-powered dog recognition
├── training_manager.py             # Automated MiewID training pipeline
├── reference_detector.py           # EfficientDet camera movement detection
├── camera_manager.py               # Multi-camera RTSP management
├── violation_detector.py           # Advanced boundary violation detection
├── tracking_system.py              # Cross-camera tracking coordination
├── project_paths.py                # Cross-platform path management
├── configs/                        # Professional configuration system
│   ├── camera_settings.json        # Multi-camera configuration
│   ├── boundary_config.json        # Advanced zone management
│   ├── alert_config.json           # Alert level configuration
│   └── reference_points.json       # Camera movement reference points
├── tts/                            # Advanced text-to-speech system
│   ├── kokoro_tts.py               # GPU-accelerated Kokoro TTS
│   └── tts_service.py              # Multi-voice service management
├── scripts/                        # Automation and memory management
│   ├── auto_approve.py             # Automated approval workflows
│   ├── scope_enforcer.py           # Scope enforcement automation
│   ├── pre_compact_memory_update.py # Memory hook for session preservation
│   └── README.md                   # Scripts documentation
├── templates/                      # Professional web interface
│   ├── index.html                  # Multi-camera dashboard
│   ├── monitor.html                # Real-time monitoring
│   ├── boundaries.html             # Advanced boundary management
│   ├── dogs.html                   # Dog profile management
│   ├── tts.html                    # Professional TTS interface
│   └── settings.html               # System configuration
├── static/                         # Enhanced web assets
│   ├── css/style.css               # Professional styling
│   ├── js/monitor.js               # Real-time monitoring
│   ├── js/boundary-drawing.js      # Interactive boundary tools
│   └── js/dogs.js                  # Dog management interface
├── models/                         # Large AI models (Git LFS)
│   ├── md_v5a.0.0.pt              # MegaDetector v5 (280MB)
│   ├── kokoro-v1_0.pth            # Kokoro TTS (327MB)
│   ├── hub/checkpoints/            # EfficientDet models
│   └── wbia-plugin-miew-id/        # MiewID plugin and training system
├── dogs/                           # Dog profile and training data
│   ├── profiles/                   # Individual dog profiles
│   ├── annotations/                # Training annotations
│   └── preprocessed/               # Processed training images
├── snapshots/                      # Organized photo capture
│   └── YYYY-MM-DD/                # Date-organized snapshots
├── mcp-memory/                     # MCP memory integration
│   ├── src/index.ts                # Memory service implementation
│   └── package.json                # Node.js dependencies
└── requirements.txt                # Python dependencies
```

## Training and Model Management

### MiewID Custom Dog Training
The system includes automated training for custom dog identification:

```bash
# Start training with your dog images
python training_manager.py --dataset dogs/annotations --config configs/yard_dogs_config.yaml

# Monitor training progress
python training_manager.py --status

# Deploy trained model
python training_manager.py --deploy --model-path models/trained/yard_dogs_v1.pth
```

### Training Data Preparation
1. **Collect Images**: Place dog photos in `dogs/annotations/`
2. **Automatic Preprocessing**: System creates training splits and augmentations
3. **Quality Check**: Validates image quality and annotations
4. **Training Pipeline**: Automated MiewID training with progress monitoring

### Model Performance Monitoring
- **Real-time Metrics**: Track detection accuracy and training progress
- **Validation Testing**: Automated model validation on test datasets
- **Performance Benchmarks**: Compare model versions and accuracy
- **Deployment Automation**: Auto-deploy best performing models

### Adding New Features
- **Extend alert_system.py**: Add new notification types and integrations
- **Modify boundary_manager.py**: Implement different detection algorithms
- **Enhance training_manager.py**: Add new model architectures and training strategies
- **Add TTS voices**: Expand voice options in kokoro_tts.py
- **Multi-camera tracking**: Extend tracking_system.py for new camera configurations

## Hardware Requirements

### Single Camera (Basic Setup)
- **CPU**: Intel i5 or AMD Ryzen 5
- **RAM**: 8GB
- **GPU**: NVIDIA GTX 1060 or better (for CUDA acceleration)
- **Storage**: 10GB free space
- **Camera**: USB webcam 720p

### Multi-Camera Professional (Recommended)
- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores)
- **RAM**: 32GB (for 10+ camera feeds)
- **GPU**: NVIDIA RTX 3070 or better (12GB+ VRAM)
- **Storage**: 1TB+ NVMe SSD (for snapshots and models)
- **Cameras**: 10-12 IP cameras with RTSP (1080p minimum)
- **Network**: Gigabit Ethernet switch for camera connectivity

### Jetson Orin Nano (Edge Deployment)
- **Platform**: NVIDIA Jetson Orin Nano (8GB)
- **JetPack**: 5.x with CUDA 11.8+
- **Storage**: 128GB+ NVMe SSD (high endurance)
- **Power**: 12V DC outdoor-rated power supply
- **Enclosure**: Weatherproof IP65+ rated enclosure
- **Cameras**: 6-8 IP cameras (optimized for edge processing)
- **Network**: Managed PoE switch for camera power and data

### GPU Memory Allocation (Multi-Camera)
- **MegaDetector**: 60% (primary detection)
- **MiewID**: 20% (dog identification)
- **EfficientDet**: 10% (reference points)
- **Kokoro TTS**: 10% (voice synthesis)

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check troubleshooting section
- Review GitHub issues
- Create new issue with details

## Deployment

### Production Setup
- Use proper WSGI server (gunicorn, uwsgi)
- Configure reverse proxy (nginx)
- Set up SSL certificates
- Monitor disk usage for snapshots

### Jetson Nano
This system is designed to run on NVIDIA Jetson Orin Nano:
- Use JetPack SDK for optimal performance
- Enable GPU acceleration
- Configure power mode for sustained performance