# Dog Tracking System

A real-time surveillance system for monitoring dogs using computer vision, boundary detection, and text-to-speech alerts. Built with Flask, OpenCV, and MegaDetector for accurate animal detection.

## Features

### Core Surveillance
- **Real-time dog detection** using MegaDetector v5
- **Boundary violation detection** with point-in-polygon algorithms
- **Live video streaming** with GPU acceleration support
- **Automatic snapshot capture** on boundary alerts
- **Cross-platform compatibility** (Windows/Linux/WSL)

### Alert System
- **Console alerts** for boundary violations
- **Automatic photo capture** with timestamped filenames
- **Text-to-Speech announcements** using Kokoro TTS
- **Web-based monitoring interface**

### Text-to-Speech
- **Offline TTS** using Kokoro-82M model
- **Multiple voice options** (af_heart, af_lollipop, af_joy, af_sarah, af_nicole)
- **System audio playback** without browser dependencies
- **GPU acceleration** for fast synthesis

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- Webcam or IP camera
- Windows 10/11 or Linux

### Dependencies
```bash
pip install flask opencv-python torch torchvision
pip install numpy soundfile pygame
pip install kokoro-tts  # For TTS functionality
```

### MegaDetector Setup
1. Download MegaDetector v5 model
2. Place in `models/` directory
3. Update path in configuration

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

### Boundary Setup
Create `boundary_config.json` with your monitoring zones:
```json
{
  "zones": [
    {
      "name": "backyard",
      "polygon": [[100, 100], [500, 100], [500, 400], [100, 400]],
      "camera_id": 0
    }
  ]
}
```

### Camera Configuration
- Default uses camera ID 0 (first webcam)
- Modify camera settings in `app.py`
- Supports USB webcams and IP cameras

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

## Architecture

### Core Components
- **app.py**: Main Flask application and web server
- **animal_detector.py**: MegaDetector integration
- **boundary_manager.py**: Boundary detection logic
- **alert_system.py**: Alert handling and notifications
- **dog_identifier.py**: Known dog recognition

### TTS System
- **tts/kokoro_tts.py**: Kokoro TTS engine wrapper
- **tts/tts_service.py**: TTS service management
- **templates/tts.html**: Web interface for TTS

### Key Features
- **Cross-platform paths**: Automatic Windows/WSL/Linux path resolution
- **GPU acceleration**: CUDA support for video processing and TTS
- **Real-time processing**: Low-latency animal detection
- **Persistent storage**: Automatic snapshot organization

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
- Verify Kokoro model is downloaded
- Check audio system configuration
- Test with different voices

### Debug Mode
Enable detailed logging by setting log level to DEBUG in app.py

## Development

### Project Structure
```
dogtracking/
├── app.py                 # Main application
├── animal_detector.py     # Animal detection
├── boundary_manager.py    # Boundary logic
├── alert_system.py        # Alert handling
├── dog_identifier.py      # Dog recognition
├── tts/                   # Text-to-speech system
│   ├── kokoro_tts.py
│   └── tts_service.py
├── templates/             # Web interface
├── static/               # CSS and JS files
├── models/               # AI models
├── snapshots/            # Captured photos
└── boundary_config.json  # Zone configuration
```

### Adding Features
- Extend alert_system.py for new notification types
- Modify boundary_manager.py for different detection algorithms
- Add new TTS voices in kokoro_tts.py

## Hardware Requirements

### Minimum
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 10GB free space
- Camera: USB webcam 720p

### Recommended
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA GTX 1060 or better (for CUDA acceleration)
- Storage: 50GB free space (for snapshots)
- Camera: 1080p webcam or IP camera

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