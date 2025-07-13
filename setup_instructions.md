# Complete Setup Instructions - Multi-Camera Dog Tracking System

## Overview
This guide provides step-by-step instructions for setting up the professional multi-camera dog tracking system on Jetson Orin Nano.

## Prerequisites

### Hardware Requirements
- **Jetson Orin Nano** with JetPack 5.x installed
- **10-12 IP cameras** with RTSP/HTTP streams
- **Network switch** for camera connectivity
- **128GB+ NVMe SSD** for storage
- **Weatherproof enclosure** for outdoor deployment
- **12V DC power supply** (outdoor rated)

### Software Requirements
- **Ubuntu 20.04** (JetPack 5.x)
- **Python 3.8+** with pip
- **CUDA 11.8+** (included with JetPack)
- **Git** for repository management

## Step 1: Initial System Setup

### 1.1 Verify Jetson Installation
```bash
# Check JetPack version
sudo apt show nvidia-jetpack

# Verify CUDA installation
nvcc --version

# Check GPU status
nvidia-smi
```

### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget vim htop
```

### 1.3 Install Python Dependencies
```bash
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libopencv-dev python3-opencv
```

## Step 2: Project Setup

### 2.1 Navigate to Project Directory
```bash
cd /mnt/c/yard
```

### 2.2 Create Virtual Environment (if not exists)
```bash
# Run the setup script
./setup.bat
```

### 2.3 Install Dependencies
```bash
# Run dependency installation
./dependency.bat
```

**Important**: The dependency.bat script will:
- Activate the virtual environment
- Install all required packages in verbose mode
- Verify installations
- Provide optimization notes for Jetson

## Step 3: Model Setup

### 3.1 Download Required Models
Follow instructions in `models.md`:

1. **MegaDetector v6-c** (Recommended for Jetson)
   - Download from PyTorchWildlife (auto-downloads on first use)
   - OR manually download from GitHub releases

2. **ArcFace Dog Identification Model**
   - Download base ArcFace model
   - OR use animal re-identification model
   - See models.md for specific URLs and options

### 3.2 Create Models Directory
```bash
mkdir -p models
mkdir -p models/tensorrt
mkdir -p models/landmark_templates
```

### 3.3 Verify Model Loading
```bash
# Activate environment
source venv/bin/activate

# Test MegaDetector loading
python -c "
from PytorchWildlife.models import detection as pw_detection
print('Loading MegaDetector v6...')
model = pw_detection.MegaDetectorV6()
print('MegaDetector loaded successfully!')
"
```

## Step 4: Camera Configuration

### 4.1 IP Camera Setup
1. Configure each camera with static IP addresses
2. Enable RTSP/HTTP streaming
3. Set resolution to 1920x1080 or 1280x720
4. Configure frame rate to 15-30 FPS
5. Test camera streams with VLC or similar

### 4.2 Network Configuration
```bash
# Example camera URL formats:
# RTSP: rtsp://username:password@192.168.1.101:554/stream1
# HTTP: http://192.168.1.101:8080/video
```

### 4.3 Camera Testing
```bash
# Test individual camera with OpenCV
python -c "
import cv2
cap = cv2.VideoCapture('rtsp://your_camera_ip:554/stream1')
ret, frame = cap.read()
if ret:
    print('Camera connected successfully!')
    print(f'Frame shape: {frame.shape}')
else:
    print('Failed to connect to camera')
cap.release()
"
```

## Step 5: System Configuration

### 5.1 Reference Point Setup
1. Identify static landmarks in each camera view:
   - Telephone poles
   - House corners
   - Deck corners
   - Large trees
   - Fence posts

2. Take template images of each landmark
3. Store in `models/landmark_templates/`

### 5.2 Initial Boundary Configuration
1. Start the web application: `./web.bat`
2. Navigate to Boundaries page
3. Set initial boundaries for each camera
4. Use reference points as anchors

## Step 6: Dog Enrollment

### 6.1 Collect Dog Training Images
1. Capture 20-50 images per dog
2. Various poses, lighting conditions
3. Different angles and distances
4. Store in organized folders

### 6.2 Train Dog Identification
1. Use Dogs page in web interface
2. Upload training images
3. Train ArcFace model for each dog
4. Test identification accuracy

## Step 7: System Startup

### 7.1 Manual Startup
```bash
# Activate environment
source venv/bin/activate

# Start web application
python app.py
```

### 7.2 Access Web Interface
- Open browser to `http://jetson_ip:5000`
- Default: `http://localhost:5000`

### 7.3 Verify System Components
1. **Monitor Page**: View all camera feeds
2. **Settings Page**: Configure cameras and detection
3. **Boundaries Page**: Set and adjust boundaries
4. **Dogs Page**: Manage dog enrollment
5. **History Page**: View activity logs

## Step 8: Performance Optimization

### 8.1 Jetson Monitoring
```bash
# Install jetson-stats (if not already installed)
sudo pip3 install jetson-stats

# Monitor system performance
jtop
```

### 8.2 TensorRT Optimization (Optional)
```bash
# Convert models to TensorRT for better performance
# See models.md for specific conversion instructions
```

### 8.3 Memory Management
- Monitor GPU memory usage
- Adjust frame rates if memory issues occur
- Consider model quantization for better performance

## Step 9: Production Deployment

### 9.1 Auto-Start Configuration
```bash
# Create systemd service for auto-start
sudo nano /etc/systemd/system/dog-tracking.service

# Add service configuration:
[Unit]
Description=Dog Tracking System
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/mnt/c/yard
ExecStart=/mnt/c/yard/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable service
sudo systemctl enable dog-tracking.service
sudo systemctl start dog-tracking.service
```

### 9.2 Outdoor Deployment Checklist
- [ ] Weatherproof enclosure installed
- [ ] Power supply connected and tested
- [ ] Network connectivity verified
- [ ] All cameras mounted and configured
- [ ] System auto-start enabled
- [ ] Remote access configured
- [ ] Backup power solution (optional)

## Step 10: Testing and Validation

### 10.1 System Testing
1. **Camera Feeds**: Verify all cameras stream properly
2. **Detection**: Test animal/person/vehicle detection
3. **Dog ID**: Verify dog identification accuracy
4. **Boundaries**: Test boundary violation detection
5. **Alerts**: Verify email/sound notifications
6. **Cross-Camera**: Test dog tracking between cameras

### 10.2 Performance Validation
- Monitor frame rates (target: 15+ FPS per camera)
- Check GPU utilization (target: <80%)
- Verify memory usage (target: <6GB)
- Test system stability (24+ hour uptime)

## Troubleshooting

### Common Issues

#### 1. Camera Connection Problems
```bash
# Check network connectivity
ping camera_ip

# Test RTSP stream
ffplay rtsp://camera_ip:554/stream1
```

#### 2. Model Loading Errors
```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check model file paths
ls -la models/
```

#### 3. Performance Issues
```bash
# Monitor system resources
jtop

# Check for memory leaks
sudo dmesg | grep -i memory
```

#### 4. Web Interface Issues
```bash
# Check Flask logs
tail -f /var/log/dog-tracking.log

# Verify port availability
netstat -tulpn | grep 5000
```

### Support Resources
- **Models Documentation**: `models.md`
- **Implementation Plan**: `plan.md`
- **Jetson Documentation**: https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit
- **PyTorchWildlife**: https://microsoft.github.io/CameraTraps/

## Maintenance

### Regular Tasks
- **Weekly**: Check system logs and performance
- **Monthly**: Update system packages
- **Quarterly**: Clean camera lenses and check connections
- **Annually**: Review and update dog training data

### Backup Procedures
- **Configuration**: Backup boundary and dog settings
- **Models**: Backup trained models and templates
- **Logs**: Archive activity logs periodically

## Success Criteria
✅ All cameras streaming at target frame rates
✅ MegaDetector detecting animals/people/vehicles accurately
✅ Dog identification working for enrolled dogs
✅ Boundary violations triggering appropriate alerts
✅ Cross-camera tracking functioning for blind spots
✅ System running stably for 24+ hours
✅ Web interface accessible and responsive
✅ Performance within Jetson Orin Nano capabilities