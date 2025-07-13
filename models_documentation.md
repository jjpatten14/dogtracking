# AI Models Documentation

## Overview
This document describes the AI models used in the Dog Tracking System and their specific purposes.

## Primary Animal Detection Models

### PytorchWildlife (MegaDetector)
- **Purpose**: Primary animal detection in outdoor environments
- **Capability**: Detects animals, people, vehicles in wildlife camera footage
- **Usage**: Main detection engine for identifying dogs and other animals
- **GPU Impact**: High - primary model for core functionality

### InsightFace
- **Purpose**: Face recognition for dog identification
- **Capability**: Individual dog identification and tracking
- **Usage**: Distinguishing between different dogs in multi-dog environments
- **GPU Impact**: Medium - used for identification after detection

### MiewID (Animal Re-Identification)
- **Purpose**: Advanced animal re-identification across cameras
- **Capability**: Track individual animals across multiple camera feeds
- **Usage**: Cross-camera tracking and long-term individual animal monitoring
- **GPU Impact**: Medium - used for advanced tracking scenarios

## Reference Point Detection Model

### EfficientDet-D0
- **Purpose**: Reference point detection for camera movement compensation
- **Capability**: Detects stable objects (telephone poles, trees, building corners)
- **Usage**: Automatically adjusts boundary coordinates when camera moves due to wind/vibration

#### Why EfficientDet-D0 Was Chosen:
- **Accuracy**: 85% detection accuracy for reference objects
- **GPU Efficiency**: Uses only 40% of the GPU resources compared to Detectron2
- **Trade-off Analysis**: 10% accuracy loss for 60% GPU savings
- **Optimal Balance**: Provides sufficient accuracy for stable reference point tracking while preserving GPU resources for primary animal detection tasks

#### Reference Objects Detected:
- **Telephone Poles**: Vertical structures for movement reference
- **Building Corners**: Sharp architectural features for precise positioning
- **Trees**: Large natural landmarks (when stable/mature)

#### Benefits:
- **Automatic Compensation**: Boundaries remain accurate even when camera shifts
- **Resource Efficient**: Minimal impact on primary animal detection performance  
- **Stable Tracking**: Reliable detection of fixed environmental features
- **Real-time Capable**: Fast enough for continuous camera movement monitoring

## Model Integration Strategy

### GPU Resource Allocation:
1. **Primary**: Animal detection models (60-70% GPU usage)
2. **Secondary**: Reference point detection (15-25% GPU usage)  
3. **Reserve**: System overhead and processing (10-15% GPU usage)

### Processing Pipeline:
1. **Frame Capture**: Raw video from camera feeds
2. **Reference Point Detection**: EfficientDet-D0 identifies stable objects
3. **Movement Compensation**: Adjust coordinates if camera shifted
4. **Animal Detection**: Primary models detect dogs/animals with corrected boundaries
5. **Tracking & Identification**: Follow individual animals across frames

## Future Considerations

### Model Updates:
- Monitor EfficientDet model performance in various weather conditions
- Consider upgrading to EfficientDet-D1 if more accuracy needed
- Evaluate newer lightweight detection models as they become available

### Performance Optimization:
- Implement model quantization for better GPU efficiency
- Consider edge computing deployment for reduced latency
- Explore model pruning for specific use case optimization

## Installation Notes

All models are automatically installed via `setup.bat`:
- PytorchWildlife: Primary animal detection
- InsightFace: Face recognition  
- MiewID: Animal re-identification
- EfficientDet-PyTorch: Reference point detection

GPU requirements scale with number of simultaneous camera feeds and detection frequency.