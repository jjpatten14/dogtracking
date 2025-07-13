# Dog Tracking System - Model Information

## Object Detection Model: Grounding DINO

### Model Details
- **Model**: Grounding DINO Tiny (Open-Vocabulary Object Detection)
- **Framework**: Hugging Face Transformers
- **Purpose**: Static reference point detection for camera movement compensation
- **Training Data**: Large-scale vision-language datasets
- **Key Feature**: Natural language object detection

### Installation Requirements
```bash
pip install transformers torch pillow
```

### Model Capabilities
- **Open-vocabulary detection**: Detect ANY object described in natural language
- **Text-prompted detection**: Use phrases like "tree trunk", "building corner"
- **Zero-shot detection**: No need for pre-defined object classes
- **High accuracy**: 52.5 AP on COCO zero-shot benchmark

### Supported Reference Point Prompts
The system uses these static object prompts:
- **Trees**: "tree trunk", "large tree"
- **Buildings**: "building corner", "house corner", "roof edge", "chimney"
- **Infrastructure**: "fence post", "utility pole", "telephone pole"

### Performance Characteristics
- **Accuracy**: Superior for architectural and landscape features
- **Speed**: Fast inference with tiny model variant
- **Memory**: Lightweight implementation
- **Flexibility**: Adaptable to any scene via text prompts

### Usage in Reference Point System
Grounding DINO detects truly static objects using natural language descriptions. Unlike traditional object detectors limited to 80 COCO classes, it can find specific architectural features like "building corner" or "fence post" that make ideal reference points.

### Reference Point Selection Criteria
Objects are selected as reference points based on:
1. **Static Nature**: Only permanent, non-moving objects
2. **Structural Stability**: Buildings, trees, poles, infrastructure
3. **Natural Language Precision**: Specific descriptions like "tree trunk" vs generic "tree"
4. **Confidence**: Detection confidence > 0.2 (lower threshold for static objects)

### Key Advantages Over Traditional Detection
- **No Moving Objects**: Excludes cars, people, animals by design
- **Architectural Focus**: Specifically targets building corners, structural elements
- **Scene Adaptability**: Can ask for objects specific to your environment
- **Better Accuracy**: Trained on diverse real-world data

### Fallback Options
If automatic detection fails:
- Manual reference point placement available
- User can click on stable objects in the video feed
- Points stored as normalized coordinates (0-1 range)
- Manual points have highest priority (confidence = 1.0)

### Model Loading and Initialization
- Model auto-downloads from Hugging Face on first use
- Uses AutoProcessor and AutoModelForZeroShotObjectDetection
- Evaluation mode for inference only
- CPU inference for compatibility
- Simple text + image input format