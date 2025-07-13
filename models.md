# Required Models for Dog Tracking System

## Core Detection & Re-ID Models

**MegaDetector** - Universal Animal/Person/Vehicle Detection  
https://github.com/microsoft/CameraTraps  
Installation: `pip install PytorchWildlife`

**MiewID** - Animal Re-Identification (EfficientNetV2 + ArcFace)  
https://github.com/WildMeOrg/wbia-plugin-miew-id  
Installation: `git clone https://github.com/WildMeOrg/wbia-plugin-miew-id && cd wbia-plugin-miew-id && pip install -e .`

**EfficientDet-D0** - Reference Point Detection for Camera Movement Compensation  
https://github.com/rwightman/efficientdet-pytorch  
Installation: `pip install effdet`  
Note: Pre-trained model auto-downloads on first use (~25MB)

## Quick Setup
```bash
# Install MegaDetector
pip install PytorchWildlife

# Install MiewID
git clone https://github.com/WildMeOrg/wbia-plugin-miew-id
cd wbia-plugin-miew-id
pip install -e .

# Install face recognition (for people ID)
pip install insightface

# Install EfficientDet-D0 (reference point detection)
pip install effdet
```

## Model Storage
Place downloaded models in: `/mnt/c/yard/models/`

## Pipeline
1. **EfficientDet-D0** → detects reference points (poles, trees, building corners) for camera movement compensation
2. **MegaDetector** → detects animals, people, vehicles (with movement-corrected boundaries)
3. **Route crops by class:**
   - People → Face ArcFace (existing)
   - Animals → MiewID ArcFace (full-body)
   - Vehicles → ALPR/Car Re-ID