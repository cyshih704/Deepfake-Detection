# Deepfake-Detection
Deepfake-Detection


## Data Preparation
- Download all video with compression rate factor 23 and 40
```
python3 download/download-Faceforensics.py ./download -d all -c c23 -t videos
python3 download/download-Faceforensics.py ./download -d all -c c40 -t videos
```

## Installation
```
# install custom layers
bash install.sh
```

## Data Preprocessing
- Sample 5 pairs of consecutive frames from each video
- Crop face region according to the first frame
```
python3 crop_face.py
```
