# Deepfake-Detection
Deepfake-Detection


## Data Preparation
- Download all video with compression rate factor 23 and 40
```
python3 download/download-Faceforensics.py ./download -d all -c c23 -t videos
python3 download/download-Faceforensics.py ./download -d all -c c40 -t videos
```

## Flownet2 Installation
```
# install custom layers
bash install.sh
```

## Usage
Set the path in **env.py**
```
DOWNLOAD_DIR: have manipulated_suquences/ and original_sequences/
PREPRO_DIR: empty folder, the folder to save proprocessed data
```
### Data Preprocessing
- Sample 100 pairs of consecutive frames from each video
- Crop face region according to the first frame
```
python3 crop_face.py
```

- Save optical flow as npy file
```
python3 save_flow.py
```
