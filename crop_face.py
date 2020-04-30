import os
import cv2
from env import PREPRO_DIR, DOWNLOAD_DIR, SEQ_DIR
import numpy as np
import face_recognition as fr
from tqdm import tqdm

YOUTUBE_SUBPATH = os.path.join('original_sequences', 'youtube')
NEURAL_SUBPATH = os.path.join('manipulated_sequences', 'NeuralTextures')
FACESWAP_SUBPATH = os.path.join('manipulated_sequences', 'FaceSwap')
FACE2FACE_SUBPATH = os.path.join('manipulated_sequences', 'Face2Face')
DEEPFAKEDETECTION_SUBPATH = os.path.join('manipulated_sequences', 'DeepFakeDetection')
DEEPFAKES_SUBPATH = os.path.join('manipulated_sequences', 'Deepfakes')

subpath = {'youtube': YOUTUBE_SUBPATH,
           'NeuralTextures': NEURAL_SUBPATH,
           'FaceSwap': FACESWAP_SUBPATH,
           'Face2Face': FACE2FACE_SUBPATH
           'DeepFakeDetection': DEEPFAKEDETECTION_SUBPATH,
           'Deepfakes': DEEPFAKES_SUBPATH}
}


def extract_frames(video_path):
    """Given the video path, extract every frame from video."""

    reader = cv2.VideoCapture(video_path)
    frameCount = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        buf[frame_num] = image
        frame_num += 1
    reader.release()

    return buf


def process(dataset, compression, num_frames, offset, x_expand, y_expand, continueous):
    """Save consecutive frames.
    
    Params:
    dataset: str, must in {'youtube', 'NeuralTextures', 'FaceSwap', 'Face2Face', 'Deepfakes'}
    compression: str, must in {'raw', 'c23', 'c40'}
    num_frames: int, the number of frames to be sampled
    offset: int, the offset between two frames
    x_expand: float, expand the cropping area of x_axis
    y_expand: float, expand the cropping area of y_axis
    """
    saved_dir = SEQ_DIR if continueous else PREPRO_DIR
    np.random.seed(0)
    assert dataset in subpath

    path = os.path.join(DOWNLOAD_DIR, subpath[dataset], compression, 'videos')
    video_name_list = os.listdir(path)

    pbar = tqdm(video_name_list)
    for video_name_ext in pbar:
        pbar.set_description('[{}|{}]'.format(dataset, compression))

        video_name = video_name_ext.split('.')[0]
        video_path = os.path.join(path, video_name_ext)

        if os.path.exists(os.path.join(saved_dir, subpath[dataset], compression, video_name)):
            continue

        # extract frames from videos
        frames = extract_frames(video_path)

        # if video has the number of frames <= 1, skip
        if len(frames) <= 1:
            continue

        
        if not continueous: # indices to randomly sampled frames
            sel_indices = np.random.choice(len(frames)-1, min(num_frames, len(frames)-1), replace=False)
        else: # indices to first num_frame frames
            sel_indices = np.arange(num_frames)

        for i, idx in enumerate(sel_indices):
            saved_path = os.path.join(saved_dir, subpath[dataset], compression, video_name, str(idx))

            # face detection
            face_locations = fr.face_locations(frames[idx])

            # extract the first person in the frame
            if len(face_locations) >= 1:
                top, right, bottom, left = face_locations[0]
                b, h, w, c = frames.shape

                # cropping area
                top = max(0, int((bottom+top)//2 - (bottom-top)//2*y_expand))  # max(0, int(top-1.5*expand))
                right = min(w-1, int((right+left)//2 + (right-left)//2*x_expand))  # min(w-1, right+expand)
                bottom = min(h-1, int((bottom+top)//2 + (bottom-top)//2*y_expand))  # min(h-1, int(bottom+1.5*expand))
                left = max(0, int((right+left)//2 - (right-left)//2*x_expand))  # max(0, left-expand)

                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)

                # save consecutive frames
                cv2.imwrite(os.path.join(saved_path, '1.png'),
                            cv2.resize(frames[idx, top:bottom, left:right], (256, 256)))
                cv2.imwrite(os.path.join(saved_path, '2.png'),
                            cv2.resize(frames[idx+offset, top:bottom, left:right], (256, 256)))


if __name__ == '__main__':

    for key, val in subpath.items():
        process(key, 'c23', num_frames=50, offset=1, x_expand=1.5, y_expand=1.8, continueous=True)
        process(key, 'c23', num_frames=20, offset=1, x_expand=1.5, y_expand=1.8, continueous=False)
