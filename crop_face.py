import os
import cv2
from env import PREPRO_DIR, DOWNLOAD_DIR
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
           'Face2Face': FACE2FACE_SUBPATH,
           'DeepFakeDetection': DEEPFAKEDETECTION_SUBPATH,
           'Deepfakes': DEEPFAKES_SUBPATH}


def extract_frames(video_path):
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


def process(dataset, compression, num_frames, offset, x_expand, y_expand):
    np.random.seed(0)
    assert dataset in subpath

    path = os.path.join(DOWNLOAD_DIR, subpath[dataset], compression, 'videos')
    video_name_list = os.listdir(path)

    pbar = tqdm(video_name_list)
    for video_name_ext in pbar:
        pbar.set_description('[{}|{}]'.format(dataset, compression))

        video_name = video_name_ext.split('.')[0]
        video_path = os.path.join(path, video_name_ext)

        if os.path.exists(os.path.join(PREPRO_DIR, subpath[dataset], compression, video_name)):
            continue

        frames = extract_frames(video_path)

        if len(frames) <= 1:
            continue
        sel_indices = np.random.choice(len(frames)-1, min(num_frames, len(frames)-1), replace=False)
        for i, idx in enumerate(sel_indices):
            saved_path = os.path.join(PREPRO_DIR, subpath[dataset], compression, video_name, str(idx))

            face_locations = fr.face_locations(frames[idx])
            if len(face_locations) >= 1:
                top, right, bottom, left = face_locations[0]
                b, h, w, c = frames.shape

                top = max(0, int((bottom+top)//2 - (bottom-top)//2*y_expand))  # max(0, int(top-1.5*expand))
                right = min(w-1, int((right+left)//2 + (right-left)//2*x_expand))  # min(w-1, right+expand)
                bottom = min(h-1, int((bottom+top)//2 + (bottom-top)//2*y_expand))  # min(h-1, int(bottom+1.5*expand))
                left = max(0, int((right+left)//2 - (right-left)//2*x_expand))  # max(0, left-expand)

                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
#
                cv2.imwrite(os.path.join(saved_path, '1.png'),
                            cv2.resize(frames[idx, top:bottom, left:right], (256, 300)))
                cv2.imwrite(os.path.join(saved_path, '2.png'),
                            cv2.resize(frames[idx+offset, top:bottom, left:right], (256, 300)))


if __name__ == '__main__':

    #process('youtube', 'c23', num_frames=5, offset=1, x_expand=1.5, y_expand=1.8)
    #process('youtube', 'c40', num_frames=5, offset=1, x_expand=1.5, y_expand=1.8)

    for key, val in subpath.items():
        process(key, 'c23', num_frames=30, offset=1, x_expand=1.5, y_expand=1.8)
        process(key, 'c40', num_frames=30, offset=1, x_expand=1.5, y_expand=1.8)
