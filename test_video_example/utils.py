
import face_recognition as fr
import numpy as np
import os
import cv2

def get_sequential_optical_flow(video_path):
    print("Get sequential optical flow on {}".format(video_path))
    x_expand, y_expand = 1.5, 1.8

    sequential_optical_flow = []

    # extract frames from video
    frames = extract_frames(video_path)
    
    for i in range(50):
        face_locations = fr.face_locations(frames[i])

        # extract the first person in the frame
        if len(face_locations) >= 1:
            top, right, bottom, left = face_locations[0]
            b, h, w, c = frames.shape

            # cropping area
            top = max(0, int((bottom+top)//2 - (bottom-top)//2*y_expand))  # max(0, int(top-1.5*expand))
            right = min(w-1, int((right+left)//2 + (right-left)//2*x_expand))  # min(w-1, right+expand)
            bottom = min(h-1, int((bottom+top)//2 + (bottom-top)//2*y_expand))  # min(h-1, int(bottom+1.5*expand))
            left = max(0, int((right+left)//2 - (right-left)//2*x_expand))  # max(0, left-expand)


            frame1 = cv2.cvtColor(cv2.resize(frames[i, top:bottom, left:right], (256, 256)), cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(cv2.resize(frames[i+1, top:bottom, left:right], (256, 256)), cv2.COLOR_BGR2RGB)

            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            optical_flow = get_farneback_flow(prvs, nxt)
            sequential_optical_flow.append(optical_flow)

    return np.array(sequential_optical_flow)


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


def get_farneback_flow(prvs, nxt):
    h, w = prvs.shape
    flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    hsv = np.zeros((h, w, 3))
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    flow = np.array([mag, ang])
    flow_16bit = np.float16(flow)

    return flow_16bit