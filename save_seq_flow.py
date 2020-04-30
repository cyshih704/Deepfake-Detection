import argparse

from dataloader.utils import get_image_path
import cv2
import torch
from models import flownet_models
import os
import numpy as np
from tqdm import tqdm
from flow_utils import flow2img
from pytorch_spynet.run import estimate as estimate_spy
from pytorch_pwc.run import estimate as estimate_pwc
from pytorch_pwc.run import Network as PWC_Net
from env import SEQ_DIR
import json
from save_flow import save_farneback_flow

ALL_DATASETS = ['youtube', 'Face2Face']
SPLIT = ['train', 'val', 'test']
RATIO = ['c23']


def get_videoname_path(dataset, compression, split, DIR):
    """Get all video name paths."""
    with open(os.path.join('dataloader', 'splits', '{}.json'.format(split))) as f:
        data = json.load(f)
    
    if dataset == 'youtube':
        subpath = 'original_sequences'
        dir_set = set([element for sublist in data for element in sublist])
    else:
        subpath = 'manipulated_sequences'
        dir_set = set(['{}_{}'.format(a, b) for a, b in data] + ['{}_{}'.format(b, a) for a, b in data])


    compression_path = os.path.join(DIR, subpath, dataset, compression)
    video_name_list = sorted(os.listdir(compression_path))

    ret_path = []
    for video_name in video_name_list:
        if video_name not in dir_set:
            continue

        path_to_video_name = os.path.join(compression_path, video_name)
        ret_path.append(path_to_video_name)
    return ret_path


def merge(path):
    """Merge all consecutive flow into one file."""

    all_files = os.listdir(path)
    npy_files = [fn for fn in all_files if fn.split('.')[1] == "npy"]
    
    if "merged_flow.npy" in npy_files:
        return
    
    npy_files.sort()
    ret = []
    for fn in npy_files:
        ret.append(np.load(os.path.join(path, fn)))
    ret = np.float16(np.array(ret))

    np.save(os.path.join(path, 'merged_flow'), ret)

def merge_flow():
    for ratio in RATIO:
        for split in SPLIT:
            for dataset in ALL_DATASETS:
                all_path = get_videoname_path(dataset, ratio, split, SEQ_DIR)
                for path in tqdm(all_path):
                    flow_path = os.path.join(path, "flow")
                    merge(flow_path)

def main():
    # get and save flow
    for ratio in RATIO:
        for split in SPLIT:
            for dataset in ALL_DATASETS:
                all_path = get_image_path(dataset, ratio, split, SEQ_DIR)
                for path in tqdm(all_path):
                    save_farneback_flow(path)
    # merge consecutive flow into one file
    merge_flow()
if __name__ == '__main__':
    main()