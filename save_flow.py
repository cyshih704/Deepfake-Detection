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


parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model', help='saved_model_name')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-l', '--load_model', help='load model')
parser.add_argument('-n', '--num_data', type=int, default=20000, help='the number of data used to train')
parser.add_argument("--rgb_max", type=float, default = 255.)
parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'

ALL_DATASETS = ['youtube', 'Face2Face']
SPLIT = ['train', 'val', 'test']
RATIO = ['c23']

def save_flow(flownet, path):
    """Save optical flow by FlowNet2.0."""

    if os.path.exists(os.path.join(path, 'flownet2_flow.png')):
        return 

    first_img = cv2.cvtColor(cv2.imread(os.path.join(path, '1.png')), cv2.COLOR_BGR2RGB)
    second_img = cv2.cvtColor(cv2.imread(os.path.join(path, '2.png')), cv2.COLOR_BGR2RGB)

    first_img = cv2.resize(first_img, (256, 256)).transpose(2, 0, 1) # c, h, w
    second_img = cv2.resize(second_img, (256, 256)).transpose(2, 0, 1) # c, h, w

    inputs = np.array([first_img, second_img])
    inputs = torch.FloatTensor(inputs).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        flow = flownet(inputs) # batch, 2, h, w

    # save optical flow
    flow = flow.squeeze().cpu().numpy()
    flow_16bit = np.float16(flow)
    np.save(os.path.join(path, 'flownet2'), flow_16bit)
    
    # convert polar to RGB image and save
    flow_img = flow2img(flow.transpose(1, 2, 0))
    cv2.imwrite(os.path.join(path, 'flownet2_flow.png'), cv2.cvtColor(flow_img, cv2.COLOR_RGB2BGR))

def save_farneback_flow(path):
    """Save optical flow by Farneback algorithm."""

    if os.path.exists(os.path.join(path, 'farnback_flow.png')):
        return 

    frame1 = cv2.imread(os.path.join(path, '1.png'))
    frame2 = cv2.imread(os.path.join(path, '2.png'))

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

    flow = np.array([mag, ang])
    flow_16bit = np.float16(flow)
    np.save(os.path.join(path, 'farnback'), flow_16bit)

    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(path, 'farnback_flow.png'), rgb)



def save_spynet_flow(path):
    """Save optical flow by SPyNet algorithm."""

    if os.path.exists(os.path.join(path, 'spynet.npy')):
        return 

    first_img = cv2.cvtColor(cv2.imread(os.path.join(path, '1.png')), cv2.COLOR_BGR2RGB)
    second_img = cv2.cvtColor(cv2.imread(os.path.join(path, '2.png')), cv2.COLOR_BGR2RGB)

    first_img = torch.FloatTensor(cv2.resize(first_img, (256, 256)).transpose(2, 0, 1)).to(DEVICE) / 255.0# c, h, w
    second_img = torch.FloatTensor(cv2.resize(second_img, (256, 256)).transpose(2, 0, 1)).to(DEVICE) / 255.0# c, h, w

    flow = estimate_spy(first_img, second_img).numpy()

    flow_16bit = np.float16(flow)
    np.save(os.path.join(path, 'spynet'), flow_16bit)


def save_pwcnet_flow(path):
    """Save optical flow by PWC-Net algorithm."""

    if os.path.exists(os.path.join(path, 'pwc.npy')):
        return 

    first_img = cv2.cvtColor(cv2.imread(os.path.join(path, '1.png')), cv2.COLOR_BGR2RGB)
    second_img = cv2.cvtColor(cv2.imread(os.path.join(path, '2.png')), cv2.COLOR_BGR2RGB)

    first_img = torch.FloatTensor(cv2.resize(first_img, (256, 256)).transpose(2, 0, 1)).to(DEVICE) / 255.0# c, h, w
    second_img = torch.FloatTensor(cv2.resize(second_img, (256, 256)).transpose(2, 0, 1)).to(DEVICE) / 255.0# c, h, w

    flow = estimate_pwc(first_img, second_img).numpy()


    flow_16bit = np.float16(flow)
    np.save(os.path.join(path, 'pwc'), flow_16bit)


def main():

    for ratio in RATIO:
        for split in SPLIT:
            for dataset in ALL_DATASETS:
                all_path = get_image_path(dataset, ratio, split)
                for path in tqdm(all_path):
                    save_flow(flownet, path) # 2 x 256 x 256
                    save_farneback_flow(path)
                    save_spynet_flow(path)
                    save_pwcnet_flow(path)
                    
if __name__ == '__main__':
    main()