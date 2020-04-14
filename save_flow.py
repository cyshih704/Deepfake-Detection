import argparse

from dataloader.utils import get_image_path
import cv2
import torch
from models import flownet_models
import os
import numpy as np
from tqdm import tqdm

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

ALL_DATASETS = ['youtube', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'Deepfakes']
SPLIT = ['train', 'val']
RATIO = ['c23']

def save_flow(flownet, path):
    
    first_img = cv2.cvtColor(cv2.imread(os.path.join(path, '1.png')), cv2.COLOR_BGR2RGB)
    second_img = cv2.cvtColor(cv2.imread(os.path.join(path, '2.png')), cv2.COLOR_BGR2RGB)

    first_img = cv2.resize(first_img, (256, 256)).transpose(2, 0, 1) # c, h, w
    second_img = cv2.resize(second_img, (256, 256)).transpose(2, 0, 1) # c, h, w

    inputs = np.array([first_img, second_img])
    inputs = torch.FloatTensor(inputs).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        flow = flownet(inputs) # batch, 2, h, w

    flow = flow.squeeze().cpu().numpy()
    flow = np.int8(flow)
    np.save(os.path.join(path, 'flow'), flow)
    
def main():

    state_dict = torch.load('saved_model/FlowNet2.tar')['state_dict']
    flownet = flownet_models.FlowNet2(args).to(DEVICE)
    flownet.load_state_dict(state_dict)
    flownet.eval()

    for ratio in RATIO:
        for split in SPLIT:
            for dataset in ALL_DATASETS:
                all_path = get_image_path(dataset, ratio, split)
                for path in tqdm(all_path):
                    if not os.path.exists(os.path.join(path, 'flow.npy')):
                        save_flow(flownet, path) # 2 x 256 x 256

if __name__ == '__main__':
    main()