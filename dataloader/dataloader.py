import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch
from dataloader.utils import image_transform, get_image_path
#from env import PREPRO_DIR
#PREPRO_DIR = '/home/tmt/ML_data/preproc_frames'
ALL_DATASETS = ['youtube', 'Face2Face', 'FaceSwap', 'NeuralTextures']
# split, batch_size, shuffle, num_workers=8

def get_loader(split, batch_size, shuffle, num_workers=8):
    dataset = DeepfakeDataset(split, 'all')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader
class DeepfakeDataset(Dataset):
    """Deepfake dataset."""

    def __init__(self, split, dataset):
        self.x_path = []
        self.y = []
        self.transforms = image_transform()
        if dataset == 'all':
            for dataset in ALL_DATASETS:
                img_path = get_image_path(dataset, 'c23', split)
                label = [0]*len(img_path) if dataset != 'youtube' else [1]*len(img_path)

                self.x_path += img_path
                self.y += label
        else:
            self.x_path = get_image_path(dataset, 'c23', split)
            self.y = [0]*len(self.x_path) if dataset != 'youtube' else [1]*len(self.x_path)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        first_img = self.transforms(cv2.imread(os.path.join(self.x_path[i], '1.png')))
        second_img = self.transforms(cv2.imread(os.path.join(self.x_path[i], '2.png')))
        label = self.y[i]

        return first_img, second_img, torch.Tensor(label).int()

if __name__ == '__main__':
    #a = get_image_path('Face2Face', 'c23', 'val')
    dataset = DeepfakeDataset('train', 'youtube')
    loader = DataLoader(dataset, batch_size=8)
    for img1, img2, label in loader:
        print(img1.size(), img2.size(), label)