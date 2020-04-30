import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch
from dataloader.utils import get_merged_flow_path
from skimage import io
from PIL import Image as pil_image


ALL_DATASETS = ['youtube', 'Face2Face']
FAKE_DATASETS = ['Face2Face', 'FaceSwap', 'NeuralTextures', 'Deepfakes']
# split, batch_size, shuffle, num_workers=8

def get_loader(split, batch_size, shuffle, num_data):
    dataset = DeepfakeDataset(split, 'all', num_data=num_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)#, num_workers=num_workers)
    return loader



class DeepfakeDataset(Dataset):
    """Deepfake dataset."""

    def __init__(self, split, dataset, num_data):
        np.random.seed(0)

        self.x_path = []
        self.y = []
        #self.transforms = image_transform()
        if dataset == 'all':
            for dataset in ALL_DATASETS:
                img_path = get_merged_flow_path(dataset, 'c23', split)
                label = [0]*len(img_path) if dataset != 'youtube' else [1]*len(img_path)

                if num_data != -1:
                    indices = np.random.choice(len(img_path), size=num_data, replace=False)
                    img_path = list(np.array(img_path)[indices])
                    label = list(np.array(label)[indices])

                self.x_path += img_path
                self.y += label
        else:
            self.x_path = get_image_path(dataset, 'c23', split)
            self.y = [0]*len(self.x_path) if dataset != 'youtube' else [1]*len(self.x_path)

            if num_data != -1:
                indices = np.random.choice(self.x_path, size=num_data, replace=False)
                self.x_path = np.array(self.x_path)[indices]
                self.y = np.array(self.y)[indices]

        print('The length of {} dataset: {}'.format(split, len(self.x_path)))


    def __len__(self):
        return len(self.y)

    def _read_image_to_rgb(self, img_path):
        """Return numpy array with shape (c, h, w), where c is rgb."""
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256)).transpose(2, 0, 1) # c, h, w
        return img

    def __getitem__(self, i):
        seq_data = np.load(self.x_path[i])
        label = self.y[i]

        return torch.FloatTensor(seq_data), torch.Tensor([label]).float()

