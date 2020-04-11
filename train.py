import argparse

from dataloader.dataloader import get_loader
from tqdm import tqdm
import torch
import models
#from models import FlowNet2

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

def train_val(model, loader, epoch, device):
    #device = model.device()
    #print(device)
    for phase in ['train', 'val']:
        total_loss = 0
        total_pic = 0
        data_loader = loader[phase]
        pbar = tqdm(iter(data_loader))

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for num_batch, (first_img, second_img, label) in enumerate(pbar):
            first_img, second_img, label = first_img.to(device), second_img.to(device), label.to(device)
            inputs = torch.cat((first_img.unsqueeze(2), second_img.unsqueeze(2)), dim=2)

            if phase == 'train':
                flow = model(inputs) # batch, 2, h, w
            else:
                with torch.no_grad():
                    pass

            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        pbar.set_description('[{}] Epoch: {}; loss: {:.4f}'.format(phase.upper(), epoch+1, total_loss/total_pic))


def main():
    train_loader = get_loader('test', batch_size=8, shuffle=True, num_workers=8)
    val_loader = get_loader('val', batch_size=8, shuffle=False, num_workers=8)
    loader = {'train': train_loader, 'val': val_loader}

    state_dict = torch.load('saved_model/FlowNet2.tar')['state_dict']
    model = models.FlowNet2(args).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    for epoch in range(1):
        train_val(model, loader, epoch, DEVICE)

if __name__ == '__main__':
    main()