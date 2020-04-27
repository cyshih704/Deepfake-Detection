import argparse

from dataloader.dataloader import get_loader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models import flownet_models
from models.classifier import FlowClassifier
from models.resnet import resnet18
from models.vgg import vgg11_bn
from tb_writer import TensorboardWriter
import os


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


def test(clf, criterion, loader, device):

    total_loss = 0
    total_correct = 0
    total_data = 0

    pbar = tqdm(iter(loader))

    clf.eval()

    for num_batch, (img, labels) in enumerate(pbar):
        #first_img, second_img = first_img.to(device), second_img.to(device)
        #farnback_img, flownet_img = farnback_img.to(device), flownet_img.to(device)
        #farnback, flownet = farnback.to(device), flownet.to(device)
        img = img.to(device)
        labels = labels.to(device)


        with torch.no_grad():
            preds = clf(img)


        pred_class = (preds > 0.5).int()
        num_correct = torch.sum((pred_class == labels)).item()

        loss = criterion(preds, labels)        


        total_correct += num_correct
        total_loss += loss.item()
        total_data += img.size(0)




        pbar.set_description('[{}] loss: {:.4f}, acc: {:.2f}%'.format("TEST", total_loss/total_data,
                        total_correct/total_data*100))


def main():
    loader = get_loader('test', batch_size=32, shuffle=False, num_data=-1)

    clf = resnet18(pretrained=False, num_classes=1, in_channels=2).to(DEVICE)
    clf = vgg11_bn(pretrained=False, num_classes=1).to(DEVICE)
    if args.load_model:
        dic = torch.load(args.load_model)
        state_dict = dic["state_dict"]
        clf.load_state_dict(state_dict)
        print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))


    criterion = nn.BCELoss(reduction='mean')


    test(clf, criterion, loader, DEVICE)


if __name__ == '__main__':
    main()