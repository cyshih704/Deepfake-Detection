import argparse

from dataloader.dataloader import get_loader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models import flownet_models
from models.classifier import FlowClassifier
from models.resnet import resnet18
from models.vgg import vgg11_bn, vgg11
from tb_writer import TensorboardWriter
import os


parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-l', '--load_model', help='load model')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'


def test(clf, criterion, loader, device):

    total_loss = 0
    total_correct = 0
    total_data = 0

    pbar = tqdm(iter(loader))

    clf.eval()

    for num_batch, (img, labels) in enumerate(pbar):
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

    #clf = resnet18(pretrained=False, num_classes=1, in_channels=2).to(DEVICE)
    clf = vgg11_bn(pretrained=False, num_classes=1).to(DEVICE)
    #clf = vgg11(pretrained=False, num_classes=1).to(DEVICE)

    if args.load_model:
        dic = torch.load(args.load_model)
        state_dict = dic["state_dict"]
        clf.load_state_dict(state_dict)
        print('Accuracy of loaded model: {:.4f}'.format(dic['val_loss']))

    criterion = nn.BCELoss(reduction='mean')
    test(clf, criterion, loader, DEVICE)


if __name__ == '__main__':
    main()