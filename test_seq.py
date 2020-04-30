import argparse

from dataloader.dataloader_seq import get_loader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models.LSTM import eqModel
from models.resnet import resnet18
from models.vgg import vgg11_bn
from tb_writer import TensorboardWriter
import os
from models.self_attn import SelfAttention


parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-l', '--load_model', help='load model')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'


def test(encoder, decoder, criterion, loader, device):

    total_loss = 0
    total_correct = 0
    total_data = 0

    pbar = tqdm(iter(loader))

    encoder.eval()
    decoder.eval()

    for num_batch, (seq, labels) in enumerate(pbar):
        seq = seq.to(device)
        labels = labels.to(device)


        with torch.no_grad():
            features = encoder(seq[0]).unsqueeze(1) # (50, 512)
            L, B, X, Y, Z = features.size()
            features = features.view(L, B, -1)

            preds = decoder(features)

        pred_class = (preds > 0.5).int()
        num_correct = torch.sum((pred_class == labels)).item()

        loss = criterion(preds, labels)        


        total_correct += num_correct
        total_loss += loss.item()
        total_data += seq.size(0)




        pbar.set_description('[{}] loss: {:.4f}, acc: {:.2f}%'.format("TEST", total_loss/total_data,
                        total_correct/total_data*100))


def main():
    loader = get_loader('test', batch_size=1, shuffle=False, num_data=-1)


    decoder = SeqModel(False).to(DEVICE)
    decoder = SelfAttention().to(DEVICE)

    if args.load_model:
        dic = torch.load(args.load_model)
        state_dict = dic["state_dict"]
        decoder.load_state_dict(state_dict)
        print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))

    #encoder = resnet18(pretrained=False, num_classes=1, in_channels=2).to(DEVICE)
    #encoder = vgg11(pretrained=False, num_classes=1).to(DEVICE)
    encoder = vgg11_bn(pretrained=False, num_classes=1).to(DEVICE)
    
    dic = torch.load("saved_model/farnback_vgg.tar")
    state_dict = dic["state_dict"]
    encoder.load_state_dict(state_dict)
    modules = list(encoder.children())[:-2]
    encoder = nn.Sequential(*modules)


    criterion = nn.BCELoss(reduction='mean')


    test(encoder, decoder, criterion, loader, DEVICE)


if __name__ == '__main__':
    main()