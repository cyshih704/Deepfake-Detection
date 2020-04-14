import argparse

from dataloader.dataloader import get_loader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models import flownet_models
from models.classifier import FlowClassifier
#from models import FlowNet2
import pretrainedmodels

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


def train_val(model, clf, criterion, optimizer, loader, epoch, device):
    #device = model.device()
    for phase in ['train', 'val']:
        total_loss = 0
        total_correct = 0
        total_data = 0

        data_loader = loader[phase]
        pbar = tqdm(iter(data_loader))

        if phase == 'train':
            #model.train()
            clf.train()
        else:
            #model.eval()
            clf.eval()

        for num_batch, (first_img, second_img, labels) in enumerate(pbar):
            first_img, second_img, labels = first_img.to(device), second_img.to(device), labels.to(device)
            inputs = torch.cat((first_img.unsqueeze(2), second_img.unsqueeze(2)), dim=2)

            if phase == 'train':
                #flow = model(inputs) # batch, 2, h, w
                flow = first_img # batch, 2, h, w
                preds = clf(flow)
            else:
                with torch.no_grad():
                    #flow = model(inputs) # batch, 2, h, w
                    flow = first_img # batch, 2, h, w
                    preds = clf(flow)
        
            if phase == 'train':
                loss = criterion(preds, labels)

                #weight = torch.ones_like(labels)
                #weight[labels == 1.0] = 4.0
                #weighted_loss = torch.mean(loss * weight)
                weighted_loss = loss
        
                weighted_loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            _, pred_class = torch.max(preds, dim=1, keepdims=True)
            num_correct = torch.sum((pred_class == labels)).item()

            total_correct += num_correct
            total_loss += weighted_loss
            total_data += first_img.size(0)

            pbar.set_description('[{}] Epoch: {}; loss: {:.4f}, acc: {:.2f}%'.format(phase.upper(), epoch+1, total_loss/total_data,
                            total_correct/total_data*100))


def main():
    clf = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
    del clf.last_linear
    clf.last_linear = nn.Sequential(nn.Linear(2048, 1), nn.Sigmoid())
    clf = clf.to(DEVICE)

    train_loader = get_loader('train', batch_size=2, shuffle=True, num_workers=8, num_data=7000)
    val_loader = get_loader('val', batch_size=2, shuffle=False, num_workers=8, num_data=500)
    loader = {'train': train_loader, 'val': val_loader}
    #state_dict = torch.load('saved_model/FlowNet2.tar')['state_dict']
    #model = flownet_models.FlowNet2(args).to(DEVICE)
    #model.load_state_dict(state_dict)
    #model.eval()
    model = 1

    #clf = FlowClassifier(3).to(DEVICE)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    for epoch in range(100):
        train_val(model, clf, criterion, optimizer, loader, epoch, DEVICE)

if __name__ == '__main__':
    main()