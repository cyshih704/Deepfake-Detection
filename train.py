import argparse

from dataloader.dataloader import get_loader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import resnet18
from models.vgg import vgg11_bn, vgg11
from tb_writer import TensorboardWriter
import os


parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model', help='saved_model_name')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-l', '--load_model', help='load model')
parser.add_argument('-n', '--num_data', type=int, default=20000, help='the number of data used to train')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'


def train_val(clf, criterion, optimizer, loader, epoch, device):
    train_acc, val_acc = 0, 0
    train_loss, val_loss = 0, 0
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

        for num_batch, (img, labels) in enumerate(pbar):
            img = img.to(device)
            labels = labels.to(device)

            if phase == 'train':
                preds = clf(img)
            else:
                with torch.no_grad():
                    preds = clf(img)


            pred_class = (preds > 0.5).int()
            num_correct = torch.sum((pred_class == labels)).item()

            loss = criterion(preds, labels)        


            total_correct += num_correct
            total_loss += loss.item()
            total_data += img.size(0)


            if phase == 'train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_acc = total_correct/total_data*100
                train_loss = total_loss/total_data
            else:
                val_acc = total_correct/total_data*100
                val_loss = total_loss/total_data


            pbar.set_description('[{}] Epoch: {}; loss: {:.4f}, acc: {:.2f}%'.format(phase.upper(), epoch, total_loss/total_data,
                            total_correct/total_data*100))
    return train_loss, val_loss, train_acc, val_acc


class EarlyStop():
    """Early stop training if validation loss didn't improve for a long time"""
    def __init__(self, patience, mode = 'min'):
        self.patience = patience
        self.mode = mode

        self.best = float('inf') if mode == 'min' else 0
        self.cur_patience = 0

    def stop(self, loss, model, epoch, saved_model_path):
        update_best = loss < self.best if self.mode == 'min' else loss > self.best

        if update_best:
            self.best = loss
            self.cur_patience = 0

            torch.save({'val_loss': loss, \
                        'state_dict': model.state_dict(), \
                        'epoch': epoch}, saved_model_path+'.tar')
            print('SAVE MODEL to {}'.format(saved_model_path))
        else:
            self.cur_patience += 1
            if self.patience == self.cur_patience:
                return True
        
        return False





def main():
    # initialize tensorboard for visualization training progress
    tensorboard_path = 'runs/{}'.format(args.saved_model_name)
    tb_writer = TensorboardWriter(tensorboard_path)

    # set the early stop
    early_stop = EarlyStop(patience=5, mode='max')

    # get train and val dataloader
    train_loader = get_loader('train', batch_size=32, shuffle=True, num_data=-1)
    val_loader = get_loader('val', batch_size=32, shuffle=False, num_data=-1)
    loader = dict(train=train_loader, val=val_loader)

    # choose the classifier
    clf = resnet18(pretrained=False, num_classes=1, in_channels=2).to(DEVICE)
    #clf = vgg11_bn(pretrained=False, num_classes=1).to(DEVICE)
    #clf = vgg11(pretrained=False, num_classes=1).to(DEVICE)

    # set criterion and optimizer
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(clf.parameters(), lr=1e-4)


    # training
    for epoch in range(1, 100):
        train_loss, val_loss, train_acc, val_acc = train_val(clf, criterion, optimizer, loader, epoch, DEVICE)
        tb_writer.tensorboard_write(epoch, train_loss, val_loss, train_acc, val_acc)

        saved_model_path = os.path.join("saved_model", "{}".format(args.saved_model_name))
        if early_stop.stop(val_acc, clf, epoch, saved_model_path):
            break

    tb_writer.close()

if __name__ == '__main__':
    main()