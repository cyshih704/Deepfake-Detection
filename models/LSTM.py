import torch
import torch.nn as nn
from models.modules import ResBlock

        
class SeqModel(nn.Module):

    def __init__(self, bidirectional=False):
        super(SeqModel, self).__init__()
        embedding_dim = 512*7*7
        hidden_dim = 512
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        mul = 1 if not bidirectional else 2
        self.clf = nn.Linear(512*mul, 1)
        self.sigmoid = nn.Sigmoid()
        self.bidirectional = bidirectional
    def forward(self, x):
        out, (h, c) = self.lstm(x) # LxBx512, 1x1x512, 1x1x512
        in_clf = h[0] if not self.bidirectional else h.permute(1, 0, 2).view(1, -1)
        ret = self.clf(in_clf)
        ret = self.sigmoid(ret)
        return ret



