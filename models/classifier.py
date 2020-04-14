import torch
import torch.nn as nn
from models.modules import ResBlock
import pretrainedmodels

class FlowClassifier(nn.Module):
    def __init__(self, in_channels):
        super(FlowClassifier, self).__init__()
        cfg = [32, 64, 128, 256, 256]
        self.in_channels = in_channels
        
        self.feature_extractor = self._make_layers(cfg)
        self.clf = nn.Linear(8*8*256, 1)
        self.sigmoid = nn.Sigmoid()
    def _make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            conv2d = ResBlock(channels_in=in_channels, num_filters=v, stride=2)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        batch = x.size(0)
        x = self.feature_extractor(x)
        x = x.view(batch, -1)
        x = self.clf(x)
        x = self.sigmoid(x)
        return x