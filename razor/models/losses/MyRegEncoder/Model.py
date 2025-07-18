import torch
import torch.nn as nn
from ....registry import MODELS


@MODELS.register_module()
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

@MODELS.register_module()
class Encoder(nn.Module):
    def __init__(self, num_obj):
        super(Encoder, self).__init__()
        num_obj = num_obj
        self.net = nn.Sequential(nn.Conv2d(num_obj, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                 Swish(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
                                 Swish(),
            			 ResBlock(64), 
                                 nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                 Swish(),
            			 ResBlock(128), 
                                 nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
                                 Swish(),
            			 ResBlock(128), 
                                 SpatialAttention(128), 
                                 nn.Conv2d(128, 256, 3, padding=1),
                                 Swish()
                                 )
        self.feature_proj = nn.Sequential(
            nn.Conv2d(256, d, 1),  
            Swish(),
            AdaptivePool2d((16, 16))  
         )
        self.energy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1024),
            Swish(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.net(x).squeeze(dim=-1)
	x = self.feature_proj(x)
        x = self.energy_head(x)
        return x
