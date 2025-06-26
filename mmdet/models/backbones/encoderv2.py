# my_encoder_module/encoderv1.py
import torch
import torch.nn as nn
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmengine.logging import print_log


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


@MODELS.register_module()
class Encoder(nn.Module):
    def __init__(self, num_obj=80, pretrained=None, freeze=False):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_obj, 64, kernel_size=3, stride=1, padding=1, bias=True),
            Swish(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            Swish(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            Swish(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            Swish(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            Swish()
        )

        if pretrained:
            self.load_weights(pretrained)

        # 冻结权重
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def load_weights(self, pretrained_path):
        """加载预训练权重并过滤不需要的层"""
        try:
            state_dict = torch.load(pretrained_path, map_location=self.device)
            if 'Net' in state_dict:
                state_dict = state_dict['Net']

            # 过滤掉不需要的层
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not (k.startswith("net.11") or k.startswith("net.13"))
            }
            missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

            # 记录缺失和意外的权重信息
            if missing_keys:
                print_log(f"Warning: Missing keys in state_dict: {missing_keys}", logger='current')
            if unexpected_keys:
                print_log(f"Warning: Unexpected keys in state_dict: {unexpected_keys}", logger='current')

            print_log(f"预训练权重成功加载到编码器模型（路径: {pretrained_path}）", logger='current')
        except Exception as e:
            print_log(f"加载预训练权重时出错: {e}", logger='current')
            raise e

    def forward(self, x):
        """返回一个包含单个张量的元组"""
        return (self.net(x).to(self.device),)  # 输出格式为 (output_tensor, )