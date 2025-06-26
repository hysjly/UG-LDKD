from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....registry import MODELS
from .base_connector import BaseConnector

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Encoder(nn.Module):
    def __init__(self, num_obj):
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
            Swish(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            Swish(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.net(x).squeeze(dim=-1)
        return x

def align(x, c, h, w):
    x.channels = x.shape[1]
    adjust_channels = nn.Sequential(
        nn.Conv2d(x.channels, c, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True))
    # 2. 调整学生特征的空间尺寸以匹配教师特征
    upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)

    x = adjust_channels(x)
    x = upsample(x)
    return x


@MODELS.register_module()
class FourierTeacherConnector(BaseConnector):
    def __init__(self, teacher_c, teacher_h, teacher_w, init_cfg=None):
        super().__init__(init_cfg)
        self.teacher_c = teacher_c
        self.teacher_h = teacher_h
        self.teacher_w = teacher_w

    def forward_train(self, x):

        x = align(x, self.teacher_c, self.teacher_h, self.teacher_w)
        # 进行FFT
        fft_result = torch.fft.fft2(x)

        # 创建一个高通滤波器掩码
        _, c, h, w = x.shape
        mask = torch.ones_like(fft_result)  # 使用ones_like创建一个全为1的张量
        cutoff_frequency_h = h // 4
        cutoff_frequency_w = w // 4
        mask[:, :, :cutoff_frequency_h, :cutoff_frequency_w] = 0  # 将低频部分设为0，即高频部分
        mask[:, :, -cutoff_frequency_h:, -cutoff_frequency_w:] = 0  # 将低频部分设为0，即高频部分

        # 应用掩码
        fft_result *= mask

        return fft_result  # 输出高频掩码

# 预训练的编码器实例化
num_obj = 1
encoder = Encoder(num_obj)
@MODELS.register_module()
class StudentEncoderConnector(BaseConnector):
    def __init__(self, num_obj,teacher_c, teacher_h, teacher_w, init_cfg=None):
        self.encoder = Encoder(num_obj)  # 实例化 Encoder
        super().__init__(init_cfg)
        self.teacher_c = teacher_c
        self.teacher_h = teacher_h
        self.teacher_w = teacher_w

        # 加载预训练权重
        self.encoder.load_state_dict(torch.load("E:/professional_tool/project/PyCharmProject/MMDetection/Dynamic-Loss-Weighting-main/ModelEncoder_coco_PreTrain.pth"))
        self.encoder.eval()  # 设置为评估模式

    def forward_train(self, x):
        x = align(x, self.teacher_c, self.teacher_h, self.teacher_w)
        absolute_difference = self.process_image(x)
        return absolute_difference

    def process_image(self, input_image):
        # 确保输入图像是浮点型
        if input_image.dtype != torch.float32:
            input_image = input_image.float()

            # 使用编码器进行前向传播
        encoder_output = self.encoder(input_image)  # (batch_size, 1)

        # 获取输入图像的尺寸
        input_size = input_image.shape[2:]  # (height, width)

        # 重新调整编码器输出的大小
        adjusted_output = F.interpolate(encoder_output.unsqueeze(1), size=input_size, mode='bilinear',
                                        align_corners=False)

        # 计算绝对差值
        absolute_difference = torch.abs(input_image - adjusted_output)

        # 将绝对差值转换到频域
        frequency_domain = torch.fft.fft2(absolute_difference)  # 进行二维快速傅里叶变换
        frequency_domain = torch.fft.fftshift(frequency_domain)  # 将零频率分量移到频域中心

        return frequency_domain  # 返回频域值
