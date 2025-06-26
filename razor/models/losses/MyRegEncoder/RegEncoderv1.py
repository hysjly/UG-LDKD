import torch
from torch import nn
from .Tools import DeviceInitialization, Sampling
from .Model import Encoder
from ....registry import MODELS


@MODELS.register_module()
class GenerateYP:
    def __init__(self, class_num=1, img_height=128, img_width=128, device='cuda:0',
                 model_path='/home/jz207/workspace/liulb/MMDetection/Encoder_Kvasir-SEG_PreTrain7679.pth'):
        """
        初始化生成伪造标签模块。
        :param class_num: 类别数量，对应不同的器官数量。
        :param img_height: 输入图像的高度。
        :param img_width: 输入图像的宽度。
        :param device: 使用的设备（例如 'cuda:0' 或 'cpu'）。
        :param model_path: 预训练编码器模型的路径。
        """
        self.class_num = class_num
        self.img_height = img_height
        self.img_width = img_width
        self.device = torch.device(device)

        # 初始化编码器并加载预训练模型
        self.Coder = Encoder(num_obj=class_num)
        Saved = torch.load(model_path)
        self.Coder.load_state_dict(Saved['Net'])
        self.Coder.to(self.device)

        # 添加 1x1 卷积层用于降维通道数
        self.conv1x1 = nn.Conv2d(in_channels=512, out_channels=80, kernel_size=1).to(self.device)

        # 添加上采样层，使得输出尺寸为 (128, 128)
        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)

    def generate_yp(self, predictions):
        """
        生成伪造标签 yp。
        :param predictions: 目标检测网络的预测结果，大小为 (batch_size, height, width)。
        :return: 生成的伪造标签 yp，大小与输入相同。
        """
        predictions = predictions.to(self.device)
        batch_size = predictions.size(0)
        fakes = torch.zeros((batch_size, self.class_num, self.img_height, self.img_width), device=self.device)

        # 应用 1x1 卷积降维通道数
        predictions = self.conv1x1(predictions)  # (batch_size, 80, height, width)

        # 使用上采样层扩展尺寸到 128x128
        predictions = self.upsample(predictions)  # 输出大小为 (batch_size, 80, 128, 128)
        _,preIdx = torch.max(predictions.detach(),dim=1)

        # 根据预测结果生成初始的伪造标签 fakes
        for ii in range(self.class_num):
            fakes[:, ii] = torch.where(preIdx == ii, torch.ones_like(preIdx),
                                       torch.zeros_like(preIdx))

        # 将 fakes 调整到 [-1, 1] 范围
        fakes = 2 * (fakes - 0.5)

        # 使用 Sampling 函数对 fakes 进行更新，生成最终的伪造标签 yp
        yp = Sampling(self.Coder, fakes, step=10, step_size=10)

        # 将 yp 调整回 [0, 1] 范围
        yp = (yp / 2) + 0.5

        return yp
