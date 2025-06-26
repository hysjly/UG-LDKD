import torch
from .Tools import DeviceInitialization, Sampling
from .Model import Encoder
from ....registry import MODELS

@MODELS.register_module()
class GenerateYPv1:
    def __init__(self, class_num=80, img_height=128, img_width=128, device='cuda:0',
                 model_path='/home/jz207/workspace/liulb/MMDetection/Encoder_coco_PreTrain.pth'):
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

    def generate_yp(self, predictions):
        """
        生成伪造标签 yp。
        :param predictions: 目标检测网络的预测结果，大小为 (batch_size, height, width)。
        :return: 生成的伪造标签 yp，大小与输入相同。
        """
        predictions = predictions.to(self.device)
        batch_size = predictions.size(0)
        fakes = torch.zeros((batch_size, self.class_num, self.img_height, self.img_width), device=self.device)

        _,preIdx = torch.max(predictions.detach(),dim=1)
        # 根据预测结果生成初始的 fakes
        for ii in range(self.class_num):
            fakes[:, ii] = torch.where(preIdx == ii, torch.ones_like(preIdx),
                                       torch.zeros_like(preIdx))

        # 将 fakes 调整到 [-1, 1] 范围
        fakes = 2 * (fakes - 0.5)

        # 使用 Sampling 函数对 fakes 进行更新，生成最终的 yp
        yp = Sampling(self.Coder, fakes, step=10, step_size=10)

        # 将 yp 调整回 [0, 1] 范围
        yp = (yp / 2) + 0.5

        return yp



# if __name__ == "__main__":
#     # 假设有一个目标检测网络的预测结果 predictions
#     predictions = torch.randint(0, 80, (16, 128, 128))  # 示例预测结果，大小为 (batch_size, height, width)
#
#     # 初始化生成伪造标签模块
#     yp_generator = GenerateYP(class_num=80, img_height=128, img_width=128, device='cuda:0',
#                               model_path='/home/jz207/workspace/liulb/MMDetection/Encoder_coco_PreTrain.pth')
#
#     # 生成伪造标签 yp
#     yp = yp_generator.generate_yp(predictions)
#     print(yp.shape)  # 输出生成的 yp 的形状
