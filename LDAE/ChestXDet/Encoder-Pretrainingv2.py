import torch
import time
from Tools import DeviceInitialization, DataReading, Sampler
from Datasetv2_2 import DataSet
from Model import Encoder

# torch.backends.cudnn.benchmark = False
data_path = '/home/jz207/workspace/liull/MMDetection/data/'  # 数据路径
json_path = '/home/jz207/workspace/liull/MMDetection/data/ChestXDet/train.json'  # JSON 文件路径
set = 'ChestXDet'  # 数据集类型
modelName = 'Encoder_' + set + '_'
epoch_num = 30  # 训练轮数
batch_size = torch.tensor([1])  # 批大小
img_size = 1024
num_obj = 13  # 类别数量
learning_rate = 3e-4
reg_weight = 0.0001
path = '/home/jz207/workspace/liull/MMDetection/data/'
modelPath = '/home/jz207/workspace/liull/MMDetection/Dynamic_ChestXDet/'
fracTrain = 80

device = DeviceInitialization('cuda:0')
trainIdx, testIdx = DataReading(data_path=path, set=set, fracTrain=fracTrain)
trainSet = DataSet(dataPath=path, set=set, dataName=trainIdx, height=img_size, width=img_size, jsonPath=json_path)
testSet = DataSet(dataPath=path, set=set, dataName=testIdx, height=img_size, width=img_size, jsonPath=json_path)
TrainSet = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size.item(), shuffle=True, num_workers=0,
                                       pin_memory=True)
TestSet = torch.utils.data.DataLoader(dataset=testSet, batch_size=batch_size.item(), shuffle=False, num_workers=0,
                                      pin_memory=True)

Coder = Encoder(num_obj=num_obj)
Coder.to(device)
optim = torch.optim.Adam(Coder.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.995)

# 将 sample_size 转换为 torch.Tensor
sample_size = torch.tensor(64)
Sampler = Sampler(device, Coder, (num_obj, img_size, img_size), sample_size, 40, 10, learning_rate)

torch.cuda.synchronize()
start_time = time.time()
IterNum = 0
tempLossReal = 0
tempLossFake = 0
for epoch in range(epoch_num):
    torch.set_printoptions(precision=4, sci_mode=False)
    for idx, (_, real_imgs) in enumerate(TrainSet, 0):
        # 获取真实和生成的图像
        batch_size[0] = real_imgs.size(0)
        real_imgs = real_imgs.to(device, dtype=torch.float32)
        real_imgs = 2.0 * (real_imgs - 0.5)
        real_imgs += 0.005 * torch.randn_like(real_imgs)
        real_imgs = real_imgs.clamp(min=-1.0, max=1.0)
        fake_imgs = Sampler.sample_new_exmps()
        imgs = torch.cat([real_imgs, fake_imgs], dim=0)

        # 更新编码器
        real_out, fake_out = Coder(imgs).chunk(2, dim=0)
        div = fake_out.mean() - real_out.mean()
        reg = (real_out ** 2).mean() + (fake_out ** 2).mean()
        loss = div + reg_weight * reg
        optim.zero_grad()
        loss.backward()
        optim.step()

        # 输出训练信息
        tempLossReal += real_out.detach().mean()
        tempLossFake += fake_out.detach().mean()
        IterNum += 1

        if IterNum % 100 == 0:
            torch.cuda.synchronize()
            print("Epoch:%02d  ||  Iteration:%04d  ||  RealLoss:%.4f  ||  FakeLoss:%.4f  ||  Time elapsed:%.2f(min)"
                  % (epoch + 1, IterNum, tempLossReal / 100, tempLossFake / 100, (time.time() - start_time) / 60))
            scheduler.step()
            tempLossReal = 0
            tempLossFake = 0

torch.save({'Net': Coder.state_dict(), 'optim': optim.state_dict()}, modelPath + modelName + 'PreTrain.pth')