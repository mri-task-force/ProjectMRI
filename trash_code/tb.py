import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models,datasets
from tensorboardX import SummaryWriter


cat_img = Image.open('./n7041.jpg')

transform_224 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
cat_img_224=transform_224(cat_img)

# 将图片展示在tebsorboard中：
writer = SummaryWriter(log_dir='./tblog', comment='cat image') # 这里的logs要与--logdir的参数一样
writer.add_image("cat",cat_img_224)
writer.close()# 执行close立即刷新，否则将每120秒自动刷新


# 更新损失函数
# 更新损失函数和训练批次我们与visdom一样使用模拟展示，这里用到的是tensorboard的SCALAR页面
x = torch.FloatTensor([100])
y = torch.FloatTensor([500])

for epoch in range(100):
    x /= 1.5
    y /= 1.5
    loss = y - x
    with SummaryWriter(log_dir='./tblog', comment='train') as writer: #with语法:自动调用close方法
        writer.add_histogram('his/x', x, epoch)
        writer.add_histogram('his/y', y, epoch)
        writer.add_scalar('data/x', x, epoch)
        writer.add_scalar('data/y', y, epoch)
        writer.add_scalar('data/loss', loss, epoch)
        writer.add_scalars('data/data_group', {'x': x,
                                               'y': y,
                                               'loss': loss}, epoch)
