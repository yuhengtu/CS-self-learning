# Simple : 0.50099
# Medium : 0.73207 Training Augmentation + Train Longer
# Strong : 0.81872 Training Augmentation + Model Design + Train Looonger (+
# Cross Validation + Ensemble)
# Boss : 0.88446 Training Augmentation + Model Design +Test Time
# Augmentation + Train Looonger (+ Cross Validation + Ensemble) 

# https://pytorch.org/vision/stable/models.html 内置model
# https://github.com/huggingface/pytorch-image-models  前沿CV
# pretrained = False

# 必须做data augmentation（train_tfm函数）
# https://pytorch.org/vision/stable/transforms.html； Coding : fill in train_tfm to gain this effect

# mixup/Test Time Augmentation/Cross-validation 见笔记

# 做了足够的augmentation就别怕用大模型
# 开源model中, model structures with downsampling work better

# on Classification
# ○ Label Smoothing Cross Entropy Loss
# ○ FocalLoss
# on Optimization
# ○ Dropout
# ○ Gradient Accumulation（数据多没法用大batch，用梯度累积模拟大batch）
# ○ BatchNorm
# ○ Image Normalization

# Residual Connection Implementation
# 有Residual_Model，modify the forward part of the model

# # 下载数据
# 如果https://www.dropbox.com/s/6l2vcvxl54b0b6w/food11.zip 下载不了可以用以下途径下载数据
# - [百度网盘下载(提取码：ml22)](https://pan.baidu.com/s/1ZeCvuPh1Oc2EwoVkdXlL0g)
# - [Kaggle Data: ml2022spring-hw3b](https://www.kaggle.com/competitions/ml2022spring-hw3b/data)

# CNN最后一层，print dim再接linear层，这里常有维度报错

# Medium Baseline (acc>0.73207): Data augmentation+dropout。
# 对train_tfm进行修改，添加了常用的augmentation 方法，包括RandomResizedCrop（随机截取并resize）、RandomHorizontalFlip（随机横向翻转）、RandomVerticalFlip（随机竖向翻转）、RandomRoation（随机旋转）、RandomAffine（随机仿射）、RandomGrayscale（随机灰度化）。
# 另外在模型的全连接层的最前面加上dropout层，注意dropout一定放到全连接层，千万不要放到卷积层。

import numpy as np
import pandas as pd
import torch
import math
import os
import random
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchviz import make_dot
# "ConcatDataset" 和"Subset" 会在半监督学习的数据用到
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Transforms
# TODO: 尝试一些图片处理及数据扩增的方法
# TODO: 尝试一些Ensemble方法
# - `Torchvision` 中有很多图片处理的方法，还有很多数据扩增 (data augmentation) 的方法， 可以参考[Pytorch官网](https://pytorch.org/vision/stable/index.html)进一步的了解学习不同的数据处理方法
# - [transforms官方使用示例](https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py)
# 从上图中我们可以看出照片的大小都是不一致的，最小的图片处理也需要将图片切成一样大小的

# 一般情况下，我们不会在验证集和测试集上做数据扩增
# 我们只需要将图片裁剪成同样的大小并转换成Tensor就行
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 当然，我们也可以再测试集中对数据进行扩增（对同样本的不同装换）
#  - 用训练数据的装化方法（train_tfm）去对测试集数据进行转化，产出扩增样本
#  - 对同个照片的不同样本分别进行预测
#  - 最后可以用soft vote / hard vote 等集成方法输出最后的预测

# train_tfm = transforms.Compose([
#     # 图片裁剪 (height = width = 128)
#     transforms.Resize((128, 128)),
#     # TODO:在这部分还可以增加一些图片处理的操作
#     transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
#     # ToTensor() 放在所有处理的最后
#     transforms.ToTensor(),
# ])

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    #transforms.CenterCrop()
    transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0)),
    #transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(180),
    transforms.RandomAffine(30),
    #transforms.RandomInvert(p=0.2),
    #transforms.RandomPosterize(bits=2),
    #transforms.RandomSolarize(threshold=192.0, p=0.2),
    #transforms.RandomEqualize(p=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    #transforms.RandomApply(torch.nn.ModuleList([]))
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
])

# 数据集
# 数据的标签在文件名中, 我们可以在加载的数据将文件名处理成标签
class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):#默认test_tfm
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        # 创建了一个文件列表 self.files，其中包含指定路径中以 ".jpg" 结尾的文件。这些文件名被按字母顺序排序
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        # 打印出路径 path 中的第一个样本文件的信息
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # 测试集没有label
        return im,label
  
# 神经网络模型
# TODO: 尝试一些现成的网络框架
# - 参考官网中的一些模型[torchvision.models](https://pytorch.org/vision/stable/models.html)
#     - AlexNet - ConvNeXt - DenseNet - EfficientNet - EfficientNetV2 - GoogLeNet - Inception V3
#     - MaxVit - MNASNet - MobileNet V2 - MobileNet V3 - RegNet - ResNet - ResNeXt - ShuffleNet V2
#     - SqueezeNet - SwinTransformer - VGG - VisionTransformer - Wide ResNet
from torchvision.models import resnet50
resNet = resnet50(pretrained=False)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64] 
           

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
          
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

batch_size = 64
_dataset_dir = "./food11"
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
# 1. FoodDataset： 用文件路径，transform方法构建数据集
# 2. DataLoader： 使用Pytorch中Dataloader类按照Batch将数据集加载
train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

n_epochs = 100
patience = 8 # If no improvement in 'patience' epochs, early stop

model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5) 

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):
    # ---------- Training ----------
    model.train()

    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()

        # 稳定训练的技巧，梯度裁剪；梯度的范数是一个用于衡量梯度向量大小的值。
        # 如果梯度的范数大于 max_norm，那么梯度将被按比例缩小，以使其范数不超过 max_norm
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    model.eval()

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
        # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Testing and generate prediction CSV
model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()  
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
# 保证ID为四位数（前面填充0）
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)

