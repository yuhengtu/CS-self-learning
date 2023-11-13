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

# # **一些重要的函数**
# 绘制计算图
def model_plot(model_class, input_sample):
    clf = model_class()
    y = clf(input_sample) 
    clf_view = make_dot(y, params=dict(list(clf.named_parameters()) + [('x', input_sample)]))
    return clf_view

# 设置全局的随机种子
def all_seed(seed = 6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed) 
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed) 
    # python 全局
    os.environ['PYTHONHASHSEED'] = str(seed) 
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')
    
# 快速观察训练集中的9张照片
def quick_observe(train_dir_root):
    pics_path = [os.path.join(train_dir_root, i) for i in os.listdir(train_dir_root)]
    labels = [i.split('_')[0] for i in os.listdir(train_dir_root)]
    idxs = np.arange(len(labels))
    sample_idx = np.random.choice(idxs, size=9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    for idx_, i in enumerate(sample_idx):
        row = idx_ // 3
        col = idx_ % 3
        img=Image.open(pics_path[i])
        axes[row, col].imshow(img)
        c = labels[i]
        axes[row, col].set_title(f'class_{c}')

    plt.show()

# 数据准备( 数据转换、数据扩增 )
train_dir_root = './food11/training'
quick_observe(train_dir_root)

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

x = torch.randn(1, 3, 128, 128).requires_grad_(True)
model_plot(Classifier, x)

# 训练部分
def trainer(train_loader, valid_loader, model, config, device, rest_net_flag=False):

    # 对于分类任务, 我们常用cross-entropy评估模型表现.
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']) 
    # 模型存储位置
    save_path =  config['save_path'] if rest_net_flag else  config['resnet_save_path']

    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_accs = []
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()             
            x, y = x.to(device), y.to(device)  
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                   
            # 稳定训练的技巧，梯度裁剪；梯度的范数是一个用于衡量梯度向量大小的值。
            # 如果梯度的范数大于 max_norm，那么梯度将被按比例缩小，以使其范数不超过 max_norm
            if config['clip_flag']:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()    
            step += 1
            acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
            l_ = loss.detach().item()
            loss_record.append(l_)
            train_accs.append(acc.detach().item())
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': f'{l_:.5f}', 'acc': f'{acc:.5f}'})
        
        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('ACC/train', mean_train_acc, step)
        
        model.eval() # 设置模型为评估模式
        loss_record = []
        test_accs = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()

            loss_record.append(loss.item())
            test_accs.append(acc.detach().item())
            
        mean_valid_acc = sum(test_accs) / len(test_accs)
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f},acc: {mean_train_acc:.4f} Valid loss: {mean_valid_loss:.4f},acc: {mean_valid_acc:.4f} ')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('ACC/valid', mean_valid_acc, step)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), save_path) # 保存最优模型
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

# 超参设置
# `config` 包含所有训练需要的超参数（便于后续的调参），以及模型需要存储的位置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 6666,
    'dataset_dir': "./food11",
    'n_epochs': 10,      
    'batch_size': 64, 
    'learning_rate': 0.0003,           
    'weight_decay':1e-5,
    'early_stop': 300,
    'clip_flag': True, 
    'save_path': './models/model.ckpt',
    'resnet_save_path': './models/resnet_model.ckpt'
}
print(device)
all_seed(config['seed'])

# 导入数据集
# 1. FoodDataset： 用文件路径，transform方法构建数据集
# 2. DataLoader： 使用Pytorch中Dataloader类按照Batch将数据集加载
_dataset_dir = config['dataset_dir']

train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

# 测试级保证输出顺序一致
test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

# 测试集数据扩增
test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=train_tfm)
test_loader_extra1 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=train_tfm)
test_loader_extra2 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=train_tfm)
test_loader_extra3 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)


# 开始训练，Classifier
model = Classifier().to(device)
trainer(train_loader, valid_loader, model, config, device)

# 测试并生成预测结果的csv
model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(config['save_path']))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

# soft vote（软投票）预测
# - 用扩增数据进行四次测试接预测，然后用soft_vote（软投票）的方法进行预测
test_loaders = [test_loader_extra1, test_loader_extra2, test_loader_extra3, test_loader]
loader_nums = len(test_loaders)
# 存储每个dataloader预测结果，一个dataloader一个数组
loader_pred_list = []
for idx, d_loader in enumerate(test_loaders):
    # 存储一个dataloader的预测结果,  一个batch一个是数组
    pred_arr_list = [] 
    with torch.no_grad():
        tq_bar = tqdm(d_loader)
        tq_bar.set_description(f"[ DataLoader {idx+1}/{loader_nums} ]")
        for data,_ in tq_bar:
            test_pred = model_best(data.to(device))
            logit_pred = test_pred.cpu().data.numpy()
            pred_arr_list.append(logit_pred)
        # 将每个batch的预测结果合并成一个数组
        loader_pred_list.append( np.concatenate(pred_arr_list, axis=0) )

# 将预测结果合并
pred_arr = np.zeros(loader_pred_list[0].shape)
for pred_arr_t in loader_pred_list:
    pred_arr += pred_arr_t

soft_vote_prediction = np.argmax(0.5 * pred_arr / len(loader_pred_list) + 0.5 * loader_pred_list[-1], axis=1)
df = pd.DataFrame()
# 保证ID为四位数（前面填充0）
df["Id"] = [str(i).zfill(4) for i in range(1, len(test_set)+1)]
df["Category"] = soft_vote_prediction
df.to_csv("submission.csv",index = False)


# # 训练残差网络（Residual Net）
# resNet = resNet.to(device)
# trainer(train_loader, valid_loader, resNet, config, device)

# model_best = resNet.to(device)
# model_best.load_state_dict(torch.load(config['resnet_save_path']))
# model_best.eval()
# test_loaders = [test_loader_extra1, test_loader_extra2, test_loader_extra3, test_loader]
# loader_nums = len(test_loaders)
# # 存储每个dataloader预测结果，一个dataloader一个数组
# loader_pred_list = []
# for idx, d_loader in enumerate(test_loaders):
#     # 存储一个dataloader的预测结果,  一个batch一个数组
#     pred_arr_list = [] 
#     with torch.no_grad():
#         tq_bar = tqdm(d_loader)
#         tq_bar.set_description(f"[ DataLoader {idx+1}/{loader_nums} ]")
#         for data,_ in tq_bar:
#             test_pred = model_best(data.to(device))
#             logit_pred = test_pred.cpu().data.numpy()
#             pred_arr_list.append(logit_pred)
#         # 将每个batch的预测结果合并成一个数组
#         loader_pred_list.append( np.concatenate(pred_arr_list, axis=0) )

# # 将预测结果合并
# pred_arr = np.zeros(loader_pred_list[0].shape)
# for pred_arr_t in loader_pred_list:
#     pred_arr += pred_arr_t

# soft_vote_prediction = np.argmax(0.5 * pred_arr / len(loader_pred_list) + 0.5 * loader_pred_list[-1], axis=1)

# df = pd.DataFrame()
# # 保证ID为四位数（前面填充0）
# df["Id"] = [str(i).zfill(4) for i in range(1, len(test_set)+1)]
# df["Category"] = soft_vote_prediction
# df.to_csv("submission.csv",index = False)
