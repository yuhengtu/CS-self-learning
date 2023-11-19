# concat_nframes参数设置+batch_size+网络架构改变+余弦退火学习率。
# concat_nframes参数设置为19。batch_size设置为2048。设置三个宽度为1024的隐藏层。
# 利用余弦退火学习率，用李宏毅老师的话，这都是古圣先贤的意思，用就对了。

# 作业 2: phoneme音位分类 ，可以理解为音标
# 相关资料：
# * Slides地址: https://docs.google.com/presentation/d/1v6HkBWiJb8WNDcJ9_-2kwVstxUWml87b9CnA16Gdoio/edit?usp=sharing
# * Kaggle地址: https://www.kaggle.com/c/ml2022spring-hw2

# 前置知识：
# - 音位： 
# 例如：Machine Learning → M AH SH IH N L ER N IH NG M M M AH AH SH SH IH IH IH N N N N ... Machine
# - MFCC：  
# 在语音识别和话者识别方面，最常用梅尔倒谱系数

# MFCC将一段音频转化为一个39维度的特征放中间，前向与后向音频各取5个做辅助判断，11*39维的一个向量，判断中间音位label
# 想要深入了解实现过程的可以查看HW2讲解或下列链接：  
# [Prof. Hung-Yi Lee[2020Spring DLHLP] Speech Recognition](https://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/ASR%20(v12).pdf)  
# [ Prof. Lin-Shan Lee’s[Introduction to Digital Speech Processing]Chap.7](http://ocw.aca.ntu.edu.tw/ntu-ocw/ocw/cou/104S204)

# #  数据集下载
# 如果下列命令无法下载，可以到下列地址下载数据
# - Kaggle下载数据:  [Kaggle: ml2022spring-hw2](https://www.kaggle.com/competitions/ml2022spring-hw2)
# - 百度云下载数据: [云盘(提取码：05zc)](https://pan.baidu.com/s/198xn8Lk9MjvUsq866mZuuw)
# pt文件可以使用torch.load方法导入

# 下载链接
# get_ipython().system('wget -O libriphone.zip "https://github.com/xraychen/shiny-robot/releases/download/v1.0/libriphone.zip"')

# 下列数据获取方式需要依靠gdown
# 备用链接 0
# !pip install --upgrade gdown
# !gdown --id '1o6Ag-G3qItSmYhTheX6DYiuyNzWyHyTc' --output libriphone.zip
# 备用链接 1
# !pip install --upgrade gdown
# !gdown --id '1R1uQYi4QpX0tBfUWt2mbZcncdBsJkxeW' --output libriphone.zip
# 备用链接 2
# !wget -O libriphone.zip "https://www.dropbox.com/s/wqww8c5dbrl2ka9/libriphone.zip?dl=1"
# 备用链接 3
# !wget -O libriphone.zip "https://www.dropbox.com/s/p2ljbtb2bam13in/libriphone.zip?dl=1"

# get_ipython().system('unzip -q libriphone.zip')
# get_ipython().system('ls libriphone')

# ## 准备数据
# Helper函数用于预处理来自每个话语的原始MFCC特征的训练数据
# concat_fatte函数连接过去和未来的k帧（总共2k+1＝n帧），我们预测中心帧。

import os
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

# 将前后的特征联系在一起，如concat_n = 11 则前后都接 5
def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n 必须是奇数
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

# 数据预处理函数
def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: 预先计算，不需要更改
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # 分割训练和验证数据
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X
# 返回的X代表数据的维度，如果不链接则为39 如果链接即为n*39 n为连接的特征总数,y为标签

# ## 定义数据集
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# 初始化，_getitem__（获取一个元素）以及__len__（获取数据长度）方法
class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


# # 神经网络模型
# TODO: 使用近似相同数量的参数实现2个模型，
# （A）一个更窄和更深（例如hidden_layers=6，hidden_dim＝1024）
# （B）另一个更宽和更浅（例如 hidden_layers＝2、hidden_dim＝1700）
#  报告两种模型的训练/验证精度

# TODO:  添加dropout层，并报告dropout率分别等于（A）0.25/（B）0.5/（C）0.75的训练/验证准确性

# Dropout层是一种可以用于减少神经网络过拟合的结构
# 在进行第一个batch的训练时，有以下步骤：
# * 设定每一个神经网络层进行dropout的概率
# * 根据相应的概率拿掉一部分的神经元，然后开始训练，更新没有被拿掉神经元以及权重的参数，将其保留

# PS: 上面的两个TODO是可以改进其他部分来提高你的成绩的方法。  
# 如下的策略是助教给出的几个优化方式：  
# ● (1%) Simple baseline: 0.45797 (sample code)  
# ● (1%) Medium baseline: 0.69747 (concat n frames, add layers)  
# ● (1%) Strong baseline: 0.75028 (concat n, batchnorm, dropout, add layers)  
# ● (1%) Boss baseline: 0.82324 (sequence-labeling(using RNN))  
# 对于boss baseline，您可以参考RNN之前的课程记录

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            # nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.35),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)], # *[]将循环得到的解压
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# ## 超参数定义
# TODO:  可以考虑进一步优化超参数来提高准确率
# data prarameters
# 用于数据处理时的参数
concat_nframes = 19             # 要连接的帧数,n必须为奇数（总共2k+1=n帧）
train_ratio = 0.8               # 用于训练的数据比率，其余数据将用于验证
# training parameters
# 训练过程中的参数
seed = 0                        # 随机种子
batch_size = 2048                # 批次数目
num_epoch = 50                   # 训练epoch数
early_stopping = 8
learning_rate = 0.0001          # 学习率
model_path = './model.ckpt'     # 选择保存检查点的路径（即下文调用保存模型函数的保存位置）
# model parameters
# 模型参数
input_dim = 39 * concat_nframes # 模型的输入维度，不应更改该值，这个值由上面的拼接函数决定
hidden_layers = 3               # hidden_layer层的数量
hidden_dim = 1024                # 隐藏维度

# ## 准备数据与模型
# 引入gc模块进行垃圾回收
import gc

# 预处理数据
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

# 将数据导入
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# 删除原始数据以节省内存
del train_X, train_y, val_X, val_y
gc.collect()

# 利用dataloader加载数据
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 检查当前是否有可用的GPU 否则使用CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# 固定随机种子
def same_seeds(seed): # 固定随机种子（CPU）
    torch.manual_seed(seed) # 固定随机种子（GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True # 固定网络结构

# 固定随机种子
same_seeds(seed)

# 创建模型、定义损失函数和优化器
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate*5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                        T_0=8, T_mult=2, eta_min=learning_rate/2)

import torchsummary
torchsummary.summary(model, input_size=(input_dim,))

# ## 训练模型
best_acc = 0.0
early_stop_count = 0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # training
    model.train() # set the model to training mode
    pbar = tqdm(train_loader, ncols=110)
    pbar.set_description(f'T: {epoch+1}/{num_epoch}')
    samples = 0
    for i, batch in enumerate(pbar):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(features) 
        
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()
       
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        correct = (train_pred.detach() == labels.detach()).sum().item()
        train_acc += correct
        samples += labels.size(0)
        train_loss += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({'lr':lr, 'batch acc':correct/labels.size(0),
                          'acc':train_acc/samples, 'loss':train_loss/(i+1)})
    scheduler.step()
    pbar.close()
    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            pbar = tqdm(val_loader, ncols=110)
            pbar.set_description(f'V: {epoch+1}/{num_epoch}')
            samples = 0
            for i, batch in enumerate(pbar):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                
                loss = criterion(outputs, labels) 
                
                _, val_pred = torch.max(outputs, 1) #get the index of the class with the highest probability
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                samples += labels.size(0)
                val_loss += loss.item()
                pbar.set_postfix({'val acc':val_acc/samples ,'val loss':val_loss/(i+1)})
            pbar.close()
            # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping:
                print(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                break
    else:
        print(f'[{epoch+1:03d}/{num_epoch:03d}] Acc: {acc:3.6f} Loss: {loss:3.6f}')

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

del train_loader, val_loader
gc.collect()

# ## 测试
# 创建测试数据集，并从保存的检查点加载模型。
# 载入数据
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 加载已经训练好的模型
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # 获得概率最高的类的索引
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

# 将预测结果写入CSV文件。
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))
