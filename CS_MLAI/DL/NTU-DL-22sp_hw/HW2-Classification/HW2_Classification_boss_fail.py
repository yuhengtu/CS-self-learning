# concat_nframes参数设置+batch_size+BiLSTM-CRF网络架构+余弦退火学习率。
# BiLSTM-CRF网络结构是序列标注中的经典模型，该结构可以综合考虑lstm的输出结果和标签顺序分布，
# 可参考pytorch官方样例：https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html，或者使用pytorchcrf库。
# 在使用BiLSTM-CRF架构的时候，需要修改数据的产生方式，之前每个sample的feature和label size分别是（batch_size, 39*concat_nframes)和（batch_size，)，现在是（batch_size，concat_nframes, 39)和（batch_size，concat_nframes），最后做推理的时候也需要相应的改变。
# 同时因为BiLSTM和CRF的收敛速度一般是不一样的，CRF的学习率要设置的大些，运行代码提交后，分数是：0.79449，还没到boss baseline，想得到更好的结果需要进行精细调参，另外可以尝试Transfromer-CRF或Bert-CRF结构。

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import numpy as np 
import pandas as pd 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pass
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import random
import pandas as pd
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

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
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

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
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
      y = torch.empty(max_len, concat_nframes, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname]).unsqueeze(1)
          label = concat_feat(label, concat_nframes)

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

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx].view(-1, 39), self.label[idx]
        else:
            return self.data[idx].view(-1, 39)

    def __len__(self):
        return len(self.data)

# get_ipython().system(' pip install pytorch-crf')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

#class BiLSTM_CRF(nn.Module):
#
#    def __init__(self, class_size=41, input_dim=39, hidden_dim=128,dropout=0.5):
#        super().__init__()
#        self.input_dim = input_dim
#        self.hidden_dim = hidden_dim
#        self.class_size = class_size
#
#        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, dropout=dropout,
#                            num_layers=3, bidirectional=True, batch_first=True)
#        self.hidden2tag = nn.Sequential(
#            nn.Dropout(dropout),
#            nn.Linear(hidden_dim, class_size)
#        )
#
#        self.crf = CRF(self.class_size, batch_first=True)
#        
#    def get_emissions(self, x):
#        feats, _ = self.lstm(x)
#        return self.hidden2tag(feats)
#
#    def likelihood(self, x, y):
#        emissions = self.get_emissions(x)
#        loss = self.crf(emissions, y)
#        return loss
#
#    def forward(self, x):  # dont confuse this with _forward_alg above.
#        emissions = self.get_emissions(x)
#        seqs = self.crf.decode(emissions)
#        return torch.LongTensor(seqs)

class BiLSTM(nn.Module):
    def __init__(self, class_size=41, input_dim=39, hidden_dim=192, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.class_size = class_size
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, dropout=dropout,
                            num_layers=3, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_size)
        )
        
    def forward(self, x):
        feats, _ = self.lstm(x)
        return self.hidden2tag(feats)
    
class Crf(nn.Module):
    def __init__(self, class_size=41):
        super().__init__()
        self.class_size = class_size
        self.crf = CRF(self.class_size, batch_first=True)
        
    def likelihood(self, x, y):
        return self.crf(x, y)
    
    def forward(self, x):
        return torch.LongTensor(self.crf.decode(x))

# data prarameters
concat_nframes = 21          # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
mid = concat_nframes//2
train_ratio = 0.95               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 0                        # random seed
batch_size = 2048                # batch size
num_epoch = 50                   # the number of training epoch
early_stopping = 8
learning_rate = 0.0001            #learning rate
model1_path = './model1.ckpt'     # the path where the checkpoint will be saved
model2_path = './model2.ckpt'
# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 3              # the number of hidden layers
hidden_dim = 1024              # the hidden dim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

import numpy as np

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import gc

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', 
                                   phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', 
                               phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# fix random seed
same_seeds(seed)

# create model, define a loss function, and optimizer
#model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
#model = BiLSTM_CRF().to(device)
bilstm = BiLSTM().to(device)
crf = Crf().to(device)
optimizer1 = torch.optim.AdamW(bilstm.parameters(), lr=learning_rate*20, weight_decay=0.015)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, 
                                        T_0=8, T_mult=2, eta_min=learning_rate/2)

optimizer2  = torch.optim.AdamW(crf.parameters(), lr=learning_rate*500, weight_decay=1e-8)

total_num = 0
for i, param in enumerate(bilstm.parameters()):
    print('Layer:', i, '    parameter num:',param.numel(), '    shape:', param.shape)
    total_num += param.numel()

print(f'Total parameters num: {total_num}')

total_num = 0
for i, param in enumerate(crf.parameters()):
    print('Layer:', i, '    parameter num:',param.numel(), '    shape:', param.shape)
    total_num += param.numel()

print(f'Total parameters num: {total_num}')

best_acc = 0.0
early_stop_count = 0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    train_item =0
    # training
    bilstm.train() # set the model to training mode
    crf.train()
    pbar = tqdm(train_loader, ncols=110)
    pbar.set_description(f'T: {epoch+1}/{num_epoch}')
    samples = 0
    for i, batch in enumerate(pbar):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        
        optimizer1.zero_grad() 
        optimizer2.zero_grad()
        loss = -crf.likelihood(bilstm(features), labels)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(bilstm.parameters(), max_norm=50)
        optimizer1.step()
        optimizer2.step()
        
        train_loss += loss.item()
        train_item += labels.size(0)
        
        lr1 = optimizer1.param_groups[0]["lr"]
        lr2 = optimizer2.param_groups[0]["lr"]
        pbar.set_postfix({'lr1':lr1, 'lr2':lr2, 'loss':train_loss/train_item})
    scheduler.step()
    pbar.close()
    # validation
    if len(val_set) > 0:
        bilstm.eval() # set the model to evaluation mode
        crf.eval()
        with torch.no_grad():
            pbar = tqdm(val_loader, ncols=110)
            pbar.set_description(f'V: {epoch+1}/{num_epoch}')
            samples = 0
            for i, batch in enumerate(pbar):
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = crf(bilstm(features))                
                val_acc += (outputs[:, mid] == labels[:, mid].cpu()).sum().item()
                samples += labels.size(0)
                pbar.set_postfix({'val acc':val_acc/samples})
            pbar.close()
            # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(bilstm.state_dict(), model1_path)
            torch.save(crf.state_dict(), model2_path)
            print('saving model with acc {:.3f}'.format(best_acc/(len(val_set))))
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stopping:
                print(f"Epoch: {epoch + 1}, model not improving, early stopping.")
                break

del train_loader, val_loader
gc.collect()

# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat',
                         phone_path='./libriphone', concat_nframes=concat_nframes)

test_set = LibriDataset(test_X)

import gc
del test_X
gc.collect()

# get dataloader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
#model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
bilstm = BiLSTM().to(device)
bilstm.load_state_dict(torch.load(model1_path))

crf = Crf().to(device)
crf.load_state_dict(torch.load(model2_path))

pred = np.array([], dtype=np.int32)

bilstm.eval()
crf.eval()
with torch.no_grad():
    for features in tqdm(test_loader):
        features = features.to(device)
        outputs = crf(bilstm(features))
        pred = np.concatenate((pred, outputs.detach().cpu()[:, mid]), axis=0)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))

