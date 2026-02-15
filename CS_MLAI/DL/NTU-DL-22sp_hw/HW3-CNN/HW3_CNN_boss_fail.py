# Boss Baseline (acc>0.88446): Data augmentation + 残差网络架构+ FocalLoss + Cross Validation + Ensemble + Test Time Augmentation
# 相对于strong baseline，加大了网络的深度和训练次数，并使用Test Time Augmentation（TTA)。
# TTA使用了原本的test_tfm构建的testloader，另外使用train_tfm构建其他5个testloader，6个testloader的权重分别为0.6和0.1，0.1，0.1，0.1，0.1。

import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

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

class FoodDataset(Dataset):

    def __init__(self,path=None,tfm=test_tfm,files=None):
        super(FoodDataset).__init__()
        self.path = path
        if path:
            self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        else:
            self.files = files
        self.transform = tfm
        print('Num of element: ', len(self.files))
  
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
            label = -1 # test has no label
        return im,label

class Residual_Block(nn.Module):
    def __init__(self, ic, oc, stride=1):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oc),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
        self.downsample = None
        if stride != 1 or (ic != oc):
            self.downsample = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=1, stride=stride),
                nn.BatchNorm2d(oc),
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample:
            residual = self.downsample(x)
            
        out += residual
        return self.relu(out)
        
class Classifier(nn.Module):
    def __init__(self, block, num_layers, num_classes=11):
        super().__init__()
        self.preconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.layer0 = self.make_residual(block, 32, 64,  num_layers[0], stride=2)
        self.layer1 = self.make_residual(block, 64, 128, num_layers[1], stride=2)
        self.layer2 = self.make_residual(block, 128, 256, num_layers[2], stride=2)
        self.layer3 = self.make_residual(block, 256, 512, num_layers[3], stride=2)
        
        #self.avgpool = nn.AvgPool2d(2)
        
        self.fc = nn.Sequential(            
            nn.Dropout(0.4),
            nn.Linear(512*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 11),
        )
        
    def make_residual(self, block, ic, oc, num_layer, stride=1):
        layers = []
        layers.append(block(ic, oc, stride))
        for i in range(1, num_layer):
            layers.append(block(oc, oc))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # [3, 128, 128]
        out = self.preconv(x)  # [32, 64, 64]
        out = self.layer0(out) # [64, 32, 32]
        out = self.layer1(out) # [128, 16, 16]
        out = self.layer2(out) # [256, 8, 8]
        out = self.layer3(out) # [512, 4, 4]
        #out = self.avgpool(out) # [512, 2, 2]
        out = self.fc(out.view(out.size(0), -1)) 
        return out

import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1, 1)
        
        log_p = probs.log()
        
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
            
        return loss
    
batch_size = 64
num_layers = [2, 4, 3, 1] # residual number layers
alpha = torch.Tensor([1, 2.3, 0.66, 1, 1.1, 0.75, 2.3, 3.5, 1.1, 0.66, 1.4])

n_epochs = 300
patience = 32 # If no improvement in 'patience' epochs, early stop

k_fold = 4

train_dir = "./food11/training"
val_dir = "./food11/validation"

train_files = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if x.endswith('.jpg')]
val_files = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.endswith('.jpg')]
total_files = train_files + val_files
random.shuffle(total_files)

num = len(total_files) // k_fold

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

test_fold = k_fold

for i in range(test_fold):
    fold = i+1
    print(f'\n\nStarting Fold: {fold} ********************************************')
    model = Classifier(Residual_Block, num_layers).to(device)
    criterion = FocalLoss(11, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=2e-5) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=1)
    stale = 0
    best_acc = 0
    
    val_data = total_files[i*num: (i+1)*num]
    train_data = total_files[:i*num] + total_files[(i+1)*num:]
    
    train_set = FoodDataset(tfm=train_tfm, files=train_data)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    valid_set = FoodDataset(tfm=test_tfm, files=val_data)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    for epoch in range(n_epochs):
        # ---------- Training ----------
        model.train()

        train_loss = []
        train_accs = []
        lr = optimizer.param_groups[0]["lr"]
        
        pbar = tqdm(train_loader)
        pbar.set_description(f'T: {epoch+1:03d}/{n_epochs:03d}')
        for batch in pbar:

            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
    
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)
            pbar.set_postfix({'lr':lr, 'b_loss':loss.item(), 'b_acc':acc.item(),
                    'loss':sum(train_loss)/len(train_loss), 'acc': sum(train_accs).item()/len(train_accs)})
        
        scheduler.step()
        
        model.eval()

        valid_loss = []
        valid_accs = []

        pbar = tqdm(valid_loader)
        pbar.set_description(f'V: {epoch+1:03d}/{n_epochs:03d}')
        for batch in pbar:

            imgs, labels = batch
            #imgs = imgs.half()
    
            with torch.no_grad():
                logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)
            pbar.set_postfix({'v_loss':sum(valid_loss)/len(valid_loss), 
                              'v_acc': sum(valid_accs).item()/len(valid_accs)})

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
    
        if valid_acc > best_acc:
            print(f"Best model found at fold {fold} epoch {epoch+1}, acc={valid_acc:.5f}, saving model")
            torch.save(model.state_dict(), f"Fold_{fold}_best.ckpt")
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break

test_dir = "./food11/test"
test_set = FoodDataset(test_dir, tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

test_dir = "./food11/test"
test_tfms = [test_tfm1, test_tfm2, test_tfm3, test_tfm4, test_tfm5]
test_loaders = []
for i in range(5):
    test_set_i = FoodDataset(test_dir, tfm=train_tfm)
    test_loader_i = DataLoader(test_set_i, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loaders.append(test_loader_i)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

models = []
for i in range(0, 4):
    fold = i + 1
    model_best = Classifier(Residual_Block, num_layers).to(device)
    model_best.load_state_dict(torch.load(f"Fold_{fold}_best.ckpt"))
    model_best.eval()
    models.append(model_best)

preds = [[], [], [], [], [], []] 
with torch.no_grad():
    for data, _ in test_loader:
        batch_preds = [] 
        for model_best in models:
            batch_preds.append(model_best(data.to(device)).cpu().data.numpy())
        batch_preds = sum(batch_preds)
        preds[0].extend(batch_preds.squeeze().tolist())
        
        #batch_label = np.argmax(batch_preds, axis=1)
        #prediction += batch_label.squeeze().tolist()
        
    for i, loader in enumerate(test_loaders):
        for data, _ in loader:
            batch_preds = []
            for model_best in models:
                batch_preds.append(model_best(data.to(device)).cpu().data.numpy())
            batch_preds = sum(batch_preds)
            preds[i+1].extend(batch_preds.squeeze().tolist())

preds = np.array(preds)
print(preds.shape)
preds = 0.6* preds[0] + 0.1 * preds[1] + 0.1 * preds[2] + 0.1 * preds[3] + 0.1 * preds[4] + 0.1 * preds[5]
print(preds.shape)
prediction = np.argmax(preds, axis=1)

#create test csv
import pandas as pd
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)