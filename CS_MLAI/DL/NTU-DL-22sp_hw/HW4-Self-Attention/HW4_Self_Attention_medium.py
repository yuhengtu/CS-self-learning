# - 给定音频区分出说话的人

# - Baselines:
#   - Easy: 运行
#   - Medium: 调参，减少复杂度，防止overfit 
#   - Strong: 改变transformer的结构，使用一种transformer变体—— [conformer](https://arxiv.org/abs/2005.08100)  
#   - Boss: 使用 [Self-Attention Pooling](https://arxiv.org/pdf/2008.01077v1.pdf) & [Additive Margin Softmax](https://arxiv.org/pdf/1801.05599.pdf)进一步提升模型表现. 

#   - Data: [link](https://drive.google.com/drive/folders/1vI1kuLB-q1VilIftiwnPOCAeOOFfBZge?usp=sharing)

# 维度更改+使用TransformerEncoder层+Dropout+全连接层修改+Train longer。
# 助教代码中的d_model维度是40，而我们需要预测的n_spks维度是600，维度相差过大，需要将d_model调整为224，经测试d_model过大过小都不好。
# TransformerEncoder使用3层TransformerEncoderLayer，dropout=0.2。
# 全连接层从2层改为1层，并加入BatchNorm。训练step由70000改为100000。
# 运行代码，提交得到kaggle分数：0.74025。

# 模型架构见笔记

# 下载数据
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partaa
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partab
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partac
# !wget https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/Dataset.tar.gz.partad
# !cat Dataset.tar.gz.part* > Dataset.tar.gz
# # unzip the file
# !tar zxvf Dataset.tar.gz
# 如果https://github.com/MachineLearningHW/ML_HW4_Dataset/releases/latest/download/ 下载不了可以用以下途径下载数据
# - [Kaggle Data: ml2022spring-hw4](https://www.kaggle.com/competitions/ml2022spring-hw4/data)

import pandas as pd 
import numpy as np
import random
from pathlib import Path

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, random_split, DataLoader
from torch import functional  as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pad_sequence
import math

import os
import sys
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchviz import make_dot
from torch.optim import AdamW
import csv

# 一些重要的函数
# - all_seed 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(87)

# 数据集
# - 原始数据集 [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).
# - The [license](https://creativecommons.org/licenses/by/4.0/) and [complete version](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/files/license.txt) of Voxceleb2.
# - 我们从Voxceleb2数据集中随机抽取600个演讲者 
# - 将数据原始波形转换为mel谱图
# 
# - 文件夹的结构如下:
#   - data directory   
#   |---- metadata.json    
#   |---- testdata.json     
#   |---- mapping.json     
#   |---- uttr-{random string}.pt   
# 
# - metadata.json中的信息，见ppt图
#   - "n_mels": 40， mel图谱的维度.
#   - "speakers": 
#     - id...
#     - value: "feature_path"：uttr-{...} 
            #  "mel_len"-特征的长度, 各不相同
# 为了更加高效, 我们在训练的时候将mel图谱分割成一定的长度。
 
class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len
        
        # 加载演讲者和id编码的映射.
        mapping_path = Path(data_dir) / 'mapping.json' # mapping.json文件的路径
        mapping = json.load(mapping_path.open()) # 加载JSON文件，解析为Python字典
        self.speaker2id = mapping['speaker2id'] # 从字典中提取了一个键为'speaker2id'的项，对应ID值

        # 加载训练数据的源数据(特征文件， 演讲者)
        metadata_path = Path(data_dir) / 'metadata.json'
        metadata = json.load(metadata_path.open())['speakers'] # 从字典中提取了一个键为'speakers'的项。这个项应该包含演讲者的特征文件
        
        # 获取总演讲者数
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker, utt in metadata.items(): # speaker是演讲者的标识符，utt是与该演讲者相关的一组语音数据
            for utt_i in utt: #每个具体的语音数据项
                self.data.append([utt_i['feature_path'], self.speaker2id[speaker]])
                # self.data是一个列表，每个元素包含特征文件路径和演讲者ID

    def __len__(self):
	    return len(self.data)
 
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # 载入经过预处理的mel图谱特征(mel-spectrogram)
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        # 分割 mel-specrogram
        if len(mel) > self.segment_len:
            # 开始的位置为随机
            start = random.randint(0, len(mel) - self.segment_len)
            # 切分语音
            mel = torch.FloatTensor(mel[start: start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel) # 未截取，长度不一致？
        # 将speaker 转成long格式便于后续计算loss
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
  
    def get_speaker_number(self):
        return self.speaker_num

# 导入数据集
# - 将数据集分割成训练集(90%)和验证集(10%).
# - 创建dataloader用于模型训练.
# - 用`pad_sequence`方法将一个batch中的数据都扩展成一样的长度(`collate_batch`)  
# Example:
# import torch
# from torch.nn.utils.rnn import pad_sequence
# # 假设你有一个batch的数据
# a = torch.ones(25, 40)
# b = torch.ones(22, 40)
# c = torch.ones(15, 40)
# # 使用pad_sequence将数据扩展成一样的长度
# padded_batch = pad_sequence([a, b, c], batch_first=True)
# # 打印结果
# print(padded_batch.size())
# # torch.Size([3, 25, 40])

def collate_batch(batch):
    # 将一个batch中的数据合并
    mel, speaker = zip(*batch)
    # 为了保持一个batch内的长度都是一样的所有需要进行padding, 同时设置batch的维度是最前面的一维
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) 一个很小的值作为填充
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
	"""Generate dataloader"""
	dataset = myDataset(data_dir)
	speaker_num = dataset.get_speaker_number()
  # 将数据拆分成训练集和验证集
	trainlen = int(0.9 * len(dataset))
	lengths = [trainlen, len(dataset) - trainlen]
	trainset, validset = random_split(dataset, lengths)

	train_loader = DataLoader(
		trainset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=n_workers,# 数据加载过程中使用多少个子进程
		pin_memory=True,
		collate_fn=collate_batch, # 用于处理batch中的样本的函数
	)
	valid_loader = DataLoader(
		validset,
		batch_size=batch_size,
		num_workers=n_workers,
		drop_last=True,
		pin_memory=True,
		collate_fn=collate_batch,
	)

	return train_loader, valid_loader, speaker_num

# Transformer模型
# TODO: encode改用Conformer 
# TODO: 增加Self-Attention Pooling Layer
# - 可以参考[https://github.com/sooftware/conformer](https://github.com/sooftware/conformerhttps://github.com/sooftware/conformer)
# - self-attetion & multi-self-attention & transformer block可以看李老师的视频
#     - [B站视频 第五讲 Transformer-2](https://www.bilibili.com/video/BV1m3411p7wD?p=33&vd_source=f209dda877a0d7be7d5309f93b340d6f)

# # Model
# - TransformerEncoderLayer:
#   - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
#   - Parameters:
#     - d_model: the number of expected features of the input (required).
# 
#     - nhead: the number of heads of the multiheadattention models (required).
# 
#     - dim_feedforward: the dimension of the feedforward network model (default=2048).
# 
#     - dropout: the dropout value (default=0.1).
# 
#     - activation: the activation function of intermediate layer, relu or gelu (default=relu).
# 
# - TransformerEncoder:
#   - TransformerEncoder is a stack of N transformer encoder layers
#   - Parameters:
#     - encoder_layer: an instance of the TransformerEncoderLayer() class (required).
# 
#     - num_layers: the number of sub-encoder-layers in the encoder (required).
# 
#     - norm: the layer normalization component (optional).

class Classifier(nn.Module):
	def __init__(self, d_model=224, n_spks=600, dropout=0.2):
		super().__init__()
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
#   TODO:
        #   尝试改变Transformer， 改成Conformer.
        #   https://arxiv.org/abs/2005.08100 

        # - TransformerEncoderLayer:
        #   - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
        #   - Parameters:
        #     - d_model: the number of expected features of the input (required).
        #     - nhead: the number of heads of the multiheadattention models (required).
        #     - dim_feedforward: the dimension of the feedforward network model (default=2048).
        #     - dropout: the dropout value (default=0.1).
        #     - activation: the activation function of intermediate layer, relu or gelu (default=relu).

		self.encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model, # self_attn [Q, K, V] shape=(d_model*3, d_model)
			dim_feedforward=d_model*2, #Transformer 的 Encoder Layer 中前馈神经网络的隐藏层大小
			nhead=2, # 自注意力机制中的头数（number of heads）。
			dropout=dropout
			# batch_first=True, # 如果设置为 True，则输入的数据形状为 (batch_size, seq_len, feature_dim)；如果设置为 False，则形状为 (seq_len, batch_size, feature_dim)。
      # activation='gelu' # GELU（Gaussian Error Linear Unit）激活函数。
			)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.BatchNorm1d(d_model),
			#nn.Linear(d_model, d_model),
			#nn.ReLU(),
      # nn.Dropout(dropout),
			nn.Linear(d_model, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# out: (length, batch size, d_model)
		out = out.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		out = self.encoder(out)
		# out: (batch size, length, d_model)
		out = out.transpose(0, 1) 
		# mean pooling
		stats = out.mean(dim=1)

		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out


# 学习率设置
# - 对于transformer结构, 学习率的设计和CNN有所不同
# - 一些相关工作表明在训练前期逐步增加学习率（Warm up）有利于模型训练transformer.
# - 按照`plot_lr`设计一个Warm up的学习变化架构
#   - 设置学习率在 0到优化器设置的学习率的区间
#   - 在初期（Warmup period）学习率从零增长到0 to 初始学习率

def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
    """
    创建一个学习率变化策略,
    学习率跟随cosine值变化,
    在warm up时间段内变化区间在:
        0 -> 优化器设置的学习率 .
    Args:
        opt (Optimizer): 优化器类
        num_warmup_steps (int): 多少步增加一下lr
        num_training_steps (int): 总训练步骤
        num_cycles (float, optional): 变化周期. 默认为 0.5.
        last_epoch (int, optional):在中断训练后从先前的epoch继续，而不是从头开始。Defaults to -1.
    Return:
		`torch.optim.lr_scheduler.LambdaLR`对象 with the appropriate schedule.
    """
    def lr_lambda(current_step):
		# Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence衰减
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def model_fn(batch, model, criterion, device):
	"""Forward a batch through the model."""

	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)

	outs = model(mels)

	loss = criterion(outs, labels)

	# Get the speaker id with highest probability.
	preds = outs.argmax(1)
	# Compute accuracy.
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy


# # Validate
# - Calculate accuracy of the validation set.
def valid(dataloader, model, criterion, device): 
	"""Validate on validation set."""

	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_fn(batch, model, criterion, device)
			running_loss += loss.item()
			running_accuracy += accuracy.item()

		pbar.update(dataloader.batch_size)
		pbar.set_postfix(
			loss=f"{running_loss / (i+1):.2f}",
			accuracy=f"{running_accuracy / (i+1):.2f}",
		)

	pbar.close()
	model.train()

	return running_accuracy / len(dataloader)


# # Main function
def parse_args():
	"""arguments"""
	config = {
		"data_dir": "./Dataset",
		"save_path": "model.ckpt",
		"batch_size": 32,
		"n_workers": 8,
		"valid_steps": 2000,
		"warmup_steps": 1000,
		"save_steps": 10000,
		"total_steps": 100000,
	}
	return config

def main(
	data_dir,
	save_path,
	batch_size,
	n_workers,
	valid_steps,
	warmup_steps,
	total_steps,
	save_steps,
):
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
	train_iterator = iter(train_loader)
	print(f"[Info]: Finish loading data!",flush = True)

	model = Classifier(n_spks=speaker_num).to(device)
	total_params = 0
	for i, param in enumerate(model.parameters()):
		print('Layer:', i, 'parameter num:', param.numel())
		total_params += param.numel()
	print(f'Total params: {total_params}')
	criterion = nn.CrossEntropyLoss()
	optimizer = AdamW(model.parameters(), lr=1e-3)
	scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
	print(f"[Info]: Finish creating model!",flush = True)

	best_accuracy = -1.0
	best_state_dict = None

	pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

	for step in range(total_steps):
		# Get data
		try:
			batch = next(train_iterator)
		except StopIteration:
			train_iterator = iter(train_loader)
			batch = next(train_iterator)

		loss, accuracy = model_fn(batch, model, criterion, device)
		batch_loss = loss.item()
		batch_accuracy = accuracy.item()

		# Updata model
		loss.backward()
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

		# Log
		pbar.update()
		pbar.set_postfix(
			loss=f"{batch_loss:.2f}",
			accuracy=f"{batch_accuracy:.2f}",
			step=step + 1,
		)

		# Do validation
		if (step + 1) % valid_steps == 0:
			pbar.close()

			valid_accuracy = valid(valid_loader, model, criterion, device)

			# keep the best model
			if valid_accuracy > best_accuracy:
				best_accuracy = valid_accuracy
				best_state_dict = model.state_dict()

			pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

		# Save the best model so far.
		if (step + 1) % save_steps == 0 and best_state_dict is not None:
			torch.save(best_state_dict, save_path)
			pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

	pbar.close()

if __name__ == "__main__":
	main(**parse_args())


# # Inference
# 推理（inference）时的数据集和数据加载器
class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]# utterances 语音项

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]# 从语音项中提取特征文件路径
		mel = torch.load(os.path.join(self.data_dir, feat_path))# 加载 mel特征

		return feat_path, mel


def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)# 分成两个元组
	return feat_paths, torch.stack(mels)
# feat_paths是包含特征文件路径的元组，mels是包含对应梅尔频谱图特征的张量

# ## Main funcrion of Inference
def parse_args():
	"""arguments"""  
	config = {
		"data_dir": "./Dataset",
		"model_path": "model.ckpt",
		"output_path": "output.csv",
	}
	return config

def main(
	data_dir,
	model_path,
	output_path,
):
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	mapping_path = Path(data_dir) / "mapping.json"
	mapping = json.load(mapping_path.open())

	dataset = InferenceDataset(data_dir)
	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		drop_last=False,
		num_workers=8,
		collate_fn=inference_collate_batch,
	)
	print(f"[Info]: Finish loading data!",flush = True)

	speaker_num = len(mapping["id2speaker"])
	model = Classifier(n_spks=speaker_num).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	print(f"[Info]: Finish creating model!",flush = True)

	results = [["Id", "Category"]]
	for feat_paths, mels in tqdm(dataloader):
		with torch.no_grad():
			mels = mels.to(device)
			outs = model(mels)
			preds = outs.argmax(1).cpu().numpy()
			for feat_path, pred in zip(feat_paths, preds):
				results.append([feat_path, mapping["id2speaker"][str(pred)]])

	with open(output_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)


if __name__ == "__main__":
	main(**parse_args())

