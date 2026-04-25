# 维度更改+ConformerBlock+Self-attention pooling+Additive margin softmax+Train longer。
# 与strong baseline相比，将mean pooling 换成了self-attention pooling
# 另外使用了简单版的additive margin softmax，batch size从32到64，step改为200000步。
# 运行代码，提交后得到分数：0.78725。
# 另外试了正常版的additive margin softmax，仍然效果有限，最后用ensemble+TTA，能到达boss baseline
# 不过这跟作业课件中的要求有违背，课件中的说明是单个模型到达boss baseline

# self-attention pooling：计算相似度，softmax归一化得到注意力权重，加权求和形成最终的池化权重。并不直接抛弃元素，而是给每个元素一个重要性权重
# Additive margin softmax：损失函数，引入额外的边缘，用于分割

import numpy as np
import torch
import random
import os
import json
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from conformer import ConformerBlock
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
import json
import csv

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

class myDataset(Dataset):
	def __init__(self, data_dir, segment_len=192):
		self.data_dir = data_dir
		self.segment_len = segment_len
	
		mapping_path = Path(data_dir) / "mapping.json"
		mapping = json.load(mapping_path.open())
		self.speaker2id = mapping["speaker2id"]
	
		metadata_path = Path(data_dir) / "metadata.json"
		metadata = json.load(open(metadata_path))["speakers"]
	
		self.speaker_num = len(metadata.keys())
		self.data = []
		for speaker in metadata.keys():
			for utterances in metadata[speaker]:
				self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
	def __len__(self):
			return len(self.data)
 
	def __getitem__(self, index):
		feat_path, speaker = self.data[index]
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		if len(mel) > self.segment_len:
			start = random.randint(0, len(mel) - self.segment_len)
			mel = torch.FloatTensor(mel[start:start+self.segment_len])
		else:
			mel = torch.FloatTensor(mel)
		speaker = torch.FloatTensor([speaker]).long()
		return mel, speaker
 
	def get_speaker_number(self):
		return self.speaker_num

def collate_batch(batch):
	mel, speaker = zip(*batch)
	mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
	return mel, torch.FloatTensor(speaker).long()

def get_dataloader(data_dir, batch_size, n_workers):
	dataset = myDataset(data_dir)
	speaker_num = dataset.get_speaker_number()
	trainlen = int(0.9 * len(dataset))
	lengths = [trainlen, len(dataset) - trainlen]
	trainset, validset = random_split(dataset, lengths)

	train_loader = DataLoader(
		trainset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=n_workers,
		pin_memory=True,
		collate_fn=collate_batch,
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

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, 1)
    
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
      
        attention_weight:
            att_w : size (N, T, 1)
    
        return:
            utter_rep: size (N, H)
        """
        att_w = F.softmax(self.W(batch_rep).squeeze(-1), dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)
        return utter_rep

class Classifier(nn.Module):
	def __init__(self, d_model=224, n_spks=600, dropout=0.3):
		super().__init__()
		self.prenet = nn.Linear(40, d_model)
		self.encoder = ConformerBlock(
				dim = d_model,
				dim_head = 4,
				heads = 4,
				ff_mult = 4,
				conv_expansion_factor = 2,
				conv_kernel_size = 20,
				attn_dropout = dropout,
				ff_dropout = dropout,
				conv_dropout = dropout,
		)
		self.sap = SelfAttentionPooling(d_model)
		self.pred_layer = nn.Sequential(
			nn.BatchNorm1d(d_model),
			nn.Linear(d_model, n_spks),
		)

	def forward(self, mels):
		out = self.prenet(mels)
		out = out.permute(1, 0, 2)
		out = self.encoder(out)
		out = out.transpose(0, 1) 
		#stats = out.mean(dim=1)
		stats = self.sap(out)
        
		out = self.pred_layer(stats)
		return out

from torch.autograd import Variable
class AMSoftmax(nn.Module):
    def __init__(self):
        super(AMSoftmax, self).__init__()

    def forward(self, input, target, scale=5.0, margin=0.35):
        # self.it += 1
        cos_theta = input
        target = target.view(-1, 1)  # size=(B,1)

        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index).bool()

        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= margin
        output = output * scale

        logpt = F.log_softmax(output, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt
        loss = loss.mean()

        return loss

# lr
def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)

def model_fn(batch, model, criterion, device):
	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)
	outs = model(mels)
	loss = criterion(outs, labels)

	preds = outs.argmax(1)
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy

# Validate
# - Calculate accuracy of the validation set.
def valid(dataloader, model, criterion, device): 
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
		"batch_size": 64,
		"n_workers": 8,
		"valid_steps": 2000,
		"warmup_steps": 1000,
		"save_steps": 10000,
		"total_steps": 200000,
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
	# Main function.
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
	criterion = AMSoftmax()
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

# Inference
# Dataset of inference
class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		return feat_path, mel

def inference_collate_batch(batch):
	feat_paths, mels = zip(*batch)
	return feat_paths, torch.stack(mels)

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
	# Main function.
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

