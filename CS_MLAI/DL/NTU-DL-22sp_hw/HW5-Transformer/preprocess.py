# 机器翻译，英译中 (繁体)
#   - 输入: an English sentence (比如: tom is a student .)
#   - 输出: the Chinese translation  (比如: 湯姆 是 個 學生 。)
# evaluation用blue score,看output和target有多像 

# ● Simple Baseline: Train a simple RNN seq2seq to acheive translation
# ● Medium Baseline: Add learning rate scheduler and train longer
# ● Strong Baseline: Switch to Transformer and tuning hyperparameter
# ● Boss Baseline: Apply back-translation

# 1、Paired data
# ○ TED2020: TED talks with transcripts translated by a global community of volunteers to
# more than 100 language
# ○ We will use (en, zh-tw) aligned pairs
# 2、Monolingual data
# ○ More TED talks in traditional Chinese

# 1. Preprocessing
# a. download raw data
# b. clean and normalize
# c. remove bad data (too long/short)
# d. tokenization
# 数据集有两个压缩包：ted2020.tgz(包含raw.en, raw.zh两个文件)，test.tgz(包含test.en, test.zh两个文件)。
# 第一步将解压后的文件全部放到一层目录中，目录位置是："DATA/rawdata/ted2020"，并将raw.en,raw.zh，test.en，test.zh分别改名称为train_dev.raw.en, train_dev.raw.zh, test.raw.en, test.raw.zh。
# 第二步是数据清洗操作，去掉或者替换一些特殊字符，例如符号单破折号’—‘会被删除，清洗后的文件名称是，train_dev.raw.clean.en, train_dev.raw.clean.zh, test.raw.clean.en, test.raw.clean.zh。
# 第三步是划分训练集和验证集，train_dev.raw.clean.en和train_dev.clean.zh被分成train.clean.en, valid.clean.en和train.clean.zh, valid.clean.zh。
# 第四步是分词，使用sentencepiece中的spm对训练集和验证集进行分词建模，模型名称是spm8000.model，同时产生词汇库spm8000.vocab，使用模型对训练集、验证集、以及测试集进行分词处理，得到文件train.en, train.zh, valid.en, valid.zh, test.en, test.zh。
# 第五步是文件二进制化，该过程使用fairseq库，这个库对于序列数据的处理很方便。运行后最终生成了一系列的文件，文件目录是"DATA/data_bin/ted2020"，这下面有18个文件，其中的一些二进制文件才是我们最终想要的训练数据。
# 数据的预处理流程基本不需要修改，如果在个人电脑上训练，跑一遍程序后，之后可直接从数据预处理结束开始跑，能节省不少时间。

# Training tips
# 1、Tokenize data with subword units
# ○ Reduce the vocabulary size
# ○ Alleviate the open vocabulary problem
# ○ example
# ■ ▁put ▁your s el ve s ▁in ▁my ▁po s ition ▁.
# ■ Put yourselves in my position.
# 2、Label smoothing见笔记
# 3、Learning rate warm-up
# 4、Back translation
# 先train一个中译英的backward model，然后把Monolingual data翻译成英文，这样就有更多data用来train英译中
# Some points to note about back-translation
# （1）Monolingual data should be in the same domain as the parallel corpus
# （2）The performance of the backward model is critical
# （3）Increase model capacity since the data amount is increased

# 导入包
# - **editdistance**： 快速实现编辑距离（Levenshtein距离）。  
# - **sacrebleu**： 计算bleu的库, 可以查看下[知乎: BLEU指标及评测脚本使用的一些误解](https://zhuanlan.zhihu.com/p/404381278)
# - **sacremoses**: 使用Python实现了Moses的tokenizer, truecaser以及normalizer功能，使用起来比较方便[官方Github（有示例）](https://github.com/alvations/sacremoses)
# - **sentencepiece**： 由谷歌将一些词-语言模型相关的论文进行复现，开发了一个开源工具——训练自己领域的sentencepiece模型，该模型可以代替预训练模型(BERT,XLNET)中词表的作用，可以参考[sentencepiece原理与实践](https://zhuanlan.zhihu.com/p/159200073)
# - **wandb**: 是Weights & Biases的缩写，这款工具能够帮助跟踪你的机器学习项目。它能够自动记录模型训练过程中的超参数和输出指标，然后可视化和比较结果，并快速与同事共享结果。[官方文档：quickstart](https://docs.wandb.ai/v/zh-hans/quickstart)
# - **fairseq**: 一个用PyTorch编写的序列建模工具包，它允许研究人员和开发人员训练用于翻译、摘要、语言建模和其他文本生成任务的自定义模型。[fairseq官方文档](https://fairseq.readthedocs.io/en/latest/)

# pip install  editdistance  sacrebleu sacremoses sentencepiece wandb
# get_ipython().system("pip install 'torch>=1.6.0' editdistance matplotlib sacrebleu sacremoses sentencepiece tqdm wandb")
# get_ipython().system('pip install --upgrade jupyter ipywidgets')

# https://fairseq.readthedocs.io/en/latest/
# get_ipython().system('git clone https://github.com/pytorch/fairseq.git')
# get_ipython().system('cd fairseq && git checkout 9a1c497')
# get_ipython().system('pip install --upgrade ./fairseq/')

# https://github.com/facebookresearch/fairseq/issues/2106
# 卸载numpy重新安装fairseq

# AttributeError: module 'numpy' has no attribute 'float'.
# sed -i 's/np.float/float/g' $(grep -rl 'np.float' /opt/conda/lib/python3.8/site-packages/fairseq)

import sys
import pdb
import pprint
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils
import matplotlib.pyplot as plt

# 数据集下载
# En-Zh Bilingual Parallel Corpus  （英-中 平行语料）
# [TED2020](#reimers-2020-multilingual-sentence-bert)
#     - Raw: 398,066 (sentences)   
#     - Processed: 393,980 (sentences)

# 测试数据
# - 大小: 4,000 (句子)
# - 中文翻译并未提供。 提供的（.zh）文件是伪翻译，每一行都是一个'。'

data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
# urls = (
#     "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted2020.tgz",
#     "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/test.tgz",
# )
# file_names = (
#     'ted2020.tgz', # train & dev
#     'test.tgz', # test
# )
prefix = Path(data_dir).absolute() / dataset_name

# prefix.mkdir(parents=True, exist_ok=True)
# for u, f in zip(urls, file_names):
#     path = prefix/f
    # 不存在则直接通过 weget进行下载
#     if not path.exists():
#         get_ipython().system('wget {u} -O {path}')
#     if path.suffix == ".tgz":
#         get_ipython().system('tar -xvf {path} -C {prefix}')
#     elif path.suffix == ".zip":
#         get_ipython().system('unzip -o {path} -d {prefix}')
# get_ipython().system("mv {prefix/'raw.en'} {prefix/'train_dev.raw.en'}")
# mv ./raw.en ./DATA/rawdata/ted2020/train_dev.raw.en
# get_ipython().system("mv {prefix/'raw.zh'} {prefix/'train_dev.raw.zh'}")
# mv ./raw.zh ./DATA/rawdata/ted2020/train_dev.raw.zh
# get_ipython().system("mv {prefix/'test/test.en'} {prefix/'test.raw.en'}")
# mv ./test/test.en ./DATA/rawdata/ted2020/test.raw.en
# get_ipython().system("mv {prefix/'test/test.zh'} {prefix/'test.raw.zh'}")
# mv ./test/test.zh ./DATA/rawdata/ted2020/test.raw.zh
# get_ipython().system("rm -rf {prefix/'test'}")

src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'

# 打印前五行
import subprocess
subprocess.run(['head', f'{data_prefix}.{src_lang}', '-n', '5'])
subprocess.run(['head', f'{data_prefix}.{tgt_lang}', '-n', '5'])

# Preprocess files
# 1. 全角转半角 `strQ2B`
# 2. 一些字符串替换 `clean_s`
# 3. 训练数据适当的清洗 `clean_corpus`
#     - 删除过短数据 `min_len`
#     - 删除过长数据 `max_len`
#     - 删除翻译前后数据长度比例超过一定值的字段 `ratio`
import re
def strQ2B(ustring):
    # 把字符串全角转半角
    # reference:https://ithelp.ithome.com.tw/articles/10233122
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  #  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)
                
def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # 删除 ([text])
        s = s.replace('-', '') # 删除 '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # 保留标点符号
    elif lang == 'zh':
        s = strQ2B(s)# 把字符串全角转半角
        s = re.sub(r"\([^()]*\)", "", s) # 删除 ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # 保留标点符号
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())

def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return
    with open(f'{prefix}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r') as l2_in_f:
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in l1_in_f:
                        s1 = s1.strip()
                        s2 = l2_in_f.readline().strip()
                        s1 = clean_s(s1, l1)
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # 删除过短数据
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # 删除过长数据
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # 删除翻译前后数据长度比例超过一定值的字段
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)

clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)

subprocess.run(['head', f'{data_prefix}.clean.{src_lang}', '-n', '5'])
subprocess.run(['head', f'{data_prefix}.clean.{tgt_lang}', '-n', '5'])


# ## Split into train/valid
valid_ratio = 0.01 # 3000~4000就足够了
train_ratio = 1 - valid_ratio

if (prefix/f'train.clean.{src_lang}').exists() and (prefix/f'train.clean.{tgt_lang}').exists() and (prefix/f'valid.clean.{src_lang}').exists() and (prefix/f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')
else:
    line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
        count = 0
        # 基于下标拆分训练和测试集
        for line in open(f'{data_prefix}.clean.{lang}', 'r'):
            if labels[count]/line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()

# ## Subword Units 
# 在机器翻译中词表外词（OOV）的翻译是主要问题。 这个问题可以通过使用子字单元(`Subword Units`) 来缓解。
# - 我们使用 [sentencepiece](#kudo-richardson-2018-sentencepiece) python包
# - 选择“unigram”或“字节对编码（BPE）”算法
import sentencepiece as spm
vocab_size = 8000
if (prefix/f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 用'bpe'也行 
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

# 用训练好的模型清洗数据: 将句子的起始终点加上标识：
# 如 ▁這個 研 討 會 給我 留 下 了 極 為 深 刻 的 印 象 ▁, ▁我想 感 謝 大家 對我 之前 演講 的 好 評 
spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))
in_tag = {
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix/f'{split}.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix/f'{split}.{lang}', 'w') as out_f:
                with open(prefix/f'{in_tag[split]}.{lang}', 'r') as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)

subprocess.run(['head', f'{data_dir}/{dataset_name}/train.{src_lang}', '-n', '5'])
subprocess.run(['head', f'{data_dir}/{dataset_name}/train.{tgt_lang}', '-n', '5'])


# 使用fairseq将数据二进制化
# 生成4个文件
# - train.en-zh.en.bin
# - train.en-zh.en.idx
# - train.en-zh.zh.bin
# - train.en-zh.zh.idx
binpath = Path('./DATA/data-bin', dataset_name)
command = f"python -m fairseq_cli.preprocess \
           --source-lang {src_lang} \
           --target-lang {tgt_lang} \
           --trainpref {prefix}/train \
           --validpref {prefix}/valid \
           --testpref {prefix}/test \
           --destdir {binpath} \
           --joined-dictionary \
           --workers 2"
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    # get_ipython().system("python -m fairseq_cli.preprocess         --source-lang {src_lang}        --target-lang {tgt_lang}        --trainpref {prefix/'train'}        --validpref {prefix/'valid'}        --testpref {prefix/'test'}        --destdir {binpath}        --joined-dictionary        --workers 2")
    subprocess.run(command, shell=True)
    print("finish all")
