# 修改get_rate函数+epoch改为30。
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

seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 超参设置
config = Namespace(
    datadir = "./DATA/data-bin/ted2020",
    savedir = "./checkpoints/rnn",
    source_lang = "en",
    target_lang = "zh",

    # 设置cpu核数：fetching & processing data.
    num_workers=2,  
    # batch size 中tokens大小设置. 梯度累积增加有效批量 gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,
    
    # NoamScheduler调整学习率。我们可以通过 lr_factor 调整最大lr。
    #     NoamScheduler：可以在numpy-ml中了解详情 https://numpy-ml.readthedocs.io/en/latest/numpy_ml.neural_nets.schedulers.html
    #         lr= lr_factor * ( model_size ** (-0.5) * min(step** (-0.5), step * warmup_steps ** (-1.5)) )
    lr_factor=2.,
    lr_warmup=4000,
    
    # 梯度裁剪norm ，防止梯度爆炸
    clip_norm=1.0,
    
    # 训练最大轮次
    max_epoch=30,
    start_epoch=1,
    
    # beam search 大小
    #    beam search 可以详细阅读《动手学深度学习》： https://d2l.ai/chapter_recurrent-modern/beam-search.html
    # （束搜索）Beam search通过在生成过程中维护多个备选的输出序列（称为“束”），并根据一定的评分准则选择最有可能的序列，从而提高最终输出的质量。
    beam=5, 
    # 生成序列最大长度 ax + b, x是原始序列长度
    max_len_a=1.2, 
    max_len_b=10, 
    # 解码时，数据后处理：删除句子符号 和 jieba对句子 。
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process = "sentencepiece",
    
    # checkpoints
    keep_last_epochs=5,
    resume=None, # 如果设置，则从config.savedir 中的对应 checkpoint name 恢复
    
    # logging
    use_wandb=False,
)

# Logging
# - logging package logs ordinary messages
# - wandb logs the loss, bleu, etc. in the training process
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO", # "DEBUG" "WARNING" "ERROR"
    stream=sys.stdout,
)
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb:
    import wandb
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

# CUDA Environments
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # 数据集加载
# ## 我们从fairseq借用TranslationTask
# * 用于加载上面创建的二进制数据
# * 实现数据迭代器 (dataloader)
# * 内置 task.source_dictionary 和 task.target_dictionary 同样方便
# * 实现 beach search decoder
from fairseq.tasks.translation import TranslationConfig, TranslationTask
## setup task
task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)

logger.info("loading data for epoch 1")
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)

sample = task.dataset("valid")[1]
pprint.pprint(sample)
pprint.pprint(
    "Source: " + \
    task.source_dictionary.string(
        sample['source'],
        config.post_process,
    )
)
pprint.pprint(
    "Target: " + \
    task.target_dictionary.string(
        sample['target'],
        config.post_process,
    )
)

# # 数据生成器
# > 和 pytorch 中的 `torch.utils.data.DataLoader`类似
# 
# * 控制每个`batch`包含不超过N个`token`，从而优化GPU内存效率
# * 每个`batch`对训练集进行重排序（乱序）
# * 忽略超过最大长度的句子
# * 将`batch`中的所有句子填充到相同的长度，从而实现GPU的并行计算
# * 在一个`token`中增加 `eos` 和 `shift`
#     - **训练的时候注意**: 为了训练模型以基于前缀预测下一个词语，我们将右移的目标序列作为decoder输入。
#     - 一般来说，将bos前置到目标就可以完成任务（如下所示）
# ![seq2seq](https://i.imgur.com/0zeDyuI.png)
#     - 在`fairseq`中进行的处理有点不样,是将`eos`移动到最开始位置。根据经验，这也有同样的效果。示例:
#     ```
#     # output target (target) and Decoder input (prev_output_tokens): 
#                    eos = 2
#                 target = 419,  711,  238,  888,  792,   60,  968,    8,    2
#     prev_output_tokens = 2,  419,  711,  238,  888,  792,   60,  968,    8
#     ```
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        # fairseq.utils
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # 将此设置为False以加快速度。However, if set to False, changing max_tokens beyond 
        # 此方法的第一次调用无效. 
    )
    return batch_iterator

demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
sample

# * 每个batch是python dict, key是`string`, values是`Tensor value`.
# ```python
# batch = {
#     "id": id, # 样例的id
#     "nsentences": len(samples), # 句子批次大小 (sentences)
#     "ntokens": ntokens, # token批次大小 (tokens)
#     "net_input": {
#         "src_tokens": src_tokens, # 翻译的句子(`source language`)
#         "src_lengths": src_lengths, # 句子padding前的长度
#         "prev_output_tokens": prev_output_tokens, # 右移动的目标（`right shifted target`)
#     },
#     "target": target, # 翻译结果
# }
# ```

# # Model Architecture
# * We again inherit fairseq's encoder, decoder and model, so that in the testing phase we can directly leverage fairseq's beam search decoder.
from fairseq.models import (
    FairseqEncoder, 
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)

# # Encoder
# - The Encoder 可以是 `RNN`  或 `Transformer Encoder`. 下面的描述是针对RNN的. 
# 对于每个输入 token, Encoder 会生成一个 output vector 和一个 hidden states vector, 并且 hidden states vector 将传入下一个rnn cell. 
# 换句话来说, Encoder 按输入顺序顺序读取句子, 并在每个时间步输出一个 single vector, 最终输出final hidden states, 或者 content vector, 在最后一个时间步.
# - 超参数:
#   - *args*
#       - encoder_embed_dim: embeddings维度, 将 one-hot 向量压缩, 实现维度降低
#       - encoder_ffn_embed_dim：  hidden states 和 output vectors 的维度
#       - encoder_layers：RNN Encoder的层数
#       - dropout：将部分神经元激活的概率被设置为0，以防止过度拟合。通常，这在训练中应用，在测试中删除。
#   - *dictionary*: fairseq 中提供的dictionary.它用于获得填充索引(`padding index`)，进而获得编码器填充掩码(`encoder padding mask`)。
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
# 
# - 输入: 
#     - *src_tokens*: 表示英语的整数序列 e.g. 1, 28, 29, 205, 2 
# - 输出: 
#     - *outputs*:  RNN每步输出,可以进行Attention处理
#     - *final_hiddens*: 每一步的`hidden states`, 传入解码器(`decoder`)用于翻译
#     - *encoder_padding_mask*: 这告诉解码器(`decoder`)忽略哪个位置
class RNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        
        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_ffn_embed_dim # 隐藏状态和输出向量的维度
        self.num_layers = args.encoder_layers # RNN编码器的层数
        
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=True # 使用双向GRU
        )
        self.dropout_out_module = nn.Dropout(args.dropout)
        self.padding_idx = dictionary.pad()# 获取填充索引
        
    def combine_bidir(self, outs, bsz: int):
        # 将双向输出合并为单一方向
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)

    def forward(self, src_tokens, **unused):
        bsz, seqlen = src_tokens.size()
        
        # 获取 embeddings
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C 时间步在第一维度
        x = x.transpose(0, 1)
        
        # 直通双向 RNN
        h0 = x.new_zeros(2 * self.num_layers, bsz, self.hidden_dim)
        x, final_hiddens = self.rnn(x, h0)
        outputs = self.dropout_out_module(x)
        # outputs = [sequence len, batch size, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        
        # 由于encode是双向的，我们需要连接两个方向的隐藏状态
        final_hiddens = self.combine_bidir(final_hiddens, bsz)
        # hidden =  [num_layers x batch x num_directions*hidden]
        
        # 生成编码器填充掩码
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return tuple(
            (
                outputs,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden
                encoder_padding_mask,  # seq_len x batch
            )
        )
    
    def reorder_encoder_out(self, encoder_out, new_order):
        # 这部分会在fairseq's beam search中使用。
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
            )
        )

# ## Attention
# - 当输入序列较长时，“内容向量”单独不能准确地表示整个序列，注意力机制(`Attention`)可以为解码器(`Encoder`)提供更多信息。
# - 根据当前时间步长的**Decoder embeddings**，将**Encoder outputs**与解码器嵌入(`Decoder embeddings`)匹配，以确定相关性，然后将编码器输出与相关性加权相加，作为**Decoder**RNN的输入。 
# - 常见的注意力实现使用神经网络/点积作为**query**（解码器嵌入`decoder embeddings`）和**key**（编码器输出`Encoder outputs`）之间的相关性，然后是**softmax**以获得分布，最后**values**（编码器输入`Encoder outputs`）被所述分布**加权和**。
#     - $Q = W_Q  I_{decoder-emb}$
#     - $K = W_K  I_{encoder-out}$
#     - $V = W_V  I_{encoder-out}$
#     - $A = K^TQ$
#     - $A'= softmax(A)$
#     - $O = VA'$
# 
# - Parameters:
#   - *source_embed_dim*: query的维度 $W_Q$
#   - *input_embed_dim*: key的维度 $W_K$
#   - *output_embed_dim*: value的维度 $W_V$
# 
# - Inputs: 
#     - *inputs*: 做attention的输入
#     - *encoder_outputs*:  作为 query / value,
#     - *encoder_padding_mask*: 这告诉解码器`decoder`忽略哪个位置

# - Outputs: 
#     - *output*: attention后的上下文向量
#     - *attention score*: attention后的分布 $A'$
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = nn.Linear(
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )

    def forward(self, inputs, encoder_outputs, encoder_padding_mask):
        # inputs: T, B, dim
        # encoder_outputs: S x B x dim
        # padding mask:  S x B
        
        # batch first调整为batch在第一维度
        inputs = inputs.transpose(1,0) # B, T, dim
        encoder_outputs = encoder_outputs.transpose(1,0) # B, S, dim
        encoder_padding_mask = encoder_padding_mask.transpose(1,0) # B, S
        
        # project to the dimensionality of encoder_outputs
        x = self.input_proj(inputs)

        # 计算注意力分数
        # (B, T, dim) x (B, dim, S) = (B, T, S)
        attn_scores = torch.bmm(x, encoder_outputs.transpose(1,2))

        # 在与padding相对应的位置取消注意
        if encoder_padding_mask is not None:
            # leveraging broadcast  B, S -> (B, 1, S)
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back

        # 对得分进行softmax操作，得到注意力分布
        attn_scores = F.softmax(attn_scores, dim=-1)

        # shape (B, T, S) x (B, S, dim) = (B, T, dim) weighted sum
        # 加权和计算上下文向量
        x = torch.bmm(attn_scores, encoder_outputs)

        # (B, T, dim)
        # 将注意力上下文向量与解码器embedding拼接后映射到输出维度
        x = torch.cat((x, inputs), dim=-1)
        x = torch.tanh(self.output_proj(x)) # concat拼接 + linear + tanh
        
        # restore shape (B, T, dim) -> (T, B, dim)时间步在第一维度
        return x.transpose(1,0), attn_scores

# # Decoder
# * **Decoder**的隐藏状态(`hidden states`)将由**Encoder**的最终隐藏状态（`final hidden states`）初始化
# * 同时，**Decoder**将根据当前时间步长的输入（之前时间步长的输出）更改其隐藏状态（`hidden states`），并生成输出
# * Attention提高了性能
# * seq2seq步骤在解码器(`decoder`)中实现 , 以便以后Seq2Seq类可以接受RNN和Transformer，而无需进一步修改。
# 
# - Parameters:
#   - *args*
#       - decoder_embed_dim: decoder 维度, 和`encoder_embed_dim`类似，
#       - decoder_ffn_embed_dim: decoder RNN 的隐含层(hidden states`)维度,和`encoder_ffn_embed_dim`类似
#       - decoder_layers: decoder RNN 的网络层数
#       - share_decoder_input_output_embed: 通常，解码器`decoder`的投影矩阵将与解码器输入嵌入(`decoder input embeddings`)共享权重
#   - *dictionary*: fairseq 中提供的dictionary.
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
# 
# - Inputs: 
#     - *prev_output_tokens*: 表示右移目标(`right-shifted target`)的整数序列  e.g. 1, 28, 29, 205, 2 
#     - *encoder_out*: encoder的输出.
#     - *incremental_state*: 为了在测试期间加快解码速度，我们将保存每个时间步长的隐藏状态(` hidden state`)。
# 
# - Outputs: 
#     - *outputs*: 解码器每个时间步的logits（softmax之前）输出
#     - *extra*: 没有使用
class RNNDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        
        # 确保解码器和编码器有相同数量的层
        assert args.decoder_layers == args.encoder_layers, f"""seq2seq rnn requires that encoder 
        and decoder have same layers of rnn. got: {args.encoder_layers, args.decoder_layers}"""
        # 确保解码器的隐藏维度是编码器隐藏维度的两倍
        assert args.decoder_ffn_embed_dim == args.encoder_ffn_embed_dim*2, f"""seq2seq-rnn requires 
        that decoder hidden to be 2*encoder hidden dim. got: {args.decoder_ffn_embed_dim, args.encoder_ffn_embed_dim*2}"""
        
        self.embed_dim = args.decoder_embed_dim
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers
        
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=False # 使用单向GRU
        )
        # 初始化注意力层
        self.attention = AttentionLayer(
            self.embed_dim, self.hidden_dim, self.embed_dim, bias=False
        ) 
        # self.attention = None
        self.dropout_out_module = nn.Dropout(args.dropout)
        
        # 如果隐藏维度与embed维度不同，需要进行额外的投影操作
        if self.hidden_dim != self.embed_dim:
            self.project_out_dim = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.project_out_dim = None

        # 如果共享解码器输入输出embed，则使用相同的权重矩阵进行投影
        if args.share_decoder_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        # 将Encoder的输出转化为最终的目标序列
        # 从编码器encoder中提取输出
        encoder_outputs, encoder_hiddens, encoder_padding_mask = encoder_out
        # outputs:          seq_len x batch x num_directions*hidden
        # encoder_hiddens:  num_layers x batch x num_directions*encoder_hidden
        # padding_mask:     seq_len x batch
        
        # 如果存在增量状态，并且长度大于0，从增量状态中提取上一个时间步的信息
        if incremental_state is not None and len(incremental_state) > 0:
         # 如果保留了上一个时间步的信息，我们可以从那里继续，而不是从头开始
            prev_output_tokens = prev_output_tokens[:, -1:]
            cache_state = self.get_incremental_state(incremental_state, "cached_state")
            prev_hiddens = cache_state["prev_hiddens"]
        else:
            # 增量状态不存在，或者这是训练时间，或者是测试时间的第一个时间步
            # 准备 seq2seq: 将encoder_hidden传递给解码器decoder隐藏状态 
            prev_hiddens = encoder_hiddens
        
        bsz, seqlen = prev_output_tokens.size()
        
        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C 时间步在第一维度
        x = x.transpose(0, 1)
                
        # decoder-to-encoder attention
        if self.attention is not None:
            x, attn = self.attention(x, encoder_outputs, encoder_padding_mask)
                        
        # 直通双向 RNN unidirectional RNN
        x, final_hiddens = self.rnn(x, prev_hiddens)
        # outputs = [sequence len, batch size, hid dim]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        x = self.dropout_out_module(x)
                
        # 投影到 embedding size （如果hidden与嵌入embed大小不同，且share_embedding为True，则需要进行额外的投影操作）
        if self.project_out_dim != None:
            x = self.project_out_dim(x)
        
        # 投影到词表大小 vocab size
        x = self.output_projection(x)
        
        # T x B x C -> B x T x C batch 在第一维度
        x = x.transpose(1, 0)
        
        # 如果是增量，记录当前时间步的隐藏状态，将在下一个时间步中恢复
        cache_state = {
            "prev_hiddens": final_hiddens,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        
        return x, None
    
    def reorder_incremental_state(
        self,
        incremental_state,
        new_order,
    ):
        # 在fairseq's beam search中使用
        cache_state = self.get_incremental_state(incremental_state, "cached_state")
        prev_hiddens = cache_state["prev_hiddens"]
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        cache_state = {
            "prev_hiddens": torch.stack(prev_hiddens),
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        return


# ## Seq2Seq
# - Composed of **Encoder** and **Decoder**
# - Recieves inputs and pass to **Encoder** 
# - Pass the outputs from **Encoder** to **Decoder**
# - **Decoder** will decode according to outputs of previous timesteps as well as **Encoder** outputs  
# - Once done decoding, return the **Decoder** outputs
class Seq2Seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
    ):
        # 前向传播： encoder -> decoder
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra


# # Model Initialization
# TODO: Encoder和Decoder修改：改用 TransformerEncoder & TransformerDecoder
from fairseq.models.transformer import (
    TransformerEncoder, 
    TransformerDecoder,
)

def build_model(args, task):
    # 基于超参数构建模型实例
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
    
    # encoder decoder
    # TODO: switch to TransformerEncoder & TransformerDecoder
    encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    # encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)
    
    # seq2seq 模型初始化很重要, 参数权重的初始化需要一些其他操作
    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)
            
    # 权重初始化
    model.apply(init_params)
    return model

# 模型框架一些参数配置
# For strong baseline, 请参考表3中*transformer-base*的超参数 [Attention is all you need](#vaswani2017)
arch_args = Namespace(
    encoder_embed_dim=256,
    encoder_ffn_embed_dim=512,
    encoder_layers=1,
    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=1,
    share_decoder_input_output_embed=True,
    dropout=0.3,
)

# HINT: 这些是关于Transformer参数的补丁 
def add_transformer_args(args):
    args.encoder_attention_heads=4
    args.encoder_normalize_before=True
    
    args.decoder_attention_heads=4
    args.decoder_normalize_before=True
    
    args.activation_fn="relu"
    args.max_source_positions=1024
    args.max_target_positions=1024
    
    # Transformer默认参数上的修补程序 (以上未列出)
    from fairseq.models.transformer import base_architecture
    base_architecture(arch_args)

# add_transformer_args(arch_args)

if config.use_wandb:
    wandb.config.update(vars(arch_args))

model = build_model(arch_args, task)
logger.info(model)

# Optimization
# 损失函数: 标签平滑正则化（`Label Smoothing Regularization`)
# * 让模型学习生成不太集中的分布，并防止过度自信(`over-confidence`)
# * 有时，事实真相可能不是唯一的答案。因此，在计算损失时，我们为不正确的标签保留一些概率
# * 防止过拟合
# code [来源：fairseq/criterions/label_smoothed_cross_entropy](https://fairseq.readthedocs.io/en/latest/_modules/fairseq/criterions/label_smoothed_cross_entropy.html)
class LabelSmoothedCrossEntropyCriterion(nn.Module):
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduce = reduce
    
    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        # nll: 负对数似然（Negative log likelihood），当目标是一个one-ho时的交叉熵。以下行与F.nll_loss相同
        nll_loss = -lprobs.gather(dim=-1, index=target)
        # 为其他标签保留一些可能性。因此当计算交叉熵时，相当于对所有标签的对数概率求和
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # 在计算交叉熵时，添加其他标签的损失
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss

# generally, 0.1 is good enough
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)


# Optimizer: Adam + lr scheduling
# 在训练Transformer时，逆平方根学习率变化对稳定性很重要. 它后来也用于RNN。
# 根据以下公式更新学习率`learning rate` . 在第一阶段，线性增加学习率，然后学习率与时间步长的平方根倒数成比例衰减。
# $$lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$$
def get_rate(d_model, step_num, warmup_step):
    # TODO: 基于上述公式更新学习率
    #lr = 0.001
    lr = (d_model**(-0.5)) * min(step_num**(-0.5), step_num*(warmup_step**(-1.5)))
    return lr

# 可以看：https://nn.labml.ai/optimizers/noam.html
class NoamOpt:
    # 实现速率的Optim包装器
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        # 将梯度乘以常数*c*            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        # 更新 parameters 和 rate
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        # 实现上面的lrate
        if step is None:
            step = self._step
        return 0 if not step else self.factor * get_rate(self.model_size, step, self.warmup)


# ## Scheduling Visualized
optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim, 
    factor=config.lr_factor, 
    warmup=config.lr_warmup, 
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))
plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
None

# Training Procedure
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

gnorms = []

def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps) 
    # 梯度累积：更新每个accum_steps采样
    
    stats = {"loss": []}
    scaler = GradScaler() # 自动混合精度`automatic mixed precision` (amp) 
    
    model.train()
    progress = tqdm.tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=False)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # 梯度累积：更新每个accum_steps采样
        for i, sample in enumerate(samples):
            if i == 1:
                 # 清空CUDA缓存在第一部之后，可以有效减少 OOM 的可能
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            
            # 混合精度训练`mixed precision training
            with autocast():
                net_output = model.forward(**sample["net_input"])
                lprobs = F.log_softmax(net_output[0], -1)            
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                
                # logging
                accum_loss += loss.item()
                # 反向传播
                scaler.scale(loss).backward()                
        
        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1 / (sample_size or 1.0)) 
        # (sample_size or 1.0) 处理零梯度的情况
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm) 
        # grad norm clipping 处理梯度爆炸
        gnorms.append(gnorm.cpu().item())
        
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        loss_print = accum_loss/sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })
        
    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats

# 模型验证 & 模型推理(预测)
# 为了防止过度拟合，每个`epoch`都需要进行验证，以验证对未见数据的性能
# - 该程序本质上与训练相同，就是增加了推理（预测）步骤
# - 验证后，我们可以保存模型权重
# 
# 仅验证损失无法描述模型的实际性能
# - 基于当前模型直接生成翻译结果，然后使用参考译文（目标值）计算BLEU
# - 我们还可以手动检查翻译结果的质量
# - 我们使用fairseq序列生成器进行`beam search`以生成多个翻译结果

# fairseq's beam search 生成器
# 基于模型和给定输入序列, 通过beam search生成翻译结果
sequence_generator = task.build_generator([model], config)

def decode(toks, dictionary):
    # 将Tensor装换成我们可阅读的句子(human readable sentence)
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):
        # for each sample, 收集输入`input`, 翻译结果`hypothesis`和 参考`reference`（label）, 后续用于计算 BLEU
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), 
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0 表示使用 beam中最靠前的翻译结果
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()), 
            task.target_dictionary,
        ))
    return srcs, hyps, refs

import shutil
import sacrebleu

def validate(model, task, criterion, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []
    
    model.eval()
    progress = tqdm.tqdm(itr, desc=f"validation", leave=False)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)
            
            # 模型推理
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)
            
    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    
    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)
    
    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])
    
    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats

# Save and Load Model Weights
def validate_and_save(model, task, criterion, optimizer, epoch, save=True):   
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # save epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)
        
        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")
    
        # save epoch samples
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # 获取验证中最佳的 bleu     
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir/f"checkpoint_best.pt")
            
        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()
    
    return stats

def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir)/name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")

# 开始训练
model = model.to(device=device)
criterion = criterion.to(device=device)

logger.info("task: {}".format(task.__class__.__name__))
logger.info("encoder: {}".format(model.encoder.__class__.__name__))
logger.info("decoder: {}".format(model.decoder.__class__.__name__))
logger.info("criterion: {}".format(criterion.__class__.__name__))
logger.info("optimizer: {}".format(optimizer.__class__.__name__))
logger.info(
    "num. model params: {:,} (num. trained: {:,})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
)
logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")

epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
try_load_checkpoint(model, optimizer, name=config.resume)
while epoch_itr.next_epoch_idx <= config.max_epoch:
    # train for one epoch
    train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
    stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
    logger.info("end of epoch {}".format(epoch_itr.epoch))    
    epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)

plt.plot(range(1, len(gnorms)+1), gnorms)
plt.plot(range(1, len(gnorms)+1), [config.clip_norm]*len(gnorms), "-")
plt.title('Grad norm v.s. step')
plt.xlabel("step")
plt.ylabel("Grad norm")
plt.show()


# Submission
# 平均几个checkpoints可以产生与ensemble类似的效果
checkdir=config.savedir
# get_ipython().system('python ./fairseq/scripts/average_checkpoints.py --inputs {checkdir} --num-epoch-checkpoints 5 --output {checkdir}/avg_last_5_checkpoint.pt')
command = f'python ./fairseq/scripts/average_checkpoints.py --inputs {checkdir} --num-epoch-checkpoints 5 --output {checkdir}/avg_last_5_checkpoint.pt'
subprocess.run(command, shell=True)

# 确认用于生成提交的模型权重
# checkpoint_last.pt : 最后个 epoch的保存点
# checkpoint_best.pt : 验证集中 bleu 最高的保存点
# avg_last_5_checkpoint.pt:　过去5个epoched的平均值
# try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
# validate(model, task, criterion, log_to_wandb=False)
try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
validate(model, task, criterion, log_to_wandb=False)
None

# 生成预测结果
def generate_prediction(model, task, split="test", outfile="./prediction.txt"):    
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    idxs = []
    hyps = []

    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)

            # do inference
            s, h, r = inference_step(sample, model)
            
            hyps.extend(h)
            idxs.extend(list(sample['id']))
            
    # 根据预处理前的顺序排序
    hyps = [x for _,x in sorted(zip(idxs,hyps))]
    
    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h+"\n")

generate_prediction(model, task)

subprocess.run(['head', '-n', '2', '/kaggle/working/DATA/rawdata/ted2020/test.en'])
subprocess.run(['head', '-n', '2', 'prediction.txt'])





# # 反向翻译`Back-translation`
# # 训练反向翻译模型
# # 1. 在**config中切换source_lang和target_lang** 
# # 2. 更改**config**中的savedir（例如“./checkpoints/transformer back”）
# # 3. 训练模型
# # 使用反向模型生成合成数据
# # 下载 monolingual 数据

# mono_dataset_name = 'mono'
# mono_prefix = Path(data_dir).absolute() / mono_dataset_name
# mono_prefix.mkdir(parents=True, exist_ok=True)

# urls = (
#     "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted_zh_corpus.deduped.gz"
# )
# file_names = (
#     'ted_zh_corpus.deduped.gz',
# )

# for u, f in zip(urls, file_names):
#     path = mono_prefix/f
#     if not path.exists():
#         else:
#             get_ipython().system('wget {u} -O {path}')
#     else:
#         print(f'{f} is exist, skip downloading')
#     if path.suffix == ".tgz":
#         get_ipython().system('tar -xvf {path} -C {prefix}')
#     elif path.suffix == ".zip":
#         get_ipython().system('unzip -o {path} -d {prefix}')
#     elif path.suffix == ".gz":
#         get_ipython().system('gzip -fkd {path}')


# # TODO: 数据清洗
# # 1. 删除过长或过短的句子
# # 2. 统一标点符号
# # hint: 你可以使用上述定义的 `clean_s()` 来完成该操作 

# # TODO: Subword Units
# # 使用反向模型的spm模型将数据标记为子字单位
# # hint: spm 模型本地位置 DATA/raw-data/\[dataset\]/spm\[vocab_num\].model

# # Binarize
# # use fairseq to binarize data
# binpath = Path('./DATA/data-bin', mono_dataset_name)
# src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
# tgt_dict_file = src_dict_file
# monopref = str(mono_prefix/"mono.tok") # whatever filepath you get after applying subword tokenization
# if binpath.exists():
#     print(binpath, "exists, will not overwrite!")
# else:
#     get_ipython().system("python -m fairseq_cli.preprocess        --source-lang 'zh'        --target-lang 'en'        --trainpref {monopref}        --destdir {binpath}        --srcdict {src_dict_file}        --tgtdict {tgt_dict_file}        --workers 2")


# # TODO: 使用反向模型生成合成数据
# # 将二进制`monolingual data`添加到原始数据目录，并使用“split_name”命名
# # 比如： . ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
# # 然后，你可以使用 `generate_prediction(model, task, split="split_name")` 去生成翻译预测

# # Add binarized monolingual data to the original data directory, and name it with "split_name"
# # ex. ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
# get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin')
# get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx')
# get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin')
# get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx')
# # hint: do prediction on split='mono' to create prediction_file
# # generate_prediction( ... ,split=... ,outfile=... )

# # TODO: Create new dataset
# # 1. Combine the prediction data with monolingual data
# # 2. Use the original spm model to tokenize data into Subword Units
# # 3. Binarize data with fairseq

# # Combine prediction_file (.en) and mono.zh (.zh) into a new dataset.
# #
# # hint: tokenize prediction_file with the spm model
# # spm_model.encode(line, out_type=str)
# # output: ./DATA/rawdata/mono/mono.tok.en & mono.tok.zh
# #
# # hint: use fairseq to binarize these two files again
# # binpath = Path('./DATA/data-bin/synthetic')
# # src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
# # tgt_dict_file = src_dict_file
# # monopref = ./DATA/rawdata/mono/mono.tok # or whatever path after applying subword tokenization, w/o the suffix (.zh/.en)
# # if binpath.exists():
# #     print(binpath, "exists, will not overwrite!")
# # else:
# #     !python -m fairseq_cli.preprocess\
# #         --source-lang 'zh'\
# #         --target-lang 'en'\
# #         --trainpref {monopref}\
# #         --destdir {binpath}\
# #         --srcdict {src_dict_file}\
# #         --tgtdict {tgt_dict_file}\
# #         --workers 2

# # create a new dataset from all the files prepared above
# get_ipython().system('cp -r ./DATA/data-bin/ted2020/ ./DATA/data-bin/ted2020_with_mono/')

# get_ipython().system('cp ./DATA/data-bin/synthetic/train.zh-en.zh.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.bin')
# get_ipython().system('cp ./DATA/data-bin/synthetic/train.zh-en.zh.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.idx')
# get_ipython().system('cp ./DATA/data-bin/synthetic/train.zh-en.en.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.bin')
# get_ipython().system('cp ./DATA/data-bin/synthetic/train.zh-en.en.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.idx')

# # Created new dataset "ted2020_with_mono"
# # 
# # 1. Change the datadir in **config** ("./DATA/data-bin/ted2020_with_mono")
# # 2. Switch back the source_lang and target_lang in **config** ("en", "zh")
# # 2. Change the savedir in **config** (eg. "./checkpoints/transformer-bt")
# # 3. Train model

# # 1. <a name=ott2019fairseq></a>Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., ... & Auli, M. (2019, June). fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations) (pp. 48-53).
# # 2. <a name=vaswani2017></a>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017, December). Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 6000-6010).
# # 3. <a name=reimers-2020-multilingual-sentence-bert></a>Reimers, N., & Gurevych, I. (2020, November). Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 4512-4525).
# # 4. <a name=tiedemann2012parallel></a>Tiedemann, J. (2012, May). Parallel Data, Tools and Interfaces in OPUS. In Lrec (Vol. 2012, pp. 2214-2218).
# # 5. <a name=kudo-richardson-2018-sentencepiece></a>Kudo, T., & Richardson, J. (2018, November). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 66-71).
# # 6. <a name=sennrich-etal-2016-improving></a>Sennrich, R., Haddow, B., & Birch, A. (2016, August). Improving Neural Machine Translation Models with Monolingual Data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 86-96).
# # 7. <a name=edunov-etal-2018-understanding></a>Edunov, S., Ott, M., Auli, M., & Grangier, D. (2018). Understanding Back-Translation at Scale. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 489-500).
# # 8. https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus
# # 9. https://ithelp.ithome.com.tw/articles/10233122
# # 10. https://nlp.seas.harvard.edu/2018/04/03/attention.html
# # 11. https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW05/HW05.ipynb



