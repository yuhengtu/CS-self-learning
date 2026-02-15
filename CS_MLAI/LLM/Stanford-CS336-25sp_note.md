# LEC1 Intro

the bitter lesson: maximize algorithm efficiency to leverage compute

History

![image-20251103160958491](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031609529.png)

5 parts

![image-20251103161851493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031618519.png)

1. Basics: tokenization, model architecture, training

Byte-Pair Encoding (BPE) tokenizer

![image-20251103162010261](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031620300.png)

Transformor and many small improvements

![image-20251103162052081](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031620108.png)

Training

![image-20251103162803942](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031628977.png)

2. Systems: kernels, parallelism, inference

Kernels

Trick: organize computation to maximize utilization of GPUs by minimizing data movement

Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens

![image-20251103162833617](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031628642.png)

Parallelism

Data movement between GPUs is even slower, but same 'minimize data movement' principle holds

Use collective operations (e.g., gather, reduce, all-reduce)

Shard (parameters, activations, gradients, optimizer states) across GPUs

How to split computation: {data,tensor,pipeline,sequence} parallelism

![image-20251103163057539](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031630566.png)

Inference

Goal: generate tokens given a prompt (needed to actually use models!)

Inference is also needed for reinforcement learning, test-time compute, evaluation

Globally, inference compute (every use) exceeds training compute (one-time cost)

Two phases: prefill and decode

- Prefill (similar to training): tokens are given, can process all at once (compute-bound)

- Decode: need to generate one token at a time (memory-bound)

Methods to speed up decoding:

- Use cheaper model (via model pruning, quantization, distillation)

- Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)
- Systems optimizations: KV caching, batching

![image-20251103163221620](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031632648.png)

3. Scaling Law

Goal: do experiments at small scale, predict hyperparameters/loss at large scale

TL;DR: (e.g., 1.4B parameter model should be trained on 28B tokens)

![image-20251103163711481](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031637513.png)

4. Data

Evaluation

- Perplexity: textbook evaluation for language models

- Standardized testing (e.g., MMLU, HellaSwag, GSM8K)

- Instruction following (e.g., AlpacaEval, IFEval, WildBench)

- Scaling test-time compute: chain-of-thought, ensembling

- LM-as-a-judge: evaluate generative tasks

- Full system: RAG, agents

Data curation

- Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.
- Appeal to fair use to train on copyright data? [Henderson+ 2023](https://arxiv.org/pdf/2303.15715.pdf)
- Might have to license data (e.g., Google with Reddit data) [article](https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/)
- Formats: HTML, PDF, directories (not text!)

Data processing

- Transformation: convert HTML/PDF to text (preserve content, some structure, rewriting)

- Filtering: keep high quality data, remove harmful content (via classifiers)

- Deduplication: save compute, avoid memorization; use Bloom filters or MinHash

![image-20251103171714575](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031717651.png)

5. Alignment

So far, a **base model** is raw potential, very good at completing the next token. Alignment makes the model actually useful.

Goals of alignment:

- Get the language model to follow instructions
- Tune the style (format, length, tone, etc.)
- Incorporate safety (e.g., refusals to answer harmful questions)

Two phases:

- Supervised finetuning (SFT)

  - Data often involves human annotation.
  - Intuition: base model already has the skills, just need few examples to surface them. [Zhou+ 2023](https://arxiv.org/pdf/2305.11206.pdf)
  - Supervised learning: fine-tune model to maximize p(response | prompt).

- Learning from Feedback (SFT annotation is expensive, pursue lighter annotation)

  - Preference data

    - Data: generate multiple responses using model (e.g., [A, B]) to a given prompt

    - User provides preferences (e.g., A < B or A > B)

  - Verifiers

    - Formal verifiers (e.g., for code, math)
    - Learned verifiers: train against an LM-as-a-judge

  - Algorithms

    - Proximal Policy Optimization (PPO) from reinforcement learning  [Schulman+ 2017](https://arxiv.org/pdf/1707.06347.pdf) [Ouyang+ 2022](https://arxiv.org/pdf/2203.02155.pdf)
    - Direct Policy Optimization (DPO): for preference data, simpler [Rafailov+ 2023](https://arxiv.org/pdf/2305.18290.pdf)
    - Group Relative Preference Optimization (GRPO): remove value function [Shao+ 2024](https://arxiv.org/pdf/2402.03300.pdf)



Efficiency drives design decisions

- Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware.

- Data processing: avoid wasting precious compute updating on bad / irrelevant data

- Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures.

- Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)

- Training: we can get away with a single epoch!
- Scaling laws: use less compute on smaller models to do hyperparameter tuning
- Alignment: if tune model more to desired use cases, require smaller base models

Tomorrow, we will become data-constrained...



# LEC1 Tokenization

This unit was inspired by Andrej Karpathy's video on tokenization; check it out!  [video](https://www.youtube.com/watch?v=zduSFxRajkE)



# LEC3 Architectures, hyperparameters

![image-20251103175345977](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031753013.png)

![image-20251103192805733](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031928796.png)

RoPE and SeiGLU are common now

![image-20251103193051477](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031930557.png)

![image-20251103193308898](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031933938.png)

1. LayerNorm

Pre-norm is a consensus

![image-20251103193620147](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031936185.png)

![image-20251103193826190](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031938231.png)

![image-20251103193957212](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031939252.png)

![image-20251103194141224](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031941269.png)

$\gamma$ and $\beta$ are learnable, RMSnorm become standard, 

![image-20251103194433800](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031944847.png)

RMSNorm does as good as LayerNorm, but simpler; this saves 0.17% of the FLOPs, but saves 25.5% runtime

![image-20251103194704949](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031947008.png)

![image-20251103195157819](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031951871.png)

![image-20251103195343933](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031953988.png)

remove bias term does not influence performance, speed up training run time and improve optimization stability

![image-20251103195407926](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031954973.png)

![image-20251103195640373](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511031956423.png)

2. Activation

![image-20251103200113953](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032001993.png)

$\otimes$ is element-wise multiply

![image-20251103200811192](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032008235.png)

![image-20251103201217841](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032012896.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032107006.png" alt="image-20251103210720897" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032109742.png" alt="image-20251103210947698" style="zoom:25%;" />

there are high-performace model with out GLU as activation

![image-20251103211224139](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032112181.png)

3. Serial & Parallel layers

serial can lead to less utilization of GPU

![image-20251103211351422](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032113467.png)

gain system efficiency

![image-20251103211522650](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032115694.png)

![image-20251103211727344](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511032117383.png)

4. Position Embedding

![image-20251104170752959](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041707087.png)

what matters is relative position, x is word, i is location of x

![image-20251104170826874](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041708934.png)

Sin Positional Embedding: add position to embedding

RoPE: rotate the query/key vectors

inner product are invariant to arbitrary rotation (Q K do inner product)

![image-20251104171046209](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041710275.png)

Cut the d dimensional vector into 2d pairs; $m = 1,2,3,...; \theta_i = 10000^{-\frac{2(i - 1)}{d}} \quad i \in [1, d/2]$

say Q is 128 dim, $Q = (q_1, q_2, ... , q_{128})$

let $z_1 = q_1 + j \cdot q_2, Z = (z_1, z_2, ... , z_{64})$

Rotate: $z_i' = z_i \cdot e^{j m \theta_i}$

![image-20251104171632681](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041716748.png)

![image-20251104172849850](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041728925.png)

![image-20251104174543910](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041745994.png)

5. Hyperparameters - d_ff / d_model

most hyperparameters stay unchanged across different successful LMs

![image-20251104181228836](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041812939.png)

![image-20251104181322665](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041813734.png)

first 3 models in table are not GLU models, others are. 8/3 = 2.666

![image-20251104181414237](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041814319.png)

![image-20251104181734173](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041817253.png)

![image-20251104181922450](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041819530.png)

![image-20251104182014158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041820236.png)

6. Hyperparameters - Head-dim*num-heads / model-dim

![image-20251104182633928](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041826025.png)

![image-20251104183124396](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041831483.png)

This paper argue if you have more and more heads, they will have lower and lower rank, very few dims per head will affect expressiveness. But we do not observe this in practice.

d: model dim (embedding dim); h: number of heads; d_p: per head dim

![image-20251104183249599](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041832669.png)

6. Hyperparameters - Aspect ratios

d_model (width) / n_layer (depth)

![image-20251104183905437](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041839521.png)

deep -> pipeline parallel: different layer on different devices (can have slow networking/high latency networking)

wide -> tensor parallel: split matrices on different devices (need fast networking)

different parallel paradigms have different constraints

![image-20251104183948144](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041839233.png)

Left fig: people usually choose 100 aspect ratio

Right fig: DM is model dimension (width); NL is number of layers (depth); from loss, deeper model may not help, number of parameters matter, but from downstream, deeper model is better

![image-20251104184704374](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041847463.png)

7. Hyperparameters - Vocab Size

![image-20251104190112375](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041901464.png)

8. Dropout and other regularization

Pre-training is usually one epoch

![image-20251104190705752](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041907806.png)

new models do wright decay but not dropout

![image-20251104190738962](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041907015.png)

when lr drop down, the loss with weight decay drop quicker than no weight decay

![image-20251104190937315](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041909380.png)

![image-20251104191407877](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041914933.png)

9. Stability tricks - Unstable Softmax and Logit soft-capping

try to turn blue curves into orange curve, so no loss explosion happens

![image-20251104192155650](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041921747.png)

exp is numerically badly behaved; 2 softmax, 1 at the end, one in self attention

![image-20251104192318393](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041923491.png)

for the softmax at the end, add a second term in loss, try to force los(Z(xi)) to be cloese to 0

![image-20251104192558993](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041925090.png)

for softmax in self attention, layer norm for QK

![image-20251104193533291](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041935396.png)

Logit soft-capping, Gemma2 and Olmo2; not very popular, and a paper argue against this

![image-20251104193805282](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041938378.png)

10. Attention heads: GQA/MQA, Sparse / sliding window attention

![image-20251104194105254](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041941350.png)

b: batch size, N: sequence length, d: hidden dim

we want high arithmetic intensity = memory / compute. GPU memory is expensive, compute is cheap

![image-20251104194508830](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041945926.png)

KV cache for inference efficiency

![image-20251104195051588](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041950696.png)

inference time arithmetic intensity; repeatedly loading KV to memory and putting back

![image-20251104195329647](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041953751.png)

multiple Qs , single K V -> MQA (multiple query attention)

![image-20251104195636768](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041956883.png)

GQA (grouped query attention), further trade off

![image-20251104195921278](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511041959393.png)

![image-20251104200017823](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042000936.png)

Sparse / sliding window attention

![image-20251104200200309](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042002417.png)

![image-20251104200257532](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042002651.png)

Block4 -> full attention, no positional embedding; block1-3 -> sliding attention with RoPE; 

![image-20251104200356245](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042003362.png)

![image-20251104200609665](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042006786.png)



# LEC4 MoE & Deepseek V3

more paras with same flop

![image-20251104203628229](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042036331.png)

Papers cleaerly shoe MoE > dense model

![image-20251104203642927](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042036034.png)

128e -> 128 experts

![image-20251104203703819](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042037966.png)

![image-20251104203719311](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042037426.png)

each expert on one device, route the token to the right device and compute on it

![image-20251104203752716](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042037822.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042039863.png" alt="image-20251104203943741" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042040430.png" alt="image-20251104204017312" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042040722.png" alt="image-20251104204033597" style="zoom:20%;" />

System complexity, not stable, heuristic routing strategy

![image-20251104204643795](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042046918.png)

![image-20251104204808216](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042048328.png)

1. Routing function

k = 2 is popular

![image-20251104205010041](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042050163.png)

now standard is token choose expert

![image-20251104205317510](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042053627.png)

now standard is topk; results show a hasing router (not smart router) can also give performance gain; methods like RL/solve linear assignment/solve optimal transform are elegent but too expensive to be used

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042058650.png" alt="image-20251104205804527" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042103954.png" alt="image-20251104210357827" style="zoom:15%;" />

standard formula (Mixtral, DBRX, DeepSeek v3 do not softmax at bottom, they softmax after g_i,t, minor difference)

u is residual input, e is learned vectors for each expert

softmax here is mostly for normalization

![image-20251104210654486](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042106608.png)

fine-grained expert and shared expert 

fine-grained expert: if we do k=2, the flop double; we want more experts, fine-grained expert divide one original expert into 2 smaller ones and do k=4 (dff/dmodel = 4/2 = 2); fine-grained expert has been proved to be quite useful

![image-20251104214114234](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042141417.png)

7 out of 63: top 7

![image-20251104214809572](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042148720.png)

![image-20251104215023816](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042150956.png)

![image-20251104215135667](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042151799.png)

Trining MoE

![image-20251104215617707](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042156837.png)

S-Base is linear assignment; S-Base > RL > Hash, not significantly better

![image-20251104215737233](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042157377.png)

stochastic exploring policy, no longer used

H(x) is the noisy score, Wg and Wnoise are learnable, Softplus scales the standard normal noise based on input

![image-20251104215959567](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042159718.png)

![image-20251104233838448](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042338599.png)

Heuristic balancing losses: what people do in practice

![image-20251104234216239](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042342335.png)

Per-expert balancing is the same as previous slide; which token go to which expert -> which token go to which device

![image-20251104234825319](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042348416.png)

bi is a fudge factor; if an expert not given enough token, it gets higher bi, otherwise lower bi; learn bi using online learning

bi is only for routing decision, not sent to gi; this makes training stable 

![image-20251104235126158](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511042351255.png)

![image-20251105000257094](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050002217.png)

System side of MoE: parallelize nicely

![image-20251105000427616](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050004711.png)

![image-20251105000652300](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050006406.png)

GPT 4 API gives different output even if temperature is 0

if a router route all tokens to an expert, the expert device do not have memory to load all tokens, as a result, tokens are dropped, MLP will do zero computation and residual will just add original tokens

![image-20251105000938396](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050009493.png)

![image-20251105002045360](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050020464.png)

![image-20251105002114177](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050021295.png)

MoE is hard to finetune especially with little data, prone to overfitting; one not popular solution: iterate dense layers and MoE layers, only update dense layers for fine-tune; another solution: deepseek uses a lot of data to fine-tune 

![image-20251105002218750](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050022856.png)

upcycle from dense model to MoE, get a large parameter model without pretraining from scratch, MoE is great at inference

copy the FFN a bunch of times and perturb them, continue training since this

![image-20251105002821895](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050028011.png)

![image-20251105002835740](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050028843.png)

![image-20251105002918993](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050029103.png)

Deepseek MoE architecture

Deepseek-MoE (not V1, V1 is before MoE), 64 expets, 4 routing activated, 2 shared

![image-20251105003006256](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050030353.png)

V2 same architecture, add device and communication constraint from system level; communication: tokens need to be routed to the device, after FFN need to go back to original device

![image-20251105004116603](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050041711.png)

same architecture, keep top M, delete communication loss; for aux-loss, instead of doing at batch levle, add a sequence level balancing, because in inference (does not matter in training), someone might send very OOD sequence that overwhelm certrain experts

![image-20251105004446144](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050044258.png)

DeepSeek V3 other than MoE: 

Similar to GQA/MQA, project head to lower dim space

C is a compressed version of H, instead of cache KV, cache C, and upproject to KV

![image-20251105004946975](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050049083.png)

Wuk is a new matrix, but Q takes inner product with K and Wuk can be merged into Q

Compare Q does not seem too necessary compared to compress KV

![image-20251105005300908](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050053028.png)

take hidden state, pass to a lightweight MTP1 (one layer transformer) amd predict one token in the future 

so the model predict 2 tokens in the future altogether

![image-20251105005919305](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050059428.png)

![image-20251105010256027](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511050102155.png)



# LEC9,11 Scaling Law

tune on small models and extrapolate to large models

History of scaling law

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082023529.png" alt="image-20251108202316440" style="zoom:5%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082023000.png" alt="image-20251108202333953" style="zoom:5%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082024899.png" alt="image-20251108202455828" style="zoom:5%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082025049.png" alt="image-20251108202518980" style="zoom:5%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082025309.png" alt="image-20251108202546267" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082027280.png" alt="image-20251108202717234" style="zoom:25%;" />

LLM scaling

![image-20251108202913647](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082029722.png)

when we scle dataset size, we want much larger model size so that the dataset size don't saturate in training

![image-20251108202956718](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082029749.png)

data scaling (scale free/power law)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082033524.png" alt="image-20251108203302479" style="zoom:50%;" />

why power law?

![image-20251108203502000](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082035039.png)![image-20251108203704245](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082037298.png)

in NN, slope is -0.13, -0.3, -0.095, much smaller than 1/n; for flexible/nonparametric fn class, slope should be -1/d, much slower

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082043974.png" alt="image-20251108204353920" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082044322.png" alt="image-20251108204413276" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082045093.png" alt="image-20251108204545022" style="zoom:25%;" />

usage of scaling law

![image-20251108204706606](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082047663.png)

Kaplan: data mixture only affect the offset not the slope

![image-20251109001231808](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090012927.png)

pretrain on new data > pretrain multiple epoch on same data

![image-20251109001427186](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090014227.png)

repeat high quality data or train on new data

![image-20251109001722205](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090017261.png)

![image-20251109001953274](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090019329.png)



more mysterious: model scaling

gold paper: Kaplan 2021

![image-20251109002611224](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090026269.png)![image-20251109002620617](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090026671.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090026102.png" alt="image-20251109002647065" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090027697.png" alt="image-20251109002713633" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090028727.png" alt="image-20251109002806672" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090028349.png" alt="image-20251109002850289" style="zoom:25%;" />![image-20251109003336520](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090033615.png)![image-20251109003402054](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090034131.png)

batch size and lr are important hyperparas

![image-20251109003837683](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090038785.png)![image-20251109003949989](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511090039103.png)











# LEC13,14 Data

Reasons for secrecy: (i) competitive dynamics and (ii) copyright liability

- Now there's less annotation, but there's still a lot of curation and cleaning.

![image-20251107172838140](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071728315.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071732527.png" alt="image-20251107173208416" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071733325.png" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071733455.png" alt="image-20251107173322351" style="zoom:25%;" />

BERT is a transition from training on sentences to training on documents. BERT training data consists of: books corpus and wikipedia

![image-20251107173918784](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071739899.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071739418.png" alt="image-20251107173932315" style="zoom:25%;" />

GPT2 and WebText

![image-20251107174323802](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071743919.png)

CommonCrawl, a polite crawling, does not include full Internet, not even all wikipedia

robot.txt states which crawler the website allow

![image-20251107174609606](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071746769.png)<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071751986.png" alt="image-20251107175136821" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071751564.png" alt="image-20251107175150178" style="zoom:15%;" />

convert WARC to WET (html to text)

![image-20251107175208487](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511071752830.png)

CCNet: model-based filtering

![image-20251108190038326](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081900458.png)

Collosal Clean Crawled corpus (C4), from Google T5 paper: heuristic filtering

![image-20251108190324000](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081903062.png)![image-20251108190842850](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081908913.png)

GPT3

![image-20251108190957012](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081909052.png)

The Pile

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081911944.png" alt="image-20251108191106908" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081913723.png" alt="image-20251108191340654" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081911940.png" alt="image-20251108191117882" style="zoom:5%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081911387.png" alt="image-20251108191137343" style="zoom:5%;" />

some sources of the pile

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081914734.png" alt="image-20251108191437687" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081917903.png" alt="image-20251108191717838" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081917617.png" alt="image-20251108191728548" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081918559.png" alt="image-20251108191857516" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081921850.png" alt="image-20251108192132809" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081921656.png" alt="image-20251108192148585" style="zoom:25%;" />

MassiveText: heuristic filtering, early days people are afraid of bias for model-based filtering due to weak model ability, later all people switch to model-based

![image-20251108192405070](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081924135.png)

Llama

classify high quality data: do they look like a reference of wikipedia or not

the dataset is reproduced by RedPajama V1 and open source and later further processed by SlimPajama

Redpajama V2 is a resource to do research on data filtering

![image-20251108192809820](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081928857.png)

RefinedWeb and Fineweb

![image-20251108193322869](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081933934.png)![image-20251108193409930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081934971.png)

Dolma from Olmo, still heuristic filtering

![image-20251108193538096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081935155.png)![image-20251108193607490](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081936563.png)

DataComp-LM use model-based filtering and demostrate that LLM trained on this performs best. Olmo 2 uses DCLM baseline

![image-20251108193842199](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081938237.png)![image-20251108194047785](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081940870.png)

Nemotron-CC

jusText preserves more tokens than trafilatura in data filtering; use LM rephrasing

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081951354.png" alt="image-20251108195145238" style="zoom:25%;" />![image-20251108195153628](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511081951683.png)



copyright

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082001537.png" alt="image-20251108200129443" style="zoom:5%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082001314.png" alt="image-20251108200157267" style="zoom:5%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082002856.png" alt="image-20251108200214796" style="zoom:5%;" />



Mid/post training data

long context usually appear at mid-training

![image-20251108200421792](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082004844.png)

2022: convert lots of existing NLP datasets into prompts (but they are not natural)

![image-20251108200833352](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082008384.png)

open-ended instructions, heavy use of synthetic data

![image-20251108201118237](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082011287.png)![image-20251108201132227](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511082011271.png)











# LEC15 SFT

SFT training data, look at 3 almost distinct paradigms to construct fintuning data

![image-20251105125939838](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051259993.png)

FLAN: from NLP datasets, large amount of data, low quality

![image-20251105130233460](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051302592.png)

Alpaca: LM generated

![image-20251105130523165](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051305297.png)

Oaast: human written, expensive

![image-20251105130700814](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051307953.png)

Trade-offs

![image-20251105131148617](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051311744.png)

![image-20251105131305564](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051313706.png)

Human/AI-judge prefer output list and lonfer output; but post-training does not want better format, actually want less hallucinations etc.

![image-20251105131318114](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051313236.png)

![image-20251105131656435](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051316583.png)

teach model to output citation for complicated input may result in hallucinated citation

![image-20251105131829627](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051318775.png)

it is dangerous to have complicated SFT data for model pretrained with easy data (from this perspective, RL ensures the signal is based on what the model already know, avoid being too complicated)

![image-20251105132030921](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051320142.png)

be careful with distillation data and SFT data, avoid being too difficult for the model

![image-20251105175358074](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051753200.png)

Safety, seem cannot be solved by instruction tuning

![image-20251105175407345](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051754465.png)

![image-20251105175513607](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051755809.png)

Trade-off between over-refusal

![image-20251105175552168](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051755375.png)

![image-20251105175911064](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051759295.png)

![image-20251105180030615](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051800844.png)

finetune method

![image-20251105180131482](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051801690.png)

difference between SFT and CPT (continous pretraining) is blur; the follwoing 123 becomes popular and standard, 2 is called mid-training; new base model is after stage 2

![image-20251105180209013](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051802240.png)

left fig is pre-train (stable stage from WSD: warmup, stable, decay); right fig is mid training (decay stage), mix high quality data and pretrain data

![image-20251105180552602](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051805820.png)

# LEC15 RLHF

In SFT, LM is a model of some distri

In RL, LM becomes a policy that gives good reward

![image-20251105182320227](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051823491.png)

samples from p* is expensive

![image-20251105182558917](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051825137.png)

some people prefer AI written summary than those written by himself

G-V Gap: Generator Validator Gap; maybe it is higher quality to validate than to verify

![image-20251105182834053](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051828270.png)

![image-20251105183034043](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051830289.png)

RLHF data standard setup is pairwise feedback, annotation guidelines

google gives a minute per question to annotate, which is very difficult

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051833984.png" alt="image-20251105183323760" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051835083.png" alt="image-20251105183512931" style="zoom:20%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051840196.png" alt="image-20251105184010972" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051842561.png" alt="image-20251105184213336" style="zoom:10%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051843200.png" alt="image-20251105184325964" style="zoom:10%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051845072.png" alt="image-20251105184514837" style="zoom:10%;" />

RLAIF

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051846106.png" alt="image-20251105184616884" style="zoom:25%;" />

Ultrafeedback is a very popular dataset for off-policy RLHF

Zephyr 7B find RLGPT4F works better than RLHF

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051851577.png" alt="image-20251105185102335" style="zoom:20%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051854014.png" alt="image-20251105185433753" style="zoom:20%;" />

Human like long response, model generate longer responses after RLHF

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051856170.png" alt="image-20251105185626939" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051902732.png" alt="image-20251105190221479" style="zoom:25%;" />

first term is reward, second is KL divergence to avoid RL goes too far from SFTed model, third is pretrain while do RL to avoid catastrophically forget

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051904423.png" alt="image-20251105190408196" style="zoom:25%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051908645.png" alt="image-20251105190846384" style="zoom:20%;" />

Attempt2: use advantage At, which is a reduced variance version of the reward; and goes off-policy with importance sampling 

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051917546.png" alt="image-20251105191724380" style="zoom:25%;" />

PPO is complicated, but the other stuff people tried don't work as well

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051944263.png" alt="image-20251105194452001" style="zoom:25%;" />

DPO simplify PPO and works as well, get rid of reward, put it into loss

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051946596.png" alt="image-20251105194659356" style="zoom:25%;" />

assume the policy $\pi_\theta$ is not an NN, it is an arbitrary fn instead

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511051949339.png" alt="image-20251105194901068" style="zoom:15%;" /><img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062314223.png" alt="image-20251106231451030" style="zoom:15%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062319054.png" alt="image-20251106231902716" style="zoom:25%;" />

SimPO take out ref and normalize by sequence length

Important: In RL many findings are setting-dependent, the first AI2 paper finds PPO > DPO, the second finds they are similar when SFT is good, normalize helps

![image-20251106232216327](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062322566.png)![image-20251106232556984](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062325277.png)

RL concern; RL overfits on human feedback and noisy AI feedback, but not on noiseless AI feedback

![image-20251106232807031](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062328271.png)![image-20251106232905811](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202511062329077.png)