# LLM

gpt实则两三个token组成一个中文文字

llm在做文字接龙

![image-20240605024926314](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406050249473.png)

![image-20240605164241940](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051642076.png)

![image-20240605160746067](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051607181.png)

***

第一阶段 pre-train：SSL (masked) 学习网上爬下来的乱七八糟资料

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406050252525.png" alt="image-20240605025241463" style="zoom:25%;" />

![image-20240605150921605](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051509657.png)

GPT3很难控制，prompt格式很重要，你问他问题他只会乱接龙；没有教他回答问题，只是教他文字接龙

****

第二阶段：人类整理问题+答案，supervised learning

人类标注资料有限，因此关键是使用第一阶段的参数作为初始参数

![image-20240605152440543](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051524662.png)

保持第一阶段的参数固定，新加一些参数xyz以供优化，更快

![image-20240605153142815](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051531955.png)

![image-20240605153822904](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051538065.png)

![image-20240605154359001](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051543149.png)

![image-20240605154558096](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051545170.png)

finetune这一步不需要大量资料，几万笔即可，重点是质量高，less is more

openai知道真实用户会问什么问题

![image-20240605155942184](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051559306.png)

GPT逆向工程

![image-20240605160115152](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051601239.png)

但是没有pretrain参数 -> llama开源了

![image-20240605160402929](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051604065.png)

![image-20240605160517493](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051605564.png)

***

第三阶段：RLHF (PPO)

人类比较轻松，只需要判断两个模型输出哪个比较好即可，人类回馈有偏见

训练一个reward model来模拟人类的判断

横轴是向reward model学习的次数，实线是人类的喜好程度，虚线是reward model的喜好程度

![image-20240605162707562](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051627680.png)

![image-20240605162823411](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051628494.png)

![image-20240605163547234](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051635381.png)

RLAIF，可以让模型自己对自己做RLAIF，模型可能没能力输出好答案，但有能力判断答案好坏

![image-20240605163751663](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051637764.png)

训练一个safety reward model，一个helpfulness reward model

![image-20240605163903897](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051639972.png)

***

Speculative Decoding: 加快语言模型输出时间

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051702260.png" alt="image-20240605170221139" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051702752.png" alt="image-20240605170246682" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051703517.png" alt="image-20240605170323403" style="zoom:25%;" />

![image-20240605170402333](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051704458.png)



# LoRA & QLoRA

![image-20240613165149324](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406131651475.png)

r = 1-64

alpha: 0-1, amount of change added to original model weights

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406131656388.png" alt="image-20240613165654285" style="zoom:25%;" />

![image-20240613172819064](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406131728206.png)



# LLM safety

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406051706556.png" alt="image-20240605170615421" style="zoom:25%;" />



<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061606890.png" alt="image-20240606160643692" style="zoom:25%;" />

gemini有谷歌查询验证功能

![image-20240606161517151](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061615237.png)

***

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061614062.png" alt="image-20240606161452911" style="zoom:25%;" />

![image-20240606161715500](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061617641.png)

让LLM排序一模一样的履历，仅名字不同，排一千次

![image-20240606162010757](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061620864.png)

![image-20240606162254276](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061622427.png)

![image-20240606162403294](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061624371.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061625733.png" alt="image-20240606162549604" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061629166.png" alt="image-20240606162956023" style="zoom:25%;" />

***

目前只能训练出比较准确的针对单一语言模型的分类器

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061632854.png" alt="image-20240606163221729" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061634296.png" alt="image-20240606163441221" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061635472.png" alt="image-20240606163523397" style="zoom:25%;" />

如果只是润稿的话检测器不会输出这么高的概率

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061636021.png" style="zoom:25%;" />

水印

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061638340.png" alt="image-20240606163855267" style="zoom:25%;" />

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061639218.png" alt="image-20240606163916143" style="zoom:25%;" />

***

诈骗LLM prompt hacking

![image-20240606164424934](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061644063.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061645831.png" alt="image-20240606164514744" style="zoom:25%;" />

对gpt4o使用注音符号

![image-20240606164648411](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061646488.png)

![image-20240606164732170](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061647253.png)

![image-20240606164817120](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061648198.png)

<img src="https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061650900.png" alt="image-20240606165028744" style="zoom:25%;" />

重复单词，突然吐出个人信息，10%准确率

![image-20240606165149186](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061651265.png)

Prompt Injection

![image-20240606165603128](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061656211.png)

![image-20240606165528407](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406061655504.png)



# LLM Audio

![image-20240625235013928](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252350141.png)

![image-20240625235030575](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252350635.png)

声音16kHz，一秒采样16000次，接龙太慢；使用speech unit

![image-20240625235527130](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252355211.png)

![image-20240625235635287](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252356362.png)

文字 + speech unit

![image-20240625235842860](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406252358924.png)

2个人同时跟model说话

![image-20240626000015930](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260000077.png)

棒读指说话非常平淡，模型大了会学出语气

![image-20240626000316442](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260003567.png)

全部用语音pretrain资料太少

![image-20240626000543953](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260005071.png)

教已经pretrain的语言模型学一门新的语言

![image-20240626001243522](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260012641.png)

![image-20240626001423722](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260014796.png)

模型讲话都是同一个人的声音；finetune往往不需要太多资料，只要少量高质量

也可以用语音转换技术转换成同一个人的声音

![image-20240626001832834](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260018898.png)

语音模型相比语言模型，需要猜什么时候开始接话；人可能会打断语音模型，也有可以在和语音模型合唱

第二个频道记录模型自己说过什么，模型输入两个频道的内容，接龙第二个频道的内容

![image-20240626002429395](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260024508.png)

![image-20240626002502628](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260025707.png)

![image-20240626002702265](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406260027342.png)



https://www.youtube.com/watch?v=2vu6u5CrMYQ

llama vocabulary size 32K

audio token vocabulary 1024

![image-20240630153652966](https://cdn.jsdelivr.net/gh/yuhengtu/typora_images@master/img/202406301536155.png)




