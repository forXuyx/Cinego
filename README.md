# Cinego

此库为minimind-v所衍生的轻量视频理解大模型。

## 使用方法

### 1.训练tokenizer（可选，你也可以用我训练好的）

```shell
python train_tokenizer.py
```

### 2.提取视频特征(可选，你也可以直接下载我提取好的特征)

```shell
python preprocess.py
```

### 3.训练预训练模型

```shell
python train_pretrain.py
```

### 4.训练SFT模型

```shell
python train_sft.py
```

### 5.测试模型

```shell
python eval.py
```


## 数据集


### 由于视频数据集实在太大了，这里我的Pretrain与SFT数据集选择的均为[LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main)中的0_30_s_academic_v0_1子集


## 实现核心

### 整体采用LlaVa架构，视频特征提取实现参照了[GPT4Video](https://arxiv.org/abs/2311.16511)，本项目的实现与该论文中的实现略有不同，原文中的公式为：
$$
Att(Q, K, V) = \mathrm{softmax}\!\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr)V
$$

$$
F_s = \mathrm{CrossAttention}(Q_s,\,[f_v, Q_s],\,[f_v, Q_s])
$$

$$
F_t = \mathrm{CrossAttention}(Q_t,\,[f_v, Q_t],\,[f_v, Q_t])
$$

$$
\hat{F}_v = F_s + F_t
$$



### 结论
- 数据集还是太小了，有许多胡言乱语的描述
- 直接取平均的操作可能不能很好地表达视频的特征


## 致谢

- 特别感谢 [MiniMind](https://github.com/jingyaogong/minimind-v) 项目，本项目的架构和实现大量借鉴了他们的优秀工作
- 感谢 [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main)数据集的提供者
