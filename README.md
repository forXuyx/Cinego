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


## 数据集


### 由于视频数据集实在太大了，这里我的Pretrain与SFT数据集选择的均为[LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main)中的0_30_s_academic_v0_1子集

## 致谢

- 特别感谢 [MiniMind](https://github.com/jingyaogong/minimind-v) 项目，本项目的架构和实现大量借鉴了他们的优秀工作
- 感谢 [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main)数据集的提供者
