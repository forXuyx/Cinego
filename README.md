# Cinego

此库为minimind-v所衍生的轻量视频理解大模型。

## 使用方法

### 1.训练tokenizer（可选，你也可以用我训练好的）

```shell
python train_tokenizer.py
```

### 2.提取视频特征(可选，你也可以直接下载我提取好的特征)

```shell
bash scripts/preprocess_pretrain.sh
bash scripts/preprocess_sft.sh
```


## 数据集
- tokenizer数据集：提取自Llava-video-178k中yutube视频QA数据集
- pretrain数据集：MSRVTT-QA数据集
- sft数据集：Llava-video-178k中yutube视频QA数据集的第一部分

## 致谢

*** 特别感谢 [MiniMind](https://github.com/jingyaogong/minimind-v) 项目，本项目的架构和实现大量借鉴了他们的优秀工作。 ***
