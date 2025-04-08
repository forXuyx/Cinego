### Cinego(Cine means 🎬, ego means 🤖)

# 📌 简介

此项目继承自开源项目[MiniMind-V](https://github.com/jingyaogong/minimind),旨在扩展MiniMind-V实现视频理解功能，同时作为一个入门视频理解的入门教程😊。

> [!NOTE]
> 请确保你已经熟悉了[MiniMind-V](https://github.com/jingyaogong/minimind)的基本使用和训练流程。


**项目功能清单（含待做）**

- [ ] 完善UI界面
- [ ] 实现多轮对话训练逻辑
- [ ] 替换VideoSummary中注意力计算为线性注意力
- [x] 复现GPT4Video中的VideoSummary结构
- [x] 实现数据特征的预提取
- [x] 修改dataset结构兼容图片、视频、抽取特征


# 📌 快速开始

**开始前注意事项**

- 请确保具有完善的训练环境（请参照Minimind-V的README.md
- 请参照Minimind-V准备Clip模型以及text tokenizer（这里你可以自己重新训练一个，本项目直接用的minimind自带的tokenizer）
- 请根据自己需求下载数据集以及checkpoint（我在这里后续会贴出我训练所用的链接）

### 第0步

```bash
git clone https://github.com/forXuyx/Cinego.git
```

## Ⅰ 测试已有模型效果

### 1.命令行问答

```bash
python eval_model.py
```

### 2.或启动WebUI （待做）

```bash
streamlit run web_demo.py
```

## Ⅱ 从0开始自己训练

### 1.开始训练

**1.1 预训练**

```bash
python train_pretrain.py
```

> 执行预训练，得到 `pretrain_videolm_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为512）


**1.2 监督微调**

```bash
python train_sft.py
```

> 执行监督微调，得到 `sft_videolm_*.pth` 作为指令微调的输出权重


### 2.测试模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。

```bash
python eval_model.py
```

# 📌 数据介绍与训练策略

在先前的项目中我选取了LLaVA-Video-178K数据集中的一个子集作为Pretrain以及SFT数据集，这是不合规的，但当初只是作为一个toy项目来做的，所以没有考虑那么多，这也导致训练出来的模型存在很大的幻视，但我发现有部分朋友在关注我这个项目，所以我现在想尝试将他做的好一点哈哈哈（大家都可以参与进来！）<br/>

目前所选数据集：
- Pretrain数据集：与Minimind中所选用的Pretrain数据集保持一致（LLaVA-Video-178K），Pretrain阶段我们会更新VideoSummary以及后接的映射层以及LLM最后一层的参数，这样做的目的是得到良好的通用视觉特征表示
- SFT数据集：使用了来自[VideoChatGPT](https://huggingface.co/datasets/lmms-lab/VideoChatGPT)大概180G的视频数据（忒大了！！！），SFT阶段我们会开放模型的全部参数进行微调


# 📌 Model Structure

与MiniMind-V的整体结构一致，只是新增了一个VideoSummary块。
其结构如下（还没画）：

<!-- ![structure](./images/LLM-structure.png) -->

# 📌 A Suggestion

其实洞察整个项目，相比于Minimind-V不同就在于我新增了一个VideoSummary的结构，所以在这里其实我们最需要主要到的就是如何得到良好的视频特征表示，其他的是图像理解模型其实大同小异，这里我列举一些结构来获得视频特征表示：
- 利用Clip得到每一帧的特征表示，然后汇聚每一帧的特征表示到一个CLS中去（本项目的方法）
- 利用预训练的Vit + Q-former来得到每一帧的特征表示，然后拼接到一起（Q-former最后得到的特征大小似乎是32？或者我记错了，反正是比较低维的，所以就比较好拼接，拼接后的数据不会太长）
- 直接使用预训练的Video模型来提取每一个视频的特征

# 📌 Acknowledge

> [!NOTE]
> 如果觉得`Cinego`对你学习视频理解大模型有所帮助，可以在 GitHub 上加一个⭐<br/>
> 长水平有限，欢迎任何形式的修改意见，请随时提issue或pr，我会尽快查看😊<br/>

## 😊鸣谢
<summary> <b>参考链接 & 感谢以下优秀的论文或项目</b> </summary>

- 特别感谢Minimind系列作者[jingyaogong](https://github.com/jingyaogong)的优秀工作
- 感谢[GPT4Video](https://arxiv.org/abs/2311.16511)论文作者所提出的Summary结构
- 感谢数据集[Chinese-LLaVA-Vision-Instructions](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)的提供者
- 感谢[VideoChatGPT](https://huggingface.co/datasets/lmms-lab/VideoChatGPT)数据集的提供者


# License

This repository is licensed under the [Apache-2.0 License](LICENSE).