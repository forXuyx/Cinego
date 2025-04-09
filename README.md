# Cinego(Cine means 🎬, ego means 🤖)

## 📌 简介

此项目继承自开源项目[MiniMind-V](https://github.com/jingyaogong/minimind),旨在扩展MiniMind-V实现视频理解功能，同时作为一个入门视频理解的入门教程😊。

> [!NOTE]
> 请确保你已经熟悉了[MiniMind-V](https://github.com/jingyaogong/minimind)的基本使用和训练流程。


**项目功能清单（含待做）**

- [ ] 完善UI界面
- [ ] 实现多轮对话训练逻辑
- [x] 复现GPT4Video中的VideoSummary结构
- [x] 实现数据特征的预提取（相当于少了clip获取embedding那一块的计算量，训练稍微能快一点）
- [x] 修改dataset结构兼容图片、视频、抽取特征


## 📌 快速开始

**开始前注意事项**

- 请确保具有完善的训练环境（请参照Minimind-V的README.md）
- 请参照Minimind-V准备Clip模型以及text tokenizer（这里你可以自己重新训练一个，本项目直接用的minimind自带的tokenizer）
- 请参照Minimind-V准备Minimind基座模型
- 请根据自己需求下载数据集以及checkpoint

**下载文件存放结构**

```bash
Cinego
├── out
│   ├── lm_512.pth
│   ├── lm_768.pth
│   ├── pretrain_videolm_512.pth
│   ├── pretrain_videolm_768.pth
│   ├── sft_videolm_512.pth
│   └── sft_videolm_768.pth
├── dataset
│   ├── pretrain_vlm_data.jsonl
│   ├── pretrain_images
│   ├── sft_vlm_data_video.jsonl
│   ├── sft_video_features
│   └── eval_videos
├── model
│   ├── text_tokenizer
│   └── vision_model
```

### 第0步

```bash
git clone https://github.com/forXuyx/Cinego.git
```

### Ⅰ 测试已有模型效果

#### 1.命令行问答

```bash
python eval.py
```

#### 2.或启动WebUI （待做）

```bash
streamlit run web_demo.py
```

### Ⅱ 从0开始自己训练

#### 1.开始训练

**1.1 预训练**

```bash
bash scripts/train/train_pretrain_512_8.sh
```

> 执行预训练，得到 `pretrain_videolm_*.pth` 作为预训练的输出权重（其中*为模型的dimension，默认为512）


**1.2 监督微调**

```bash
python scripts/train/train_sft_512_8.sh
```

> 执行监督微调，得到 `sft_videolm_*.pth` 作为指令微调的输出权重


#### 2.测试模型效果

确保需要测试的模型`*.pth`文件位于`./out/`目录下。

```bash
bash scripts/eval/eval_512_8.sh
```

> [!NOTE]
> 以上运行脚本均可根据自身情况自行修改，详情请见`scripts`目录下的脚本文件。

## 📌 数据介绍

在先前的项目中我选取了LLaVA-Video-178K数据集中的一个子集作为Pretrain以及SFT数据集，这是不合规的，但当初只是作为一个toy项目来做的，所以没有考虑那么多，这也导致训练出来的模型存在很大的幻视，但我发现有部分朋友在关注我这个项目，所以我现在想尝试将他做的好一点哈哈哈（大家都可以参与进来！）<br/>

目前所选数据集：
- Pretrain数据集：与Minimind中所选用的Pretrain数据集保持一致（LLaVA-Video-178K）
- SFT数据集：使用了来自[VideoChatGPT](https://huggingface.co/datasets/lmms-lab/VideoChatGPT)大概180G的视频数据集

## 📌 数据以及checkpoint下载

- SFT文本对：[百度网盘](https://pan.baidu.com/s/1CXRDig2P-Fm7D73kqJmfvA?pwd=x4fn)
- SFT视频数据：[百度网盘](https://pan.baidu.com/share/init?surl=0hJ_U7wVmYTUo75YHc_n8g&pwd=g1hf)
- SFT视频特征数据（建议使用）：上传中......
- 验证数据集：[百度网盘](https://pan.baidu.com/s/14I5ta7rnhzBmuuEBUij4vQ)
- tokenizer：[百度网盘](https://pan.baidu.com/s/1bb0HDw5lmO1BYxr3WEoreQ) (其实就是用Minimind的hq数据重新训了一遍哈哈哈)
- 全部文件：上传中......

> [!NOTE]
> 我的所以数据均位于百度网盘，对于Linux用户，我建议使用[bypy + aria2](https://lala.im/7182.html)加速下载。

## 📌 Model Structure

与MiniMind-V的整体结构一致，只是新增了一个VideoSummary块。
其结构如下（还没画）：

<!-- ![structure](./images/LLM-structure.png) -->

## 📌 A Suggestion

其实洞察整个项目，相比于Minimind-V不同就在于我新增了一个VideoSummary的结构，所以在这里其实我们最需要主要到的就是如何得到良好的视频特征表示，其他的是图像理解模型其实大同小异，这里我列举一些结构来获得视频特征表示：
- 利用Clip得到每一帧的特征表示，然后汇聚每一帧的特征表示到一个CLS中去（本项目的方法）
- 利用Blip得到每一帧的特征表示，然后在token维度拼接起来（不太建议这种方法，因为Blip预训练模型太大了）
- 直接每一帧做平均（最简单的方法）

## 📌 Acknowledge

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