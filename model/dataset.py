import json
import av
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import cv2
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def video2image(video_path, num_frames=4, size=224):
    def preprocess(size, image):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(image)

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    
    # 解码所有帧
    frames = [frame for frame in container.decode(video_stream)]
    total_frames = len(frames)
    if total_frames == 0:
        print("ERROR: problem reading video file:", video_path)
        return torch.zeros([1, 3, size, size], dtype=torch.float32)
    
    # 均匀选择 num_frames 帧
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    images = []
    for i in indices:
        # 将 pyAV 的帧转换为 PIL Image
        pil_img = frames[i].to_image()
        img_tensor = preprocess(size, pil_img)
        images.append(img_tensor.numpy())
    
    images = np.stack(images, axis=0)
    video_frames = torch.tensor(images)
    return video_frames


class VideoDataset(Dataset):
    def __init__(self, jsonl_path, videos_path, tokenizer, max_length=512,
                 video_special_token='@' * 197):

        super().__init__()
        self.samples = self.load_data(jsonl_path)
        self.videos_path = videos_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.video_token = video_special_token
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<video>', self.video_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        video_name = sample['video']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        video_path = f'{self.videos_path}/{video_name}'
        feature_path = video_path.replace('.mp4', '.npy').replace('_video', '_video_feature')
        feature = np.load(feature_path).astype(np.float32)
        video_tensor = torch.tensor(feature, dtype=torch.float32)

        return X, Y, loss_mask, video_tensor
