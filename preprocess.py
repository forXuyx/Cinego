import torch
import torch.nn.functional as F
from tqdm import tqdm

import os
import argparse
import numpy as np
from PIL import Image
from einops import rearrange
import sys
from transformers import CLIPModel
from torch.utils.data import Dataset, DataLoader
from model.dataset import video2image

class ImageDataset(Dataset):
    def __init__(self, images_path):

        super().__init__()
        self.images_path = images_path
        self.images_list = os.listdir(images_path)


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index: int):
        image_path = os.path.join(self.images_path, self.images_list[index])
        image_tensor = video2image(image_path, num_frames=8, size=224)

        return image_tensor, image_path

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # create and load model
    model = CLIPModel.from_pretrained('model/vision_model/clip-vit-base-patch16')
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = ImageDataset(args.video_path)
    data_loader = DataLoader(dataset, batch_size=128, num_workers=16)

    for image_tensors, image_paths in tqdm(data_loader):
        image_tensors = image_tensors.to(device)
        bs, frame, c, h, w = image_tensors.shape
        image_tensors = image_tensors.view(-1, c, h, w)
        with torch.no_grad():
            outputs = model.vision_model(pixel_values=image_tensors)
        img_embeddings = outputs.last_hidden_state.squeeze()
        img_embeddings = rearrange(img_embeddings, '(bs frame) token_num dim -> bs frame token_num dim', bs=bs)
        
        # save the video feature
        img_embeddings = img_embeddings.cpu().numpy()
        for img_embedding, image_path in zip(img_embeddings, image_paths):
            save_path = image_path.replace(args.video_path, args.output_dir).replace('.mp4', '.npy').replace('.avi', '.npy').replace('.mkv', '.npy').replace('.jpg', '.npy').replace('.png', '.npy')
            np.save(save_path, img_embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default='dataset/pretrain_images')
    parser.add_argument("--output-dir", type=str, default='dataset/pretrain_images_features')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)