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
from model.dataset import video2image



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

    video_paths = []
    for root, dirs, files in os.walk(args.video_path):
        for file in files:
            if file.endswith('.mp4'):
                video_paths.append(os.path.join(root, file))
    
    for video_path in tqdm(video_paths):
        video_tensors = video2image(video_path).to(device)
        with torch.no_grad():
            outputs = model.vision_model(pixel_values=video_tensors)
            # vid_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
            vid_embedding = outputs.last_hidden_state.squeeze()
        
        # save the video feature
        vid_embedding = vid_embedding.cpu().numpy()
        save_path = video_path.replace(args.video_path, args.output_dir).replace('.mp4', '.npy')
        np.save(save_path, vid_embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default='dataset/all_video')
    parser.add_argument("--output-dir", type=str, default='dataset/all_video_feature')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)