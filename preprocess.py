import torch
import torch.nn.functional as F
from tqdm import tqdm

import os
import argparse
import numpy as np
import av
from PIL import Image
from einops import rearrange
import sys
from transformers import CLIPModel
from torch.utils.data import Dataset, DataLoader
from model.dataset import video2image

class ImageDataset(Dataset):
    def __init__(self, images_list, num_frames=16):

        super().__init__()
        self.images_list = images_list
        self.num_frames = num_frames


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index: int):
        image_path = self.images_list[index]
        image_tensor = video2image(image_path, num_frames=self.num_frames, size=224)

        return image_tensor, image_path

def main(args):
    # Setup PyTorch:

    images_list = os.listdir(args.video_path)
    images_list = [os.path.join(args.video_path, image) for image in images_list]
        
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # create and load model
    model = CLIPModel.from_pretrained('model/vision_model/clip-vit-base-patch16')
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = ImageDataset(images_list, num_frames=args.num_frames)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

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
        for image_tensor, img_embedding, image_path in zip(image_tensors, img_embeddings, image_paths):
            if torch.all(torch.eq(image_tensor, torch.zeros([args.num_frames, 3, 224, 224]).to(device))) and args.type == 'video':
                print("ERROR: video frames shape mismatch:", image_tensor.shape)
                continue
            save_path = image_path.replace(args.video_path, args.output_dir).replace('.mp4', '.npy').replace('.avi', '.npy').replace('.mkv', '.npy').replace('.jpg', '.npy').replace('.png', '.npy')
            np.save(save_path, img_embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default='dataset/pretrain_images')
    parser.add_argument("--output-dir", type=str, default='dataset/pretrain_images_features')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--type", type=str, default='image')
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)