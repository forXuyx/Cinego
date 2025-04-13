import argparse
import os
import random
import numpy as np
import torch
import warnings
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.llm_model import Cinego
from model.LMConfig import LMConfig
from model.dataset import video2image
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config, device):
    tokenizer = AutoTokenizer.from_pretrained('model/text_tokenizer')
    moe_path = '_moe' if args.use_moe else ''
    modes = {0: 'pretrain_videolm', 1: 'sft_videolm_image', 2: 'sft_videolm_video'}
    ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'
    model = Cinego(lm_config)
    state_dict = torch.load(ckp, map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)

    print(f'VLM参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

    vision_model = Cinego.get_vision_model()
    return model.eval().to(device), tokenizer, vision_model.eval().to(device)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Cinego")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--temperature', default=0.65, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--data_type', default=0, type=int,
                        help="0: 单图，1: 多图，2: 视频")
    parser.add_argument('--model_mode', default=0, type=int,
                        help="0: Pretrain模型，1: SFT图片模型，2: SFT视频模型")
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    model, tokenizer, vision_model = init_model(lm_config, args.device)


    def chat_with_vlm(prompt, pixel_tensors, image_names):
        messages = [{"role": "user", "content": prompt}]

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        print(f'[Image]: {image_names}')
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                pad_token_id=tokenizer.pad_token_id,
                pixel_tensors=pixel_tensors
            )
            print('🤖️: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')


    # 单图推理
    if args.data_type == 0:
        image_dir = 'dataset/eval_images/'
        prompt = f"{model.params.image_special_token}\nDescribe the image."

        for image_file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_file)
            pixel_tensors = video2image(image_path, num_frames=32, size=224).to(args.device).unsqueeze(0)
            chat_with_vlm(prompt, pixel_tensors, image_file)

    # 多图推理（涌现）
    if args.data_type == 1:
        image_dir = 'dataset/eval_images_mul/'
        prompt = f"{model.params.image_special_token}\nCompare these two images."

        for sub_dir in os.listdir(image_dir):
            sub_dir_path = os.path.join(image_dir, sub_dir)
            image_files = os.listdir(sub_dir_path)
            image_paths = [os.path.join(sub_dir_path, image_file) for image_file in image_files]
            pixel_tensors = torch.cat([video2image(image_path, num_frames=32, size=224).to(args.device) for
                                       image_path in image_paths], dim=0).unsqueeze(0)
            chat_with_vlm(prompt, pixel_tensors, sub_dir)
    
    # 视频推理
    if args.data_type == 2:
        image_dir = 'dataset/eval_videos/'
        prompt = f"{model.params.image_special_token}\nDescribe the video."

        for image_file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_file)
            pixel_tensors = video2image(image_path, num_frames=32, size=224).to(args.device).unsqueeze(0)
            chat_with_vlm(prompt, pixel_tensors, image_file)