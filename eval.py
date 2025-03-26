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
from transformers import logging as hf_logging
from model.dataset import video2image

hf_logging.set_verbosity_error()

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config, device):
    tokenizer = AutoTokenizer.from_pretrained('model/text_tokenizer')
    moe_path = '_moe' if args.use_moe else ''
    ckp = f'{args.out_dir}/sft_videolm_{args.dim}{moe_path}.pth'
    model = Cinego(lm_config)
    state_dict = torch.load(ckp, map_location=device)
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)

    print(f'参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

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
    parser.add_argument('--dim', default=768, type=int)
    parser.add_argument('--n_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--stream', default=True, type=bool)
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)

    model, tokenizer, vision_model = init_model(lm_config, args.device)


    def chat_with_vlm(prompt, pixel_tensors, video_names):
        messages = [{"role": "user", "content": prompt}]

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        print(f'[Video]: {video_names}')
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

    prompts = [
        f"{model.params.image_special_token}\nIllustrate the video's narrative journey, scene by scene, with attention to detail.",
        f"{model.params.image_special_token}\nWrite an in-depth depiction of the video, covering all its aspects.",
        f"{model.params.image_special_token}\nWalk through the video, detailing its key moments and features.",
        f"{model.params.image_special_token}\nExplore the thematic and visual aspects of the video comprehensively.",
        f"{model.params.image_special_token}\nProvide a comprehensive analysis of the video's content and themes.",
        f"{model.params.image_special_token}\nShare a detailed breakdown of the video's storyline and visuals.",
        ]

    answers = [
        f"The video begins with a person standing in a doorway, holding a large, checkered blanket. The person is dressed in a black shirt with white text and khaki pants. The background reveals a cluttered room filled with various items, including a chair and some equipment. The person adjusts the blanket, wrapping it around themselves and shifting it in their arms, suggesting they might be preparing to use it or move it to another location. The narrative develops as the person continues to stand in the doorway, still holding the large, checkered blanket. They appear to be adjusting their grip on the blanket, securing it more tightly around themselves, possibly in preparation for moving or using it. The background remains consistent, with the cluttered room and various items still visible. The video wraps up with the person, now wearing a gray sweatshirt with black sleeves and khaki pants, still standing in the doorway. They adjust the blanket and eventually drop it to the floor. Bending down, they pick it up with both hands, managing the blanket as if preparing to move it or use it in some way. The video concludes with the person standing upright, holding the blanket in their hands.",
        f"The video begins with a warmly lit hallway, creating a cozy atmosphere. A chair with a blanket draped over it is positioned in front of a door adorned with a colorful floral curtain. On the left wall, a black and white portrait of a woman hangs above another framed picture that leans against the wall. A person dressed in a dark sweater and pants enters the scene, walks towards the chair, and adjusts the blanket. After making the adjustment, the person exits through the door with the floral curtain, leaving the hallway empty once again, with the chair and blanket remaining as they were."
        f"The video begins with a view of a room featuring a blue vacuum cleaner on a tiled floor. The room has wooden walls, a door with glass panes partially covered by yellow floral curtains, and various items including a small stool with a cup, a table with a microwave and some books, and a cricket bat leaning against the wall. A person wearing a blue jacket and red pants enters the frame, approaches the vacuum cleaner, adjusts it, and stands up holding the vacuum hose. The scene develops as the person stands next to the vacuum cleaner, holding the hose and appearing ready to use it. The room's setting remains consistent, with the same furniture and items in place. The person slightly adjusts their stance and the vacuum hose but does not take any significant actions. The video wraps up with the person still standing next to the vacuum cleaner, holding the hose, and appearing ready to start vacuuming.",
        f"The video takes place in a well-lit kitchen with wooden cabinets and a window that lets in natural light. A person wearing a dark long-sleeve shirt and blue jeans is seen throughout the video, engaging in various kitchen activities. Initially, the person is focused on cooking, using a red pan on the stove. The kitchen counter is adorned with various items, including a glass, a green container, and a large plant near the window, contributing to a homely atmosphere. The person is seen stirring or mixing something in the pan, occasionally adjusting their position and handling different kitchen utensils or ingredients, creating a calm and focused cooking environment. The scene then transitions to the person holding a glass and drinking from it while standing near the counter. After finishing the drink, the person moves towards the sink to rinse or wash the glass, maintaining the calm and focused atmosphere. The video concludes with the person returning to the stove, holding the red pan, and continuing to stir or mix its contents, similar to the initial activity. The consistent setting and activities throughout the video emphasize a serene and dedicated approach to cooking and kitchen chores.",
        f"The video begins with a view of a small, dimly lit room with yellowish walls and visible pipes running along the walls and ceiling. A white chest freezer is positioned against the wall. A person wearing a dark jacket enters the room carrying a stack of papers and a red folder, places them on the chest freezer, and exits the frame. They return wearing a dark hoodie and unfold a large purple plastic bag, placing and adjusting it on the chest freezer. The person stands in front of the chest freezer, looking towards the camera.\n\nThe narrative continues with the person in the dark hoodie standing in the same room, now in front of the chest freezer with the large purple plastic bag on it. They cross their arms and look towards the camera before turning to adjust the purple plastic bag, ensuring it is properly laid out on the chest freezer. The person remains focused on manipulating the bag.\n\nThe video wraps up with the person still in the dark hoodie, standing in the same room. The purple plastic bag is no longer visible on the chest freezer. The person holds a red folder and a stack of papers, appearing to organize or examine them. They then hug the stack of papers and the red folder close to their chest, looking towards the camera, and remain standing in front of the chest freezer.",
        f"The video begins with a woman standing in a well-lit kitchen, unpacking groceries from a plastic bag on the countertop. She is dressed in a black long-sleeve shirt and light-colored shorts. The kitchen features wooden cabinets, a gas stove with pots and pans, and various appliances such as a toaster and a microwave. The woman methodically removes items from the bag, including a box of pasta, which she places on the counter. She continues to unpack the groceries, organizing them as she goes. The scene develops with the woman still focused on her task, adding a can of cooking spray and a package of vegetables to the items on the counter. She finishes unpacking and stands back, having neatly organized all the groceries. The video wraps up with the woman now standing near the gas stove, adjusting the knobs to turn on the burner. The blue flame of the burner becomes visible, indicating that the stove is on. The previously unpacked items, including the box of pasta, can of cooking spray, and package of vegetables, are arranged on the countertop next to the stove. The woman appears focused as she prepares to cook, standing by the lit burner, ready to begin her culinary task.",
    ]

    
    video_paths = [
        f'dataset/eval_videos/0AGCS.mp4',
        f'dataset/eval_videos/0AMBV.mp4',
        f'dataset/eval_videos/0AYPZ.mp4',
        f'dataset/eval_videos/0BLSL.mp4',
        f'dataset/eval_videos/0BX9N.mp4',
        f'dataset/eval_videos/0BXRP.mp4',
    ]

    for prompt, answer, video_path in zip(prompts, answers, video_paths):
        video_tensor = video2image(video_path).to(args.device).unsqueeze(0)
        pixel_tensors = model.get_video_embeddings(video_tensor, vision_model)
        video_name = os.path.basename(video_path)  

        print('\n')
        chat_with_vlm(prompt, pixel_tensors, video_name)
        print('👨: ', end='')
        print('\n')
        print(answer)
