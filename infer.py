import os
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from pycocotools.coco import COCO
from ifa_dataset import create_dataloader
from if_adapter.resampler import Resampler, PositionNet
from if_adapter.if_adapter import IFAdapter
from if_adapter.sdxl_pipeline import StableDiffusionXLPipeline
from if_adapter.attention_processor import (
    AttnProcessor2_0 as AttnProcessor,
)
from if_adapter.attention_processor import (
    IFAttnProcessor2_0 as IFAttnProcessor,
)


font_path = "arial.ttf"
font_size = 20
font = ImageFont.truetype(font_path, font_size)


def encode_phrases(all_obj_ids, all_obj_ids_2, text_encoder, text_encoder_2, all_obj_attention_mask, device, extract_embedding_layers=3):
    """
    prompt_embeds: total_phrases_in_one_batch, extract_embedding_layers, l, dim1+dim2
    pooled_prompt_embeds: total_phrases_in_one_batch, dim1
    """

    eot_pos = (all_obj_attention_mask.sum(axis=-1)-1).to(device)
    prompt_embeds_list = []
    eot_embeds_list = []

    for te, ids in zip([text_encoder, text_encoder_2], [all_obj_ids, all_obj_ids_2]):
        prompt_embeds = te(ids.to(device), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        eot_embeddings = prompt_embeds.hidden_states[-2][torch.arange(all_obj_ids.shape[0]), eot_pos, :]
        eot_embeds_list.append(eot_embeddings)

        prompt_embeds = prompt_embeds.hidden_states[-2:-(2*extract_embedding_layers+1):-2]
        n, l, dim = prompt_embeds[0].shape
        prompt_embeds = torch.concat(list(prompt_embeds)).reshape(extract_embedding_layers, n, l, dim)
        prompt_embeds = prompt_embeds.permute(1,0,2,3)
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds = prompt_embeds[0]
    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    eot_embeds = torch.concat(eot_embeds_list, dim=-1)
    
    all_obj_attention_mask = all_obj_attention_mask.unsqueeze(1).repeat(1,extract_embedding_layers,1)

    # eot_embeddings: should be B 1 2048
    return prompt_embeds, pooled_prompt_embeds, all_obj_attention_mask, eot_embeds

def prepare_phrase_embeddings(batch, text_encoder, text_encoder_2, device, extract_embedding_layers):
    all_obj_ids = batch["all_obj_ids"]
    all_obj_ids_2 = batch["all_obj_ids_2"]
    all_obj_attention_mask = batch["all_obj_attention_mask"]
    phrase_num_arr = [len(obj_ids) for obj_ids in all_obj_ids]

    all_obj_ids = list(itertools.chain(*all_obj_ids))
    all_obj_ids = torch.cat(all_obj_ids).to(device)

    all_obj_ids_2 = list(itertools.chain(*all_obj_ids_2))
    all_obj_ids_2 = torch.cat(all_obj_ids_2).to(device)

    all_obj_attention_mask = list(itertools.chain(*all_obj_attention_mask))
    all_obj_attention_mask = torch.cat(all_obj_attention_mask).to(device)

    prompt_embeds, pooled_prompt_embeds, all_obj_attention_mask, eot_embeds = encode_phrases(all_obj_ids, all_obj_ids_2, text_encoder, text_encoder_2, all_obj_attention_mask, device, extract_embedding_layers)


    return prompt_embeds, pooled_prompt_embeds, all_obj_attention_mask, eot_embeds, phrase_num_arr



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="https://huggingface.co/WuYW/IFAdapter/resolve/main/spd_adapter.bin",
        help="Path to pretrained if adapter model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Path to save your results.",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=1,
        help=(
            "Number of samples you want to generate."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "The seed."
        ),
    )
    parser.add_argument(
        "--control_ratio",
        type=float,
        default=1,
        help=(
            "Number of denoising steps to exert layout control."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help=(
            "Resolution of generated image."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help=(
            "Denoising steps."
        ),
    )
    parser.add_argument(
        "--dataset_txt_path",
        type=str,
        default="infer.txt",
        help=(
            "The test dataset."
        ),
    )
    
    args = parser.parse_args()

    return args




def main():
    args = parse_args()
    accelerator = Accelerator(
            mixed_precision="fp16"
        )
    device = accelerator.device
    # adapter_path = "./pretrained_models/spd_adapter.bin"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        add_watermarker=False,
    ).to("cuda")

    #-------------------------- model hyper params ----------------------
    resampler_dim = 2560
    resampler_depth = 4
    resampler_dim_head = 128
    resampler_num_heads = 20
    resampler_num_queries = 4
    resampler_ff_mult = 4
    fourier_freqs = 64
    #-------------------------- model hyper params ----------------------

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    text_encoder_2 = pipe.text_encoder_2
    tokenizer_2 = pipe.tokenizer_2
    unet = pipe.unet

    text_proj_model = Resampler(
            dim=resampler_dim,
            depth=resampler_depth,
            dim_head=resampler_dim_head,
            heads=resampler_num_heads,
            num_queries=resampler_num_queries,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=resampler_ff_mult,
            phrase_embeddings_dim=text_encoder.config.projection_dim + text_encoder_2.config.projection_dim,
        ).to(device, dtype=torch.float32)
    pos_net = PositionNet(unet.config.cross_attention_dim, fourier_freqs=fourier_freqs)

    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            sub_block_id = int(name.split(".")[-3])
            if sub_block_id >= 4:
                cross_attention_dim = None 
            hidden_size = unet.config.block_out_channels[-1]
            num_heads = unet.config.attention_head_dim[-1]
            
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            sub_block_id = int(name.split(".")[-3])
            if block_id == 1:
                cross_attention_dim = None 
            if sub_block_id >= 4:
                cross_attention_dim = None 

            hidden_size = list(reversed(unet.config.block_out_channels))[block_id] 
            num_heads = list(reversed(unet.config.attention_head_dim))[block_id]

        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            num_heads = unet.config.attention_head_dim[block_id]
            cross_attention_dim = None 

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = IFAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    num_heads=num_heads,
                    scale=1,
                    num_tokens=resampler_num_queries,
                    enable_alpha=True,
                    max_obj=10
                    
                )
            
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    ifa_model = IFAdapter(unet, text_proj_model, pos_net, adapter_modules=adapter_modules, device=device, num_tokens=4, ckpt_path=args.adapter_path)
    ifa_model.to(device, dtype=torch.float32)

    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    test_dataloader = create_dataloader(tokenizer=tokenizer, tokenizer_2=tokenizer_2, img_size=args.resolution, txt_file=args.dataset_txt_path, batch_size=1, num_workers=0, max_obj=10, use_label=False)
    
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            for idx in range(args.num_sample):
                with torch.no_grad():
                    phrase_embeds, pooled_phrase_embeds, all_obj_attention_mask, phrase_eot_embeds, phrase_num_arr = prepare_phrase_embeddings(batch, text_encoder, text_encoder_2, device, 3)
                
                phrase_embeds_uncond = torch.zeros_like(phrase_embeds)
                phrase_eot_embeds_uncond = torch.zeros_like(phrase_eot_embeds)
                pooled_phrase_embeds_uncond = torch.zeros_like(pooled_phrase_embeds)

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                                
                all_boxes = torch.stack(list(itertools.chain(*batch["all_boxes"]))).to(device)
                text_embeds_uncond = torch.zeros_like(text_embeds)
                pooled_text_embeds_uncond = torch.zeros_like(pooled_text_embeds)
                images = ifa_model.generate(
                    pipe=pipe,
                    phrase_embeds=phrase_embeds.to(device, dtype=torch.float32), 
                    negative_phrase_embeds=phrase_embeds_uncond.to(device, dtype=torch.float32), 
                    pooled_phrase_embeds=pooled_text_embeds.to(device, dtype=torch.float32),
                    negative_pooled_phrase_embeds=pooled_text_embeds_uncond,
                    phrase_eot_embeds=phrase_eot_embeds.to(device, dtype=torch.float32), 
                    negative_phrase_eot_embeds=phrase_eot_embeds_uncond.to(device, dtype=torch.float32), 
                    text_embeds=text_embeds.to(device, dtype=torch.float32), 
                    negative_text_embeds=text_embeds_uncond,
                    pooled_text_embeds=pooled_text_embeds,
                    negative_pooled_text_embeds=pooled_text_embeds_uncond,
                    cond_ratio=args.control_ratio, 
                    all_obj_attention_mask=all_obj_attention_mask, 
                    phrase_num_arr=phrase_num_arr, 
                    boxes=all_boxes.to(device),
                    scale=1,
                    seed=args.seed,
                    guidance_scale=7.5,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_inference_steps)

                save_path = os.path.join(args.output_path)
                os.makedirs(save_path, exist_ok=True)
                
                color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
                font = ImageFont.truetype('Rainbow-Party-2.ttf', 20)

                for file_name, boxes, captions, image in zip(batch["file_names"], batch["all_boxes"], batch["all_text"], images):
                    os.makedirs(os.path.join(save_path,  "images"), exist_ok=True)
                    image.save(os.path.join(save_path, "images", f"{file_name[:-5]}_{idx}.png"))
                    try:
                        white_board = Image.new('RGB', (args.resolution, args.resolution), (255, 255, 255))
                        draw = ImageDraw.Draw(white_board)
                        W, H = white_board.size
                        for i in range(len(boxes)):
                            color = color_list[i]
                            box = boxes[i]
                            adjusted_bbox = (
                                int(box[0]*W),
                                int(box[1]*W),
                                int(box[2]*W),
                                int(box[3]*W),
                            )
                            text_x = adjusted_bbox[0]+10
                            text_y = adjusted_bbox[1]+10
                            draw.text((int(text_x), int(text_y)), captions[i], fill=color, font=font)
                            draw.rectangle(adjusted_bbox, outline="red", width=4)
                        os.makedirs(os.path.join(save_path,  "layouts"), exist_ok=True)
                        white_board.save(os.path.join(save_path,  "layouts", f"{file_name[:-5]}_{idx}.png"))
                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")


if __name__ == "__main__":
    main() 