import os
import requests
from typing import List
from urllib.parse import urlparse
from tqdm import tqdm

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .attention_processor import (
    AttnProcessor2_0 as AttnProcessor,
)

from .attention_processor import (
    IFAttnProcessor2_0 as IFAttnProcessor,
)
from .resampler import Resampler

from .resampler import PositionNet



class IFAdapter(torch.nn.Module):
    def __init__(self, unet, text_proj_model, pos_net, adapter_modules=None, device="cuda", num_tokens=4, ckpt_path=None):
        super().__init__()
        self.device = device
        self.adapter_modules = adapter_modules
        self.ckpt_path = ckpt_path
        self.num_tokens = num_tokens
        self.pos_net = pos_net

        self.unet = unet.to(self.device)
        # self.set_spd_adapter()
        self.text_proj_model = text_proj_model

        # if self.adapter_modules is None:
        #     self.set_spd_adapter()

        if ckpt_path is not None:
            self.load_spd_adapter(ckpt_path)

    def load_spd_adapter(self, ckpt_path: str):
        # 1. Check and create pretrained_models directory
        pretrained_models_dir = "./pretrained_models"
        if not os.path.exists(pretrained_models_dir):
            os.makedirs(pretrained_models_dir)
            print(f"Created directory: {pretrained_models_dir}")
        
        # 2. Determine if ckpt_path is a local path or HTTP address
        parsed_url = urlparse(ckpt_path)
        is_http_url = parsed_url.scheme in ['http', 'https']
        
        if is_http_url:
            # If it's an HTTP address, download the model to local
            local_path = os.path.join(pretrained_models_dir, "spd_adapter.bin")
            
            # Check if file already exists
            if os.path.exists(local_path):
                print(f"Model file already exists: {local_path}")
                ckpt_path = local_path
            else:
                print(f"Downloading model from {ckpt_path}...")
                try:
                    response = requests.get(ckpt_path, stream=True)
                    response.raise_for_status()
                    
                    # Get total file size
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(local_path, 'wb') as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading model") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    
                    print(f"Model download completed: {local_path}")
                    ckpt_path = local_path
                except Exception as e:
                    raise Exception(f"Failed to download model: {e}")
        else:
            # If it's a local path, check if file exists
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Model file not found: {ckpt_path}")
        
        # Handle directory case (original logic)
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.safetensors")

        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        orig_pos_net_sum = torch.sum(torch.stack([torch.sum(p) for p in self.pos_net.parameters()]))

        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"text_proj_model": {}, "spd_adapter": {}, "pos_net": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules when using resampler
        print(state_dict.keys())
        self.text_proj_model.load_state_dict(state_dict["text_proj_model"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["adapter_modules"], strict=False)
        self.pos_net.load_state_dict(state_dict["pos_net"], strict=True)
        # self.load_state_dict({"dummy_image_tokens": state_dict["dummy_image_tokens"]}, strict=False)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        # new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        new_pos_net_sum = torch.sum(torch.stack([torch.sum(p) for p in self.pos_net.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of text_proj_model did not change!"
        # assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
        assert orig_pos_net_sum != new_pos_net_sum, "Weights of pos_net did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    def set_scale(self, scale): 
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IFAttnProcessor):
                attn_processor.scale = scale
    
    def generate(
        self,
        pipe,
        phrase_embeds=None, 
        negative_phrase_embeds=None, 
        image=None,
        adapter_conditioning_scale=1,
        pooled_phrase_embeds=None,
        negative_pooled_phrase_embeds=None,
        phrase_eot_embeds=None, 
        negative_phrase_eot_embeds=None, 
        text_embeds=None, 
        negative_text_embeds=None,
        pooled_text_embeds=None,
        negative_pooled_text_embeds=None,   
        all_obj_attention_mask=None, 
        phrase_num_arr=None, 
        boxes=None,
        scale=1.0,
        cond_ratio=1,
        guidance_scale=0,
        seed=None,
        height=1024,
        do_classifier_free_guidance=False,
        width=1024,
        num_inference_steps=30,
        GUI_progress=None,
        **kwargs,
        ):
        self.set_scale(scale)


        if guidance_scale > 0:
            phrase_num_arr = phrase_num_arr * 2
            boxes = boxes.repeat(2,1)
            all_obj_attention_mask = torch.cat([torch.zeros_like(all_obj_attention_mask), all_obj_attention_mask], dim=0)
            phrase_embeds = torch.cat([negative_phrase_embeds, phrase_embeds], dim=0)
            phrase_eot_embeds = torch.cat([negative_phrase_eot_embeds, phrase_eot_embeds], dim=0)

        ap_tokens = self.text_proj_model(phrase_embeds, all_obj_attention_mask)
        B, n_embedding_layers, n_q, dim = ap_tokens.shape
        ap_tokens = ap_tokens.view(B, n_embedding_layers*n_q, dim)
        # ap_tokens = ap_tokens + self.pos_net(boxes).view(B, 1, -1)

        phrase_eot_embeds = phrase_eot_embeds.view(B,1,dim)
        # ap_tokens = torch.zeros_like(ap_tokens)
        # phrase_eot_embeds = torch.zeros_like(phrase_eot_embeds)
        ap_tokens = torch.concat([ap_tokens, phrase_eot_embeds], dim=1) # B n_embedding_layers*n_q+1 dim
        boxes = boxes.to(ap_tokens)
        grounding_embeddings = self.pos_net(boxes).view(B, 1, -1)
        ap_tokens = ap_tokens + grounding_embeddings

        # expand phrase_num_arr, ap_tokens, boxes
        

        cross_attention_kwargs = {
            "phrase_num_arr": phrase_num_arr,
            "ap_tokens": ap_tokens,
            "boxes": boxes
        }
                
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        
        if image is not None and adapter_conditioning_scale is not None:

            images = pipe(
                prompt_embeds=text_embeds,
                negative_prompt_embeds=negative_text_embeds,
                image = image,
                adapter_conditioning_scale = adapter_conditioning_scale,
                pooled_prompt_embeds=pooled_text_embeds,
                negative_pooled_prompt_embeds=negative_pooled_text_embeds,
                cond_ratio=cond_ratio,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                height=height,
                width=width,
                **kwargs,
            ).images
        else:
            images = pipe(
                prompt_embeds=text_embeds,
                negative_prompt_embeds=negative_text_embeds,
                
                pooled_prompt_embeds=pooled_text_embeds,
                negative_pooled_prompt_embeds=negative_pooled_text_embeds,
                cond_ratio=cond_ratio,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                height=height,
                width=width,
                GUI_progress=GUI_progress,
                **kwargs,
            ).images

        return images

    def forward(self, 
        noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, phrase_embeds, all_obj_attention_mask, eot_embeds, phrase_num_arr, boxes):
        # ip_tokens = self.image_proj_model(image_embeds)
        ap_tokens = self.text_proj_model(phrase_embeds, all_obj_attention_mask)
        B, n_embedding_layers, n_q, dim = ap_tokens.shape
        ap_tokens = ap_tokens.view(B, n_embedding_layers*n_q, dim)
        # ap_tokens = ap_tokens + self.pos_net(boxes).view(B, 1, -1)

        eot_embeds = eot_embeds.view(B,1,dim)
        ap_tokens = torch.concat([ap_tokens, eot_embeds], dim=1) # B n_embedding_layers*n_q+1 dim

        grounding_embeddings = self.pos_net(boxes).view(B, 1, -1)
        ap_tokens = ap_tokens + grounding_embeddings

        cross_attention_kwargs = {
            "phrase_num_arr": phrase_num_arr,
            "ap_tokens": ap_tokens,
            "boxes": boxes
        }
        # encoder_hidden_states = torch.cat([text_embeds, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, text_embeds, added_cond_kwargs=unet_added_cond_kwargs, cross_attention_kwargs=cross_attention_kwargs).sample
        return noise_pred
