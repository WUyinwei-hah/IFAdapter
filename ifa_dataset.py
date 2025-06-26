import torch
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import random
from torchvision import transforms
import json
import torch
import os
from transformers import CLIPTokenizer
import random
from pycocotools.coco import COCO

def convert_coco_box(bbox, img_info):
    x0 = bbox[0]/img_info['width']
    y0 = bbox[1]/img_info['height']
    x1 = (bbox[0]+bbox[2])/img_info['width']
    y1 = (bbox[1]+bbox[3])/img_info['height']
    return [x0, y0, x1, y1]


def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

def to_valid(x0, y0, x1, y1, image_size, min_box_size):
    valid = True

    if x0>image_size or y0>image_size or x1<0 or y1<0:
        valid = False # no way to make this box vide, it is completely cropped out 
        return valid, (None, None, None, None)

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_size)
    y1 = min(y1, image_size)

    if (x1-x0)*(y1-y0) / (image_size*image_size) < min_box_size:
        valid = False
        return valid, (None, None, None, None)
     
    return valid, (x0, y0, x1, y1)

def recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, image_size, min_box_size):
    """
    x,y,w,h:  the original annotation corresponding to the raw image size.
    trans_info: what resizing and cropping have been applied to the raw image 
    image_size:  what is the final image size  
    """

    x0 = x * trans_info["performed_scale"] - trans_info['crop_x'] 
    y0 = y * trans_info["performed_scale"] - trans_info['crop_y'] 
    x1 = (x + w) * trans_info["performed_scale"] - trans_info['crop_x'] 
    y1 = (y + h) * trans_info["performed_scale"] - trans_info['crop_y'] 


    # at this point, box annotation has been recalculated based on scaling and cropping
    # but some point may fall off the image_size region (e.g., negative value), thus we 
    # need to clamp them into 0-image_size. But if all points falling outsize of image 
    # region, then we will consider this is an invalid box. 
    valid, (x0, y0, x1, y1) = to_valid(x0, y0, x1, y1, image_size, min_box_size)

    if valid:
        # we also perform random flip. 
        # Here boxes are valid, and are based on image_size 
        if trans_info["performed_flip"]:
            x0, x1 = image_size-x1, image_size-x0

    return valid, (x0, y0, x1, y1)



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    WW, HH = pil_image.size

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )

    scale = image_size / min(*pil_image.size)

    
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )

    # at this point, the min of pil_image side is desired image_size
    performed_scale = image_size / min(WW, HH)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    
    info = {"performed_scale":performed_scale, 'crop_y':crop_y, 'crop_x':crop_x, "WW":WW, 'HH':HH}

    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size], info



# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, text_file, tokenizer, tokenizer_2, size=512, center_crop=True, random_flip=True, image_root_path="", max_obj=8, min_box_size=0.01, use_label=False):
        super().__init__()

        self.use_label = use_label
        self.random_flip = random_flip
        self.tokenizer = tokenizer
        self.min_box_size = min_box_size
        self.max_obj = max_obj
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.image_root_path = image_root_path
    
        # self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        with open(text_file, 'r') as f:
            lines = f.readlines()
            train_files = [line.strip() for line in lines]
            self.train_files = train_files
            # print(f"totally {len(train_files)} samples")

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.to_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    
    def transform_image(self, pil_image):

        arr, info = center_crop_arr(pil_image, self.size)
        info["performed_flip"] = False

        image_tensor = self.to_tensor_transform(arr.copy())


        return image_tensor, info

    def __getitem__(self, idx):
        out = {}
        f = open(os.path.join(self.train_files[idx]))
        try:
            data = json.load(f)
            f.close()
        except:
            # show error message and return None
            assert 1 == 0, os.path.join(self.root_dir, self.train_files[idx])

        file_name = os.path.join(self.train_files[idx]).split("/")[-1]
        caption = data["caption"]

        all_boxes = []
        all_obj_ids = []
        all_obj_ids_2 = []
        all_obj_attention_mask = []
        all_text = []
        all_label = []

        for ann_idx, anno in enumerate(data["annotations"]):
            x, y, w, h = anno['box']
            x0 = x
            y0 = y 
            x1 = (x + w)
            y1 = (y + h)
            if True:
                all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.size ) # scale to 0-1

                # text = anno['caption']
                text = anno["caption"] if isinstance(anno["caption"], str) else anno["caption"][0]
                all_text.append(text)
                output1 = self.tokenizer(
                    text,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                text_input_ids = output1.input_ids
                attention_mask = output1.attention_mask
                output2 = self.tokenizer_2(
                    text,
                    max_length=self.tokenizer_2.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                text_input_ids_2 = output2.input_ids
                # all_obj_captions.append( anno['caption'] )
                all_obj_ids.append(text_input_ids)
                all_obj_ids_2.append(text_input_ids_2)
                all_obj_attention_mask.append(attention_mask)

        # get text and tokenize
        output1 = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        text_input_ids = output1.input_ids
        attention_mask = output1.attention_mask
        
        output2 = self.tokenizer_2(
            caption,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        text_input_ids_2 = output2.input_ids

        if len(all_obj_ids) == 0:
            all_obj_ids.append(text_input_ids)
            all_obj_ids_2.append(text_input_ids_2)
            all_obj_attention_mask.append(attention_mask)
            all_boxes.append( torch.tensor([0.0,0.0,1.0,1.0]) )
        
        if len(all_obj_ids)>self.max_obj:

            random_indices = random.sample(range(len(all_obj_ids)), self.max_obj)
            
            all_text = [all_text[i] for i in random_indices]
            all_obj_ids = [all_obj_ids[i] for i in random_indices]
            all_boxes = [all_boxes[i] for i in random_indices]
            all_obj_ids_2 = [all_obj_ids_2[i] for i in random_indices]
            all_obj_attention_mask = [all_obj_attention_mask[i] for i in random_indices]
        
        return {
            "caption": caption,
            "all_text": all_text,
            "file_name": file_name,
            "boxes": all_boxes,
            "text_input_ids": text_input_ids,
            "all_obj_ids": all_obj_ids,
            "all_obj_ids_2": all_obj_ids_2,
            "all_obj_attention_mask": all_obj_attention_mask,
            "text_input_ids_2": text_input_ids_2,
            "attention_mask": attention_mask,
        }
        
    
    def __len__(self):
        return len(self.train_files)

def collate_fn(data):
    file_names = [example["file_name"] for example in data]
    

    caption = [example["caption"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    attention_mask = torch.cat([example["attention_mask"] for example in data], dim=0)

    # boxes = torch.cat([example["boxes"] for example in data], dim=0)
    boxes = [example["boxes"] for example in data]
    all_text = [example["all_text"] for example in data]

    all_obj_ids = [example["all_obj_ids"] for example in data]
    all_obj_ids_2 = [example["all_obj_ids_2"] for example in data]
    all_obj_attention_mask = [example["all_obj_attention_mask"] for example in data]
    # all_obj_ids_2 = torch.cat([example["all_obj_ids_2"] for example in data], dim=0)
    # all_obj_attention_mask = torch.cat([example["all_obj_attention_mask"] for example in data], dim=0)


    return {
        "caption": caption,
        "file_names": file_names,
        "all_text": all_text,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "attention_mask": attention_mask,
        "all_obj_ids": all_obj_ids,
        "all_obj_ids_2": all_obj_ids_2,
        "all_obj_attention_mask": all_obj_attention_mask,
        "all_boxes": boxes,
    }


def create_dataloader(tokenizer=None, tokenizer_2=None, img_size=512, txt_file="/group/40021/wuyinwei/SpatialGuidanceDiffusion/vis.txt", batch_size=2, num_workers=0, max_obj=8, min_box_size=0.01, use_label=False):
    dataset = MyDataset(txt_file, tokenizer, tokenizer_2, size=img_size, max_obj=max_obj, min_box_size=min_box_size, use_label=use_label)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return dataloader

if __name__ == "__main__":
    

    dataloader = create_dataloader(txt_file="/group/40021/wuyinwei/SpatialGuidanceDiffusion/vis.txt", batch_size=2, sdxl_name='stabilityai/stable-diffusion-xl-base-1.0')
    for step, batch in enumerate(dataloader):
        image = batch["images"]
        continue

