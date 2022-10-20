# This file is code for making dataset for visual grounding tasks.
# Author: Qianyu Chen
# Date: 2022-10
 
# Copyright (c) THUNLP, Tsinghua University. All rights reserved. 
# See LICENSE file in the project root for license information.
import os
import re
import PIL
import json
import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from dataset.randaugment import RandomAugment


class Grounding_train_dataset(Dataset):
    def __init__(self, ann_file, max_words=200,  resize_ratio=0.25, img_res=256, half=None):   
        super().__init__()
        self.img_res = img_res     
        self.ann = []
        print("Creating dataset")
        for f in ann_file:
            self.ann.extend(json.load(open(f)))
        
        print(len(self.ann))
        self.max_words = max_words
        #image augmentation func
        self.aug_transform = Augfunc(True, resize_ratio)
        #the number of position tokens is 512
        self.pos_dict = {x:f"[pos_{x}]" for x in range(512)}
        if half is not None:
            length = len(self.ann)/2.0
            if half==0:
                self.ann = self.ann[:length]
            elif half==1:
                self.ann = self.ann[length:]
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        ann = self.ann[index].copy()
        image = Image.open(ann['file_name']).convert('RGB')
        
        bbox_list = torch.as_tensor(ann['bbox_list'], dtype=torch.float32).clamp(min=0)
        w, h = image.size
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = torch.min(bbox_list.reshape(-1, 2, 2), max_size)
        ann['bbox_list'] = cropped_boxes.reshape(-1,4).numpy().tolist()
        image, ann, do_horizontal = self.aug_transform.random_aug(image, ann, True, True, self.img_res)

        assert len(ann['tokens_positive']) == len(ann['bbox_list'])
        seq = ann['normal_caption'] if 'normal_caption' in ann else ann['caption']
        tokens2bbox={}
        for tokens, bbox in zip(ann['tokens_positive'], ann['bbox_list']):
            token_id = str(tokens[0])+str(tokens[1])
            pos_seq = ['  @@ '] 
            bbox_512 = [int(xy*512/self.img_res) if int(xy*512/self.img_res) <=511 else 511  for xy in bbox]
            pos_seq.extend([self.pos_dict[int(x)] for x in bbox_512])
            pos_seq.append(' ## ')
            tokens2bbox[token_id] = ' '.join(pos_seq)
        tokens_end = ann['tokens_positive'][1:]
        tokens_end.append([10000,0])
        new_seq = seq[:ann['tokens_positive'][0][0]]
        for s, e in zip(ann['tokens_positive'], tokens_end):
            id = str(s[0])+str(s[1])
            pos_seq =  tokens2bbox[id]
            new_seq += seq[s[0]:s[1]]
            new_seq += pos_seq
            new_seq += seq[s[1]:e[0]]
        caption = new_seq
        if do_horizontal:
            caption = caption.replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right")
        caption = pre_caption(caption, self.max_words)
        
        return image, caption


class Grounding_eval_dataset(Dataset):
    def __init__(self, ann_file, img_res, ):        
        self.ann = []
        print("Creating dataset")
        print(ann_file)
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        print(len(self.ann))
        self.img_res = img_res
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([ 
            transforms.Resize((self.img_res, self.img_res),interpolation=Image.BICUBIC),\
            transforms.ToTensor(),\
            normalize,\
        ])

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        image = Image.open(ann['file_name']).convert('RGB')

        image = self.transform(image)
        caption = ann['pseudo_caption']
        tt = caption.split(' ')
        new_caption = []
        for x in tt:
            if '[pos' in x:
                new_caption.append('[pos_1]')
            else:
                new_caption.append(x)
        caption = ' '.join(new_caption)
        caption = pre_caption(caption, 500)
        if 'bbox' in ann:
            bbox = torch.tensor(ann['bbox'],dtype=torch.float32) 
        elif 'gt_bbox' in ann:
            bbox = torch.tensor(ann['gt_bbox'],dtype=torch.float32) 
        img_wh = torch.tensor([ann['width'], ann['height']])  
        return image, caption, bbox, img_wh


#the number of position tokens is 512
pos_dict = {x:f"[pos_{x}]" for x in range(512)}

def make_pseudo_pos_seq(name, bbox, img_h, img_w):
    hh = 512/int(img_h)
    ww = 512/int(img_w)
    bbox_xyxy_resize = resize_bbox(bbox, hh, ww)
    if bbox_xyxy_resize == ' ':
        return name
    else:
        pos_seq = [name,' @@ ' ]
        pos_seq.extend([pos_dict[m] for m in bbox_xyxy_resize])
        pos_seq.append(' ## ')
        pseudo_seq = ' '.join(pos_seq)
        return pseudo_seq            


def resize_bbox(bbox, h, w):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3] 
        x1 = max(int(x_min * w,), 0)
        y1 = max(int(y_min * h,), 0)
        x2 = min(int(x_max * w,), 511)
        y2 = min(int(y_max * h,), 511)
        if 512 in [x1, y1, x2, y2]:
            return ' '
        else:
            return [x1, y1, x2, y2]


class Augfunc(object):
    def __init__(self, horizontal=True, resize_ratio=0.25):
        self.resize_ratio = resize_ratio
        max_size=1333
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.random_size_crop = Compose(
                                            [
                                                RandomResize([400, 500, 600]),
                                                RandomSizeCrop(384, max_size),
                                            ]
                                        )    
        self.horizontal = horizontal
        if self.horizontal:
            self.random_horizontal = RandomHorizontalFlip()
        self.final_transform = transforms.Compose([
            # RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),
            transforms.ToTensor(),
            normalize,
        ])
    def random_aug(self, image, ann, do_hori=True, do_aug=True, img_res=256):
        do_horizontal=False
        if random.random() < self.resize_ratio:
            image, ann = resize(image, ann, (img_res, img_res))
        else:
            if do_aug:
                image, ann = self.random_size_crop(image, ann)
            image, ann = resize(image, ann, (img_res, img_res))
        if do_hori:
            image, ann, do_horizontal = self.random_horizontal(image, ann)
            ann['caption'].replace('[TMP', 'right').replace('left_', 'right')
        image = self.final_transform(image)
        return image, ann, do_horizontal


def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')
    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


def pre_question(question,max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')  
    question = question.rstrip(' ')
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
    return question


def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "bbox_list" in target:
        boxes = torch.as_tensor(target["bbox_list"], dtype=torch.float32)
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1], dtype=torch.float32) + torch.as_tensor([w, 0, w, 0], dtype=torch.float32)
        target["bbox_list"] = boxes.numpy().tolist()
    do_horizontal = True
    return flipped_image, target, do_horizontal


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, target):
        do_horizontal = False
        if random.random() < self.p:
            return hflip(img, target)
        return img, target, do_horizontal
    

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = True):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes  # if True we can't crop a box out
    def __call__(self, img: PIL.Image.Image, target: dict):
        init_boxes = len(target["not_crop_bbox_list"])
        max_patience = 100
        for i in range(max_patience):
            w = random.randint(self.min_size, min(img.width, self.max_size))
            h = random.randint(self.min_size, min(img.height, self.max_size))
            region = T.RandomCrop.get_params(img, [h, w])
            result_img, result_target = crop(img, target, region)
            if not self.respect_boxes or len(result_target["not_crop_bbox_list"]) == init_boxes or i < max_patience - 1:
                return result_img, result_target
            elif not self.respect_boxes or len(result_target["not_crop_bbox_list"]) == init_boxes or i == max_patience - 1:
                return img, target
        #return result_img, result_target


def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w]).numpy().tolist()
    if "not_crop_bbox_list" in target:
        not_crop_bboxes = torch.as_tensor(target["not_crop_bbox_list"], dtype=torch.float32)
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = not_crop_bboxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["not_crop_bbox_list"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        positive_bboxes = torch.as_tensor(target["bbox_list"], dtype=torch.float32)
        positive_cropped_bboxes = positive_bboxes - torch.as_tensor([j, i, j, i], dtype=torch.float32)
        positive_cropped_bboxes = torch.min(positive_cropped_bboxes.reshape(-1, 2, 2), max_size)
        positive_cropped_bboxes = positive_cropped_bboxes.clamp(min=0)
        target["bbox_list"] = positive_cropped_bboxes.reshape(-1, 4).numpy().tolist()
    cropped_boxes = target["not_crop_bbox_list"].reshape(-1, 2, 2)
    keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
    crop_bbox = target["not_crop_bbox_list"][keep]
    target["not_crop_bbox_list"] = crop_bbox.reshape(-1, 4).numpy().tolist()
    return cropped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    if target is None:
        return rescaled_image, None
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    target = target.copy()
    if "bbox_list" in target:
        boxes = target["bbox_list"]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height], dtype=torch.float32)
        target["bbox_list"] = scaled_boxes.numpy().tolist()
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
    h, w = size
    target["size"] = torch.tensor([h, w]).numpy().tolist()
    return rescaled_image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
