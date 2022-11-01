import re
import os
import PIL
import json
import torch
import random
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.randaugment import RandomAugment
import torchvision.transforms as T
import torchvision.transforms.functional as F

class GQA_train_dataset(Dataset):
    def __init__(self, ann_file, max_words=200, resize_ratio=0.25,img_res=None,tokenizer=None,answer_dict=None,image_path=None):        
        self.ann = []
        for f in ann_file:
            print(f)
            self.ann += json.load(open(f,'r'))
        print(len(self.ann))
        self.image_path=image_path
        self.answer_dict=answer_dict
        self.tokenizer=tokenizer
        self.img_res = img_res
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.final_transform = transforms.Compose([ 
            transforms.Resize((img_res,img_res),interpolation=Image.BICUBIC),\
            transforms.ToTensor(),\
            normalize,\
        ])
        self.max_words = max_words
        self.aug_transform = Augfunc(True, resize_ratio, img_res)
        self.pos_dict = {x:f"[pos_{x}]" for x in range(512)}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        ann = self.ann[index].copy()
        file_name = os.path.join(self.image_path, ann['file_name'])
        image = Image.open(file_name).convert('RGB')
        bbox_list = torch.as_tensor(ann['bbox_list'], dtype=torch.float32).clamp(min=0)
        w, h = image.size
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = torch.min(bbox_list.reshape(-1, 2, 2), max_size)
        ann['bbox_list'] = cropped_boxes.reshape(-1,4).numpy().tolist()
        image, ann, do_horizontal = self.aug_transform.random_aug(image, ann, True)

        if ann['no_bbox'] == 0:
            seq = ann['question']
            tokens2bbox={}
            for tokens, bbox in zip(ann['tokens_positive'], ann['bbox_list']):
                token_id = str(tokens[0])+str(tokens[1])
                pos_seq = ['  @@ ']
                bbox_512 = [int(xy*512/self.img_res) if int(xy*512/self.img_res) <=511 else 511  for xy in bbox ]
                pos_seq.extend([self.pos_dict[int(x)] for x in bbox_512])
                #pos_seq.extend([self.pos_dict[int(x)] for x in bbox])
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
            #caption = pre_question(new_seq, self.max_words) 
            caption = new_seq
            # caption = question_output 
            if do_horizontal:
                answer = ann['answer']
                answer = answer.replace("left", " [TMP] ").replace("right", "left").replace(" [TMP] ", "right")
                caption = caption.replace("left", " [TMP] ").replace("right", "left").replace(" [TMP] ", "right")
                caption = pre_question(caption, self.max_words) 
                length = len(self.tokenizer(answer, padding='longest', truncation=True, max_length=250, return_tensors="pt").input_ids[0][1:]) 
                mask_list = ['[MASK]' for l in range(6)]
                padding_num = 6 - length
                label_padding = ['[PAD]' for x in range(padding_num)]
                seq_input = ' '.join([caption, '[SEP]', ' '.join(mask_list)])  
                seq_label = ' '.join([caption, '[SEP]', answer, ' '.join(label_padding)]) 
            else:
                answer = ann['answer']
                caption = pre_question(caption, self.max_words) 
                length = len(self.tokenizer(answer, padding='longest', truncation=True, max_length=250, return_tensors="pt").input_ids[0][1:]) 
                mask_list = ['[MASK]' for l in range(6)]
                padding_num = 6 - length
                label_padding = ['[PAD]' for x in range(padding_num)]
                seq_input = ' '.join([caption, '[SEP]', ' '.join(mask_list)])  
                seq_label = ' '.join([caption, '[SEP]', answer, ' '.join(label_padding)])
            #print(caption)
        elif ann['no_bbox'] == 1:
            seq = ann['question']
            caption = seq 
            if do_horizontal:
                answer = ann['answer']
                answer = answer.replace("left", " [TMP] ").replace("right", "left").replace(" [TMP] ", "right")
                caption = caption.replace("left", " [TMP] ").replace("right", "left").replace(" [TMP] ", "right")
                caption = pre_question(caption, self.max_words) 
                length = len(self.tokenizer(answer, padding='longest', truncation=True, max_length=250, return_tensors="pt").input_ids[0][1:])
                mask_list = ['[MASK]' for l in range(6)]
                padding_num = 6 - length
                label_padding = ['[PAD]' for x in range(padding_num)]
                seq_input = ' '.join([caption, '[SEP]', ' '.join(mask_list)])  
                seq_label = ' '.join([caption, '[SEP]', answer, ' '.join(label_padding)]) 
            else:
                answer = ann['answer']
                caption = pre_question(caption, self.max_words) 
                length = len(self.tokenizer(answer, padding='longest', truncation=True, max_length=250, return_tensors="pt").input_ids[0][1:])
                mask_list = ['[MASK]' for l in range(6)]
                padding_num = 6 - length
                label_padding = ['[PAD]' for x in range(padding_num)]
                seq_input = ' '.join([caption, '[SEP]', ' '.join(mask_list)])  
                seq_label = ' '.join([caption, '[SEP]', answer,  ' '.join(label_padding)])
        return image, seq_label, seq_input 




class GQA_val_dataset(Dataset):
    def __init__(self, ann_file, max_words=200, resize_ratio=0.25,img_res=None, image_path=None):        
        self.ann = []
        for f in ann_file:
            print(f)
            self.ann += json.load(open(f,'r'))
        self.img_res = img_res
        self.image_path = image_path
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.final_transform = transforms.Compose([ 
            transforms.ToTensor(),\
            normalize,\
        ])
        self.max_words = max_words
        self.pos_dict = {x:f"[pos_{x}]" for x in range(512)}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        ann = self.ann[index].copy()
        # image = Image.open(ann['file_name']).convert('RGB')
        file_name = os.path.join(self.image_path, ann['file_name'])
        image = Image.open(file_name).convert('RGB')
        bbox_list = torch.as_tensor(ann['bbox_list'], dtype=torch.float32).clamp(min=0)
        w, h = image.size
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = torch.min(bbox_list.reshape(-1, 2, 2), max_size)
        ann['bbox_list'] = cropped_boxes.reshape(-1,4).numpy().tolist()
        image, ann= resize(image, ann, (self.img_res, self.img_res))
        image = self.final_transform(image)
        if ann['no_bbox'] == 0: 
            seq = ann['question'] 
            tokens2bbox={}
            for tokens, bbox in zip(ann['tokens_positive'], ann['bbox_list']):
                token_id = str(tokens[0])+str(tokens[1])
                pos_seq = ['  @@ ']
                bbox_512 = [int(xy*512/self.img_res) if int(xy*512/self.img_res) <=511 else 511  for xy in bbox ]
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
            #caption = pre_question(new_seq, self.max_words) 
            caption = new_seq
            # caption = question_output 
            answer = ann['answer']
            caption = pre_question(caption, self.max_words) 
            # length = len(tokenizer(answer, padding='longest', truncation=True, max_length=250, return_tensors="pt").input_ids[0][1:]) 
            mask_list = ['[MASK]' for l in range(6)]
            seq_input = ' '.join([caption, '[SEP]', ' '.join(mask_list)])
            q_id = ann['q_id']
            #print(caption)
        elif ann['no_bbox'] == 1:
            seq = ann['normal_question'] if 'normal_question' in ann else ann['question']
            caption = seq
            answer = ann['answer']
            # answer  = self.answer_dict[answer]
            # caption = pre_question(caption, self.max_words) 
            # caption_ = caption + f" [SEP] {answer} "
            caption = pre_question(caption, self.max_words)
            mask_list = ['[MASK]' for l in range(6)]
            seq_input = ' '.join([caption, '[SEP]', ' '.join(mask_list)])
            q_id = ann['q_id']
        return image, seq_input, answer, torch.tensor(int(q_id))


    

class Augfunc(object):
    def __init__(self, horizontal=True, resize_ratio=0.25, img_res=None):
        self.resize_ratio = resize_ratio
        max_size=1333
        self.img_res=img_res
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
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),\
            transforms.ToTensor(),\
            normalize,\
        ])
    def random_aug(self, image, ann, do_hori):
        do_horizontal=False
        if random.random() < self.resize_ratio:
            image, ann = resize(image, ann, (self.img_res,self.img_res))
        else:
            image, ann = self.random_size_crop(image, ann)
            image, ann = resize(image, ann, (self.img_res, self.img_res))
        if do_hori:
            image, ann, do_horizontal = self.random_horizontal(image, ann)
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
