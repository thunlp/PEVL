# This file is code for making dataset for visual commonsense reasoning tasks.
# Author: Qianyu Chen
# Date: 2022-10
 
# Copyright (c) THUNLP, Tsinghua University. All rights reserved. 
# See LICENSE file in the project root for license information.

import json
import os
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import re
import cv2 as cv
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from dataset.randaugment import RandomAugment

pos_dict = {x:f"[pos_{x}]" for x in range(512)}
class VCR_train_dataset(Dataset):
    def __init__(self, ann_file, max_words=200, resize_ratio=0.25, img_res=256, data_load_mode='pevl'):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        self.img_res = img_res
        self.pos_dict = {x:f"[pos_{x}]" for x in range(512)}
        self.max_words = max_words
        self.aug_transform = Augfunc(resize_ratio, img_res=self.img_res)
        self.imgid_dict = {}
        n = 0
        self.pos_dict = {x:f"[pos_{x}]" for x in range(512)}
        for x in self.ann:
            id = x['file_name']
            if id not in self.imgid_dict.keys():
                self.imgid_dict[id]=n
                n+=1
        self.hori_imgid_dict = {}
        for x in self.ann:
            id = x['file_name']
            if id not in self.hori_imgid_dict.keys():
                self.hori_imgid_dict[id]=n
                n+=1

    def __len__(self):
        return len(self.ann)
        
    def __getitem__(self, index):   
        ann = self.ann[index].copy()
        image = Image.open(ann['file_name']).convert('RGB')
        ann_bbox_list = []
        for x in ann['bbox_list']:
            ann_bbox_list.append(x[:4])
        ann['bbox_list'] = ann_bbox_list
        bbox_list = torch.as_tensor(ann['bbox_list'], dtype=torch.float32).clamp(min=0)
        w, h = image.size
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = torch.min(bbox_list.reshape(-1, 2, 2), max_size)
        ann['bbox_list'] = cropped_boxes.reshape(-1,4).numpy().tolist()
        if 'with_answer' in ann:
            if ann['label'] == 1:
                pos_qa_image, pos_qa_seq, pos_qa_img_id = self.make_positive_QA(image, ann)
                return pos_qa_image, pos_qa_seq, torch.tensor([1])
            else:
                neg_img, neg_seq, neg_imgid, imgid = self.make_neg_QA(image, ann)
                return neg_img, neg_seq, torch.tensor([0])

        elif 'with_rationale' in ann:
            if ann['label'] == 1:
                pos_qa_image, pos_qa_seq, pos_qa_img_id = self.make_positive_QAR(image, ann)
                return pos_qa_image, pos_qa_seq, torch.tensor([1])
            else:
                neg_img, neg_seq, neg_imgid, imgid = self.make_neg_QAR(image, ann)
                return neg_img, neg_seq, torch.tensor([0])

    def resize_bbox(self, bbox, h, w):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3] 
        x1 = max(int(x_min * w,), 0)
        y1 = max(int(y_min * h,), 0)
        x2 = min(int(x_max * w,), 511)
        y2 = min(int(y_max * h,), 511)
        return [x1, y1, x2, y2]
    def make_pseudo_pos_seq(self, name, bbox, img_h, img_w):
        hh = 512/int(img_h)
        ww = 512/int(img_w)
        bbox_xyxy_resize = self.resize_bbox(bbox, hh, ww)
        pos_seq = [name,' @@ ' ]
        pos_seq.extend([self.pos_dict[m] for m in bbox_xyxy_resize])
        pos_seq.append(' ## ')
        pseudo_seq = ' '.join(pos_seq)
        return pseudo_seq
    def make_positive_QA(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['right_answer']
        target = target_update(target, bbox_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        # vcr_right_qa_seq += ' [yeschoice] '
        img_id = self.imgid_dict[ann['file_name']]
        if do_horizontal:
            img_id = self.hori_imgid_dict[ann['file_name']]
        return image, vcr_right_qa_seq, img_id
    def make_positive_QAR(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['right_answer']
        target['rationale'] = ann['right_rationale']
        target = target_update(target, bbox_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        # vcr_right_qa_seq += ' [yeschoice] '
        img_id = self.imgid_dict[ann['file_name']]
        if do_horizontal:
            img_id = self.hori_imgid_dict[ann['file_name']]
        return image, vcr_right_qa_seq, img_id
    def make_neg_QA(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['wrong_answer']
        target = target_update(target, bbox_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        # vcr_right_qa_seq += ' [nochoice] '
        img_id = -100
        if do_horizontal:
            image, vcr_right_qa_seq, img_id, self.hori_imgid_dict[ann['file_name']]
        return image, vcr_right_qa_seq, img_id, self.imgid_dict[ann['file_name']]
    def make_neg_QAR(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['right_answer']
        target['rationale'] = ann['wrong_rationale']
        target = target_update(target, bbox_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        # vcr_right_qa_seq += ' [nochoice] '
        img_id = -100
        if do_horizontal:
            image, vcr_right_qa_seq, img_id, self.hori_imgid_dict[ann['file_name']]
        return image, vcr_right_qa_seq, img_id, self.imgid_dict[ann['file_name']]
    def make_hard_negative_same_obj_QA(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['right_answer']
        target = target_update(target, bbox_dict)
        if self.hard_neg_aug:
            bbox_neg_dict, target, do_bbox_neg_gen = self.hard_neg_gen._same_name_neg_bbox_gen(target, bbox_dict)
        else:
            bbox_neg_dict = {}
            for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
                bbox_neg_dict[index] = {'bbox':bbox, 'name':name}
            do_bbox_neg_gen = False
        target = target_update(target, bbox_neg_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        if do_bbox_neg_gen:
            img_id = -100
            # vcr_right_qa_seq += '  [nochoice] '
        else:
            img_id = self.imgid_dict[ann['file_name']]
            # vcr_right_qa_seq += ' [yeschoice] '
        return image, vcr_right_qa_seq, img_id
    def make_hard_negative_same_obj_QAR(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['right_answer']
        target['rationale'] = ann['right_rationale']
        target = target_update(target, bbox_dict)
        if self.hard_neg_aug:
            bbox_neg_dict, target, do_bbox_neg_gen = self.hard_neg_gen._same_name_neg_bbox_gen(target, bbox_dict)
        else:
            bbox_neg_dict = {}
            for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
                bbox_neg_dict[index] = {'bbox':bbox, 'name':name}
            do_bbox_neg_gen = False
        target = target_update(target, bbox_neg_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        if do_bbox_neg_gen:
            img_id = -100
            # vcr_right_qa_seq += ' [nochoice] '
        else:
            img_id = self.imgid_dict[ann['file_name']]
            # vcr_right_qa_seq += '  [yeschoice] '
        return image, vcr_right_qa_seq, img_id
    def make_hard_negative_diff_obj_QA(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['right_answer']
        target = target_update(target, bbox_dict)
        if self.hard_neg_aug:
            bbox_neg_dict, target, do_bbox_neg_gen = self.hard_neg_gen._dif_name_neg_bbox_gen(target, bbox_dict)
        else:
            bbox_neg_dict = {}
            for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
                bbox_neg_dict[index] = {'bbox':bbox, 'name':name}
            do_bbox_neg_gen = False
        target = target_update(target, bbox_neg_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        if do_bbox_neg_gen:
            img_id = -100
            # vcr_right_qa_seq += ' [nochoice] '
        else:
            img_id = self.imgid_dict[ann['file_name']]
            # vcr_right_qa_seq += ' [yeschoice] '
        return image, vcr_right_qa_seq, img_id
    def make_hard_negative_diff_obj_QAR(self, image, ann):
        target = ann.copy()
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}
        target['answer'] = ann['right_answer']
        target['rationale'] = ann['right_rationale']
        target = target_update(target, bbox_dict)
        if self.hard_neg_aug:
            bbox_neg_dict, target, do_bbox_neg_gen = self.hard_neg_gen._dif_name_neg_bbox_gen(target, bbox_dict)
        else:
            bbox_neg_dict = {}
            for index, (bbox, name) in enumerate(zip(target['bbox_list'], target['names'])):
                bbox_neg_dict[index] = {'bbox':bbox, 'name':name}
            do_bbox_neg_gen = False
        target = target_update(target, bbox_neg_dict)
        if len(target['not_crop_bbox_list']) > 0:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target)
        else:
            image, target, do_horizontal = self.aug_transform.random_aug(image, target, False)
        vcr_right_qa_seq = pseudo_seq_gen(target, do_horizontal, self.max_words)
        if do_bbox_neg_gen:
            img_id = -100
            # vcr_right_qa_seq += '  [nochoice] '
        else:
            img_id = self.imgid_dict[ann['file_name']]
            # vcr_right_qa_seq += '  [yeschoice] '
        return image, vcr_right_qa_seq, img_id


class VCR_test_dataset(Dataset):
    def __init__(self, ann_file, img_res, dataload_mode, max_words=500):
        self.ann=[]
        self.img_res=img_res
        self.dataload_mode = dataload_mode
        self.position_token_dict = {x:f"[pos_{x}]" for x in range(512)}
        for x in ann_file:
            self.ann += json.load(open(f))
        self.max_words = max_words
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    def __len__(self):
        return len(self.ann)

    def resize_bbox(self, bbox, h, w):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3] 
        x1 = max(int(x_min * w,), 0)
        y1 = max(int(y_min * h,), 0)
        x2 = min(int(x_max * w,), 511)
        y2 = min(int(y_max * h,), 511)
        return [x1, y1, x2, y2]
    
    def make_pseudo_pos_seq(self, name, bbox, img_h, img_w):
        if self.dataload_mode=='pevl':
            hh = 512/int(img_h)
            ww = 512/int(img_w)
            bbox_xyxy_resize = self.resize_bbox(bbox, hh, ww)
            pos_seq = [name,' @@ ' ]
            pos_seq.extend([self.position_token_dict[m] for m in bbox_xyxy_resize])
            pos_seq.append(' ## ')
            pseudo_seq = ' '.join(pos_seq)
            return pseudo_seq
        elif self.dataload_mode=='albef':
            pseudo_seq = name
            return pseudo_seq

    def __getitem__(self, index):
        ann = self.ann[index].copy()
        image = Image.open(ann['file_name']).convert('RGB')
        ann_bbox_list = []
        for x in ann['bbox_list']:
            ann_bbox_list.append(x[:4])
        ann['bbox_list'] = ann_bbox_list
        bbox_list = torch.as_tensor(ann['bbox_list'], dtype=torch.float32).clamp(min=0)
        w, h = image.size
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = torch.min(bbox_list.reshape(-1, 2, 2), max_size)
        ann['bbox_list'] = cropped_boxes.reshape(-1,4).numpy().tolist()
        image, ann = resize(image, ann, (self.img_res,self.img_res))
        image = self.final_transform(image)
        bbox_dict = {}
        for index, (bbox, name) in enumerate(zip(ann['bbox_list'], ann['names'])):
            bbox_dict[index] = {'bbox':bbox, 'name':name}  
        normal_question = ann['question']
        normal_answer_list = ann['answer_choices'] if "answer_choices" in ann else ann["rationale_choices"]
        img_w = ann['width']
        img_h = ann['height']
        assert int(img_w) == self.img_res
        assert int(img_h) == self.img_res
        pseudo_question_list = []
        test_seq_list = []
        for question_token in normal_question:
            if isinstance(question_token, list):
                for obj_index in question_token:
                    name = bbox_dict[obj_index]['name']
                    bbox = bbox_dict[obj_index]['bbox']
                    pseudo_seq = self.make_pseudo_pos_seq(name, bbox, img_h, img_w)
                    pseudo_question_list.append(pseudo_seq)
            else:
                pseudo_question_list.append(question_token)
        for normal_answer in normal_answer_list:
            pseudo_answer = []
            for answer_token in normal_answer:
                if isinstance(answer_token, list):
                    for obj_index in answer_token:
                        name = bbox_dict[obj_index]['name']
                        bbox = bbox_dict[obj_index]['bbox']
                        pseudo_seq = self.make_pseudo_pos_seq(name, bbox, img_h, img_w)
                        pseudo_answer.append(pseudo_seq)
                else:
                    pseudo_answer.append(answer_token)
            pseudo_question = ' '.join(pseudo_question_list)
            pseudo_question = pre_question(pseudo_question,1000).replace('[sep]','[SEP]').split(' ')
            pseudo_answer = ' '.join(pseudo_answer)
            pseudo_answer = pre_question(pseudo_answer, 1000).split(' ')
            vcr_caption = []
            vcr_caption.extend(pseudo_question)
            vcr_caption.append('[SEP]')
            vcr_caption.extend(pseudo_answer)
            vcr_caption_seq = ' '.join(vcr_caption)
            vcr_caption_seq = pre_question(vcr_caption_seq, self.max_words)
            test_seq_list.append(vcr_caption_seq)
        label = ann["answer_label"] if "answer_label" in ann else ann["rationale_label"]
        label = torch.tensor(label)
        image= image.view((1,3,self.img_res,self.img_res))
        image = torch.cat([image,image,image,image],dim=0)
        return image, test_seq_list, label, ann['annot_id']
        
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    try:
        return float(inter) / union
    except ZeroDivisionError:
        return 0


def resize_bbox(bbox, h, w):
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3] 
    x1 = max(int(x_min * w,), 0)
    y1 = max(int(y_min * h,), 0)
    x2 = min(int(x_max * w,), 511)
    y2 = min(int(y_max * h,), 511)
    return [x1, y1, x2, y2]



def make_pseudo_pos_seq(name, bbox, img_h, img_w):
    hh = 512/int(img_h)
    ww = 512/int(img_w)
    bbox_xyxy_resize = resize_bbox(bbox, hh, ww)
    pos_seq = [name,' @@ ' ]
    pos_seq.extend([pos_dict[m] for m in bbox_xyxy_resize])
    pos_seq.append(' ## ')
    pseudo_seq = ' '.join(pos_seq)
    return pseudo_seq


def pseudo_seq_gen(ann, do_horizontal, max_words):
    ann = ann.copy()
    bbox_dict = {}
    normal_question = ann['question']
    normal_answer = ann['answer']
    img_w = ann['width']
    img_h = ann['height']
    bbox_dict = {}
    for index, (bbox, name) in enumerate(zip(ann['bbox_list'], ann['names'])):
        bbox_dict[index] = {'bbox':bbox, 'name':name} 
    # assert int(img_w) == 384
    # assert int(img_h) == 384
    assert int(img_w) == 256
    assert int(img_h) == 256
    pseudo_question = []
    pseudo_answer = []
    pseudo_rationale = []
    for question_token in normal_question:
        if isinstance(question_token, list):
            for index, obj_index in enumerate(question_token):
                name = bbox_dict[obj_index]['name']
                bbox = bbox_dict[obj_index]['bbox']
                pseudo_seq = make_pseudo_pos_seq(name, bbox, img_h, img_w)
                pseudo_question.append(pseudo_seq)
                if index < len(question_token)-1:
                    pseudo_question.append('and')
        else:
            pseudo_question.append(question_token)
    for answer_token in normal_answer:
        if isinstance(answer_token, list):
            for index, obj_index in enumerate(answer_token):
                name = bbox_dict[obj_index]['name']
                bbox = bbox_dict[obj_index]['bbox']
                pseudo_seq = make_pseudo_pos_seq(name, bbox, img_h, img_w)
                pseudo_answer.append(pseudo_seq)
                if index < len(answer_token)-1:
                    pseudo_answer.append('and')
        else:
            pseudo_answer.append(answer_token)
    vcr_caption = []
    vcr_caption.extend(pseudo_question)
    vcr_caption.append('[sep]')
    vcr_caption.extend(pseudo_answer)
    if "rationale" in ann:
        vcr_caption.append('[sep]')
        normal_rationale = ann["rationale"]
        for rationale_token in normal_rationale:
            if isinstance(rationale_token, list):
                for index, obj_index in enumerate(rationale_token):
                    name = bbox_dict[obj_index]['name']
                    bbox = bbox_dict[obj_index]['bbox']
                    pseudo_seq = make_pseudo_pos_seq(name, bbox, img_h, img_w)
                    pseudo_rationale.append(pseudo_seq)
                    if index < len(answer_token)-1:
                        pseudo_answer.append('and')
            else:
                pseudo_rationale.append(rationale_token)
        vcr_caption.extend(pseudo_rationale)
    vcr_caption_seq = ' '.join(vcr_caption)
    vcr_caption_seq = pre_caption(vcr_caption_seq, max_words-2)
    vcr_caption_seq = vcr_caption_seq.replace("[sep]", "[SEP]")
    if do_horizontal:
        vcr_caption_seq = vcr_caption_seq.replace("left", "[TMP]").replace("right", "left").replace("[TMP]", "right") 
    return vcr_caption_seq 



def target_update(target, bbox_dict):
    target = target.copy()
    bbox_dict = bbox_dict.copy()
    obj_set = set([])
    for x in target['question']:
        if isinstance(x, list):
            for y in x:
                obj_set.add(y)
    for x in target['answer']:
        if isinstance(x, list):
            for y in x:
                obj_set.add(y)
    if 'rationale' in target:
        for x in target['rationale']:
            if isinstance(x, list):
                for y in x:
                    obj_set.add(y)
    not_crop_list = []
    obj_list = list(obj_set)
    for x in obj_list:
        bbox = bbox_dict[x]['bbox']
        not_crop_list.append(bbox)
    target['not_crop_bbox_list'] = not_crop_list
    bbox_list = []
    names = []
    for index, (key,value) in enumerate(bbox_dict.items()):
        assert index == key
        names.append(value['name'])
        bbox_list.append(value['bbox'])
    target['bbox_list'] = bbox_list

    target['not_crop_bbox_list'] = bbox_list

    target['names'] = names
    return target 
    



class Augfunc(object):
    def __init__(self, resize_ratio=0.25, img_res=img_res):
        self.resize_ratio = resize_ratio
        self.img_res = img_res
        max_size=1333
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.random_size_crop = Compose(
                                            [
                                                RandomResize([400, 450]),
                                                RandomSizeCrop(384, max_size),
                                            ]
                                        ) 
        #if self.horizontal:
        self.random_horizontal = RandomHorizontalFlip()
        self.final_transform = transforms.Compose([ 
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),
            transforms.ToTensor(),\
            normalize,\
        ])
    def random_aug(self, image, ann):
        do_horizontal=False
        image, ann = resize(image, ann, (self.img_res,self.img_res))
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
    target['width']=w
    target['height']=h
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



