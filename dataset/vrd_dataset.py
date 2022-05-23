import os
import re
import PIL
import json
import pickle
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


pos_dict = {x:f"[pos_{x}]" for x in range(512)}
class VRD_train_dataset(Dataset):
    def __init__(self, ann_file, max_words=200, resize_ratio=0.25, img_res=256, vg_dict_path=''):     
        super().__init__()
        self.img_res = img_res   
        self.ann = []
        print("Creating dataset")
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        print(len(self.ann))
        self.pos_dict = {x:f"[pos_{x}]" for x in range(512)}
        self.max_words = max_words
        self.aug_transform = Augfunc(True, resize_ratio)

        self.vg_dict = json.load(open(vg_dict_path))
        self.gt_classes = self.vg_dict['idx_to_label']
        self.predicate_label = self.vg_dict['idx_to_predicate']
        self.predicate_label['0'] = ' [unrel] [unrel] [unrel] '

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):   
        ann = self.ann[index].copy()
        image = Image.open(ann['file_name']).convert('RGB')

        ann['height'] = ann['img_info']['height']
        ann['width'] = ann['img_info']['width']
        BOX_SCALE = 1024
        w,h = image.size
        assert w == ann['width']
        assert h == ann['height']
        SCALE = max(w,h)
        bbox_list = np.array(ann['bbox_list']) * SCALE / BOX_SCALE
        bbox_list = bbox_list.tolist()
        ann['bbox_list'] = bbox_list.copy()
        ann['not_crop_bbox_list'] = bbox_list.copy()
        image, ann = self.aug_transform.random_aug(image, ann, do_hori=True, do_aug=False, img_res=self.img_res)
        obj_name_bbox_dict = {}
        assert len(ann['gt_classes']) == len(ann['bbox_list'])
        for index, name, bbox in zip(range(len(ann['gt_classes'])), ann['gt_classes'], ann['bbox_list']):
            obj_name_bbox_dict[index] = {}
            obj_name_bbox_dict[index]['name'] = self.gt_classes[str(name)]
            obj_name_bbox_dict[index]['bbox'] = bbox
        
        img_w, img_h = ann['width'], ann['height']
        assert img_w == self.img_res
        assert img_h == self.img_res

        #with_rel's 1 which means this this a positive sample, ortherwise negative sample, and 
        #the ratio between them is 1:3 
        if ann['with_rel'] == 1:
            sub_index = ann['rel_tri'][0]
            obj_index = ann['rel_tri'][1]
            predicate = self.predicate_label[str(ann['rel_tri'][2])]
            predicate_length = len(predicate.split(' '))
            predicate_list = predicate.split(' ')
            if predicate_length == 1:
                predicate_list.append('[unrel]')
                predicate_list.append('[unrel]')
            elif predicate_length == 2:
                predicate_list.append('[unrel]')
            elif predicate_length == 3:
                pass
            else:
                assert 1==2
            predicate = ' '.join(predicate_list)
            sub_seq = self.make_pseudo_pos_seq(obj_name_bbox_dict[sub_index]['name'], obj_name_bbox_dict[sub_index]['bbox'], img_w, img_h)
            obj_seq = self.make_pseudo_pos_seq(obj_name_bbox_dict[obj_index]['name'], obj_name_bbox_dict[obj_index]['bbox'], img_w, img_h)
            seq = ' '.join([sub_seq, predicate, obj_seq])
            mask_seq = ' '.join([sub_seq, ' [MASK] [MASK] [MASK] ', obj_seq])
            sgg_seq = ' [SPLIT] '.join([seq, mask_seq])
            return image, sgg_seq
        elif ann['with_rel'] == 0:
            neg_pair_list = ann['un_rel_pair']
            neg_pair = random.choice(neg_pair_list)
            sub_index, obj_index = int(neg_pair.split('_')[0]), int(neg_pair.split('_')[1])
            predicate = ' [unrel] [unrel] [unrel] '
            sub_seq = self.make_pseudo_pos_seq(obj_name_bbox_dict[sub_index]['name'], obj_name_bbox_dict[sub_index]['bbox'], img_w, img_h)
            obj_seq = self.make_pseudo_pos_seq(obj_name_bbox_dict[obj_index]['name'], obj_name_bbox_dict[obj_index]['bbox'], img_w, img_h)
            seq = ' '.join([sub_seq, predicate, obj_seq])
            mask_seq = ' '.join([sub_seq, ' [MASK] [MASK] [MASK] ', obj_seq])
            sgg_seq = ' [SPLIT] '.join([seq, mask_seq])
            return image, sgg_seq

    def resize_bbox(self, bbox, h, w):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3] 
        x1 = min(max(int(x_min * w,), 0), 511)
        y1 = min(max(int(y_min * h,), 0), 511)
        x2 = max(min(int(x_max * w,), 511),0)
        y2 = max(min(int(y_max * h,), 511),0)
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
    

class VRD_eval_dataset(Dataset):
    def __init__(self, ann_file, img_res, vg_dict_path, vg_root):
        super().__init__()
        self.img_res = img_res
        self.ann = []
        print("Creating dataset")
        
        self.ann = pickle.load(open(ann_file, 'rb'))

        print(len(self.ann))

        #path of Visual Genome images
        self.vg_root = vg_root

        self.pos_dict = {x:f"[pos_{x}]" for x in range(512)}

        self.vg_dict = json.load(open(vg_dict_path))
        self.gt_classes = self.vg_dict['idx_to_label']
        self.predicate_label = self.vg_dict['idx_to_predicate']
        
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.final_transform = transforms.Compose([ 
            transforms.ToTensor(),\
            normalize,\
        ])
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):   
        ann = self.ann[index]
        t = {}
        image_path = os.path.join(self.vg_root, ann['img_path'].split('/')[-1])
        image = Image.open(image_path).convert('RGB')
        t['bbox_list'] = ann['boxes'].tolist()
        image, t = resize(image, t, (self.img_res, self.img_res))
        image = self.final_transform(image)
        imgid = ann['img_path'].split('/')[-1]
        
        t['objects'] = ann['labels'].tolist()
        obj_name = ann['labels'].tolist()
        object_dict = {}
        for index, obj in enumerate(obj_name):
            object_dict[index] = {}
            object_dict[index]['bbox'] = t['bbox_list'][index]
            object_dict[index]['name'] = self.gt_classes[str(obj)] 
            object_dict[index]['id'] = index
            object_dict[index]['seq_input'] = self.make_pseudo_pos_seq(self.gt_classes[str(obj)],\
                                                                        t['bbox_list'][index],\
                                                                        512.0,512.0)
        seq_input = []
        id_pair = []
        for A_id, A_seq in object_dict.items():
            for B_id, B_seq in object_dict.items():
                if A_id == B_id:
                    continue
                else:
                    seq_input.append(' '.join([A_seq['seq_input'], ' [MASK] [MASK] [MASK] ', B_seq['seq_input']]))
                    id_pair.append('_'.join([str(A_id), str(B_id)])) 
        seq_input = '__'.join(seq_input)
        id_pair = '#'.join(id_pair)
        return image, seq_input, id_pair, imgid

    def resize_bbox(self, bbox, h, w):
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3] 
        x1 = min(max(int(x_min * w,), 0), 511)
        y1 = min(max(int(y_min * h,), 0), 511)
        x2 = max(min(int(x_max * w,), 511),0)
        y2 = max(min(int(y_max * h,), 511),0)
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



class Augfunc(object):
    def __init__(self, do_horizontal, resize_ratio=0.25,):
        self.resize_ratio = resize_ratio
        max_size=1333
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.random_size_crop = Compose(
                                            [
                                                RandomResize([400, 500]),
                                                RandomSizeCrop(384, max_size),
                                            ]
                                        ) 
        if do_horizontal:
            self.random_horizontal = RandomHorizontalFlip()
        self.final_transform = transforms.Compose([ 
            # RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),
            transforms.ToTensor(),\
            normalize,\
        ])
    def random_aug(self, image, ann, do_hori=True, do_aug=False, img_res=384):
        do_horizontal=False
        if random.random() < self.resize_ratio:
            image, ann = resize(image, ann, (img_res, img_res))
        else:
            if do_aug:
                image, ann = self.random_size_crop(image, ann)
            image, ann = resize(image, ann, (img_res, img_res))
        if do_hori:
            image, ann, do_horizontal = self.random_horizontal(image, ann)
        image = self.final_transform(image)
        return image, ann, do_horizontal
        

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



