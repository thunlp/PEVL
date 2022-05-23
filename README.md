## PEVL: Position-enhanced Pre-training and Prompt Tuning for Vision-language Models

This is the official PyTorch implementation of the <a href="https://openreview.net/forum?id=Sg_xKgWgG9">PEVL paper</a>. PEVL show big gains of detector-free VLP models on position-sensitive tasks such as referring expression comprehension and phrase grounding, and also improves the performance on position-insensitive tasks with grounded inputs such as visual commomsense reasoning, visual relation detection and visual question answering(GQA).

This repository is currently under construction and will support pre-training on custom image-text datasets and datasets with object annotations, as well as fine-tuning on phrase grounding task (Flickr30k), referring expression comprehension (RefCOCO, RefCOCO+ and RefCOCOg), visual relation detection, visual commonsense reasoning and question answering(GQA).

<img src="img.png" width="800">

PEVL enhances the pre-training and prompt tuning of VLP models with explicit object position modeling. Specifically, PEVL reformulates discretized object positions and language in a unified language modeling framework, which facilitates explicit VL alignment during pre-training, and also enables flexible prompt tuning for various downstream tasks. 

### Requirements
- apex 0.1
- timm 0.5.4
- yaml 0.2.5
- CUDA 11.1
- numpy 1.21.5
- pytorch 1.8.0
- torchvision 0.9.0
- transformers 4.8.1
- Python 3.7.11

### Pretraining Instructions
Before pretraining, we initialize PEVL's weights with the parameters of **[ALBEF\[14M\]](https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth)**

Our raw pretraining corpus is from **[Visual Commonsense Reasoning(VCR)](https://visualcommonsense.com/download/)** and **[MDETR](https://arxiv.org/abs/2104.12763)** that collects images from Flickr30k entities, COCO, Visual Genome datasets. However, differently from MDETR, we split the sentences rather than use the combination of them.
- **[MDETR Data](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1)**
- Download VCR data from the original websites.


### Second Stage Pre-training and Fine-tuning

We conduct second stage pre-training and fine-tuning for all downstream tasks.

#### Referring Expression Comprehension
1. <a href="https://thunlp.oss-cn-qingdao.aliyuncs.com/grounding.pth"> Second stage pre-trained checkpoint </a> for position output tasks.
2. <a href="https://thunlp.oss-cn-qingdao.aliyuncs.com/pevl_grounding.tar.gz"> Dataset json files for position output downstream tasks</a>.(the 'file_name' in each json file need to be changed to your own directory)
3. In configs/visual_grounding.yaml, set the paths for the json files.
4. Fine-tuning the model:
```bash
##RefCOCO:
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12451 --use_env PEVL/run_grounding_train.py --pretrain 0 --test_dataset refcoco --config ./configs/visual_grounding.yaml --output_dir ./output/visual_grounding/refcoco --checkpoint grounding.pth 

##RefCOCOg
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12451 --use_env PEVL/run_grounding_train.py --pretrain 0 --test_dataset refcocog --config ./configs/visual_grounding.yaml --output_dir ./output/visual_grounding/refcocog --checkpoint grounding.pth

##RefCOCO+
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12451 --use_env PEVL/run_grounding_train.py --pretrain 0 --test_dataset refcocop --config ./configs/visual_grounding.yaml --output_dir ./output/visual_grounding/refcocop --checkpoint grounding.pth

```

#### Phrase Grounding
1. <a href="https://thunlp.oss-cn-qingdao.aliyuncs.com/grounding.pth"> Second stage pre-trained checkpoint </a> for position output tasks.
2. <a href="https://thunlp.oss-cn-qingdao.aliyuncs.com/pevl_grounding.tar.gz"> Dataset json files for position output downstream tasks</a>.(the 'file_name' in each json file need to be changed to your own directory)
3. In configs/visual_grounding.yaml, set the paths for the json files.
4. Fine-tuning the model:
```bash
##Flickr30k
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12451 --use_env PEVL/run_grounding_train.py --pretrain 0 --test_dataset flickr --config ./configs/visual_grounding.yaml --output_dir ./output/phrase_grounding --checkpoint grounding.pth 
```

#### Visual Relation Detection
1. <a href="https://thunlp.oss-cn-qingdao.aliyuncs.com/vrd.pth"> Second stage pre-trained checkpoint </a> for position output tasks.
2. <a href="https://thunlp.oss-cn-qingdao.aliyuncs.com/pevl_vrd.tar.gz"> Dataset json files for position output downstream tasks</a>.(the 'file_name' in each json file need to be changed to your own directory)
3. In configs/visual_grounding.yaml, set the paths for the json files.
4. Fine-tuning the model:
```bash
##for finetuning on visual genome:
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12451 --use_env PEVL/run_vrd_train.py --train 1 --pretrain 0 --config ./configs/vrd.yaml --output_dir ./output/vrd --checkpoint vrd.pth

##for evaluation on visual genome:
python -m torch.distributed.launch --nproc_per_node=8 --master_port=12451 --use_env PEVL/run_vrd_train.py --train 0 --pretrain 0 --config ./configs/vrd.yaml  --checkpoint [Finetuned checkpoint]
```

### Acknowledgement
The implementation of PEVL relies on resources from <a href="https://github.com/salesforce/ALBEF">ALBEF</a> especially, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing and excellent work.
