'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''


import os
import utils
import argparse
import random
import time
import datetime
import json
import torch

import numpy as np
import ruamel_yaml as yaml
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_grounding import PEVL_Grounding
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset import create_sampler, create_loader
from dataset.grounding_dataset import Grounding_train_dataset, Grounding_eval_dataset



unus = ['[unused{}]'.format(x) for x in range(200,800)]
pos_token = ['@@']
pos_token.extend([f'[pos_{x}]' for x in range(512)])
pos_token.append('##')
tokenizer_ = BertTokenizer.from_pretrained('./configs/vocab.txt')
postoken_dict = {}
for x,y in zip(unus, pos_token):
    un_index = tokenizer_.vocab[x]
    tokenizer_.vocab[y] = un_index
    postoken_dict[y] = un_index
    _ = tokenizer_.vocab.pop(x)
    tokenizer_.basic_tokenizer.never_split.add(y)

postoken_dict.pop('##')
postoken_dict.pop('@@')

postoken_index = torch.randn(30522).bool()
postoken_index[:] = False
for x in postoken_dict.values():
    postoken_index[x]=True



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


def pretrain(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, postoken_index):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_soft', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        image = image.to(device,non_blocking=True) 
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)  
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        loss_soft, loss_ita, loss_itm = model(image, text_input, alpha = alpha)  
        loss = loss_soft + loss_ita+loss_itm
        loss.backward()
        optimizer.step() 
        
        metric_logger.update(loss_soft=loss_soft.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
    

def finetune(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, postoken_index):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        image = image.to(device,non_blocking=True) 
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=300, return_tensors="pt").to(device)  
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader)) 
        
        loss = model(image, text_input, alpha = alpha, mode='finetune')  
        loss.backward()

        optimizer.step() 
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def grounding_test(model, data_loader, tokenizer, device, config, dataname=None):
    #test
    model.eval()
    test_results = []
    results = []
    for i, (image, text, bbox, imgs_wh) in enumerate(data_loader):
        image = image.to(device,non_blocking=True)  
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=300, return_tensors="pt").to(device) 
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        input_ids = text_input.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, model.mlm_probability)
        input_ids, labels, masked_indices = model.postoken_mask(input_ids, targets=labels, probability_matrix = probability_matrix)
        mlm_output = model.text_encoder(input_ids, 
                                        attention_mask = text_input.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,)
        masked_indices = masked_indices.cpu()
        pos_logits = mlm_output.logits.detach().cpu()[masked_indices][:,postoken_index].view(-1,4,512)
        res = []        
        for x,y,m in zip(pos_logits, bbox, imgs_wh):
            assert x.shape[0]==4
            img_h = m[1]
            img_w = m[0]  
            res = []     
            for n in x:
                res.append(float(n.argmax()/512))
            res = [res[0]*img_w, res[1]*img_h, res[2]*img_w, res[3]*img_h]
            res = torch.tensor([res[0], res[1], res[2]-res[0]+1, res[3]-res[1]+1])
            iou = computeIoU(res, y)
            if iou >= 0.5:
                results.append(iou)
            test_results.append(iou)
    print(len(test_results))  
    print('grounding accuracy: ', len(results)/len(test_results))      



def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']    


    #### Dataset #### 
    print("Creating dataset")
    if args.pretrain:
        datasets = [Grounding_train_dataset(config['train_file'], img_res=config['image_res']) ]
    elif args.test_dataset == 'refcoco':
        print('.....................REFCOCO TRAIN DATASET.....................')
        datasets = [Grounding_train_dataset(config['refcoco_train_file'], img_res=config['image_res'])]
    elif args.test_dataset == 'refcocog':
        print('.....................REFCOCOG TRAIN DATASET.....................')
        datasets = [Grounding_train_dataset(config['refcocog_train_file'], img_res=config['image_res'])]
    elif args.test_dataset == 'refcocop':
        print('.....................REFCOCO+ TRAIN DATASET.....................')
        datasets = [Grounding_train_dataset(config['refcocop_train_file'], img_res=config['image_res'])]
    elif args.test_dataset == 'flickr':
        print('.....................FLICKR TRAIN DATASET.....................') 
        datasets = [Grounding_train_dataset(config['flickr_train_file'], img_res=config['image_res'])]

    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True], num_tasks, global_rank)         
    else:
        samplers = [None]

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    
    
    ##our tokenizer
    unus = ['[unused{}]'.format(x) for x in range(200,800)]
    pos_token = ['@@']
    pos_token.extend([f'[pos_{x}]' for x in range(512)])
    pos_token.append('##')
    postoken_dict = {}
    tokenizer = BertTokenizer.from_pretrained('./configs/vocab.txt')
    for x,y in zip(unus, pos_token):
        un_index = tokenizer.vocab[x]
        tokenizer.vocab[y] = un_index
        postoken_dict[y] = un_index
        _ = tokenizer.vocab.pop(x)
        tokenizer.basic_tokenizer.never_split.add(y)
    postoken_dict.pop('@@')
    postoken_dict.pop('##')
    postoken_index = torch.randn(30522).bool()
    postoken_index[:] = False
    for x in postoken_dict.values():
        postoken_index[x]=True


    #### Model #### 
    print("Creating model")
    model = PEVL_Grounding(config=config, tokenizer=tokenizer, postoken_dict = postoken_dict,init_deit=False)
    
    model = model.to(device)   
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']+1         
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)  
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped       
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped               
        model.load_state_dict(state_dict, strict=False)    
        print('load checkpoint from %s'%args.checkpoint)


    model_without_ddp = model


    if args.distributed:
        if args.pretrain:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
         
        model_without_ddp = model.module    
    
    if args.train:
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, max_epoch):
            if epoch>0:
                lr_scheduler.step(epoch+warmup_steps)  

            if args.pretrain:
                train_stats = pretrain(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, postoken_index) 
            else:
                train_stats = finetune(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, postoken_index)

            if utils.is_main_process():  
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            }                     
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'grounding_checkpoint_%02d.pth'%epoch))  
                
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if torch.distributed.get_rank() == 0:
                val_model = model.module
                if args.test_dataset == 'refcoco':
                    print('.....................REFCOCO VAL BEGIN VAL.....................')
                    val_dataset = [Grounding_eval_dataset(config['refcoco_val'], config['image_res']) ]
                    val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                    grounding_test(val_model, val_data_loader, tokenizer, device, config)

                elif args.test_dataset == 'refcocog':
                    print('.....................REFCOCOG VAL BEGIN EVAL.....................')
                    val_dataset = [Grounding_eval_dataset(config['refcocog_val'], config['image_res']) ]
                    val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                    grounding_test(val_model, val_data_loader, tokenizer, device, config)

                elif args.test_dataset == 'refcocop':
                    print('.....................REFCOCO+ VAL BEGIN EVAL.....................')
                    val_dataset = [Grounding_eval_dataset(config['refcocop_val'], config['image_res']) ]
                    val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                    grounding_test(val_model, val_data_loader, tokenizer, device, config)

                elif args.test_dataset == 'flickr':
                    print('.....................FLICKR EVAL.....................') 
                    val_dataset = [Grounding_eval_dataset(config['flickr_val'], config['image_res']) ]
                    val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                    grounding_test(val_model, val_data_loader, tokenizer, device, config,) 

            dist.barrier() 
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str)) 
    else:
        if torch.distributed.get_rank() == 0:
            val_model = model.module
            if args.test_dataset == 'refcoco':
                print('.....................REFCOCO TESTA BEGIN EVAL.....................')
                val_dataset = [Grounding_eval_dataset(config['refcoco_testA'], config['image_res']) ]
                val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                grounding_test(val_model, val_data_loader, tokenizer, device, config) 
        
                print('.....................REFCOCO TESTB BEGIN EVAL.....................')
                val_dataset = [Grounding_eval_dataset(config['refcoco_testB'], config['image_res']) ]
                val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                grounding_test(val_model, val_data_loader, tokenizer, device, config) 

            elif args.test_dataset == 'refcocog':
                print('.....................REFCOCOG VAL BEGIN EVAL.....................')
                val_dataset = [Grounding_eval_dataset(config['refcocog_val'], config['image_res']) ]
                val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                grounding_test(val_model, val_data_loader, tokenizer, device, config)
                
                print('.....................REFCOCOG TEST BEGIN EVAL.....................')
                val_dataset = [Grounding_eval_dataset(config['refcocog_test'], config['image_res']) ]
                val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                grounding_test(val_model, val_data_loader, tokenizer, device, config) 

            elif args.test_dataset == 'refcocop':
                print('.....................REFCOCO+ TESTA BEGIN EVAL.....................')
                val_dataset = [Grounding_eval_dataset(config['refcocop_testA'], config['image_res']) ]
                val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                grounding_test(val_model, val_data_loader, tokenizer, device, config)
                
                print('.....................REFCOCO+ TESTB BEGIN EVAL.....................')
                val_dataset = [Grounding_eval_dataset(config['refcocop_testB'], config['image_res']) ]
                val_data_loader = create_loader(val_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                grounding_test(val_model, val_data_loader, tokenizer, device, config)

            elif args.test_dataset == 'flickr':
                test_dataset = [Grounding_eval_dataset(config['flickr_test'], config['image_res']) ]
                test_data_loader = create_loader(test_dataset,[None],batch_size=[config['test_batch_size']], num_workers=[1], is_trains=[False], collate_fns=[None])[0]
                print('.....................FLICKR TEST EVAL.........................')
                grounding_test(val_model, test_data_loader, tokenizer, device, config,) 

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--aug', default=0, type=int)
    parser.add_argument('--test_dataset', default='', type=str)
    parser.add_argument('--softlabel_ratio', default=0.15, type=float)
    parser.add_argument('--test_before', default=0, type=int)
    parser.add_argument('--test_all', default=0, type=int)
    parser.add_argument('--pretrain', default=0, type=int)
    parser.add_argument('--train', default=1, type=int)
    

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
    
