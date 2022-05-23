'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import random
import time
import datetime
import json
import utils
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from pathlib import Path
import ruamel_yaml as yaml
from itertools import zip_longest
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset import create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_pretrain import PEVL_pretrain
from dataset.vcr_dataset import VCR_train_dataset
from dataset.grounding_dataset import Grounding_train_dataset


def grounding_vcr_train(model, data_loader_1, data_loader_2, vcr_data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, postoken_index):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_soft', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    if args.distributed:
        vcr_data_loader.sampler.set_epoch(epoch)

    i=0
    for vcr_data, data_1, data_2 in zip_longest(metric_logger.log_every(vcr_data_loader, print_freq, header), data_loader_2, data_loader_1, fillvalue=-1):
        optimizer.zero_grad()
        vcr_image, text_list, itm_labels = vcr_data
        vcr_image = vcr_image.to(device,non_blocking=True) 
        itm_labels = itm_labels.view(-1)
        text_itm = text_list.copy()
        text_list_itm = []
        for x in text_itm:
            if '[yeschoice]' in x:
                text_list_itm.append(x.replace('[yeschoice]', ' '))
            else:
                text_list_itm.append(x.replace('[nochoice]', ' '))
                
        text_input_itm = tokenizer(text_list_itm, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)  
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(vcr_data_loader)) 
        
        loss_mlm, loss_soft, loss_ita, loss_itm = model(vcr_image, text_input_itm, epoch, alpha, itm_labels, mode='vcr')  
            
        loss_vcr = loss_mlm + loss_ita + loss_itm + loss_soft 
        
        loss_vcr.backward()
        optimizer.step()   

        metric_logger.update(loss_mlm=loss_mlm.item())
        metric_logger.update(loss_soft=loss_soft.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 
        i+=1 
        
        if data_1 != -1:
            optimizer.zero_grad()
            image, text = data_1
            image = image.to(device,non_blocking=True) 
            text_input = tokenizer(text, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)  
            
            loss_mlm, loss_soft, loss_ita, loss_itm = model(image, text_input, alpha = alpha, mode='grounding')  
                
            loss = loss_mlm + loss_ita + loss_itm + loss_soft 
            
            loss.backward()
            optimizer.step()    
            
            metric_logger.update(loss_mlm=loss_mlm.item())
            metric_logger.update(loss_soft=loss_soft.item())
            metric_logger.update(loss_ita=loss_ita.item())
            metric_logger.update(loss_itm=loss_itm.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
            
            if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                scheduler.step(i//step_size)         
            i+=1

        if data_2 != -1:
            optimizer.zero_grad()
            image, text = data_2
            image = image.to(device,non_blocking=True) 
            text_input = tokenizer(text, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)  
            
            loss_mlm, loss_soft, loss_ita, loss_itm = model(image, text_input, alpha = alpha, mode='grounding')  
                
            loss = loss_mlm + loss_ita + loss_itm + loss_soft 
            
            loss.backward()
            optimizer.step()    
            
            metric_logger.update(loss_mlm=loss_mlm.item())
            metric_logger.update(loss_soft=loss_soft.item())
            metric_logger.update(loss_ita=loss_ita.item())
            metric_logger.update(loss_itm=loss_itm.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
            
            if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                scheduler.step(i//step_size)         
            i+=1
     
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


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
    datasets_1 = [Grounding_train_dataset(ann_file = config['train_grounding_file'], img_res=config['image_res'], half=0)]
    datasets_2 = [Grounding_train_dataset(ann_file = config['train_grounding_file'], img_res=config['image_res'], half=1)]
    vcr_datasets = [VCR_train_dataset(config['train_vcr_file'], img_res=config['image_file'])]
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers_1 = create_sampler(datasets_1, [True], num_tasks, global_rank)     
        samplers_2 = create_sampler(datasets_2, [True], num_tasks, global_rank)    
        samplers_3 = create_sampler(vcr_datasets, [True], num_tasks, global_rank)   
    else:
        samplers = [None]

    data_loader_1 = create_loader(datasets_1,samplers_1,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    data_loader_2 = create_loader(datasets_2,samplers_2,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    vcr_data_loader = create_loader(vcr_datasets,samplers_3,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    ##our tokenizer
    unus = ['[unused{}]'.format(x) for x in range(200,800)]
    pos_token = ['@@']
    pos_token.extend([f'[pos_{x}]' for x in range(512)])
    pos_token.append('##')
    postoken_dict = {}
    tokenizer = BertTokenizer.from_pretrained('../configs/vocab.txt')
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
    model = PEVL_pretrain(config=config, tokenizer=tokenizer, postoken_dict = postoken_dict, init_deit=False)
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
        model.load_state_dict(state_dict)    
        print('load checkpoint from %s'%args.checkpoint)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    print("Start training")
    start_time = time.time()


    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
            
        train_stats = grounding_vcr_train(model, data_loader_1, data_loader_2, vcr_data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, postoken_index)
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
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        dist.barrier()  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

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
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)