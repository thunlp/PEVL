
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset import create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_vqa import PEVL_VQA
from dataset.vqa_dataset import GQA_train_dataset, GQA_val_dataset
from eval.eval import gqa_val
import utils

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, args):
    # train
    model.train()  
    if args.training_mode == 'pretrain':
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
            data_loader.sampler.set_epoch(epoch)

        for i, (image, text, mask_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            optimizer.zero_grad()
            if epoch == 0:
                if i%100==0:
                    print(text[:4])
                    print('\nmask\n')
                    print(mask_caption[:4])
            image = image.to(device,non_blocking=True) 
            text_input = tokenizer(text, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)  
            mask_text_input = tokenizer(mask_caption, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)
            if epoch>0:
                alpha = config['alpha']
            else:
                alpha = config['alpha']*min(1,i/len(data_loader))
            loss_mlm, loss_soft, loss_ita, loss_itm = model(image=image, text=text_input, text_mask=mask_text_input, alpha = alpha, mode=args.mode)  
            loss =  loss_soft + loss_mlm + loss_ita + loss_itm
            loss.backward()
            optimizer.step()    
            
            metric_logger.update(loss_mlm=loss_mlm.item())
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
    elif args.training_mode == 'finetune':
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50   
        step_size = 100
        warmup_iterations = warmup_steps*step_size  
        
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        for i, (image, text, mask_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            optimizer.zero_grad()
            if epoch == 0:
                if i%100==0:
                    print(text[:4])
                    print('\nmask\n')
                    print(mask_caption[:4])
            image = image.to(device,non_blocking=True) 
            text_input = tokenizer(text, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)  
            mask_text_input = tokenizer(mask_caption, padding='longest', truncation=True, max_length=200, return_tensors="pt").to(device)
            if epoch>0:
                alpha = config['alpha']
            else:
                alpha = config['alpha']*min(1,i/len(data_loader))
            loss = model(image=image, text=text_input, text_mask=mask_text_input, alpha = alpha, mode=args.mode)  
            loss.backward()
            optimizer.step()    
            
            metric_logger.update(loss_mlm=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
            
            if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                scheduler.step(i//step_size)         
            
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

    train_dataset = [GQA_train_dataset(config[args.train_file], img_res=config['image_res'], images_path=config['image_path'])]
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank() 
        samplers_train = create_sampler(train_dataset, [True], num_tasks, global_rank)   
    else:
        samplers = [None]

    train_data_loader = create_loader(train_dataset,samplers_train,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
    
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
    model = PEVL_VQA(config=config, tokenizer=tokenizer, postoken_dict = postoken_dict, init_deit=False)
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
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module   

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        train_stats = train(model, train_data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, args)
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
    
    if args.evaluate:
        if utils.is_main_process():
            val_dataset = [GQA_val_dataset(config[args.eval_file], img_res=config['image_res'], images_path=config['image_path'])] 
            val_data_loader = create_loader(val_dataset, [None], batch_size=[config['test_batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
            gqa_val(model.module, val_data_loader, tokenizer, device, config['answer_dict_path'])
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
    parser.add_argument('--find_unused_parameters', default=False, type=bool, help=' When using distributed training, the value of the flag find_unused_parameters passed to DistributedDataParallel')
    parser.add_argument('--train_file', default='')
    parser.add_argument('--evaluate', default=False, type=bool)
    parser.add_argument('--training_mode', default='pretrain')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
