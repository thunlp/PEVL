'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import utils
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from collections import Counter
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from itertools import zip_longest
from tqdm import tqdm
from models.model_vrd import PEVL_Vrd
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset.vrd_dataset import VRD_train_dataset, VRD_eval_dataset
from dataset import create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, mode):
    # train
    model.train()  
    if mode=='finetune':
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50   
        step_size = 100
        warmup_iterations = warmup_steps*step_size  
        
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            optimizer.zero_grad()
    
            image = image.to(device,non_blocking=True) 
            text_mask_list = []
            text_gt_list = []

            for x in text:
                text_gt, text_mask = x.split('[SPLIT]')
                text_gt_list.append(text_gt)
                text_mask_list.append(text_mask)

            text_input = tokenizer(text_gt_list, padding='longest', truncation=True, max_length=250, return_tensors="pt").to(device)  
            text_mask_input = tokenizer(text_mask_list, padding='longest', truncation=True, max_length=250, return_tensors="pt").to(device) 
            if epoch>0:
                alpha = config['alpha']
            else:
                alpha = config['alpha']*min(1,i/len(data_loader)) 
            
            loss_mlm = model(image, text_input, text_mask_input, alpha = alpha, mode='finetune')  
                
            loss_mlm.backward()
            optimizer.step()    
            
            metric_logger.update(loss_mlm=loss_mlm.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
            
            if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                scheduler.step(i//step_size)         
            
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())     
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

    elif mode =='pretrain':
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

        for i, (image, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            optimizer.zero_grad()
    
            image = image.to(device,non_blocking=True) 
            text_mask_list = []
            text_gt_list = []
            for x in text:
                text_gt, text_mask = x.split('[SPLIT]')
                text_gt_list.append(text_gt)
                text_mask_list.append(text_mask)

            text_input = tokenizer(text_gt_list, padding='longest', truncation=True, max_length=250, return_tensors="pt").to(device)  
            text_mask_input = tokenizer(text_mask_list, padding='longest', truncation=True, max_length=250, return_tensors="pt").to(device) 
            if epoch>0:
                alpha = config['alpha']
            else:
                alpha = config['alpha']*min(1,i/len(data_loader)) 
            
            loss_mlm,  loss_soft, loss_ita, loss_itm = model(image, text_input, text_input, alpha = alpha, mode='pretrain')  
                
            loss = loss_mlm + loss_soft + loss_ita + loss_itm
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


@torch.no_grad()
def vrd_inference(model, data_loader, tokenizer, device, config,):
    model.eval()
    results = {}
    vg_dict = json.load(open(config['vg_path_dict']))
    rels = vg_dict['idx_to_predicate']
    rels_dict = {}
    rels_dict[0] = ' [unrel] [unrel] [unrel] '
    for x in range(1,51):
        rel = rels[str(x)]
        if len(rel.split(' ')) == 1:
            q = rel.split(' ')
            q.append('[unrel]')
            q.append('[unrel]')
        elif len(rel.split(' '))==2:
            q = rel.split(' ')
            q.append('[unrel]')
        elif len(rel.split(' '))==3:
            q = rel.split(' ')
        else:
            assert len(rel.split(' ')) > 0, " length of relation tokens is {}".format(len(rel.split(' ')))
            assert len(rel.split(' '))<4, " length of relation tokens is {}".format(len(rel.split(' ')))
        rels_dict[x] = ' '.join(q)
    rels_list = []
    for x in range(51):
        rels_list.append(rels_dict[x])
    relations_cls = tokenizer(rels_list, padding='longest', return_tensors='pt').input_ids
    relation_input = []
    for relation in relations_cls:
        relation_input.append(relation[1:])
    relation_input = torch.stack(relation_input, dim=0).to(device)
    first_index = relation_input[:,0]
    second_index = relation_input[:,1]
    third_index = relation_input[:,2]
    predicate_length_list = []
    predicate_length_list.append(1.0/3.0)
    for x in rels.values():
        assert len(rels.values()) == 50
        predicate_length_list.append(1/float(len(x.split(' '))))
    predicate_length_list = torch.tensor(predicate_length_list)
    for i, (image, text, id_pair, imgids) in tqdm(enumerate(data_loader)):
        image = image.view((1,3,config['image_res'],config['image_res']))
        image = image.to(device,non_blocking=True)
        assert len(image) == len(text)
        text_mask_list = text[0].split('__')
        id_pair_list = id_pair[0].split('#')
        assert len(id_pair_list) == len(text_mask_list)
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) 
        
        num = int(len(text_mask_list)/16.0)
        split_text_input_list = []
        split_id_pair_list = []
        for split_index in range(num+1):
            split_text_input_list.append(text_mask_list[16*split_index:16*(split_index+1)])
            split_id_pair_list.append(id_pair_list[16*split_index:16*(split_index+1)])
        for mask_list, id_pairs in zip(split_text_input_list, split_id_pair_list):
            if len(mask_list) == 0:
                continue
            text_mask_input = tokenizer(mask_list, padding='longest', truncation=True, max_length=250, return_tensors="pt").to(device)
            n=len(mask_list)
            image_embeds_n = [image_embeds]*n
            image_atts_n = [image_atts]*n
            image_embeds_n = torch.stack(image_embeds_n,0).view(n,1025,768)
            image_atts_n = torch.stack(image_atts_n,0).view(n,1025)
            input_ids = text_mask_input.input_ids.clone()
            relation_mask = input_ids == 103
            mlm_output = model.text_encoder(input_ids, 
                                            attention_mask = text_mask_input.attention_mask,encoder_hidden_states = image_embeds_n,
                                            encoder_attention_mask = image_atts_n, return_dict = True,)                           
            mlm_logits = F.softmax(mlm_output.logits[relation_mask].view(-1,3,30522), \
                                        dim=2).detach().cpu()
            assert len(mlm_logits) == len(mask_list)
            for pair_id, sub_obj_relation_logit in zip(id_pairs, mlm_logits):
                sub_obj_relation_logit = sub_obj_relation_logit.view(3,30522)
                first_relation_token_log_probs = sub_obj_relation_logit[0][first_index].log().view(-1,1)
                second_relation_token_log_probs = sub_obj_relation_logit[1][second_index].log().view(-1,1)
                second_relation_token_log_probs[second_index == 719] = 0
                third_relation_token_log_probs = sub_obj_relation_logit[2][third_index].log().view(-1,1)
                third_relation_token_log_probs[third_index == 719] = 0
                relation_log_probs = torch.cat([first_relation_token_log_probs, second_relation_token_log_probs, \
                                            third_relation_token_log_probs], dim=1).sum(1)
                relation_log_probs = relation_log_probs * predicate_length_list
                assert len(relation_log_probs) == 51
                img_pair_id = imgids[-1]+'_'+pair_id
                relation_log_probs = relation_log_probs
                results[img_pair_id] = relation_log_probs
    return results



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
    
    ##our tokenizer
    unus = ['[unused{}]'.format(x) for x in range(200,800)]
    pos_token = ['@@']
    pos_token.extend([f'[pos_{x}]' for x in range(512)])
    pos_token.append('##')
    pos_token.append('[unrel]')
    postoken_dict = {}
    tokenizer = BertTokenizer.from_pretrained('vocab.txt')
    for x,y in zip(unus, pos_token):
        un_index = tokenizer.vocab[x]
        tokenizer.vocab[y] = un_index
        postoken_dict[y] = un_index
        _ = tokenizer.vocab.pop(x)
        tokenizer.basic_tokenizer.never_split.add(y)
    postoken_dict.pop('@@')
    postoken_dict.pop('##')
    postoken_dict.pop('[unrel]')
    postoken_index = torch.randn(30522).bool()
    postoken_index[:] = False
    for x in postoken_dict.values():
        postoken_index[x]=True
   

    #### Model #### 
    print("Creating model")
    model = PEVL_Vrd(config=config, tokenizer=tokenizer, postoken_dict = postoken_dict, init_deit=False)
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
        model.load_state_dict(state_dict,strict=False)    
        print('load checkpoint from %s'%args.checkpoint)
    
    model_without_ddp = model
    
    if args.distributed:
        if args.pretrain:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
    

    print("Start training")
    start_time = time.time()

    if args.train:
        vrd_datasets = [VRD_train_dataset(config['train_file'], img_res=config['image_res'], vg_dict_path=config['vg_dict_path'])]
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler(vrd_datasets, [True], num_tasks, global_rank)         
        else:
            samplers = [None]
        data_loader = create_loader(vrd_datasets, samplers, batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

        for epoch in range(start_epoch, max_epoch):
            if epoch>0:
                lr_scheduler.step(epoch+warmup_steps)  
            
            train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, args.mode)
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
                torch.save(save_obj, os.path.join(args.output_dir, 'vrd_checkpoint_%02d.pth'%epoch))   
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            
            if torch.distributed.get_rank() == 0:
                start_time = time.time()
                val_model = model_without_ddp
                val_gts = json.load(open(config['val_gt_file']))
                sgg_val_dataset = [VRD_eval_dataset(config['val_file'], config['image_res'], config['vg_dict_path'], config['vg_root'])]
                sgg_val_loader = create_loader(sgg_val_dataset, [None], batch_size=[1], 
                                            num_workers=[1],
                                            is_trains=[False],
                                            collate_fns=[None])[0]
                val_results = vrd_inference(val_model, sgg_val_loader, tokenizer, device, config)
                print(eval_vg(val_results, val_gts))
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('SGG predcls val time {}'.format(total_time_str))
            
            dist.barrier()  
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str)) 
    else:
        # evaluation
        if torch.distributed.get_rank() == 0:
            start_time = time.time()
            val_model = model_without_ddp
            test_gts = json.load(open(config['test_gt_file']))
            vrd_test_dataset = [VRD_eval_dataset(config['test_file'], config['image_res'], config['vg_dict_path'], config['vg_root'])]
            vrd_test_loader = create_loader(vrd_test_dataset, [None], batch_size=[1], 
                                        num_workers=[1],
                                        is_trains=[False],
                                        collate_fns=[None])[0]
            test_results = vrd_inference(val_model, vrd_test_loader, tokenizer, device, config)
            print(eval_vg(test_results, test_gts))
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('VRD predcls val time {}'.format(total_time_str))
                
    

def eval_vg(preds, gts):
    imkey2pair = lambda n: [int(x) for x in n.split("_")[-2:]]
    new_preds = {}
    for imkey, prd in preds.items():
        img_name = imkey.split(".jpg")[0]+".jpg"
        pair = imkey2pair(imkey)
        if img_name not in new_preds:
            new_preds[img_name] = []
        new_preds[img_name].append({"pair":pair, "pred": prd})
    for g in gts:
        if g["img_path"] not in new_preds:
            new_preds[g["img_path"]] = [{"pair": [0, 0], "pred": torch.zeros(51, dtype=torch.float)}]
    print(len(new_preds))
    gts = [g for g in gts if g["img_path"] in new_preds]
    assert len(new_preds) == len(gts), "{}, {}".format(len(new_preds), len(gts))
    preds = [new_preds[k["img_path"]] for k in gts]
    recall = {20:[], 50:[], 100:[]}
    mrecall = {20: [[] for i in range(51)], 50: [[] for i in range(51)], 100: [[] for i in range(51)]}
    for p_list, gt in zip(preds, gts):
        pairs = [p['pair'] for p in p_list]
        prds = [p['pred'] for p in p_list]
        pairs = torch.tensor(pairs)
        prds = torch.stack(prds, 0)
        rels = prds[:, 1:].argmax(1)+1
        scores = prds[torch.arange(len(prds)), rels]
        idxs = scores.argsort(descending=True)
        rels = rels[idxs]
        pairs = pairs[idxs]
        rels = torch.cat([pairs, rels[:, None]], -1)
        gt_rels = torch.from_numpy(np.array(gt["relations"]))
        #calculate recall
        for mode in recall:
            pred_rels = rels[:mode]
            rcl = (gt_rels[:,:,None] == pred_rels.T[None, :, :] ).all(1).any(1)
            recall[mode].append(sum(rcl)/float(len(gt_rels)))
            tmp_cnt = Counter(gt_rels[:, 2].tolist())
            tmp_m_recall = {}
            assert len(gt_rels) == len(rcl)
            for r, c in zip(gt_rels[:, 2].tolist(), rcl):
                tmp_m_recall[r] = tmp_m_recall.get(r, 0) + int(c)
            for r in tmp_m_recall:
                mrecall[mode][r].append(tmp_m_recall[r]/tmp_cnt[r])
    recall = {k: np.mean(v) for k, v in recall.items()}
    mrecall = {k: np.mean( [ np.mean(v) if len(v)>0 else 0 for v in v_list[1:] ] ) for k, v_list in mrecall.items()}
    rst = "R@20: {:.4f}\tR@50: {:.4f}\tR@100: {:.4f}".format(recall[20], recall[50], recall[100]) + "\n"
    rst += "mR@20: {:.4f}\tmR@50: {:.4f}\tmR@100: {:.4f}".format(mrecall[20], mrecall[50], mrecall[100]) + "\n"
    return rst





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--mode', default='', type=str)
    parser.add_argument('--pretrain', default=0, type=int)
    parser.add_argument('--train', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)