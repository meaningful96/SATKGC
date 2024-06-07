import json
import torch
import time
import datetime
import math
import random
import networkx as nx
import torch.nn as nn
import torch.utils.data
import numpy as np 
from collections import defaultdict

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import Dataset, collate, load_data, make_negative_examples
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer, count_dict
from logger_config import logger

from negative import node_degree, negative_sampling 
from dfs import make_graph

class Trainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()
        self.batch_size = args.batch_size
        # define loss function (criterion) and optimizer
        #self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        self.graph, self.directed_graph, train_len = make_graph(args.train_path)
        self.valid_graph, self.valid_directed_graph, valid_len = make_graph(args.valid_path)
        
        train_pathes, self.train, self.train_pathes_dict = load_data(args.train_path, self.graph, self.directed_graph, args.walks_num//2) 
        valid_pathes, self.valid, self.valid_pathes_dict = load_data(args.valid_path, self.valid_graph, self.valid_directed_graph, args.walks_num//2)
      
        self.train_batch_count, self.train_triplet_mapping, self.train_count_check, self.train_batch_appear = count_dict(train_pathes, args.walks_num//2)
        self.valid_batch_count, self.valid_triplet_mapping, self.valid_count_check, self.valid_batch_appear = count_dict(valid_pathes, args.walks_num//2)
        
        
        #degree_weighted & degree proportional or degree uniform 
        #self.train_deg_dict, self.train_deg_dist, self.train_pathes_dist, self.train_neighbors_degree= node_degree(self.graph, self.train_pathes_dict)
        #self.valid_deg_dict, self.valid_deg_dist, self.valid_pathes_dist, self.valid_neighbors_degree= node_degree(self.valid_graph, self.valid_pathes_dict)
        
        #original mcmc: normal & uniform 
        self.train_deg_dict, self.train_deg_dist = node_degree(self.graph, self.train_pathes_dict)
        self.valid_deg_dict, self.valid_deg_dist = node_degree(self.valid_graph, self.valid_pathes_dict)       
        
        self.training_steps = math.ceil(train_len // max(args.batch_size//2, 1))
        #self.training_steps += 60
        self.valid_steps = math.ceil(valid_len // max(args.batch_size//2, 1))
        #self.valid_steps += 10
        
        total_training_steps = args.epochs * self.training_steps
        args.warmup = min(args.warmup, total_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(total_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(total_training_steps)
        self.best_metric = None

        self.train_loader = None
        self.valid_loader = None

        self.train_loss = []
        self.valid_loss = []
        
    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        batch_indices = None
        valid_batch_indices = None
        
        torch.cuda.set_device(4)
        for epoch in range(self.args.epochs):
            # train for one epoch
            batch_indices = self.train_epoch(epoch, batch_indices, self.args.N_steps, self.args.N_negs)
            valid_batch_indices = self._run_eval(epoch=epoch, valid_batch_indices = valid_batch_indices, N_steps=self.args.N_steps, N_negs=self.args.N_negs)
        
        logger.info('train_loss: {}'.format(self.train_loss))
        logger.info('valid_loss: {}'.format(self.valid_loss))
        not_seen_train = 0
        not_seen_valid = 0

        
        for v in self.train_count_check.values():
            if v==0:
                not_seen_train +=1
        for v in self.valid_count_check.values():
            if v==0:
                not_seen_valid +=1
        
        with open(f'triplet_count/{self.args.task}/{self.args.walks_num}pos_{self.args.N_negs}Neg_dwdp_existdup_train.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.train_count_check,ensure_ascii=False, indent=4))
        with open(f'triplet_count/{self.args.task}/{self.args.walks_num}pos_{self.args.N_negs}Neg_dwdp_existdup_valid.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.train_count_check,ensure_ascii=False, indent=4))
                  
        logger.info('never seen in train: {}'.format(not_seen_train))
        logger.info('never seen in validation: {}'.format(not_seen_valid))
        
                    
    @torch.no_grad()
    def _run_eval(self, epoch, valid_batch_indices, N_steps, N_negs, step=0):
        eval_output = self.eval_epoch(epoch, valid_batch_indices, N_steps, N_negs)
        logger.info(eval_output)
        metric_dict, valid_batch_indices = eval_output['metric_dict'], eval_output['valid_batch_indices']
        is_best = self.valid_loader and (self.best_metric is None or metric_dict['Acc@1'] > self.best_metric['Acc@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)
        
        return valid_batch_indices

    @torch.no_grad()
    def eval_epoch(self, epoch, valid_batch_indices, N_steps, N_negs) -> Dict:

        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        if valid_batch_indices is None:
                valid_batch_indices = random.sample(range(len(self.valid)), self.valid_steps)
                for idx in valid_batch_indices:
                    self.valid_batch_appear[idx] +=1        
        dynamic_valid_data = Dataset(data = self.valid, task=self.args.task, batch_indices=valid_batch_indices)
        self.valid_loader = torch.utils.data.DataLoader(dynamic_valid_data, batch_size=self.args.walks_num, shuffle=False, collate_fn=collate, 
                                                        num_workers= self.args.workers, pin_memory=True)
        
        for i, batch_dict in enumerate(self.valid_loader):
            self.model.eval()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            #batch_dict['only_bfs_embedding'] = True
                
            hr_vector = self.model(**batch_dict)['hr_vector']
            head_ids = []
            for b in range(self.args.walks_num//2): 
                head_ids.append(batch_dict['batch_data'][b].head_id)
                
            sampled_negative_triplets = negative_sampling(self.model, hr_vector[:self.args.walks_num//2], self.valid_pathes_dict, self.valid_deg_dist, 
                                         self.args.walks_num, N_steps, N_negs, self.valid_graph, self.valid_directed_graph, head_ids)  
                   
            negative_examples = make_negative_examples(sampled_negative_triplets)
            negative_examples_dataset = Dataset(data=negative_examples, task=self.args.task, negative=True)   
            all_examples = batch_dict['input_batch_data'] + negative_examples_dataset.get_examples()
            examples_dict = collate(all_examples)
            
            batch_size = len(examples_dict['batch_data'])
            batch_tail_ids = []
            
            #count entities in batch
            for ex in range(batch_size):
                ex_obj = examples_dict['batch_data'][ex]
                head_id = str(ex_obj.head_id)
                relation = str(ex_obj.relation)
                tail_id = str(ex_obj.tail_id)
                batch_tail_ids.append(tail_id)
                
                k = (head_id, relation, tail_id)
                self.valid_count_check[str(k)] +=1
                batch_ids = self.valid_triplet_mapping[str(k)]
                for id in batch_ids:
                    self.valid_batch_count[id] +=1
                
            assert batch_size == self.batch_size
            examples_dict = move_to_cuda(examples_dict)
            
            outputs = self.model(**examples_dict)
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=examples_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            #loss = self.criterion(logits, labels)
            log_tail_deg = torch.tensor([np.log(self.valid_deg_dict[tail]+1) for tail in batch_tail_ids], dtype=torch.float32).to(logits.device)
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()
            # head + relation -> tail
            loss = criterion(logits, labels)
            loss *= log_tail_deg
            loss = sum(loss)/len(loss)
            losses.update(loss.item(), batch_size)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

        self.valid_loss.append(round(losses.avg,3))

        valid_count = sorted(self.valid_batch_count.items(), key=lambda x: x[1])
        valid_batch_indices =  [item[0] for item in valid_count[:self.valid_steps]] 
        valid_batch_indices = list(valid_batch_indices)
            
        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        eval_output =  {'metric_dict': metric_dict, 
                        'valid_batch_indices': valid_batch_indices}
        
        return eval_output

    def train_epoch(self, epoch, batch_indices, N_steps, N_negs):
        
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        inv_t = AverageMeter('InvT', ':6.2f')
        progress = ProgressMeter(
            self.training_steps,
            [losses, inv_t, top1, top3],
            prefix="Epoch: [{}]".format(epoch))
        
        if batch_indices is None:
                batch_indices = random.sample(range(len(self.train)), self.training_steps)
                
                for idx in batch_indices:
                    self.train_batch_appear[idx] +=1
                    
        dynamic_train_data = Dataset(data=self.train, task=self.args.task, batch_indices=batch_indices)
        self.train_loader = torch.utils.data.DataLoader(dynamic_train_data, batch_size=self.args.walks_num, shuffle=False, collate_fn=collate, num_workers= self.args.workers, pin_memory=True)
        
        for i, batch_dict in enumerate(self.train_loader):
            # switch to train mode
            self.model.train()

            if torch.cuda.is_available():
                batch_dict = move_to_cuda(batch_dict)
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    hr_vector = self.model(**batch_dict)['hr_vector']
            else:
                hr_vector = self.model(**batch_dict)['hr_vector']
            
            #start_node = batch_dict['batch_data'][0].head_id
            
            head_ids = []
            
            for b in range(self.args.walks_num//2): 
                head_ids.append(batch_dict['batch_data'][b].head_id)
            
            sampled_negative_triplets = negative_sampling(self.model, hr_vector[:self.args.walks_num//2], self.train_pathes_dict, self.train_deg_dist, 
                                         self.args.walks_num, N_steps, N_negs, self.graph, self.directed_graph, head_ids)  
            
            negative_examples = make_negative_examples(sampled_negative_triplets)
            negative_examples_dataset = Dataset(data=negative_examples, task=self.args.task, negative=True)
            all_examples = batch_dict['input_batch_data'] + negative_examples_dataset.get_examples()
            examples_dict = collate(all_examples)
            batch_size = len(examples_dict['batch_data'])
            batch_head_ids = []
            batch_tail_ids = []
            
            #count entities in batch
            for ex in range(batch_size):
                ex_obj = examples_dict['batch_data'][ex]
                head_id = str(ex_obj.head_id)
                relation = str(ex_obj.relation)
                tail_id = str(ex_obj.tail_id)
                batch_head_ids.append(head_id)
                batch_tail_ids.append(tail_id)
                
                k = (head_id, relation, tail_id)
                self.train_count_check[str(k)] +=1
                batch_ids = self.train_triplet_mapping[str(k)]
                for id in batch_ids:
                    self.train_batch_count[id] +=1
                
            assert batch_size == self.batch_size
            examples_dict = move_to_cuda(examples_dict)
            
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**examples_dict)
            else:
                outputs = self.model(**examples_dict)
            
            outputs = get_model_obj(self.model).compute_logits(output_dict=outputs, batch_dict=examples_dict)
            outputs = ModelOutput(**outputs)
            logits, labels = outputs.logits, outputs.labels
            
            assert logits.size(0) == batch_size
            
            
            log_tail_deg = torch.tensor([np.log(self.train_deg_dict[tail]+1) for tail in batch_tail_ids], dtype=torch.float32).to(logits.device)
            log_head_deg = torch.tensor([np.log(self.train_deg_dict[head]+1) for head in batch_head_ids], dtype=torch.float32).to(logits.device)
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()
            # head + relation -> tail
            loss = criterion(logits, labels)
            loss *= log_tail_deg
            loss = sum(loss)/len(loss)
                 
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()
            # tail -> head + relation
            inv_loss = criterion(logits[:, :batch_size].t(), labels)
            inv_loss *= log_head_deg
            inv_loss = sum(inv_loss)/len(inv_loss)
            loss += inv_loss
            
            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), batch_size)
            top3.update(acc3.item(), batch_size)

            inv_t.update(outputs.inv_t, 1)
            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
            self.scheduler.step()

            
            if i % self.args.print_freq == 0:
                progress.display(i)
            
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)
        
        count = sorted(self.train_batch_count.items(), key=lambda x: x[1])
        batch_indices = [item[0] for item in count[:self.training_steps]]
        batch_indices = list(batch_indices)
        
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))
        logger.info('{} Epoch Training loss: {}'.format(epoch, round(losses.avg,3)))
        self.train_loss.append(round(losses.avg,3))

        return batch_indices

    def _setup_training(self):
        self.model = torch.nn.DataParallel(self.model, device_ids=[4,1,2,3,0]).cuda('cuda:4')
      

    def _create_lr_scheduler(self, total_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=total_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=total_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
