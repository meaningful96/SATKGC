import os
import json
import torch
import torch.utils.data.dataset
import random
import time
import datetime
from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

from dfs import DFS_PATH


entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))
        
        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, data, task, batch_indices = None, negative=False):
               
        if negative == False: 
            self.examples = [item for idx in batch_indices for item in data[idx]]
        else:
            self.examples = data

        self.task = task 
                 
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()
    
    def get_examples(self):
        return [example.vectorize() for example in self.examples]



def load_data(path: str, graph, directed_graph, walks_num):
#def load_data(path, path1, path2, walks_num=100):
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    logger.info('In test mode: {}'.format(args.is_test))
    logger.info('start dfs !!!')
    #start = time.time()
    D = DFS_PATH(graph, directed_graph, args.walks_num)
    all_pathes, pathes_dict = D.intermediate()
    '''
    all_pathes = json.load(open(path1, 'r', encoding='utf-8'))  
    logger.info('load train dfs pathes: done')
    pathes_dict = json.load(open(path2, 'r', encoding = 'utf-8'))
    logger.info('load train dfs path triplets: done')
    '''
    triplets  = json.load(open(path, 'r', encoding='utf-8'))
    
    count_check = {}
    for triple in triplets:
        head_id = triple['head_id']
        tail_id = triple['tail_id']
        relation = triple['relation']
        k = (head_id, relation, tail_id)
        count_check[k] = 0
    
    for path in all_pathes:
        for triplet in path:
            h,r,t = triplet['head_id'], triplet['relation'], triplet['tail_id']
            k = (h, r, t)
            count_check[k] +=1
    
    not_seen_triplets = []
    not_seen_path = 0
    for k, v in count_check.items():
        if v ==0:
            not_seen_path +=1
            h,r,t = k[0], k[1], k[2]
            not_seen_triplets.append({'head_id':h, 'relation':r, 'tail_id':t})        

    if not_seen_path > walks_num:
        a, b = divmod(not_seen_path, walks_num)
        for i in range(a):
            not_seen = not_seen_triplets[i*walks_num:i*walks_num + walks_num]    
            all_pathes.append(not_seen)
    
        not_seen = not_seen_triplets[-b:]
        for t in random.sample(triplets, walks_num-b):
            h,r,t = t['head_id'], t['relation'], t['tail_id']
            not_seen.append({'head_id':h, 'relation':r, 'tail_id':t})
        all_pathes.append(not_seen)

    else:
        for t in random.sample(triplets, walks_num-not_seen_path):
            h,r,t = t['head_id'], t['relation'], t['tail_id']
            not_seen_triplets.append({'head_id':h, 'relation':r, 'tail_id':t})
        all_pathes.append(not_seen_triplets)
    
    num_dfs_path = len(all_pathes)
    logger.info('Load {} dfs pathes'.format(num_dfs_path))     
    vectorized_examples = []
    for i in range(num_dfs_path):
        examples = []
        reverse_examples = []
        for j in range(len(all_pathes[i])):
            obj = all_pathes[i][j]
            examples.append(Example(**obj))
            reverse_examples.append(Example(**reverse_triplet(obj)))
        examples.extend(reverse_examples)
        vectorized_examples.append(examples)
    
    #del all_pathes
    logger.info('vectorize tripets: done')
    return all_pathes, vectorized_examples, pathes_dict

def load_test_data(test_path: str, add_forward_triplet: bool = True,
              add_backward_triplet: bool = True):
    logger.info('In test mode: {}'.format(args.is_test))
    data = json.load(open(test_path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), test_path))
    
    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples
    
def make_negative_examples(negatives, add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    
    examples = []
    for neg in negatives:
        if add_forward_triplet:
            examples.append(Example(**neg))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(neg)))

    return examples



def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'input_batch_data': batch_data,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
