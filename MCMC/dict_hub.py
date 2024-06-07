import os
import glob
import json

from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph
from logger_config import logger
from typing import Optional, List
from collections import defaultdict

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
link_graph: LinkGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))


def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    _init_link_graph()
    return link_graph


def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer


def count_dict(examples: List[list], path_length):
    cnt = len(examples)
    batch_count = dict()
    batch_appear = dict()
    triplet_mapping = defaultdict(list)
    count_check = defaultdict()
    
    for i in range(cnt):
        batch_count[i] = 0
        batch_appear[i] = 0  
        for j in range(path_length):
            head_id = examples[i][j]['head_id']
            tail_id = examples[i][j]['tail_id']
            relation = examples[i][j]['relation']
            inv_relation = 'inverse '+ relation
            k = (head_id, relation, tail_id)
            inv_k = (tail_id, inv_relation, head_id)
            triplet_mapping[str(k)].append(i)
            triplet_mapping[str(inv_k)].append(i)
            
            count_check[str(k)] = 0
            count_check[str(inv_k)] = 0
            

    return batch_count, triplet_mapping, count_check, batch_appear



