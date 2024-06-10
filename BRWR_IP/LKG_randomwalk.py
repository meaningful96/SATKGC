from collections import defaultdict, deque
from multiprocessing import Pool, Manager, Process
from typing import List, Dict, Tuple
from logger_config import logger


import multiprocessing
import networkx as nx
import numpy as np
import datetime
import argparse
import random
import pickle
import time
import json
import gc
import os
import re

def build_nxGraph(path:str):
    s = time.time()
    Graph = nx.Graph()
    diGraph = nx.DiGraph()
    data = json.load(open(path, 'r', encoding='utf-8'))
    for ex in data:
        h,r,t = ex['head_id'], ex['relation'], ex['tail_id']
        Graph.add_node(h)
        Graph.add_node(t)
        Graph.add_edge(h,t, relation=r)

        diGraph.add_node(h)
        diGraph.add_node(t)
        diGraph.add_edge(h, t, relation=r)

    del data

    entities = list(Graph.nodes())
    e = time.time() 

    logger.info("Done building NetworkX Graph: {}".format(datetime.timedelta(seconds = e - s)))
    return Graph, diGraph, entities

def build_appearance(data):
    appearance = defaultdict(int)
    for ex in data:
        h,r,t = ex['head_id'], ex['relation'], ex['tail_id'] 
        triple = (h,r,t)
        appearance[triple] = 0

    del data

    return appearance

def rearrange_list(input_list):
    output_list = [(h, r, t) for (t, r, h) in input_list]
    return output_list

def weighted_random_selection(prob_distribution):
    # Generating a list of indices based on the probability distribution
    indices = list(range(len(prob_distribution)))
    selected_index = random.choices(indices, weights=prob_distribution, k=1)[0]
    
    # Output: int
    return selected_index

class Biased_RandomWalk:
    def __init__(self, train_path:str, distribution:str, nxGraph, nxDiGraph, entities):
        s = time.time()

        self.Graph, self.diGraph, self.entities = build_nxGraph(train_path)

        if distribution == 'proportional':
            self.degree_prob = defaultdict(list)        
            for entity in self.entities:
                neighbors = list(self.Graph.neighbors(entity))
                prob_proportional = list(np.zeros(len(neighbors)))
                for i, neighbor in enumerate(neighbors):
                    d = self.Graph.degree(neighbor)
                    prob_proportional[i] = d
                self.degree_prob[entity].extend(prob_proportional)

        elif distribution == 'antithetical':
            self.degree_prob = defaultdict(list)        
            for entity in self.entities:
                neighbors = list(self.Graph.neighbors(entity))
                prob_antithetical = list(np.zeros(len(neighbors)))
                for i, neighbor in enumerate(neighbors):
                    d = self.Graph.degree(neighbor)
                    prob_antithetical[i] = 1/d
                self.degree_prob[entity].extend(prob_antithetical)

        del self.entities

        self.cnt = 0
        e = time.time()
        logger.info("Done Class Initilaization: {}".format(datetime.timedelta(seconds=e-s)))
 

    def find_triple(self, diGraph, e1, e2):
        if diGraph.has_edge(e1, e2):
            relation = diGraph.edges[e1, e2]['relation']
            triple = (e1, relation, e2)
        elif diGraph.has_edge(e2, e1):
            relation = diGraph.edges[e2, e1]['relation']
            triple = (e2, relation, e1)
        return triple
                

    def biased_randomwalk(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int, distribution: str) -> List[list]:
        center_triple = (head_id, relation, tail_id)
        
        # Step 0. Initialization 
        subgraph_candidates = []         
        nh, nt = len(list(self.Graph.neighbors(head_id))), len(list(self.Graph.neighbors(tail_id)))
        center_ent = [head_id, tail_id]    
        center_pro, center_ant = [nh, nt], [1/nh, 1/nt]

        # Hard Negative is the close neighbors from the center triple
        # For example, if the triple is placed far away from the center triple,
        # It's not Hard negative. It is called easy negative
        # The Step number is for the boundary and the reset probabaility is the half of the step number 

        prob_reset = k_steps//2 # Mean of the path length 
        r_prob = 1 / prob_reset
        s_prob = 1 - r_prob

        # Step 1. Selecting the Start Point
        cnt =  0
        iterations = num_iter

        margin = 1000
        while cnt <= iterations:            
            # Uniform Distribution
            if distribution == "uniform":
                current_entity = random.choice([head_id, tail_id])
            # Degree Proportional
            if distribution == 'proportional':
                current_entity = center_ent[weighted_random_selection(center_pro)]            
            # Degree Antithetical
            if distribution == 'antithetical': 
                current_entity = center_ent[weighted_random_selection(center_ant)]


            # Step 2. Biased Random Walk with Restart(BRWR)
            prob_list = [r_prob, s_prob]
            standard = weighted_random_selection(prob_list)
            
            triples_list = []
            visited = set()
            visited.add(current_entity)
            candidate = None
            for _ in range(k_steps):
                append = False

                # Restart
                if standard == 0:
                    subgraph_candidates.extend(triples_list)
                    break
                
                # Walking toward the neighbors
                if standard == 1:

                    d_current = self.Graph.degree(current_entity)
                    neighbors = list(self.Graph.neighbors(current_entity))                                      
                    if set(neighbors) - visited == 0:
                        candidate = random.choice(neighbors)
                        break
                    else:
                        # Uniform Distribution
                        if distribution == 'uniform':
                            candidate = random.choice(neighbors)
                        
                        # Degree Proportional
                        if distribution == 'proportional':
                            if d_current >= margin:
                                candidate = random.sample(neighbors, 1)[0]
                            else:
                                candidate_prob = self.degree_prob[current_entity]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]
                       
                        # Degree Antithetical
                        if distribution == 'antithetical':
                            if d_current >= margin:
                                candidate = random.sample(neighbors, 1)[0]
                            else:
                                candidate_prob = self.degree_prob[current_entity]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]

                    visited.add(candidate)
                    triple = self.find_triple(self.diGraph, current_entity, candidate)
                    triples_list.append(triple)                                
                    current_entity = candidate                                                
                standard = weighted_random_selection(prob_list)

            subgraph_candidates.extend(triples_list)
            subgraph_candidates = list(set(subgraph_candidates))
            cnt += 1
        
        subgraph_candidates = subgraph_candidates
        return subgraph_candidates
    
    def biased_randomwalk_for_LKG(self, head_id:str, relation:str, tail_id:str, k_steps:int, num_iter:int, distribution: str, subgraph_size:int) -> List[list]:
        center_triple = (head_id, relation, tail_id)
        
        # Step 0. Initialization 
        subgraph_candidates = []         
        nh, nt = len(list(self.Graph.neighbors(head_id))), len(list(self.Graph.neighbors(tail_id)))
        center_ent = [head_id, tail_id]    
        center_pro, center_ant = [nh, nt], [1/nh, 1/nt]

        # Hard Negative is the close neighbors from the center triple
        # For example, if the triple is placed far away from the center triple,
        # It's not Hard negative. It is called easy negative
        # The Step number is for the boundary and the reset probabaility is the half of the step number 

        prob_reset = k_steps//2 # Mean of the path length 
        r_prob = 1 / prob_reset
        s_prob = 1 - r_prob

        # Step 1. Selecting the Start Point
        cnt =  0
        iterations = num_iter

        margin = 1000
        hr_set = set()
        while cnt < iterations  and  len(subgraph_candidates) < subgraph_size:            
            # Uniform Distribution
            if distribution == "uniform":
                current_entity = random.choice([head_id, tail_id])
            # Degree Proportional
            if distribution == 'proportional':
                current_entity = center_ent[weighted_random_selection(center_pro)]            
            # Degree Antithetical
            if distribution == 'antithetical': 
                current_entity = center_ent[weighted_random_selection(center_ant)]


            # Step 2. Biased Random Walk with Restart(BRWR)
            prob_list = [r_prob, s_prob]
            standard = weighted_random_selection(prob_list)
            
            triples_list = []
            visited = set()
            visited.add(current_entity)
            candidate = None
            sKG = []

            for _ in range(k_steps):
                append = False

                # Restart
                if standard == 0:
                    subgraph_candidates.extend(triples_list)
                    break
                
                # Walking toward the neighbors
                if standard == 1:

                    d_current = self.Graph.degree(current_entity)
                    neighbors = list(self.Graph.neighbors(current_entity))                                      
                    if set(neighbors) - visited == 0:
                        candidate = random.choice(neighbors)
                        break
                    else:
                        # Uniform Distribution
                        if distribution == 'uniform':
                            candidate = random.choice(neighbors)
                        
                        # Degree Proportional
                        if distribution == 'proportional':
                            if d_current >= margin:
                                candidate = random.sample(neighbors, 1)[0]
                            else:
                                candidate_prob = self.degree_prob[current_entity]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]
                       
                        # Degree Antithetical
                        if distribution == 'antithetical':
                            if d_current >= margin:
                                candidate = random.sample(neighbors, 1)[0]
                            else:
                                candidate_prob = self.degree_prob[current_entity]
                                selected_index = weighted_random_selection(candidate_prob)
                                candidate = neighbors[selected_index]

                    visited.add(candidate)
                    triple = self.find_triple(self.diGraph, current_entity, candidate)
                    if triple[0] not in hr_set and triple[2] not in hr_set:                        
                        triples_list.append(triple)
                        hr_set.add(triple[0])
                        hr_set.add(triple[2])
                    current_entity = candidate
                    sKG.append(triple)
                standard = weighted_random_selection(prob_list)

            subgraph_candidates.extend(triples_list)
            subgraph_candidates = list(set(subgraph_candidates))
            cnt += 1
        
        subgraph_candidates = subgraph_candidates
        return subgraph_candidates, sKG

def counting(appearance, subgraph):
    for triple in subgraph:
        if not triple in appearance:
            appearance[triple] = 0
        appearance[triple] += 1
    return appearance

def process_triple(data, example, obj, k_steps, num_iter, distribution):
    head_id, relation, tail_id = example[0], example[1], example[2]
    subgraph = obj.biased_randomwalk(head_id, relation, tail_id, k_steps, num_iter, distribution)
    subgraph[0] = example
    assert subgraph[0] == example
    return subgraph

def process_triple_lkg(data, example, obj, k_steps, num_iter, distribution, subgraph_size):
    head_id, relation, tail_id = example[0], example[1], example[2]
    subgraph, sKG = obj.biased_randomwalk_for_LKG(head_id, relation, tail_id, k_steps, num_iter, distribution, subgraph_size)
    subgraph[0] = example
    assert subgraph[0] == example
    return subgraph, sKG

def random_sample(data, length, k_steps, num_iter):
    margin = k_steps*num_iter - length
    tmp = random.sample(data, min(margin, len(data)))
    all_path = []
    for ex in tmp:
        all_path.append((ex['head_id'], ex['relation'], ex['tail_id']))
    all_path = list(set(all_path))
    return all_path

def process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, subgraph_total_dict, LKG):
    
    if LKG == True:
        cnt = 0
        for center in centers:
            subgraph, skG = process_triple_lkg(data, center, obj, k_steps, num_iter, distribution, subgraph_size)
            
            if len(subgraph) < subgraph_size:
                tmp = random_sample(data, len(subgraph), k_steps, num_iter)
                subgraph.extend(tmp)
            subgraph = subgraph[:subgraph_size]
            assert len(subgraph) == subgraph_size
            cnt += 1
            if cnt % 5000 == 0:
                logger.info(f"Subgraph Total Dict Done: {cnt}")
            assert subgraph[0] == center

            subgraph_total_dict[center] = [subgraph, skG]   

    if LKG == False:
        for center in centers:
            if center in subgraph_total_dict:
                continue

            subgraph = process_triple(data, center, obj, k_steps, num_iter, distribution)
            if len(subgraph) <= 1024:
                tmp = random_sample(data, len(subgraph), k_steps, num_iter)
                subgraph.extend(tmp)
            assert subgraph[0] == center

            subgraph_total_dict[center].extend(subgraph)              
        
    return subgraph_total_dict


def get_subgraph_dict(data, centers, appearance, subgraph_total_dict, subgraph_size, subgraph_dict):
    
    p = len(centers)
    for center in centers:
        subgraph = []
        subgraph_total = subgraph_total_dict[center]
        random.shuffle(subgraph_total)
        
        hr_set = set()
        subgraph.append(center)
        hr_set.add(center[0])
        hr_set.add(center[2])

        for triple in subgraph_total:
            h, t = triple[0], triple[2]
            if h not in hr_set and t not in hr_set:
                subgraph.append(triple)
                hr_set.add(h)
                hr_set.add(t)
            
            if len(subgraph) >= subgraph_size:
                break

        if len(subgraph) < subgraph_size:
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1])
            num_diff = subgraph_size - len(subgraph)
            new_triples = sorted_triples[p:p + num_diff]
            triples = [item[0] for item in new_triples]
            p += num_diff
            
            subgraph.extend(triples)
            if p + subgraph_size >= len(appearance.keys()):
                p = 0
        
        if len(subgraph) >= subgraph_size:
            subgraph = subgraph[:subgraph_size]
        assert len(subgraph) == subgraph_size
    
        subgraph_dict[center].append(subgraph)
        appearance = counting(appearance, subgraph)
    
    return subgraph_dict, appearance


def get_subgraph_dict_LKG(data, centers, appearance, subgraph_total_dict, subgraph_size, subgraph_dict, sKG_dict):
    
    p = len(centers)
    for center in centers:
        subgraph = []
        subgraph.append(center)
        lists = subgraph_total_dict[center]
        subgraph_total = lists[0]
        sKG = lists[1]
        
        tmp = random.sample(subgraph_total, subgraph_size)
        subgraph.extend(tmp)
        subgraph = subgraph[:subgraph_size]
        assert len(subgraph) == subgraph_size

        subgraph_dict[center].append(subgraph)
        appearance = counting(appearance, subgraph)
        sKG_dict[center].extend(sKG)
    return subgraph_dict, appearance, sKG_dict

def Path_Dictionary_for_LKG(data, appearance, obj, k_steps, num_iter, distribution, subgraph_size, phase, subgraph_total_dict, LKG):
    logger.info(f"Building Subgraph Dictionary with BRWR!")
    s = time.time()
    subgraph_dict = defaultdict(list)
    total_candidates = len(data) // subgraph_size * phase
    phase_candidates = len(data) // subgraph_size
    
    logger.info("Total Phase: {}".format(phase))
    logger.info("Total number of center tripels: {}".format(total_candidates))
    logger.info("Center triples per phase: {}".format(phase_candidates))
    
    cnt = 0
    center_list = []
    sKG_dict = defaultdict(list)
    if LKG:
        logger.info("Case 1")
        num_candidates = phase_candidates
        centers = random.sample(list(appearance.keys()), num_candidates) 
        subgraph_total_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, subgraph_total_dict, LKG)
        _, appearance, sKG_dict = get_subgraph_dict_LKG(data, centers, appearance, subgraph_total_dict, subgraph_size, subgraph_dict, sKG_dict)   

        logger.info("Selecting centers Done!!")
        logger.info("Start to Extrac the subgraphs!!")

        subgraph_total_dict = defaultdict(list)
        subgraph_dict = defaultdict(list)
        for _ in range(phase):
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1]) # low frequency
            sorted_triples = sorted_triples[:num_candidates] 
            centers = [ex[0] for ex in sorted_triples]
            center_list.extend(centers)
            cnt += len(centers)
            subgraph_total_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, subgraph_total_dict, LKG)
            logger.info("Final Steps for Extracting Subgraph!!")
            subgraph_dict, appearance, sKG_dict = get_subgraph_dict_LKG(data, centers, appearance, subgraph_total_dict, subgraph_size, subgraph_dict, sKG_dict)
            if cnt % 5000 == 0:
                logger.info(f"Done: {cnt}")
        logger.info(f"Done: {len(center_list)}")
        assert total_candidates == len(center_list)
        result = [subgraph_dict, center_list]
        logger.info("Done building Path Dictionary with sKG")
        return appearance, result, subgraph_total_dict, sKG_dict

    elif not LKG:
        logger.info("Case 2")
        num_candidates = phase_candidates
        centers = random.sample(list(appearance.keys()), num_candidates) 
        subgraph_total_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, subgraph_total_dict, LKG)
        subgraph_dict, appearance = get_subgraph_dict(data, centers, appearance, subgraph_total_dict, subgraph_size, subgraph_dict)
        center_list.extend(centers)
        cnt += len(centers)
        logger.info(f"Done: {cnt}")

        for _ in range(phase-1):
            sorted_triples = sorted(appearance.items(), key=lambda x: x[1]) # low frequency
            sorted_triples = sorted_triples[:num_candidates] 
            centers = [ex[0] for ex in sorted_triples]
            center_list.extend(centers)
            cnt += len(centers)

            subgraph_total_dict = process_centers(data, obj, k_steps, num_iter, distribution, subgraph_size, centers, subgraph_total_dict, LKG)
            subgraph_dict, appearance = get_subgraph_dict(data, centers, appearance, subgraph_total_dict, subgraph_size, subgraph_dict)
            logger.info(f"Done: {cnt}")     

        assert total_candidates == len(center_list)
        result = [subgraph_dict, center_list]
        logger.info("Done building Path Dictionary")
        return appearance, result, subgraph_total_dict

def Making_Subgraph_for_LKG(subgraph_dict, centers, count_dict, subgraph_size):
    total_subgraph = []
    for center in centers:
        tmp = subgraph_dict[center][count_dict[center]]
        total_subgraph.extend(tmp)
        
    total_subgraph = [{'head_id': head, 'relation': relation, 'tail_id': tail} for head, relation, tail in total_subgraph]
    return total_subgraph

def get_degree_dict(subgraph_dict, nxGraph, subgraph_size):
    # degree_dict[Center Triple][subgraph index][o: dh_list, 1: dt_list]
    logger.info("Get DW Dictionary!!")
    s = time.time()
    keys = list(subgraph_dict.keys())
    degree_dict = defaultdict(list)
    for key in keys:
        subgraph_list = subgraph_dict[key]
        for subgraph in subgraph_list:
            dh_list = []
            dt_list = []
            for triple in subgraph:
                dh = nxGraph.degree(triple[0]) + 1
                dt = nxGraph.degree(triple[2]) + 1
                dh_list.extend([dh, dt])
                dt_list.extend([dt, dh])
            assert len(dh_list) == subgraph_size * 2
            assert len(dt_list) == subgraph_size * 2
            degree_dict[key].append([dh_list, dt_list])
    e = time.time()
    logger.info(f"Time for building DW Dictionary: {datetime.timedelta(seconds=e-s)}")
    return degree_dict

def log_progress(queue, total_centers):
    processed_count = 0
    while processed_count < total_centers:
        msg = queue.get()  
        if msg == 'done':
            processed_count += 1
            if processed_count % 2000 == 0:
                logger.info(f"Processed {processed_count} centers out of {total_centers}")



def get_spw_dict(subgraph_dict, nxGraph, subgraph_size):
    logger.info("Start to build Shortest Path Weight Dictionary")
    s = time.time()
    centers = list(set(subgraph_dict.keys()))
    logger.info(f"Total Centers: {len(list(subgraph_dict.keys()))}")
    logger.info(f"Candidates: {len(centers)}")

    logger.info("SPW_Dictionary!!")
    total_sw = defaultdict(list)
    
    manager = Manager()
    queue = manager.Queue()
    
    process = 30
    logger.info(f"Number of Process: {process}")

    progress_process = Process(target=log_progress, args=(queue, len(centers)))
    progress_process.start()
    
    with Pool(process) as p:
        results = p.starmap(get_shortest_distance, [(nxGraph, center, subgraph_dict, subgraph_size, queue) for center in centers])
    
    queue.put('done')
    progress_process.join()

    for sub_list, center in results:
        total_sw[center].extend(sub_list)

    e = time.time()
    logger.info(f"Time for building SPW_Dict: {datetime.timedelta(seconds=e-s)}")
    return total_sw


def get_shortest_distance(nxGraph, center, subgraph_dict, subgraph_size, queue=None):
    head = center[0]
    res = []
    subgraphs = subgraph_dict[center]
    
    for subgraph in subgraphs:
        sub_list = list(np.zeros(subgraph_size * 2))
        assert len(subgraph) == subgraph_size
        for i, triple in enumerate(subgraph):
            tail = triple[2]
            try:
                st = nx.shortest_path_length(nxGraph, source=head, target=tail)
                if st == 0:
                    st = 1
            except nx.NetworkXNoPath:
                st = 999
            except nx.NodeNotFound:
                st = 999
            sub_list[2 * i] = 1 / st
            sub_list[2 * i + 1] = 1 / st
        res.append(sub_list)
    
    if queue:
        queue.put('done')

    return res, center

def get_spw_dict_LKG(subgraph_dict, sKG_dict, subgraph_size):
    logger.info("Start to build Shortest Path Weight Dictionary")
    s = time.time()
    centers = list(set(subgraph_dict.keys()))
    logger.info(f"Total Centers: {len(list(subgraph_dict.keys()))}")
    logger.info(f"Candidates: {len(centers)}")

    logger.info("SPW_Dictionary!!")
    total_sw = defaultdict(list)
    
    manager = Manager()
    queue = manager.Queue()
    
    process = 10
    logger.info(f"Number of Process: {process}")

    progress_process = Process(target=log_progress, args=(queue, len(centers)))
    progress_process.start()
    
    with Pool(process) as p:
        results = p.starmap(get_shortest_distance_LKG, [(sKG_dict[center], center, subgraph_dict, subgraph_size, queue) for center in centers])
    
    queue.put('done')
    progress_process.join()

    for sub_list, center in results:
        total_sw[center].extend(sub_list)

    e = time.time()
    logger.info(f"Time for building SPW_Dict: {datetime.timedelta(seconds=e-s)}")
    return total_sw

def get_shortest_distance_LKG(sKG, center, subgraph_dict, subgraph_size, queue=None):
    head = center[0]
    res = []
    subgraphs = subgraph_dict[center]
    
    nxGraph = nx.Graph()
    for ex in sKG:
        h, r, t = ex[0], ex[1], ex[2]
        nxGraph.add_node(h)
        nxGraph.add_node(t)
        nxGraph.add_edge(h,t, relation=r)

    for subgraph in subgraphs:
        sub_list = list(np.zeros(subgraph_size * 2))
        assert len(subgraph) == subgraph_size
        for i, triple in enumerate(subgraph):
            tail = triple[2]
            try:
                st = nx.shortest_path_length(nxGraph, source=head, target=tail)
                if st == 0:
                    st = 1
                elif st > 10:
                    st = 10
            except nx.NetworkXNoPath:
                st = 999
            except nx.NodeNotFound:
                st = 999
            sub_list[2 * i] = 1 / st
            sub_list[2 * i + 1] = 1 / st
        res.append(sub_list)
    
    if queue:
        queue.put('done')

    return res, center



def main(base_dir, dataset, k_steps, num_iter, distribution, phase, subgraph_size, mode, LKG):
    ## Step 1. Define the all paths
    if mode =='valid':
        num_k_steps = k_steps
        num_iteration = num_iter
    else:
        num_k_steps = k_steps
        num_iteration = num_iter

    logger.info("Start Sampling using BRWR!!")
    logger.info(f"Maximum Step Size: {k_steps}")
    logger.info(f"Maximum iteration: {num_iter}")
    logger.info(f"Total Phase: {phase}")
    logger.info(f"Subgraph Size: {subgraph_size}")
    logger.info(f"LKG: {LKG}")
    s = time.time()
    data_file = f'{mode}.txt.json'
    data_path = os.path.join(base_dir, dataset, data_file) 
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    if mode == 'valid':
        if LKG:
            train_data_path = os.path.join(base_dir, dataset, 'train.txt.json')
            data_length = len(json.load(open(train_data_path, 'r', encoding='utf-8')))
            cnt = data_length // (2500 * subgraph_size * 2) # subgraph_size * 2 == batch_size during training
            logger.info(f"CNT: {cnt}")
            phase = cnt * phase + phase
            logger.info(f'Phase: {phase}')

    subgraph_total_dict_path = os.path.join(base_dir, dataset, f"subgraph_total_dict_{mode}_{k_steps}_{num_iter}.pkl")
    if os.path.exists(subgraph_total_dict_path):
        with open(subgraph_total_dict_path, "rb") as file:
            subgraph_total_dict = pickle.load(file)
        logger.info("Subgraph Total Dictionary loaded successfully!!")
    else:
        subgraph_total_dict = defaultdict(list)
        logger.info("Subgraph Total Dictionary does not exists. Proceeding with the next steps!!")

    logger.info("Data Loading Done!!")
    subgraph_out = os.path.join(base_dir, dataset, f"{mode}_{distribution}_{k_steps}_{num_iter}.pkl")
    appear_out = os.path.join(base_dir, dataset, f"appearance_{mode}_{distribution}_{k_steps}_{num_iter}.pkl")
    degree_out = os.path.join(base_dir, dataset, f"Degree_{mode}_{distribution}_{k_steps}_{num_iter}.pkl")
    shortest_out = os.path.join(base_dir, dataset, f"ShortestPath_{distribution}_{k_steps}_{num_iter}.pkl")

    ## Step 2. BRWR for extracting subgraphs !!
    appearance = build_appearance(data)
    nxGraph, nxDiGraph, entities = build_nxGraph(data_path)
    obj = Biased_RandomWalk(data_path, distribution, nxGraph, nxDiGraph, entities)
    if not LKG:
        total_appearance, result, subgraph_total_dict = Path_Dictionary_for_LKG(data, appearance, obj, num_k_steps, 
                                                                                num_iteration, distribution, subgraph_size, 
                                                                                phase, subgraph_total_dict, LKG)
    
    if LKG:
        total_appearance, result, subgraph_total_dict, total_sKG = Path_Dictionary_for_LKG(data, appearance, obj, num_k_steps, 
                                                                                num_iteration, distribution, subgraph_size, 
                                                                                phase, subgraph_total_dict, LKG)

    """
    subgraph_dict
    {'center1': [[subgraph1], [subgraph2], [subgraph3], ...]
     'center2': [[subgraph1], [subgraph2]]
     'center3': [[subgraph1]]}
     
     subgraph = [(h1,r1,t1), (h2,r2,t2), ...]
    
    """
    
    with open(subgraph_out, 'wb') as f:
        pickle.dump(result, f)

    with open(appear_out, 'wb') as f:
        pickle.dump(total_appearance, f)

    if not LKG:
        with open(subgraph_total_dict_path, 'wb') as f:
            pickle.dump(subgraph_total_dict, f)

    subgraph_dict = result[0]
            
    del obj
    del data
    del result
    del appearance
    del nxDiGraph
    del entities
    del total_appearance
    del subgraph_total_dict
    gc.collect()


    ## Building Degree_weights_dictionary
    degree_dict = get_degree_dict(subgraph_dict, nxGraph, subgraph_size)
    with open(degree_out, 'wb') as f:
        pickle.dump(degree_dict, f)
    del degree_dict

    if mode == 'train':
        if not LKG:
            spw_dict = get_spw_dict(subgraph_dict, nxGraph, subgraph_size)
        elif LKG:
            del nxGraph
            spw_dict = get_spw_dict_LKG(subgraph_dict, total_sKG, subgraph_size)

        with open(shortest_out, 'wb') as f:
            pickle.dump(spw_dict, f)
    logger.info("Done building SPW Dictionary: {}".format(shortest_out))
    logger.info("Done building BRWR Subgraph Dictionary: {}".format(subgraph_out))
    logger.info("Done building DW Dictioanry: {}".format(degree_out))
    logger.info("Done building Subgraph Total Dictionary: {}".format(subgraph_total_dict_path))
    e = time.time()
    logger.info(f"Time: {datetime.timedelta(seconds = e - s)}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save path dictionary.")
    parser.add_argument("--base-dir", type=str, required=True, help="Your base directory")    
    parser.add_argument("--k-steps", type=int, default=50, required=True, help="Maximum Path Length")
    parser.add_argument("--n-iter", type=int, default=20, required=True, help="Number of Iteration")
    parser.add_argument("--dataset", type=str, choices=['WN18RR', 'FB15k237', 'wiki5m_ind', 'wiki5m_trans', 'YAGO3-10', 'NELL-995'], required=True, help="Dataset name")
    parser.add_argument("--distribution", type=str, choices=['uniform', 'proportional', 'antithetical'], required=True, help="Distribution type")
    parser.add_argument("--phase", type=int, required=True, help="Training Phase")
    parser.add_argument("--subgraph-size", type=int, default=512, required=True, help="Subgraph Size")
    parser.add_argument("--mode", type=str, choices=['train', 'valid'], required=True, help="Mode")
    args = parser.parse_args()


    if args.dataset == 'wiki5m_ind':
        LKG = True
    elif args.dataset == 'wiki5m_trans':
        LKG = True
    else:
        LKG = False
    main(args.base_dir, args.dataset, args.k_steps, args.n_iter, args.distribution, args.phase, args.subgraph_size, args.mode, LKG)

