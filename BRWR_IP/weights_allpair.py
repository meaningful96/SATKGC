import networkx as nx
import json
import datetime
import time
import pickle
from collections import defaultdict
from multiprocessing import Pool
import argparse
import os

from logger_config import logger

def load_pkl(path:str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json(path:str):
    data = json.load(open(path, 'r', encoding='utf-8'))
    return data

def calculate_shortest_paths(source, all_entities2, graph):
    sub_dict = {}
    for target in all_entities2:
        try:
            st = nx.shortest_path_length(graph, source=source, target=target)
            if st == 0:
                st = 1
        except nx.NetworkXNoPath:
            st = len(all_entities2)
        sub_dict[target] = st
    return source, sub_dict

def nxGraph(path:str):
    # We only need a undirected graph (e.g., nx.Graph())
    data = load_json(path)
    nx_G = nx.Graph()
    for ex in data:
        h, r, t = ex['head_id'], ex['relation'], ex['tail_id']
        nx_G.add_node(h)
        nx_G.add_node(t)
        nx_G.add_edge(h, t, relation=r)
    return nx_G

def get_degree_dict(nxGraph):
    logger.info("Stage1: Degree Weight Dictionary")
    s = time.time()
    entities = list(nxGraph.nodes())
    degree_dict = defaultdict(int)
    for entity in entities:
        d = nxGraph.degree(entity)
        degree_dict[entity] = d
    e = time.time()
    logger.info(f"Time for building degree_dict: {datetime.timedelta(seconds = e-s)}")

    return degree_dict


def main(base_dir, dataset, num_cpu):
    inpath_train = os.path.join(base_dir, dataset, 'train.txt.json')
    inpath_valid = os.path.join(base_dir, dataset, 'valid.txt.json')

    outpath_degree_train = os.path.join(base_dir, dataset, "Degree_train.pkl")
    outpath_degree_valid = os.path.join(base_dir, dataset, "Degree_valid.pkl")
    outpath_shortest_train = os.path.join(base_dir, dataset, "ShortestPath_train.pkl")

    logger.info("Build NetworkX Graph!!")
    nx_G_train = nxGraph(inpath_train)
    nx_G_valid = nxGraph(inpath_valid)

    s1 = time.time()
    logger.info("Step 1. Make degree dictionary!!") 
    degree_dict_train = get_degree_dict(nx_G_train)
    degree_dict_valid = get_degree_dict(nx_G_valid)

    with open(outpath_degree_train, 'wb') as f:
        pickle.dump(degree_dict_train, f)
    with open(outpath_degree_valid, 'wb') as f:
        pickle.dump(degree_dict_valid, f)

    del degree_dict_train
    del degree_dict_valid
    del nx_G_valid
    e1 = time.time()
    logger.info(f"Time for degree_dict: {datetime.timedelta(seconds=e1-s1)}")


    s2 = time.time()
    st_dict = defaultdict(dict)
    logger.info("Step 2. Make shortest path dictionary!!")

    all_entities_list = list(nx_G_train.nodes())

    with Pool(processes=args.num_cpu) as pool:  
        results = pool.starmap(calculate_shortest_paths, [(entity, all_entities_list, nx_G_train) for entity in all_entities_list])

    for source, sub_dict in results:
        st_dict[source] = sub_dict
    
    with open(outpath_shortest_train, 'wb') as f:
        pickle.dump(st_dict, f)
    e2 = time.time()
    logger.info(f"Time for spw_dict: {datetime.timedelta(seconds=e2-s2)}")
    logger.info("Done")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate shortest paths in a graph.")
    parser.add_argument("--base-dir", required=True, help="base path")
    parser.add_argument("--dataset", type=str, choices=['WN18RR', 'FB15k237', 'UMLS'], required=True, help="Dataset name")
    parser.add_argument("--num-cpu", type =int, required=True, help="process number") 

    args = parser.parse_args()

    main(args.base_dir, args.dataset, args.num_cpu)
