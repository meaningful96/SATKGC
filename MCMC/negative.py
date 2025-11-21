import torch
import numpy as np 
import time
import datetime
from collections import defaultdict
from scipy.stats import norm
import random
import networkx as nx
from doc import Example, collate

def node_degree(graph, candidates):
    #make uniform dist 
    deg_dict = dict()
    deg_distribution = dict()
    #candidates_dist = defaultdict(list)
    
    for key, value in graph.items():
        deg_dict[key] = len(value)

    deg_distribution = {key: 1/len(graph) for key in deg_dict.keys()}

        
    '''
    for key in candidates.keys():
        candidates_dist[key] = [deg_dict[node] for node in candidates[key]]
        candidates_dist[key] = [i/np.sum(candidates_dist[key]) for i in candidates_dist[key]]
    '''
    return deg_dict, deg_distribution

def negative_sampling(model, hr_vector, pathes_dict, deg_distribution, walks_num, N_steps, N_negs, graph, directed_graph, head_ids):    
    
    distribution = norm.pdf(np.arange(0,walks_num,1), 50, 10)   
    distribution = [i/np.sum(distribution) for i in distribution ]
    
    cur_state = None
    count = 0 
    negatives = list()
    seen = set(pathes_dict[head_ids[0]])
    sampled_triplets = []
    #for checking overlapping entities
    neighbors = list()
    start = time.time()
    for i in range(len(hr_vector)):
        
        #initialize current negative node at random 
        if cur_state is None:
            cur_state = random.choice(list(deg_distribution.keys()))
        
        while count < N_steps:
            
            count +=1
            sample_num = np.random.random() 
            #Sampling from uniform dist 
            if sample_num < 0.5:
                #sample a node y from q(y|x)
                y = random.choices(list(deg_distribution.keys()), weights= list(deg_distribution.values()), k=1)[0]
                #q(y|x) 
                q_prob = deg_distribution[y] 
                #q(x|y) 
                q_prob_next = deg_distribution[cur_state] 

            #Sampling from the nearest k nodes 
            else:         
                y = random.choices(pathes_dict[cur_state], weights=distribution, k=1)[0]
                index = pathes_dict[cur_state].index(y)
                q_prob = distribution[index]
                    
                node_list_next = pathes_dict[y]
                
                if cur_state in node_list_next:
                    index_next = node_list_next.index(cur_state)
                    q_prob_next = distribution[index_next]
                else:
                    q_prob_next = deg_distribution[cur_state]
                
            #Generate a uniform random number r 
            u = np.random.rand()
                
            y_vector_list = []
            cur_vector_list = []
            
            y_vector = Example(head_id='', relation = '', tail_id = y)
            cur_vector = Example(head_id ='', relation = '', tail_id = cur_state)
                
            y_vector_list.append(y_vector.vectorize())
            cur_vector_list.append(cur_vector.vectorize())
                
            y_vector = collate(y_vector_list)
            cur_vector = collate(cur_vector_list)
                
            y_vector['only_ent_embedding'] = True 
            cur_vector['only_ent_embedding'] = True 
            y_emb = model(**y_vector)['ent_vectors']
            cur_emb = model(**cur_vector)['ent_vectors']
                
            #alpha = 0.25
            p_prob = (torch.unsqueeze(hr_vector[i],0).mm(y_emb.t())) ** 0.25 
            p_prob_next = (torch.unsqueeze(hr_vector[i],0).mm(cur_emb.t())) ** 0.25
                
            #calculate acceptance ratio
            A_a = (p_prob.detach() * q_prob_next) / (p_prob_next.detach() * q_prob)
            alpha = min(1, A_a)
                
            #accept
            if u < alpha:
                next_state = y
            #reject
            else:
                next_state = cur_state
            cur_state = next_state          
        
        sample_num = np.random.random(N_negs) 
        num_dist1 = len([i for i in sample_num if i<0.5])
        num_dist2 = len([i for i in sample_num if i>0.5])
        sampled_list1 = np.random.choice(list(deg_distribution.keys()), size=num_dist1, replace=False, p=list(deg_distribution.values()))
        negatives.extend(sampled_list1)
        num=0
        while num < num_dist2:
            y = random.choices(pathes_dict[cur_state], weights=distribution, k=1)[0]
            negatives.append(y)
            cur_state = y
            num+=1
        '''
        sample_num = np.random.random(N_negs) 
        num_dist1 = len([i for i in sample_num if i<0.5])
        num_dist2 = len([i for i in sample_num if i>0.5])
        #sample a node y from q(y|x) (degree proportional distribution)
        #with replacement
        #sampled_list1 = random.choices(list(deg_distribution.keys()), weights = list(deg_distribution.values()), k=num_dist1)          
        #withoout replacement
        sampled_list1 = np.random.choice(list(deg_distribution.keys()), size=num_dist1, replace=False, p=list(deg_distribution.values()))
        #Sampling from the nearest k nodes  
        #with replacement           
        #sampled_list2= random.choices(pathes_dict[head_ids[i]], weights=path_distribution[head_ids[i]], k=num_dist2)
        sampled_list2 = np.random.choice(pathes_dict[head_ids[i]], size=num_dist2, replace=False, p=distribution)
        
        negatives.extend(sampled_list1)
        negatives.extend(sampled_list2)
        '''
    
    for n in negatives:        
        index = random.choices(range(len(graph[n])),k=1)[0]
        neg_triplet = list(graph[n])[index]
                
        if n in directed_graph.keys() and neg_triplet in directed_graph[n]:
            head_id, relation, tail_id = neg_triplet[0], neg_triplet[1], neg_triplet[2]
            neighbors.append(tail_id)
            #seen.add(tail_id)
        else:
            head_id, relation, tail_id = neg_triplet[2], neg_triplet[1], neg_triplet[0]
            neighbors.append(head_id)
            #seen.add(head_id)
        neg = {'head_id': head_id,'relation':relation,'tail_id':tail_id}
        sampled_triplets.append(neg)
                
    end = time.time()
    #print('time check: ', datetime.timedelta(seconds= (end-start)))
    #print(len(negatives), len(set(negatives)))
    #print(len(neighbors), len(set(neighbors)))
    #print(len(set(negatives).union(set(seen))))
    #print(len(set(neighbors).union(set(negatives))))
    #print(len(set(seen).union(set(neighbors).union(set(negatives)))))
    return sampled_triplets

'''
def negative_sampling(model, hr_vector, candidates, deg_distribution, walks_num, N_steps, N_negs, graph, directed_graph, start_node, G=None, head_ids=None):    
   
    #####################################
    # In MCNS paper, they explained 
    # 𝑞(𝑦|𝑥) is defined by mixing uniform sampling 
    # and sampling from the nearest 𝑘 nodes with 1/2 probability each.
    # candidates : dfs path
    # n_steps = 10.. burn in period
    # output -> sampled_negatives : negatives entity id, negative_vector: negatives embedding 
    #####################################
    
    distribution = norm.pdf(np.arange(0,walks_num,1), 50, 10)   
    distribution = [i/np.sum(distribution) for i in distribution ]
    
    cur_state = None
    count = 0 
    negatives = list()
    seen = set(candidates[start_node])
    sampled_negative_triplets = []
    
    neighbors = list()
    
    start = time.time()
    for i in range(len(hr_vector)):
        #initialize sampled 
        sampled = 0
    
        while sampled < N_negs:
            #initialize current negative node at random 
            if cur_state is None:
                cur_state = random.choice(list(deg_distribution.keys()))
    
            count +=1
            sample_num = np.random.random() 
            #Sampling from uniform dist 
            if sample_num < 0.5:
                #sample a node y from q(y|x)
                y = np.random.choice(list(deg_distribution.keys()), 1, p = list(deg_distribution.values()))[0]
                #q(y|x) 
                q_prob = deg_distribution[y] 
                #q(x|y) 
                q_prob_next = deg_distribution[cur_state] 

            #Sampling from the nearest k nodes 
            else:
                y = np.random.choice(candidates[cur_state], 1, p=distribution)[0]    
                index = candidates[cur_state].index(y)
                q_prob = distribution[index]    
                node_list_next = candidates[y]
                
                if cur_state in node_list_next:
                    index_next = node_list_next.index(cur_state)
                    q_prob_next = distribution[index_next]
                else:
                    q_prob_next = deg_distribution[cur_state]
            
            if count > N_steps:
                
                if cur_state not in seen:
                    negatives.append(cur_state)
                #    seen.add(cur_state)
                    sampled +=1
                    
                    index = np.random.choice(range(len(graph[cur_state])), 1)[0]
                    neg_triplet = graph[cur_state][index]
                    
                    if cur_state in directed_graph.keys() and neg_triplet in directed_graph[cur_state]:
                        head_id, relation, tail_id = neg_triplet[0], neg_triplet[1], neg_triplet[2]
                        neighbors.append(tail_id)
                    #   seen.add(tail_id)
                    else:
                        head_id, relation, tail_id = neg_triplet[2], neg_triplet[1], neg_triplet[0]
                        neighbors.append(head_id)
                    #   seen.add(head_id)
                    neg = {'head_id': head_id,'relation':relation,'tail_id':tail_id}
                    sampled_negative_triplets.append(neg)
                    
                    cur_state = y
                
                else:
                    cur_state = y
                            
            else: 
                #Generate a uniform random number r 
                u = np.random.rand()
                
                y_vector_list = []
                cur_vector_list = []
            
                y_vector = Example(head_id='', relation = '', tail_id = y)
                cur_vector = Example(head_id ='', relation = '', tail_id = cur_state)
                
                y_vector_list.append(y_vector.vectorize())
                cur_vector_list.append(cur_vector.vectorize())
                
                y_vector = collate(y_vector_list)
                cur_vector = collate(cur_vector_list)
                
                y_vector['only_ent_embedding'] = True 
                cur_vector['only_ent_embedding'] = True 
                y_emb = model(**y_vector)['ent_vectors']
                cur_emb = model(**cur_vector)['ent_vectors']
                
                #alpha = 0.25
                p_prob = (torch.unsqueeze(hr_vector[i],0).mm(y_emb.t())) ** 0.25 
                p_prob_next = (torch.unsqueeze(hr_vector[i],0).mm(cur_emb.t())) ** 0.25
                
                #calculate acceptance ratio
                A_a = (p_prob * q_prob_next) / (p_prob_next * q_prob)
                next_state = list()
              
                for i in list(range(len(cur_state))):
                    alpha = min(1, A_a)
                    #accept
                    if u < alpha:
                        next_state = y
                    #reject
                    else:
                        next_state = cur_state
                cur_state = next_state         
    
    end = time.time()
    print('time to sample negative: ', datetime.timedelta(seconds= (end-start)))
    print(len(set(negatives)))
    print(len(set(neighbors)))
    print(len(set(neighbors).union(set(negatives))))
    
    
    return sampled_negative_triplets
'''


