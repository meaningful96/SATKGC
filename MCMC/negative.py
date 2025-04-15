import torch
import numpy as np 
import time
from scipy.stats import norm
import random
from doc import Example, collate

def node_degree(graph):
    #make uniform dist 
    deg_dict = dict()
    deg_distribution = dict()
       
    for key, value in graph.items():
        deg_dict[key] = len(value)

    deg_distribution = {key: 1/len(graph) for key in deg_dict.keys()}

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
                
    return sampled_triplets



