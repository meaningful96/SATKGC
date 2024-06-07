import pickle
path = './data/FB15k237/train_antithetical_50_400.pkl'
duple = []

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

data = load_pkl(path)
centers = data[1]
subgraphs = data[0]
from collections import defaultdict

cnt_center = defaultdict(int)
for center in list(set(centers)):
    cnt_center[center] = 0


for center in centers:
    index = cnt_center[center]
    subgraph = subgraphs[center][index]
    
    cnt_center[center] += 1

    hr_set = set()
    for triple in subgraph:
        hr_set.add(triple[0])
        hr_set.add(triple[2])

    duple.append(len(hr_set))

print("Duplications!!")
print(3072 - sum(duple)/len(duple))
