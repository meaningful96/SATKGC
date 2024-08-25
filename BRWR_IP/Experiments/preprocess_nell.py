import os
import json
import argparse
import multiprocessing as mp

from multiprocessing import Pool
from typing import List
from collections import defaultdict

def preprocessing_nell(task):
    entities_dict_path = './data/NELL-995/entities.dict'
    realtions_dict_path = './data/NELL-995/relations.dict'
    entities_tmp = {}
    relations_tmp = []
    with open(entities_dict_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split('\t')
            entities_tmp[value] = int(key)
    
    with open(entities_dict_path, 'r', encoding='utf-8') as file:
        for line in file:
            key, value = line.strip().split('\t')
            relations_tmp.append(value)
   
    relations_tmp = list(set(relations_tmp))
    entities_final = []
    relations_final = {}
    
    #entitiy.json#
    for key in list(entities_tmp.keys()):
        sub_dict = {}
        key_wo_ = key.replace('_', ' ')
        sub_dict["entity_id"] = key
        sub_dict['entity'] = key_wo_
        sub_dict['entity_desc'] = key_wo_
        entities_final.append(sub_dict)
    entities_out_path = './data/NELL-995/entities.json'
    relations_out_path = './data/NELL-995/relations.json'
    with open(entities_out_path, 'w', encoding='utf-8') as f:
        json.dump(entities_final, f)

    #relations.json#
    for ex in relations_tmp:
        key = ex
        val = key.replace(':', ' ')
        relations_final[key] = val

    with open(relations_out_path, 'w', encoding='utf-8') as f:
        json.dump(relations_final, f)

    print("Done Saving {}".format(entities_out_path))
    print("Done Saving {}".format(relations_out_path))

    # train, valid, test
    train_txt = './data/NELL-995/train.txt'
    valid_txt = './data/NELL-995/valid.txt'
    test_txt = './data/NELL-995/test.txt'
    train_json = './data/NELL-995/train.txt.json'
    valid_json = './data/NELL-995/valid.txt.json'
    test_json = './data/NELL-995/test.txt.json'

    for path in [train_txt, valid_txt, test_txt]:
        json_data = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    sub_dict = {}
                    sub_dict['head_id'] = parts[0]  # A
                    sub_dict['relation'] = parts[1]  # B
                    sub_dict['tail_id'] = parts[2]  # C
                    json_data.append(sub_dict)

        if path == train_txt:
            output_file_path = train_json
        elif path == valid_txt:
            output_file_path = valid_json
        elif path == test_txt:
            output_file_path = test_json
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
            print("Done Saving {}".format(output_file_path))

def main(task):
    if task.lower() == 'nell-995':
        preprocessing_nell(task)
    print('Done')


if __name__ == '__main__':
    main("NELL-995")
