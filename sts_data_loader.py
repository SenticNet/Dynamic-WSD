import os
import time
import json
import torch
import random
import itertools
import numpy as np
import copy
import csv
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence


import nltk
nltk.download('averaged_perceptron_tagger')
pos_tag2id = {'JJ': 0,'NN': 1,'IN': 2,'NNP': 3,'MD': 4,'VB': 5,'CD': 6,'.': 7,',': 8,
 'VBZ': 9,'VBG': 10,'PRP': 11,'VBP': 12,':': 13,'CC': 14,'NNS': 15,'VBN': 16,'VBD': 17,'POS': 18,
 'RB': 19,'WRB': 20,'RP': 21,'WP': 22,'FW': 23,'$': 24,'#': 25,')': 26,'DT': 27,'NNPS': 28,
 'RBR': 29,"''": 30,'JJR': 31,'JJS': 32,'SYM': 33,'TO': 34,'PRP$': 35,'EX': 36,'PDT': 37,'UH': 38,'(': 39, 'WP$': 40, 'WDT':41, 'LS':42, '``':43}


def batch_convert_ids_to_tensors(batch_token_ids, ignore_index):
    bz = len(batch_token_ids)
    batch_tensors = [batch_token_ids[i].squeeze(0) for i in range(bz)]
    batch_tensors = pad_sequence(batch_tensors, True, padding_value=ignore_index).long()
    return batch_tensors


class My_data_set(data.Dataset):
    def __init__(self, file_path, tokenizer):
        if not os.path.exists(file_path):
            raise Exception("[ERROR] Data file does not exist!")
        else:
            self.raw_data = self.__load_data_from_file__(file_path)
            
        self.tokenizer = tokenizer 


    def __load_data_from_file__(self, file_path):
        raw_data = []
        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    raw_data.append(row)
                line_count += 1
        return raw_data

    
    def __getitem__(self, index):
        data_item = self.raw_data[index]
        sent = self.tokenizer.encode(data_item['tweet'])
        text = [self.tokenizer.decode(i) for i in sent[1:-1]]
        pos = nltk.pos_tag(text)
        pos_tags = [-1]
        for i in pos:
            if i[1] not in pos_tag2id.keys():
                pos_tag2id[i[1]] = len(pos_tag2id.keys())
            pos_tags += [pos_tag2id[i[1]]]
        pos_tags += [-1]

        label = int(data_item['polarity'])
        if label==4:
            label = 1
        
        new_data_item = {}
        new_data_item['input'] = torch.LongTensor(sent)
        new_data_item['labels'] = label
        new_data_item['pos'] = torch.LongTensor(pos_tags)
        
        return new_data_item
    
    def __len__(self):
        return len(self.raw_data)


            
def collate_fn(data):
    batch_data = {'input': [], 'labels':[], 'pos':[] }
    
    for data_item in data:
        for k, v in batch_data.items():
            batch_data[k].append(data_item[k])
            
    batch_data['input'] =  batch_convert_ids_to_tensors(batch_data['input'], ignore_index=1)  # ignore_index=pad index, we use roberta
    batch_data['labels'] =  torch.LongTensor(batch_data['labels'])
    batch_data['pos'] =  batch_convert_ids_to_tensors(batch_data['pos'], ignore_index=1)  # 0=no, 1=yes, 2=pad

    return batch_data


def get_loader(file_path, tokenizer, batch_size, num_workers=0):
    dataset = My_data_set(file_path, tokenizer)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    train_loader = data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,
                                 )
    test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn,
                                 )                             
    return train_loader, test_loader
    

