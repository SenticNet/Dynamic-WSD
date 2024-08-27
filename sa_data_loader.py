import logging
import torch

from datasets import load_dataset, ClassLabel
import nltk
nltk.download('averaged_perceptron_tagger')


# logger = logging.getLogger(Path(__file__).name)
# logger.setLevel(logging.INFO)

pos_tag2id = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5,
                'JJ':6,  'JJR': 7, 'JJS': 8,
                'LS': 9, 'MD': 10, 
                'NN': 11, 'NNS': 12, 'NNP': 13, 'NNPS': 14,
                'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18,
                'RB': 19, 'RBR': 20, 'RBS': 21,
                'RP': 22, 'TO': 23, 'UH': 24,
                'VB':25, 'VBD': 26, 'VBG': 27, 'VBN': 28, 'VBP': 29, 'VBZ': 30,
                'WDT': 31, 'WP': 32, 'WP$': 33, 'WRB': 34}
pos_id2tag = dict((v, k) for k, v in pos_tag2id.items())



def get_s140_loader(tokenizer, batch_size=10, split='train', num_workers=4):
    datasets = load_dataset('sentiment140', split=split)
    tokenizer = tokenizer
    pos_tag2id = {'JJ': 0,'NN': 1,'IN': 2,'NNP': 3,'MD': 4,'VB': 5,'CD': 6,'.': 7,',': 8,
                    'VBZ': 9,'VBG': 10,'PRP': 11,'VBP': 12,':': 13,'CC': 14,'NNS': 15,'VBN': 16,'VBD': 17,'POS': 18,
                    'RB': 19,'WRB': 20,'RP': 21,'WP': 22,'FW': 23,'$': 24,'#': 25,')': 26,'DT': 27,'NNPS': 28,
                    'RBR': 29,"''": 30,'JJR': 31,'JJS': 32,'SYM': 33,'TO': 34,'PRP$': 35,'EX': 36,'PDT': 37,'UH': 38,'(': 39}

    def encode(examples):
    # encodes bos and eos
        return tokenizer(examples['text'], truncation=True, padding="max_length")

    datasets = datasets.map(encode, batched=True)
    datasets = datasets.map(lambda examples: {"input": examples["input_ids"]}, batched=True, remove_columns=["input_ids"])
    datasets = datasets.map(lambda examples: {"labels": examples["sentiment"]}, batched=True, remove_columns=["sentiment"])
    datasets = datasets.class_encode_column("labels")


    def pos_tag(examples):
        token = tokenizer(examples['text'], truncation=True, padding="do_not_pad")['input_ids']
        text = [tokenizer.decode(i) for i in token[1:-1]]
        pos = nltk.pos_tag(text)
        padding_length = len(examples['input']) - len(pos) -1
        examples['pos'] = [-1]
        for i in pos:
            if i[1] not in pos_tag2id.keys():
                pos_tag2id[i[1]] = len(pos_tag2id.keys())
            examples['pos'] += [pos_tag2id[i[1]]]
        examples['pos'] += [-1] * padding_length
        return examples

    datasets = datasets.map(pos_tag)
    pos_id2tag = dict((v, k) for k, v in pos_tag2id.items())
    
    new_features = datasets.features.copy()
    new_features["labels"] = ClassLabel(names=['0', '1', '2'])
    datasets = datasets.cast(new_features)
    datasets.set_format(type="torch", columns=["input", "attention_mask", "labels", "pos"])


    data_loader = torch.utils.data.DataLoader(dataset=datasets,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=0,
                                 )


    return data_loader, pos_id2tag

