import torch
from torch.utils.data import DataLoader
from typing import Iterator, List, Dict, Union, Tuple, Optional
import logging
import datetime
from datetime import datetime
from pprint import pprint as pp
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.optim as optim
import sys
import operator
import os
import copy
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import gc

from subst_model import Subst_model

import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

MAXIMUM_POST_SEQ_SIZE = 100
ATTENTION_OPTION_NONE = 0  # no attention
ATTENTION_OPTION_ATTENTION_WITH_POST = 1
ATTENTION_OPTION_ATTENTION_WITH_METAPHOR = 1
POST_ENCODER_OPTION_LSTM = 1
METAPHOR_ENCODER_OPTION_LSTM = 1

print("CUDA AVAILABILITY: {}".format(torch.cuda.is_available()))



#Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


TAG = {'JJ': wn.ADJ, 'JJR': wn.ADJ, 'JJS': wn.ADJ,
        'NN': wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
        'RB': wn.ADV, 'RBR': wn.ADV, 'RBS': wn.ADV,
        "MD": wn.VERB, 'VB': wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB}
# pos_tag2id = {'CC': 0, 'CD': 1, 'DT': 2, 'EX': 3, 'FW': 4, 'IN': 5,
#                 'JJ':6,  'JJR': 7, 'JJS': 8,
#                 'LS': 9, 'MD': 10, 
#                 'NN': 11, 'NNS': 12, 'NNP': 13, 'NNPS': 14,
#                 'PDT': 15, 'POS': 16, 'PRP': 17, 'PRP$': 18,
#                 'RB': 19, 'RBR': 20, 'RBS': 21,
#                 'RP': 22, 'TO': 23, 'UH': 24,
#                 'VB':25, 'VBD': 26, 'VBG': 27, 'VBN': 28, 'VBP': 29, 'VBZ': 30,
#                 'WDT': 31, 'WP': 32, 'WP$': 33, 'WRB': 34}
# TAG = {6: wn.ADJ, 7: wn.ADJ, 8: wn.ADJ,
#         11: wn.NOUN, 12: wn.NOUN, 13: wn.NOUN, 14: wn.NOUN,
#         19: wn.ADV, 20: wn.ADV, 21: wn.ADV,
#         10: wn.VERB, 25: wn.VERB, 26: wn.VERB, 27: wn.VERB, 28: wn.VERB, 29: wn.VERB, 30: wn.VERB}


class DotProductAttention(nn.Module):
    """
    Compute the dot products of the query with all values and apply a softmax function to obtain the weights on the values
    """
    def __init__(self, hidden_dim: int):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value)

        return context, attn
    
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, hidden_dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(hidden_dim)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, key)
        return context, attn
    
class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
     Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), key)
        return context, attn
    
class HAN_block(nn.Module):
    '''
    Attention mechanism is one of the above attentions.
    Args:
        hidden_dim: dimesion of embedding matrix
    Inputs: query, key
         - **query** (batch_size, 1, hidden_dim): for the first layer, it is a trainable randomly initialized vectors in a batch; for non-first layer it is the context output of previous layer.
         - **key** (batch_size, max_len, hidden_dim): embedding matrix.
    '''
    def __init__(self, hidden_dim):
        super(HAN_block, self).__init__()
        self.att = ScaledDotProductAttention(hidden_dim)
        self.linear_observer = nn.Linear(hidden_dim,hidden_dim)
        self.linear_matrix = nn.Linear(hidden_dim,hidden_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.2) # need to modify the dropout rate accordingly
        
    def forward(self, query: Tensor, key: Tensor):
        query_ = query[:key.size(0),:,:] # make sure that the batch size of query matches the size of key in the last batch of an epoech
        context, att_weight = self.att(query_,key)
        new_query_vec = self.dropout(self.layer_norm(self.activation(self.linear_observer(context))))
        new_key_matrix = self.dropout(self.layer_norm(self.activation(self.linear_matrix(key))))
        
        return new_query_vec, new_key_matrix, att_weight


class SentimentClassifier(nn.Module):
    def __init__(self,  
                 encoder,
                 tokenizer,
                 subst_generator,
                 pos_id2tag,
                 num_class = 3,
                 hidden_dim = 1024,
                 sig_words_num = 2,
                 num_of_sub = 3,
                 max_cand_num = 15,
                 subst_mode = 'top2',
                 alpha = 0.7,
                 beta = 0.005,
                 max_length = 140,
                 cuda_device: int = -1,
                 case_study=False,
                 ) -> None:

        nn.Module.__init__(self)
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.sig_words_num = sig_words_num
        self.num_of_sub = num_of_sub
        self.max_cand_num = max_cand_num
        self.subst_mode = subst_mode
        self.pos_id2tag = pos_id2tag
        self.pos_id2tag[-1] = "pad"
        # self.vocab = vocab
        self.query0 = None

        self.encoder = encoder
        self.dropout = nn.Dropout(p=0.5)

        self.HAN_1 = HAN_block(hidden_dim)
        self.HAN_2 = HAN_block(hidden_dim)
        self.ffn_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.ffn_layer2 = nn.Linear(hidden_dim, self.num_class)

        self.tokenizer = tokenizer
        self.subst_generator = subst_generator
        # self.alpha = alpha
        self.beta = beta
        self.case_study = case_study

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("CURRENT CUDA DEVICE ", self.device)

        self.loss_func = torch.nn.CrossEntropyLoss()



    def forward(self, data_item):
        sentences = data_item["input"]  # (bsz, max_length)
        mask = (sentences != self.tokenizer.pad_token_id).bool().to(self.device) 

        encoder_output = self.encoder(sentences, attention_mask=mask)
        last_hidden_state = encoder_output["last_hidden_state"]
        last_hidden_state = self.dropout(last_hidden_state)

        query0 = torch.rand([last_hidden_state.size(0), 1, last_hidden_state.size(2)], device=self.device)
        query1, key1, _ = self.HAN_1(query0, last_hidden_state)
        _, _, att_weight_2 = self.HAN_2(query1, key1)

        # find the words that have high att_weight_2
        sig_words_idx = []  # (bsz, sig_words_num)
        att_weight_2 = att_weight_2.squeeze(1)
        for idx, sentence_weights in enumerate(att_weight_2):
            temp_idx = self.get_subst_target(sentence_weights[mask[idx]!=False], data_item["input"][idx][mask[idx]!=False], last_hidden_state[idx][mask[idx]!=False], self.subst_mode)
            sig_words_idx.append(temp_idx)
        sig_words_idx = torch.stack(sig_words_idx)
        sig_words = []  # (bsz, sig_words_num)
        for idx, sentence in enumerate(sentences):
            temp_words = []
            for word_idx in sig_words_idx[idx]:
                # if word_idx != -1:
                temp_words.append(self.tokenizer.decode(sentence[word_idx].int()))
            sig_words.append(temp_words)
            
        # free cuda memory
        # del query1
        # del key1
        # del att_weight_2
        # gc.collect()
        # torch.cuda.empty_cache()

        batch_paraphrase_sentences = [] # (bsz, sig_words_num, max_length)
        subst_S_index = [] # (bsz, sig_words_num)
        subst_word_index = [] # (bsz*sig_words_num, length of unpadded num of subst)
        batch_subst = []
        for sentence_idx, sentence_sig_words in enumerate(sig_words):
            # paraphrase_sentences = []
            # temp_subst_S_index = []
            pos_tags = data_item["pos"][sentence_idx]
            # each sentence has sig_words_num of targets
            for count, sig_word in enumerate(sentence_sig_words):
                pos = pos_tags[sig_words_idx[sentence_idx][count]]
                # if sig_word isn't N/V/A/R, don't replace (replace by itself)
                subst = []
                sig_word = sig_word.strip()
                if self.pos_id2tag[pos.item()] in TAG.keys():
                    # get synonym from all synsets associated with the word
                    synsets = wn.synsets(sig_word, TAG[self.pos_id2tag[pos.item()]])
                    subst = self.get_synonym_tokens(synsets)
                # use sig_word itself as candidate padding
                if len(subst) < self.max_cand_num:
                    subst += [[sig_word]] * (self.max_cand_num - len(subst))
                else:
                    subst = random.sample(subst, self.max_cand_num)
                    if [sig_word] not in subst:
                        subst[random.randint(0,len(subst)-1)] = [sig_word]

                # remove padding from sentences
                ori_sentence = sentences[sentence_idx]
                paraphrase_sentence = ori_sentence[mask[sentence_idx]!=False]
                subst_S_index.append(paraphrase_sentence.size(-1))
                
                if self.case_study:
                    with open("results/output/"+self.subst_mode+str(self.num_class)+"_subst.txt", "a") as f:
                        f.write(str(sentence_idx) + " " + sig_word + ": ")
                        for s in subst:
                            f.write(s[0] + " ")
                        f.write("\n")
                
                # subst = [self.tokenizer.convert_tokens_to_ids(s) for s in subst]
                subst_str = ''
                for s in subst:
                    for w in s:
                        subst_str += w
                    subst_str += ','
                comma_id = self.tokenizer.convert_tokens_to_ids(',')
                temp_subst = self.tokenizer.encode(subst_str[:-1])[1:]
                temp_size = len(temp_subst)
                idx_list = [idx + 1 for idx, val in enumerate(temp_subst) if val == comma_id]
                if len(idx_list) == (self.max_cand_num-1):
                    subst = [temp_subst[i: j-1] for i, j in zip([0] + idx_list, idx_list + ([temp_size] if idx_list[-1] != temp_size else []))]
                else:
                    subst = [self.tokenizer.convert_tokens_to_ids(s) for s in subst]
                    
                # add subst to sentence with eos
                prompt = []
                temp_subst_word_index = []
                for s in subst:
                    tmp = range(paraphrase_sentence.size(-1)+len(prompt), paraphrase_sentence.size(-1)+len(prompt)+len(s) )
                    temp_subst_word_index.append(list(tmp))
                    prompt.extend(s)
                    prompt.append(self.tokenizer.eos_token_id)
                # temp_subst_word_index = [paraphrase_sentence.size(-1) + i for i in range(subst.size(-1))]
                subst_word_index.append(temp_subst_word_index)
                paraphrase_sentence = torch.cat((paraphrase_sentence, torch.tensor(prompt, device=self.device).long()),-1)
                batch_paraphrase_sentences.append(paraphrase_sentence) #[:self.tokenizer.model_max_length]
                batch_subst.append(subst)
            # batch_paraphrase_sentences.append(self.batch_convert_ids_to_tensors(paraphrase_sentences))  
            # subst_S_index.append(torch.LongTensor(temp_subst_S_index))


        # reshape input to (bsz*sig_words_num, max_length)
        bsz = sentences.size(0)
        # batch_paraphrase_sentences = torch.reshape(self.batch_convert_ids_to_tensors(batch_paraphrase_sentences), (bsz*self.sig_words_num, -1))
        batch_paraphrase_sentences = self.batch_convert_ids_to_tensors(batch_paraphrase_sentences)
        subst_S_index = torch.tensor(subst_S_index, device=self.device).long()
        # subst_S_index = torch.reshape(torch.stack(subst_S_index), (bsz*self.sig_words_num, -1))
        target_position = torch.reshape(sig_words_idx, (-1,))
        subst_input = {'sent':batch_paraphrase_sentences, 'target_position':target_position, 
                        'subst_S_index':subst_S_index, 'subst_word_index': subst_word_index, 'candidates': batch_subst}
        # subst_pred: (bsz * sig_words_num, num_of_sub)
        # subst_pred_best: (bsz * sig_words_num, 1)
        subst_pred, subst_pred_best = self.subst_generator.predict(subst_input, num_of_sub=self.num_of_sub)
        # subst_pred = torch.reshape(subst_pred, (bsz, self.sig_words_num, -1))  
        # subst_pred_best = torch.reshape(subst_pred, (bsz, self.sig_words_num))
        
        # free cuda memory
        del batch_paraphrase_sentences
        del target_position
        del subst_S_index
        del subst_word_index
        del batch_subst
        gc.collect()
        torch.cuda.empty_cache()

        
        # use subst_pred as golden candidates to finetune subst_generator
        # bsz*sig_words_num*num_of_sub
        finetune_subst_instances = []
        finetune_subst_S_index = []
        finetune_sig_words_idx = []
        finetune_subst_label = []
        finetune_subst_index = []
        finetune_candidates = []
        sa_sentences = [] # for computing subst_loss_weight
        sa_gold_label = []
        for idx, sentence in enumerate(sentences):
            for i in range(self.sig_words_num):
                subst_instance = sentence[mask[idx]!=False]
                prompt = []
                temp_subst_index = []
                for j in range(self.num_of_sub):
                    # make prompts
                    tmp = range(len(subst_instance)+len(prompt), len(subst_instance)+len(prompt)+len(subst_pred[idx*self.sig_words_num+i][j]) )
                    temp_subst_index.append(list(tmp))
                    prompt.extend(subst_pred[idx*self.sig_words_num+i][j])
                    prompt.append(self.tokenizer.eos_token_id)
                    # make sa sentences for computing weights
                    sa_sentence = torch.cat((sentence[:sig_words_idx[idx][i]], subst_pred[idx*self.sig_words_num+i][j], sentence[sig_words_idx[idx][i]+1:]))
                    sa_sentences.append(sa_sentence)
                    sa_gold_label += [data_item["labels"][idx]] * self.num_of_sub
                subst_instance = torch.cat((subst_instance, torch.tensor(prompt, device=self.device).long()),-1)
                finetune_subst_index += [temp_subst_index] * self.num_of_sub
                finetune_subst_instances += [subst_instance] * self.num_of_sub
                finetune_subst_S_index += [sentence.size(-1)] * self.num_of_sub
                finetune_sig_words_idx += [sig_words_idx[idx][i]] * self.num_of_sub
                finetune_subst_label.append(torch.diag(torch.ones(self.num_of_sub)))
                finetune_candidates += [subst_pred[idx*self.sig_words_num+i]] * self.num_of_sub
        finetune_subst_instances = self.batch_convert_ids_to_tensors(finetune_subst_instances)  # (bsz*sig_words_num*num_of_sub, max_length)
        finetune_subst_S_index = torch.tensor(finetune_subst_S_index, device=self.device).long()
        # finetune_sig_words_idx = torch.LongTensor(finetune_sig_words_idx)
        # finetune_subst_label: a (len=bsz*sig_words_num) list of (num_of_sub, num_of_sub)  ==>  (bsz*sig_words_num*num_of_sub, num_of_sub)
        finetune_subst_label = torch.cat(finetune_subst_label, 0).to(self.device)
        subst_input = {'sent':finetune_subst_instances, 'target_position':finetune_sig_words_idx, 
                        'subst_S_index':finetune_subst_S_index, 'subst_label': finetune_subst_label, 
                        'subst_word_index':finetune_subst_index, 'candidates': finetune_candidates}
        subst_loss, _ = self.subst_generator(subst_input, finetune=True)  # (bsz*sig_words_num*num_of_sub)
        
        # free cuda memory
        # del finetune_subst_instances
        # del finetune_subst_label
        del finetune_sig_words_idx
        del finetune_subst_S_index
        del finetune_subst_index
        del finetune_candidates
        gc.collect()
        torch.cuda.empty_cache()
        
        # compute subst_loss_weight
        sa_sentences = self.batch_convert_ids_to_tensors(sa_sentences).detach()
        sa_mask = (sa_sentences != self.tokenizer.pad_token_id).bool().to(self.device).detach()
        sa_prob = F.softmax(self.get_sa_prob(sa_sentences,sa_mask)[0], dim=-1).detach()  # (bsz*sig_words_num*num_of_sub, num_class)
        subst_loss_weight = torch.zeros((bsz*self.sig_words_num*self.num_of_sub), device=self.device)
        for count, prob in enumerate(sa_prob):
            gold_idx = sa_gold_label[count].item()
            subst_loss_weight[count] = self.weight_func(prob[gold_idx])
        weighted_subst_loss = torch.matmul(subst_loss_weight, subst_loss).mean()  # single value
        
        # free cuda memory
        del sa_sentences
        del sa_mask
        del sa_prob
        del subst_loss_weight
        gc.collect()
        # torch.cuda.empty_cache()

        # replace the original words with substitute
        
        # A) replace at token level
        # new_sentences = []
        # for idx, sentence in enumerate(sentences):
        #     new_sentence = sentence
        #     offset = 0
        #     for i in range(self.sig_words_num):
        #         if subst_pred_best[idx*self.sig_words_num+i].tolist() != [3]:
        #             subst_word_len = len(subst_pred_best[idx*self.sig_words_num+i])
        #             new_sentence = torch.cat((new_sentence[:(sig_words_idx[idx][i]+offset)], subst_pred_best[idx*self.sig_words_num+i], new_sentence[(sig_words_idx[idx][i]+1+offset):]))
        #             offset += (len(subst_pred_best[idx*self.sig_words_num+i]) - 1)
        #     new_sentences.append(new_sentence)
            
        # B) replace at sentence level
        new_sentences = []
        for idx, sentence in enumerate(sentences):
            new_sentence = ""
            offset = 0
            for count, w in enumerate(sentence):
                if count in sig_words_idx[idx]:
                    subst_idx = sig_words_idx[idx].tolist().index(count)
                    new_sentence += self.tokenizer.decode(subst_pred_best[idx*self.sig_words_num+subst_idx].int())
                elif w != self.tokenizer.eos_token_id and w != self.tokenizer.pad_token_id:
                    new_sentence += self.tokenizer.decode(w.int())
            new_sentence = new_sentence.replace('<s>','')
            new_sentence = self.tokenizer.encode(new_sentence)
            new_sentence = torch.tensor(new_sentence, device=self.device).long()
            new_sentences.append(new_sentence)
            
            if self.case_study:
                with open("results/output/"+self.subst_mode+str(self.num_class)+"_sentence.txt", "a") as f:
                    f.write(str(idx) + "\n")
                    for count, w in enumerate(sentence):
                        if count in sig_words_idx[idx]:
                            f.write("["+ self.tokenizer.decode(w.int()) + "] ")
                        else:
                            f.write(self.tokenizer.decode(w.int()))
                    f.write("\n")
                    for count, w in enumerate(new_sentence):
                        if count in sig_words_idx[idx]:
                            f.write("["+ self.tokenizer.decode(w.int()) + "] ")
                        else:
                            f.write(self.tokenizer.decode(w.int()))
                    f.write("\n\n")
                    
        del sentences
        del mask
        del sig_words_idx
        gc.collect()
        
        new_sentences = self.batch_convert_ids_to_tensors(new_sentences)
        new_mask = (new_sentences != self.tokenizer.pad_token_id).bool().to(self.device) 
        # compute sentiment analysis loss
        logits, att = self.get_sa_prob(new_sentences, new_mask, True)
        sa_loss = self.loss_func(logits, data_item["labels"])  # single value
        
        if self.case_study:
            att = att.squeeze(1)
            with open("results/output/"+self.subst_mode+str(self.num_class)+"_score.txt", "a") as f:
                f.write("\n")
                for idx, sentence in enumerate(new_sentences):
                    for c, w in enumerate(sentence):
                        f.write(self.tokenizer.decode(w.int()) + " " + str(att[idx][c].item()) + "\t")
                    f.write("\n")

        # loss = self.alpha * sa_loss + (1 - self.alpha) * weighted_subst_loss 
        loss = sa_loss + weighted_subst_loss 

        pred = torch.argmax(logits, dim=-1)

        return loss.view(-1), sa_loss.view(-1), weighted_subst_loss.view(-1), pred



    def get_sa_prob(self, sentences, mask, backwards=False):
        encoder_output = self.encoder(sentences, attention_mask=mask)
        last_hidden_state = encoder_output["last_hidden_state"]
        last_hidden_state = self.dropout(last_hidden_state)
        
        if self.query0 == None:
            self.query0 = torch.rand([last_hidden_state.size(0), 1, last_hidden_state.size(2)], requires_grad = True, device=self.device)
        if backwards:
            query1, key1, att_weight_1 = self.HAN_1(self.query0, last_hidden_state)
        else:
            query1, key1, att_weight_1 = self.HAN_1(self.query0.detach(), last_hidden_state.detach())
        query2, key2, att_weight_2 = self.HAN_2(query1, key1)
        HAN_output = query2.squeeze(1)   # (bsz, hidden_dim)
        logits = F.relu(self.ffn_layer1(HAN_output))
        logits = self.ffn_layer2(logits)  # (bsz, num_class)

        return logits, att_weight_2


    def weight_func(self, prob):
        return self.beta * prob.pow(2)

    def get_synonym_tokens(self, synsets):
        synonym_tokens = []
        for synset in synsets:
            for lemma in synset.lemmas():
                tokens = lemma.name().replace('_', ' ').split()
                if tokens not in synonym_tokens:
                    synonym_tokens += [tokens]
        return synonym_tokens


    def get_subst_target(self, sentence_weights, sentence, hidden, mode):
        if mode == 'top2':
            values, indices = torch.topk(sentence_weights, 2)
            if self.case_study:
                with open("results/output/"+self.subst_mode+str(self.num_class)+"_score.txt", "a") as f:
                    for idx, w in enumerate(sentence):
                        f.write(self.tokenizer.decode(w.int()) + " " + str(sentence_weights[idx].item()) + "\t")
                    f.write("\n")
            return indices

        if mode == 'rand':
            values, indices = torch.topk(sentence_weights, 5)
            return torch.index_select(indices, 0, torch.randint(0, 4, (2,)))

        if mode == 'ling':        
            max_len = 5
            if sentence_weights.size(-1) < 5:
                max_len = sentence_weights.size(-1)
            values, indices = torch.topk(sentence_weights, max_len)
            candidates = [sentence[idx].int() for idx in indices]

            least_sim = torch.zeros((max_len,))
            for idx, candidate in enumerate(candidates):
                candidate = self.tokenizer.decode(candidate)
                synsets = wn.synsets(candidate)
                subst = self.get_synonym_tokens(synsets)
                # compute the minimum similarity between the candidate and subst words
                encoder_input = []
                subst_idx = []
                for s in subst:
                    tmp = range(len(encoder_input), len(encoder_input)+len(s))
                    subst_idx.append(list(tmp))
                    encoder_input.extend(self.tokenizer.convert_tokens_to_ids(s))
                    encoder_input.append(self.tokenizer.eos_token_id)
                if len(encoder_input) > 1:
                    encoder_input = torch.tensor(encoder_input, device=self.device).long()
                    mask = (encoder_input != self.tokenizer.pad_token_id).bool().to(self.device) 
                    encoder_output = self.encoder(encoder_input.unsqueeze(1), attention_mask=mask.unsqueeze(1))
                    hidden_states = encoder_output["last_hidden_state"].detach()  # (subst word count, hidden_size)
                    subst_hidden = torch.stack([torch.sum(hidden_states[i], dim=0) for i in subst_idx])
                    cand_hidden = hidden[indices[idx]].unsqueeze(1)
                    sim = (torch.pow(cand_hidden - subst_hidden, 2)).sum(dim=-1) / 0.05
                    least_sim[idx] = torch.argmax(sim, dim=-1)[0]
            least_sim_idx = torch.argmax(least_sim)
            _, least_sim_idx_2 = torch.kthvalue(least_sim, 2)
            
            if self.case_study:
                with open("results/output/"+self.subst_mode+str(self.num_class)+"_score.txt", "a") as f:
                    count = 0
                    for idx, w in enumerate(sentence):
                        f.write(self.tokenizer.decode(w.int()) + " " + str((sentence_weights[idx].item())))
                        if idx in indices:
                            f.write(" " + str(least_sim[count].item()))
                            count += 1
                        f.write("\t")
                    f.write("\n")
            
            return torch.stack([least_sim_idx, least_sim_idx_2])



    
    def batch_convert_ids_to_tensors(self, batch_token_ids):
        bz = len(batch_token_ids)
        batch_tensors = [batch_token_ids[i].squeeze(0) for i in range(bz)]
        batch_tensors = pad_sequence(batch_tensors, True, self.tokenizer.pad_token_id).long()
        return batch_tensors



#     def predict(self, data_item):
#         sentences = data_item["input"]  # (bsz, max_length)
#         mask = (sentences != self.tokenizer.pad_token_id).bool().to(self.device) 

#         encoder_output = self.encoder(sentences, attention_mask=mask)
#         last_hidden_state = encoder_output["last_hidden_state"]
#         last_hidden_state = self.dropout(last_hidden_state)

#         query0 = torch.rand([last_hidden_state.size(0), 1, last_hidden_state.size(2)], device=self.device)
#         query1, key1, _ = self.HAN_1(query0, last_hidden_state)
#         _, _, att_weight_2 = self.HAN_2(query1, key1)
        
#         with open("results/output/att_weights.txt", "a") as f:
#             for n, sentence in enumerate(sentences):
#                 for idx, w in enumerate(sentence):
#                     f.write(w + " " + str(att_weight_2[n][idx].item()) + " ")
#                 f.write("\n")
#             f.write("\n")

#         # find the words that have high att_weight_2
#         sig_words_idx = []  # (bsz, sig_words_num)
#         att_weight_2 = att_weight_2.squeeze(1)
#         for idx, sentence_weights in enumerate(att_weight_2):
#             temp_idx = self.get_subst_target(sentence_weights[mask[idx]!=False], data_item["input"][idx][mask[idx]!=False], last_hidden_state[idx][mask[idx]!=False], self.subst_mode)
#             sig_words_idx.append(temp_idx)
#         sig_words_idx = torch.stack(sig_words_idx)
#         sig_words = []  # (bsz, sig_words_num)
#         for idx, sentence in enumerate(sentences):
#             temp_words = []
#             for word_idx in sig_words_idx[idx]:
#                 # if word_idx != -1:
#                 temp_words.append(self.tokenizer.decode(sentence[word_idx].int()))
#             sig_words.append(temp_words)

#         batch_paraphrase_sentences = [] # (bsz, sig_words_num, max_length)
#         subst_S_index = [] # (bsz, sig_words_num)
#         subst_word_index = [] # (bsz*sig_words_num, length of unpadded num of subst)
#         batch_subst = []
#         for sentence_idx, sentence_sig_words in enumerate(sig_words):
#             pos_tags = data_item["pos"][sentence_idx]
#             # each sentence has sig_words_num of targets
#             for count, sig_word in enumerate(sentence_sig_words):
#                 pos = pos_tags[sig_words_idx[sentence_idx][count]]
#                 # if sig_word isn't N/V/A/R, don't replace (replace by itself)
#                 subst = []
#                 if self.pos_id2tag[pos.item()] in TAG.keys():
#                     # get synonym from all synsets associated with the word
#                     synsets = wn.synsets(sig_word.strip(), TAG[self.pos_id2tag[pos.item()]])
#                     subst = self.get_synonym_tokens(synsets)
#                 # use sig_word itself as candidate padding
#                 if len(subst) < self.max_cand_num:
#                     subst += [[sig_word]] * (self.max_cand_num - len(subst))
#                 else:
#                     subst = random.sample(subst, self.max_cand_num)
#                     if [sig_word] not in subst:
#                         subst[random.randint(0,len(subst)-1)] = [sig_word]

#                 # remove padding from sentences
#                 ori_sentence = sentences[sentence_idx]
#                 paraphrase_sentence = ori_sentence[mask[sentence_idx]!=False]
#                 subst_S_index.append(paraphrase_sentence.size(-1))
                
#                 subst = [self.tokenizer.convert_tokens_to_ids(s) for s in subst]
#                 # remove unknown tokens
#                 # if len(subst) > 1:
#                 #     subst = list(filter(lambda x: x != self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token), subst))
                    
#                 # add subst to sentence with eos
#                 prompt = []
#                 temp_subst_word_index = []
#                 for s in subst:
#                     tmp = range(paraphrase_sentence.size(-1)+len(prompt), paraphrase_sentence.size(-1)+len(prompt)+len(s) )
#                     temp_subst_word_index.append(list(tmp))
#                     prompt.extend(s)
#                     prompt.append(self.tokenizer.eos_token_id)
#                 # temp_subst_word_index = [paraphrase_sentence.size(-1) + i for i in range(subst.size(-1))]
#                 subst_word_index.append(temp_subst_word_index)
#                 paraphrase_sentence = torch.cat((paraphrase_sentence, torch.tensor(prompt, device=self.device).long()),-1)
#                 batch_paraphrase_sentences.append(paraphrase_sentence) #[:self.tokenizer.model_max_length]
#                 batch_subst.append(subst)
#             # batch_paraphrase_sentences.append(self.batch_convert_ids_to_tensors(paraphrase_sentences))  
#             # subst_S_index.append(torch.LongTensor(temp_subst_S_index))


#         # reshape input to (bsz*sig_words_num, max_length)
#         bsz = sentences.size(0)
#         # batch_paraphrase_sentences = torch.reshape(self.batch_convert_ids_to_tensors(batch_paraphrase_sentences), (bsz*self.sig_words_num, -1))
#         batch_paraphrase_sentences = self.batch_convert_ids_to_tensors(batch_paraphrase_sentences)
#         subst_S_index = torch.tensor(subst_S_index, device=self.device).long()
#         # subst_S_index = torch.reshape(torch.stack(subst_S_index), (bsz*self.sig_words_num, -1))
#         target_position = torch.reshape(sig_words_idx, (-1,))
#         subst_input = {'sent':batch_paraphrase_sentences, 'target_position':target_position, 
#                         'subst_S_index':subst_S_index, 'subst_word_index': subst_word_index, 'candidates': batch_subst}
#         # subst_pred: (bsz * sig_words_num, num_of_sub)
#         # subst_pred_best: (bsz * sig_words_num, 1)
#         subst_pred, subst_pred_best = self.subst_generator.predict(subst_input, num_of_sub=self.num_of_sub)
#         # subst_pred = torch.reshape(subst_pred, (bsz, self.sig_words_num, -1))  
#         # subst_pred_best = torch.reshape(subst_pred, (bsz, self.sig_words_num))
        
#         # free cuda memory
#         del batch_paraphrase_sentences
#         del target_position
#         del subst_S_index
#         del subst_word_index
#         del batch_subst
#         gc.collect()
#         torch.cuda.empty_cache()

#         # replace the original words with substitute
#         new_sentences = []
#         for idx, sentence in enumerate(sentences):
#             new_sentence = sentence
#             offset = 0
#             for i in range(self.sig_words_num):
#                 subst_word_len = len(subst_pred_best[idx*self.sig_words_num+i])
#                 new_sentence = torch.cat((new_sentence[:(sig_words_idx[idx][i]+offset)], subst_pred_best[idx*self.sig_words_num+i], new_sentence[(sig_words_idx[idx][i]+subst_word_len+offset):]))
#                 offset += (len(subst_pred_best[idx*self.sig_words_num+i]) - 1)
#             new_sentences.append(new_sentence)

            
#         new_sentences = self.batch_convert_ids_to_tensors(new_sentences)
#         # compute sentiment analysis loss
#         encoder_output = self.encoder(new_sentences, attention_mask=mask)
#         last_hidden_state = encoder_output["last_hidden_state"]
#         last_hidden_state = self.dropout(last_hidden_state)
        
#         if self.query0 == None:
#             self.query0 = torch.rand([last_hidden_state.size(0), 1, last_hidden_state.size(2)], requires_grad = True, device=self.device)
#         if backwards:
#             query1, key1, att_weight_1 = self.HAN_1(self.query0, last_hidden_state)
#         else:
#             query1, key1, att_weight_1 = self.HAN_1(self.query0.detach(), last_hidden_state.detach())
#         query2, key2, att_weight_2 = self.HAN_2(query1, key1)
#         HAN_output = query2.squeeze(1)   # (bsz, hidden_dim)
#         logits = F.relu(self.ffn_layer1(HAN_output))
#         logits = self.ffn_layer2(logits)  # (bsz, num_class)
        
#         with open("results/output/att_weights.txt", "a") as f:
#             for n, sentence in enumerate(sentences):
#                 for idx, w in enumerate(sentence):
#                     f.write(self.tokenizer.decode(w.int()) + " " + str(att_weight_2[n][idx].item()) + " ")
#                 f.write("\n")
#             f.write("\n")
        
#         pred = torch.argmax(logits, dim=-1)
        
#         with open("results/output/att_weights.txt", "a") as f:
#             for n, sentence in enumerate(sentences):
#                 print(str(pred.item()) + '\n\n')

#         return pred
