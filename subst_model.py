import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Subst_model(nn.Module):
    def __init__(self, sentence_encoder, tokenizer, device, prompt_tau=0.05, dis_method="euclidean", candidate_max_length=50):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_tau = prompt_tau
        self.dis_method = dis_method
        self.candidate_max_length = candidate_max_length
        self.dropout = nn.Dropout(p=0.5)
        
        # self.num_class = 2
        # self.metaphor_layer = nn.Linear(1024, self.num_class)
        
        # self.metaphor_ignore_index = 2
        # self.metaphor_loss = nn.CrossEntropyLoss(ignore_index=self.metaphor_ignore_index, 
        #                                          reduction="sum", weight=torch.FloatTensor([1,10]).to(self.device))
        self.prompt_loss = nn.CrossEntropyLoss()
        self.prompt_loss_finetune = nn.CrossEntropyLoss(reduction="none")
        
    def __dist__(self, x, y, dim=-1, tau=0.05, method="euclidean"): 
        x = x.unsqueeze(1)
        if method is None:
            method = self.method
            
        if method == 'dot':
            sim = (x * y).sum(dim)/ tau
        elif method == 'euclidean':
            sim = -(torch.pow(x - y, 2)).sum(dim) / tau
        elif method == 'cosine':
            sim = torch.abs(F.cosine_similarity(x, y, dim=dim) / tau)
        elif method == 'KL':
            kl_mean_1 = F.kl_div(F.log_softmax(x, dim=-1), F.softmax(y, dim=-1), reduction='sum')
            kl_mean_2 = F.kl_div(F.log_softmax(y, dim=-1), F.softmax(x, dim=-1), reduction='sum')
            sim = (kl_mean_1 + kl_mean_2)/2
        return sim
    
    def get_prompt_emb(self, last_hidden_state, data_item):  
        batch_metaphor_embed = []
        batch_prompt_embed = []
        for batch_index, each_hidden in enumerate(last_hidden_state): 
            sent = data_item["sent"][batch_index] 
            metaphor_index = data_item["target_position"][batch_index]
            # metaphor_emb = torch.sum(last_hidden_state[batch_index][metaphor_index], dim=0)
            metaphor_emb = last_hidden_state[batch_index][metaphor_index]
            batch_metaphor_embed.append(metaphor_emb)
            
            prompt_word_index = data_item["subst_word_index"][batch_index]
            prompt_word_emb = torch.stack([torch.sum(last_hidden_state[batch_index][i], dim=0) for i in prompt_word_index])
            # if len(prompt_word_emb) < self.candidate_max_length:
            #     prompt_word_emb += [torch.tensor(0.)] * (self.candidate_max_length - len(prompt_word_emb)) 
            # batch_prompt_embed.append(torch.stack(prompt_word_emb))
            batch_prompt_embed.append(prompt_word_emb)

        return torch.stack(batch_metaphor_embed), torch.stack(batch_prompt_embed)
    
    def get_head_mask(self, input_id, data_item):
        """head_mask = [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        head_mask = []
        for batch_index in range(input_id.size(0)):
            tmp = torch.zeros(input_id.size(1), input_id.size(1)).long().to(self.device) 
            sent_len = data_item["subst_S_index"][batch_index]
            zero_matrix = torch.LongTensor([list(range(sent_len, input_id.size(-1))) for i in range(sent_len)])
            one_sent_mask = torch.ones(input_id.size(1), input_id.size(1)).scatter_(1, zero_matrix, 0)
            head_mask.append(one_sent_mask)
        head_mask = torch.stack(head_mask).unsqueeze(0).unsqueeze(2)
        head_mask = head_mask.repeat(self.sentence_encoder.config.num_hidden_layers, 1, self.sentence_encoder.config.num_attention_heads, 1, 1)
        
        return head_mask.to(self.device)
    
    def forward(self, data_item, finetune=False):
        input_id = data_item["sent"]
        attention_mask = (input_id != self.tokenizer.pad_token_id).bool().to(self.device) 
        
        temp_bz, temp_sent_len= input_id.size(0), input_id.size(1)
        position_ids = torch.LongTensor(list(range(temp_sent_len))).repeat(temp_bz, 1).to(self.device) 
        msk = torch.arange(temp_sent_len).unsqueeze(0).expand(temp_bz, temp_sent_len).to(self.device) >= data_item["subst_S_index"].unsqueeze(1).long()
        position_ids[msk] = data_item["subst_S_index"].unsqueeze(1).expand(temp_bz, temp_sent_len)[msk]
        
        # head_mask = self.get_head_mask(input_id, data_item)
        # encoder_output = self.sentence_encoder(input_id, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask)
        encoder_output = self.sentence_encoder(input_id, attention_mask=attention_mask, position_ids=position_ids)
        last_hidden_state = encoder_output["last_hidden_state"]
        last_hidden_state = self.dropout(last_hidden_state)
        
        # Task 1
        # label each word in the sentence as metaphor or non-metaphor word
        # metaphor_logits = self.metaphor_layer(last_hidden_state)
        # metaphor_pred = torch.argmax(metaphor_logits, dim=-1)
        # metaphor_loss = self.metaphor_loss(metaphor_logits.reshape(-1,2), data_item["metaphor_label"].view(-1)) * self.args.metaphor_loss_weight
        
        # metaphor_pred = torch.zeros((input_id.size(0), input_id.size(1))).to(self.device)
        # metaphor_loss = torch.zeros(1).to(self.device)
        
        # Task 2
        # label each prompt word as synonym or non-synonym
        # batch_metaphor_embed: (bsz); batch_prompt_embed: (bsz, num_of_prompt_word)
        batch_metaphor_embed, batch_prompt_embed = self.get_prompt_emb(last_hidden_state, data_item)
        prompt_logits = self.__dist__(batch_metaphor_embed, batch_prompt_embed, tau=self.prompt_tau, method=self.dis_method)
        prompt_pred_idx = torch.argmax(prompt_logits, dim=-1)
        prompt_pred = torch.zeros(prompt_pred_idx.size(0), self.candidate_max_length).to(self.device).scatter_(1, prompt_pred_idx.unsqueeze(1), 1)
        # changed prompt_pred to list of words
        # prompt_pred = []
        # for sent_idx, cand_idx in enumerate(prompt_pred_idx):
        #     if cand_idx >= len(data_item['candidates'][sent_idx]):
        #         prompt_pred.append(self.tokenizer.pad_token_id)
        #     else:
        #         prompt_pred.append(data_item['candidates'][sent_idx][cand_idx])
        if finetune:
            prompt_loss = self.prompt_loss_finetune(prompt_logits, data_item["subst_label"])
        else:
            prompt_loss = self.prompt_loss(prompt_logits, data_item["subst_label"]) #[:,:prompt_logits.size(-1)]) #* self.args.prompt_loss_weight 
        
        # prompt_pred = torch.zeros((metaphor_logits.size(0), self.args.wrong_word_num+1)).to(self.device)
        # prompt_loss = torch.zeros(1).to(self.device)
        
        return prompt_loss.view(-1), prompt_pred


    def predict(self, data_item, num_of_sub=3):
        # generate n substitution candidates
        input_id = data_item["sent"]
        attention_mask = (input_id != self.tokenizer.pad_token_id).bool().to(self.device) 
        
        temp_bz, temp_sent_len= input_id.size(0), input_id.size(1)
        position_ids = torch.LongTensor(list(range(temp_sent_len))).repeat(temp_bz, 1).to(self.device) 
        msk = torch.arange(temp_sent_len).unsqueeze(0).expand(temp_bz, temp_sent_len).to(self.device) >= data_item["subst_S_index"].unsqueeze(1).long()
        position_ids[msk] = data_item["subst_S_index"].unsqueeze(1).expand(temp_bz, temp_sent_len)[msk]
        
        # head_mask = self.get_head_mask(input_id, data_item)
        # encoder_output = self.sentence_encoder(input_id, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask)
        encoder_output = self.sentence_encoder(input_id, attention_mask=attention_mask, position_ids=position_ids)
        last_hidden_state = encoder_output["last_hidden_state"]
        last_hidden_state = self.dropout(last_hidden_state)

        batch_metaphor_embed, batch_prompt_embed = self.get_prompt_emb(last_hidden_state, data_item)
        prompt_logits = self.__dist__(batch_metaphor_embed, batch_prompt_embed, tau=self.prompt_tau, method=self.dis_method)
        prompt_pred_best = torch.argmax(prompt_logits, dim=-1)  # (bsz)
        _, prompt_pred = torch.topk(prompt_logits, num_of_sub, dim=-1)  # (bsz, num_of_sub)
        # prompt_pred_onehot = torch.zeros(prompt_pred.size(0), self.candidate_max_length).to(self.device).scatter_(1, prompt_pred.unsqueeze(1), 1)
        prompt_pred_idx = torch.add(prompt_pred, data_item['subst_S_index'].unsqueeze(1).expand(prompt_pred.size()))
        # changed prompt_pred to list of words
        # prompt_pred_topk = torch.zeros(prompt_pred.size(0), num_of_sub).to(self.device)
        pred_best = []
        pred_topk = []
        for sentence_idx, words in enumerate(data_item['candidates']):
            pred_best.append(torch.tensor(words[prompt_pred_best[sentence_idx]], device=self.device).long())
            # prompt_pred_best[sentence_idx] = sentence[data_item['subst_S_index'][sentence_idx] + prompt_pred_best[sentence_idx]]
            temp_pred_topk = []
            for count, prompt_idx in enumerate(prompt_pred[sentence_idx]):
                # prompt_pred_topk[sentence_idx][count] = sentence[data_item['subst_S_index'][sentence_idx] + prompt_idx]
                temp_pred_topk.append(torch.tensor(words[prompt_idx], device=self.device).long())
            pred_topk.append(temp_pred_topk)

        return pred_topk, pred_best        
        