import os
import sys
import time
import copy
import json
import torch
import shutil
from tqdm import tqdm
import torch

from torchmetrics.classification import F1Score, Precision, Recall, Accuracy


class SA_Tester:

    def __init__(self, model, data_loader, dir_path, device, num_of_class, case_study=False):
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.record_test_path = dir_path
        self.case_study = case_study
        if num_of_class == 2:
            self.pre = Precision(task="binary", average='macro').to(device)
            self.r = Recall(task="binary", average='macro').to(device)
            self.f1 = F1Score(task="binary", average='macro').to(device)
            self.accuracy = Accuracy(task="binary").to(device)
        else:
            self.pre = Precision(task="multiclass", average='macro', num_classes=num_of_class).to(device)
            self.r = Recall(task="multiclass", average='macro', num_classes=num_of_class).to(device)
            self.f1 = F1Score(task="multiclass", average='macro', num_classes=num_of_class).to(device)
            self.accuracy = Accuracy(task="multiclass", num_classes=num_of_class).to(device)

    def recored_eval_fn(self, record_test_path, pre, r, f1, acc):
        with open(record_test_path, "a") as f:
            f.write("EVALUATION pre: {:2.2f}, r: {:2.2f}, f1: {:2.2f}, acc: {:2.2f},\n".format(round(pre,2), round(r,2), round(f1,2), round(acc,2)))

    def test(self):
        self.model.eval()
        # sa_f1 = 0
        tbar = tqdm(self.data_loader, total=len(self.data_loader), disable=False, desc="Eval", ncols=170)
        total_pre = 0
        total_r = 0
        total_f1 = 0
        total_accuracy = 0
        iter = 0
        for it, data_item in enumerate(tbar):
            data_item["input"] = data_item["input"].to(self.device)
            data_item["labels"] = data_item["labels"].to(self.device)
            _, _, _, pred = self.model(data_item)
            
            if self.case_study:
                for idx, p in enumerate(pred):
                    if p != data_item["labels"][idx]:
                        with open("results/output/"+self.subst_mode+self.num_of_class+"_error.txt", "a") as f:
                            f.write(idx + "\n")
                            f.write(data_item["input"][idx] + "\n")
                            f.write("True:" +  data_item["labels"][idx] + "Pred:" + p + "\n\n")

            pre = self.pre(pred, data_item["labels"])
            r = self.r(pred, data_item["labels"])
            f1 = self.f1(pred, data_item["labels"])
            accuracy = self.accuracy(pred, data_item["labels"])
            iter += 1
            total_pre += pre.detach().item()
            total_r += r.detach().item()
            total_f1 += f1.detach().item()
            total_accuracy += accuracy.detach().item()
            
        self.recored_eval_fn(self.record_test_path, total_pre/iter*100, total_r/iter*100, total_f1/iter*100, total_accuracy/iter*100)
        torch.cuda.empty_cache()
        # tbar.set_postfix_str("pre: {:2.2f}, r: {:2.2f}, f1: {:2.2f} ".format( total_pre/iter*100, total_r/iter*100, total_f1/iter*100, 2) )





class Subst_Tester:

    def __init__(self, model, data_loader, dir_path, device):
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.metrics = Subst_Metrics()
        self.record_test_path = dir_path+"/subst_records.txt"


    def recored_eval_fn(self, record_test_path, pre, r, f1):
        with open(record_test_path, "a") as f:
            f.write("EVALUATION pre: {:2.2f}, r: {:2.2f}, f1: {:2.2f}, \n".format(round(pre,2), round(r,2), round(f1,2)))


    def test(self):
        self.model.eval()
        pred_cnt = 1e-9
        label_cnt = 1e-9
        correct_cnt = 0

        tbar = tqdm(self.data_loader, total=len(self.data_loader), disable=False, desc="Testing", ncols=170)
        for it, data_item in enumerate(tbar):
            data_item["sent"] = data_item["sent"].to(self.device)
            data_item["subst_S_index"] = data_item["subst_S_index"].to(self.device)
            data_item["subst_label"] = data_item["subst_label"].to(self.device)

            _, pred = self.model(data_item)

            tmp_pred_cnt, tmp_label_cnt, tmp_correct_cnt = \
                self.metrics.metrics_by_prompt(pred, data_item["subst_label"])
            pred_cnt += tmp_pred_cnt
            label_cnt += tmp_label_cnt
            correct_cnt += tmp_correct_cnt
            
        precision = correct_cnt / pred_cnt *100
        recall = correct_cnt / label_cnt *100
        f1 = 2 * precision * recall / (precision + recall + 1e-9) 
        self.recored_eval_fn(self.record_test_path, precision, recall, f1)


class Subst_Metrics():
    def __init__(self):
        pass

    def metrics_by_prompt(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pred_cnt = torch.sum(pred).item()
        label_cnt = torch.sum(label).item()
        # correct_cnt: number of correctly predicted true prompts (true positive)
        correct_cnt = torch.sum(torch.mul(pred==label, pred==1)).item()
        return int(pred_cnt), int(label_cnt), int(correct_cnt) 