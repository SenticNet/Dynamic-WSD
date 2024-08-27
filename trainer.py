import os
import sys
import time
import copy
import json
import torch
import shutil
from tqdm import tqdm
from tester import SA_Tester, Subst_Metrics, Subst_Tester
from torchmetrics.classification import F1Score, Precision, Recall, Accuracy



class SA_Trainer:

    def __init__(self, model, data_loader, tokenizer, optimizer, test_loader, device, num_of_epo, dir_path, num_of_class=3, update_every=1, print_every=50, case_study=False):
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.num_of_epo = num_of_epo
        self.update_every = update_every
        self.print_every = print_every
        if num_of_class == 2:
            # self.pre = Precision(task="binary", average='macro')
            # self.r = Recall(task="binary", average='macro')
            self.f1 = F1Score(task="binary", average='macro').to(device)
            self.accuracy = Accuracy(task="binary").to(device)
        else:
            # self.pre = Precision(task="multiclass", average='macro', num_classes=num_of_class)
            # self.r = Recall(task="multiclass", average='macro', num_classes=num_of_class)
            self.f1 = F1Score(task="multiclass", average='macro', num_classes=num_of_class).to(device)
            self.accuracy = Accuracy(task="multiclass", num_classes=num_of_class).to(device)
        self.tester = SA_Tester(self.model, test_loader, dir_path, device, num_of_class, case_study)

        self.record_train_path = dir_path


    def recored_loss_fn(self, record_train_path, epoch_iter, it, total_loss, total_sa_loss, total_subst_loss, f1, acc):
        with open(record_train_path, "a") as f:
            f.write("epoch_iter: {:4d}, it: {:7d}, total_loss: {:5.6f}, total_sa_loss: {:5.6f}, total_subst_loss: {:5.6f}, f1: {:2.2f}, acc: {:2.2f},\n".format(\
                epoch_iter, it, round(total_loss,6), round(total_sa_loss,6), round(total_subst_loss,6), round(f1,2), round(acc,2)))


    def train(self):
        # iter_sample = 0

        for epoch_iter in range(self.num_of_epo):
            total_loss = 0
            total_sa_loss = 0
            total_subst_loss = 0
            total_f1 = 0
            total_accuracy = 0
            print("epoch_iter", epoch_iter)

            tbar = tqdm(self.data_loader, total=len(self.data_loader), disable=False, desc="Training", ncols=170)
            for it, data_item in enumerate(tbar):
                self.model.train()
                data_item["input"] = data_item["input"].to(self.device)
                data_item["labels"] = data_item["labels"].to(self.device)
                loss, sa_loss, subst_loss,pred = self.model(data_item)

                loss = loss.mean()
                sa_loss = sa_loss.mean()
                subst_loss = subst_loss.mean()

                loss.backward()

                if (it % self.update_every == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # self.scheduler.step()

                total_loss += loss.detach().item()
                total_sa_loss += sa_loss.detach().item()
                total_subst_loss += subst_loss.detach().item()
                # iter_sample += 1

                # pre = self.pre(pred, data_item["labels"])
                # r = self.r(pred, data_item["labels"])
                f1 = self.f1(pred, data_item["labels"])
                accuracy = self.accuracy(pred, data_item["labels"])
                total_f1 += f1.detach().item()
                total_accuracy += accuracy.detach().item()

                if it%self.print_every==0 and it >1:
                    total_loss = float(total_loss) / self.print_every
                    total_sa_loss = float(total_sa_loss) / self.print_every
                    total_subst_loss = float(total_subst_loss) / self.print_every
                    total_f1 = float(total_f1) / self.print_every
                    total_accuracy = float(total_accuracy) / self.print_every
                    self.recored_loss_fn(self.record_train_path, epoch_iter, it, total_loss, total_sa_loss, total_subst_loss, total_f1*100, total_accuracy*100)
                    tbar.set_postfix_str("loss: {:2.6f}, f1: {:2.2f},  acc: {:2.2f},".format(total_loss, total_f1*100, total_accuracy*100, 2) )
                    torch.cuda.empty_cache()
                    total_loss = 0
                    total_sa_loss = 0
                    total_subst_loss = 0
                    total_f1 = 0
                    total_accuracy = 0
                    # iter_sample = 0

            # Epoch ends - evaluation
            torch.cuda.empty_cache()
            self.tester.test()

            print("\n\n")




class Subst_Trainer:

    def __init__(self, model, data_loader, tokenizer, optimizer, test_loader, device, num_of_epo, dir_path, update_every=1, print_every=50):
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.num_of_epo = num_of_epo
        self.update_every = update_every
        self.print_every = print_every
        self.metrics = Subst_Metrics()
        self.tester = Subst_Tester(self.model, test_loader, dir_path, device)

        self.record_train_path = dir_path+"/subst_records.txt"


    def recored_train_fn(self, record_train_path, epoch_iter, it, loss, f1):
        with open(record_train_path, "a") as f:
            f.write("epoch_iter: {:4d}, it: {:7d}, loss: {:5.6f}, f1: {:2.2f}, \n".format(\
                epoch_iter, it, round(loss,6), round(f1,2)))


    def train(self):
        total_loss = 0
        total_sa_loss = 0
        total_subst_loss = 0

        # iter_sample = 0

        pred_cnt = 1e-9
        label_cnt = 1e-9
        correct_cnt = 0

        for epoch_iter in range(self.num_of_epo):
            self.model.train()
            
            print("epoch_iter", epoch_iter)
            tbar = tqdm(self.data_loader, total=len(self.data_loader), disable=False, desc="Training", ncols=170)
            
            for it, data_item in enumerate(tbar):
                data_item["sent"] = data_item["sent"].to(self.device)
                data_item["subst_S_index"] = data_item["subst_S_index"].to(self.device)
                data_item["subst_label"] = data_item["subst_label"].to(self.device)
                loss, pred = self.model(data_item)
                loss = loss.mean()

                tmp_pred_cnt, tmp_label_cnt, tmp_correct_cnt = \
                    self.metrics.metrics_by_prompt(pred, data_item["subst_label"])

                loss.backward()

                if (it % self.update_every == 0):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += tmp_correct_cnt
                # iter_sample += 1
                
                
                if it%self.print_every==0 and it >1:
                    total_loss = float(total_loss) / self.print_every
                    precision = correct_cnt / pred_cnt *100
                    recall = correct_cnt / label_cnt *100
                    f1 = 2 * precision * recall / (precision + recall + 1e-9) 
                    tbar.set_postfix_str("loss: {:2.6f}, f1: {:2.2f} ".format(total_loss, f1, 2) )
                    self.recored_train_fn(self.record_train_path, epoch_iter, it, total_loss, f1)
                    torch.cuda.empty_cache()
                    total_loss = 0
                    pred_cnt = 1e-9
                    label_cnt = 1e-9
                    correct_cnt = 0
                    # iter_sample = 0

            # Epoch ends - evaluation
            self.tester.test()

            print("\n\n")

