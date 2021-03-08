#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *


def get_cos_sin(x, y):
    cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val


def get_output_TDE_list(outputs, outputs_bias_list, alpha1, alpha2):
    return outputs - alpha2.item() * ((1-alpha1.item()) * outputs_bias_list[0] + alpha1.item()*outputs_bias_list[1])


def finetune_TDE_alpha_list(nb_inc, nb_cl_fg, nb_cl, epochs, bias_model, tg_model, bias_optimizer, bias_lr_scheduler, \
            trainloader, testloader, causal_embed_list,\
            device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tg_model.eval()
    bias_model.train()

    for epoch in range(epochs):
        total = 0
        correct = 0
        train_loss = 0
        #train
        bias_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(bias_lr_scheduler.get_lr())
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_bias_list = []
            bias_optimizer.zero_grad()
            with torch.no_grad():               
                outputs_feature, outputs = tg_model(inputs, return_feat=True)
                tg_model_norm_weight = torch.cat([F.normalize(tg_model.fc.fc1.weight, p=2, dim=1), 
                        F.normalize(tg_model.fc.fc2.weight, p=2, dim=1)]).permute(1,0)
                if len(causal_embed_list) == 1:
                    outputs_bias_list.append(torch.zeros(outputs.size(1)).to(device))
                    cos_val, sin_val = get_cos_sin(outputs_feature, causal_embed_list[0])
                    outputs_bias_list.append(cos_val*torch.mm(causal_embed_list[0], tg_model_norm_weight)*tg_model.fc.sigma)
                else:
                    assert len(causal_embed_list) == 2
                    cos_val, sin_val = get_cos_sin(outputs_feature, causal_embed_list[0])
                    outputs_bias_list.append(cos_val*torch.mm(causal_embed_list[0], tg_model_norm_weight)*tg_model.fc.sigma)                    
                    cos_val, sin_val = get_cos_sin(outputs_feature, causal_embed_list[1])
                    outputs_bias_list.append(cos_val*torch.mm(causal_embed_list[1], tg_model_norm_weight)*tg_model.fc.sigma)                    
            biased_outputs = bias_model(outputs, outputs_bias_list)
            biased_outputs = F.softmax(biased_outputs, dim=1)
            biased_outputs = torch.log(biased_outputs)
            loss = F.nll_loss(biased_outputs, targets) * 5
            loss.backward()
            bias_optimizer.step()

            _, predicted = biased_outputs.max(1)
            total += targets.size(0)
            train_loss += loss.item()
            correct += predicted.eq(targets).sum().item()

        print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        
        # eval
        test_loss = 0
        correct = 0
        total = 0
        print([bias_model.alpha1.item(), bias_model.alpha2.item()])
        # print(bias_model.alpha1.item())
        # if isinstance(bias_model.alpha, nn.ParameterList):
        #     print(bias_model.alpha._parameters.values())
        # else:
        #     print(bias_model.alpha.detach())
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs_bias_list = []
                outputs_feature, outputs = tg_model(inputs, return_feat=True)
                tg_model_norm_weight = torch.cat([F.normalize(tg_model.fc.fc1.weight, p=2, dim=1), 
                        F.normalize(tg_model.fc.fc2.weight, p=2, dim=1)]).permute(1,0)
                if len(causal_embed_list) == 1:
                    outputs_bias_list.append(torch.zeros(outputs.size(1)).to(device))
                    cos_val, sin_val = get_cos_sin(outputs_feature, causal_embed_list[0])
                    outputs_bias_list.append(cos_val*torch.mm(causal_embed_list[0], tg_model_norm_weight)*tg_model.fc.sigma)
                else:
                    assert len(causal_embed_list) == 2
                    cos_val, sin_val = get_cos_sin(outputs_feature, causal_embed_list[0])
                    outputs_bias_list.append(cos_val*torch.mm(causal_embed_list[0], tg_model_norm_weight)*tg_model.fc.sigma)                    
                    cos_val, sin_val = get_cos_sin(outputs_feature, causal_embed_list[1])
                    outputs_bias_list.append(cos_val*torch.mm(causal_embed_list[1], tg_model_norm_weight)*tg_model.fc.sigma)  
                # outputs = outputs - bias_model.alpha.detach() * outputs_bias
                # outputs = get_output_TDE_list(outputs, outputs_bias_list, bias_model.alpha1, bias_model.alpha1)
                outputs = get_output_TDE_list(outputs, outputs_bias_list, bias_model.alpha1, bias_model.alpha2)
                loss = nn.CrossEntropyLoss()(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    return bias_model