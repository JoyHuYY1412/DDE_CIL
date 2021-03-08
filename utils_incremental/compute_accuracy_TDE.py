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
import math


def get_output_TDE_list(outputs, outputs_bias_list, alpha1, alpha2):
    return outputs - alpha2.item() * ((1-alpha1.item()) * outputs_bias_list[0] + alpha1.item()*outputs_bias_list[1])


def get_cos_sin(x, y):
    cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val


def compute_accuracy_TDE(tg_model, tg_feature_model, class_means, evalloader, causal_embed, iteration=0, scale=None, print_info=True, device=None, num_crnt_class=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()
    causal_embed.to(device)

    total = 0
    alpha_TDE = 0.5
    correct_cnn = 0
    correct_icarl = 0
    correct_ncm = 0
    correct_TDE = 0
    all_accu = [0]*4
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            if iteration==0:
                outputs = tg_model(inputs)
                if num_crnt_class is not None:
                    outputs = outputs[:, :num_crnt_class]
                outputs = F.softmax(outputs, dim=1)
                if scale is not None:
                    assert(scale.shape[0] == 1)
                    assert(outputs.shape[1] == scale.shape[1])
                    outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
                _, predicted = outputs.max(1)
                correct_cnn += predicted.eq(targets).sum().item()
                outputs_feature = np.squeeze(tg_feature_model(inputs))
                # Compute score for iCaRL
                sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += predicted_icarl.eq(targets).sum().item()
                # Compute score for NCM
                sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += predicted_ncm.eq(targets).sum().item()
            else:
                outputs_feature = np.squeeze(tg_feature_model(inputs))
                # TDE_outputs_feature = outputs_feature - alpha*cos_val*causal_embed*torch.norm(outputs_feature, 2, 1, keepdim=True)

                tg_model_norm_weight = torch.cat([F.normalize(tg_model.fc.fc1.weight, p=2, dim=1), 
                    F.normalize(tg_model.fc.fc2.weight, p=2, dim=1)]).permute(1,0)

                # outputs = torch.mm(TDE_outputs_feature/(torch.norm(outputs_feature, 2, 1, keepdim=True)), tg_model_norm_weight) *tg_model.fc.sigma
                outputs = torch.mm(outputs_feature/(torch.norm(outputs_feature, 2, 1, keepdim=True)), tg_model_norm_weight) *tg_model.fc.sigma

                _, predicted = outputs.max(1)
                correct_cnn += predicted.eq(targets).sum().item()

                # Compute score for iCaRL
                sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += predicted_icarl.eq(targets).sum().item()
                # Compute score for NCM
                sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += predicted_ncm.eq(targets).sum().item()

                if iteration > 0:
                    cos_val, sin_val = get_cos_sin(outputs_feature, causal_embed)
                    outputs_bias = cos_val*torch.mm(causal_embed, tg_model_norm_weight)*tg_model.fc.sigma
                    outputs_TDE = outputs - alpha_TDE*outputs_bias
                    outputs_TDE = F.softmax(outputs_TDE, dim=1)
                    if scale is not None:
                        assert(scale.shape[0] == 1)
                        assert(outputs_TDE.shape[1] == scale.shape[1])
                        outputs_TDE = outputs_TDE / scale.repeat(outputs_TDE.shape[0], 1).type(torch.FloatTensor).to(device)
                    _, predicted_TDE = outputs_TDE.max(1)
                    correct_TDE += predicted_TDE.eq(targets).sum().item()

    all_accu[0] = round(correct_cnn/total, 4)
    all_accu[1] = round(correct_icarl/total, 4)
    all_accu[2] = round(correct_ncm/total, 4)
    if iteration > 0:
        all_accu[3] = round(correct_TDE/total, 4)
    else:
        all_accu[3] = all_accu[0]
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct_cnn/total))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))
        if iteration > 0:
            print("  top 1 accuracy CNN-TDE          :\t\t{:.2f} %".format(100.*correct_TDE/total))
    return torch.tensor(100*np.array(all_accu))


def compute_accuracy_finetune_alpha_TDE(nb_inc, nb_cl_fg, nb_cl, tg_model, tg_feature_model, class_means, 
            evalloader, causal_embed_list, finetune_alpha_list, iteration=0, scale=None, print_info=True, device=None, num_crnt_class=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()
    causal_embed_list = [causal_embed_i.to(device) for causal_embed_i in causal_embed_list]

    correct_cnn = 0
    correct_icarl = 0
    correct_ncm = 0
    correct_cbf_TDE = 0
    total = 0
    all_accu = [0]*4
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            if iteration==0:
                outputs = tg_model(inputs)
                if num_crnt_class is not None:
                    outputs = outputs[:, :num_crnt_class]
                outputs = F.softmax(outputs, dim=1)
                if scale is not None:
                    assert(scale.shape[0] == 1)
                    assert(outputs.shape[1] == scale.shape[1])
                    outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
                _, predicted = outputs.max(1)
                correct_cnn += predicted.eq(targets).sum().item()

                outputs_feature = np.squeeze(tg_feature_model(inputs))
                # Compute score for iCaRL
                sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += predicted_icarl.eq(targets).sum().item()
                # Compute score for NCM
                sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += predicted_ncm.eq(targets).sum().item()
            else:
                outputs_feature = np.squeeze(tg_feature_model(inputs))
                # TDE_outputs_feature = outputs_feature - alpha*cos_val*causal_embed*torch.norm(outputs_feature, 2, 1, keepdim=True)

                tg_model_norm_weight = torch.cat([F.normalize(tg_model.fc.fc1.weight, p=2, dim=1), 
                    F.normalize(tg_model.fc.fc2.weight, p=2, dim=1)]).permute(1,0)

                # outputs = torch.mm(TDE_outputs_feature/(torch.norm(outputs_feature, 2, 1, keepdim=True)), tg_model_norm_weight) *tg_model.fc.sigma
                outputs = torch.mm(outputs_feature/(torch.norm(outputs_feature, 2, 1, keepdim=True)), tg_model_norm_weight) *tg_model.fc.sigma

                _, predicted = outputs.max(1)
                correct_cnn += predicted.eq(targets).sum().item()

                # Compute score for iCaRL
                sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += predicted_icarl.eq(targets).sum().item()
                # Compute score for NCM
                sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += predicted_ncm.eq(targets).sum().item()

                if iteration > 0:
                    outputs_bias_list = []
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
                    outputs_TDE = get_output_TDE_list(outputs, outputs_bias_list, finetune_alpha_list[0], finetune_alpha_list[1])
                    _, predicted_TDE = outputs_TDE.max(1)
                    correct_cbf_TDE += predicted_TDE.eq(targets).sum().item()

    all_accu[0] = round(correct_cnn/total, 4)
    all_accu[1] = round(correct_icarl/total, 4)
    all_accu[2] = round(correct_ncm/total, 4)
    if iteration > 0:
        all_accu[3] = round(correct_cbf_TDE/total, 4)
    else:
        all_accu[3] = all_accu[0]
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct_cnn/total))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))
        if iteration > 0:
            print("  top 1 accuracy CNN-CBF-TDE          :\t\t{:.2f} %".format(100.*correct_cbf_TDE/total))
    return torch.tensor(100*np.array(all_accu))
