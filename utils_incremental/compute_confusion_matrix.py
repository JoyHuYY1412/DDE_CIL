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

def compute_confusion_matrix(tg_model, tg_feature_model, class_means, evalloader, print_info=False, device=None, num_crnt_class=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0
    num_classes = tg_model.fc.out_features
    if num_crnt_class is not None:
        num_classes = num_crnt_class
    cm = np.zeros((3, num_classes, num_classes))
    all_targets = []
    all_predicted = []
    all_predicted_icarl = []
    all_predicted_ncm = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets)

            outputs = tg_model(inputs)
            if num_crnt_class is not None:
                outputs = outputs[:, :num_crnt_class]
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted)

            outputs_feature = np.squeeze(tg_feature_model(inputs))
            # Compute score for iCaRL
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            all_predicted_icarl.append(predicted_icarl)
            # Compute score for NCM
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()
            all_predicted_ncm.append(predicted_ncm)
            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
                  # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)

    cm[0, :, :] = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted))
    cm[1, :, :] = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted_icarl))
    cm[2, :, :] = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted_ncm))

    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format( 100.*correct/total ))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format( 100.*correct_icarl/total ))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format( 100.*correct_ncm/total ))
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format( 100.*np.mean(np.diag(cm[0])/np.sum(cm[0],axis=1)) ))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format( 100.*np.mean(np.diag(cm[1])/np.sum(cm[1],axis=1)) ))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format( 100.*np.mean(np.diag(cm[2])/np.sum(cm[2],axis=1)) ))
    
    return cm
