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
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
from scipy.spatial.distance import cdist

import modified_resnet_cifar
import modified_linear
import utils_pytorch
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_confusion_matrix import compute_confusion_matrix
from utils_incremental.compute_accuracy_TDE import get_cos_sin, compute_accuracy_TDE, compute_accuracy_finetune_alpha_TDE
import pdb
import pandas as pd


def cal_forget(accu_matrix):
    forget = torch.zeros([accu_matrix.size(0)])
    for step_i in range(1, accu_matrix.size(0)):
        forget_step_i = 0
        for task_j in range(0, step_i):
            forget_step_i += torch.max(accu_matrix[:step_i+1, task_j]) - accu_matrix[step_i, task_j]
        # step_i pairs 
        forget_step_i = forget_step_i/step_i
        forget[step_i] = forget_step_i
    return forget


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(precision=2)
######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--nb_cl_fg', default=50, type=int, \
    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=10, type=int, \
    help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int, \
    help='Examplars per class')
parser.add_argument('--ckp_prefix', \
    default='seed_1993_rs_ratio_0.0_class_incremental_MR_LFAD_cosine_cifar100', \
    type=str)
parser.add_argument('--run_id', default=0, type=int, \
    help='ID of run')
parser.add_argument('--order', \
    default='./checkpoint/cifar100_order_run_0.pkl', \
    type=str)
parser.add_argument('--DCE', action='store_true', \
    help='train with DCE')
parser.add_argument('--top_k', default=10, type=int, \
    help='top_k used for DCE')
parser.add_argument('--finetune_TDE', action='store_true', \
    help='use finetune stage to get alphas')
args = parser.parse_args()
print(args)

nb_inc = int((100-args.nb_cl_fg)/args.nb_cl +1)
order_list = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, \
49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, \
67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, \
37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75,\
38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43,\
5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, \
99, 7, 34, 55, 54, 26, 35, 39]
order = np.array(order_list)

args.ckp_prefix = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos)
if args.DCE:
    args.ckp_prefix += '_top_' + str(args.top_k)
ckp_path = './checkpoint/{}'.format(args.ckp_prefix)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
])
evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform_test)
input_data = evalset.test_data
input_labels = evalset.test_labels
map_input_labels = np.array([order_list.index(i) for i in input_labels])
#evalset.test_labels = map_input_labels
#evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
#    shuffle=False, num_workers=2)
all_step_num = int((100-args.nb_cl_fg)/args.nb_cl + 1)
cnn_cumul_acc = []
icarl_cumul_acc = []
ncm_cumul_acc = []
num_classes = []
nb_cl = args.nb_cl
start_iter = int(args.nb_cl_fg/nb_cl)-1
all_accu_matrix = torch.zeros([all_step_num, all_step_num+1, 4])
# top 1 accuracy CNN
# top 1 accuracy iCaRL
# top 1 accuracy NCM
# top 1 accuracy CNN-CBF-TDE
step_i = 0
for iteration in range(start_iter, int(100/nb_cl)):
    #print("###########################################################")
    #print("For iteration {}".format(iteration))
    #print("###########################################################")
    if iteration == start_iter:
        ckp_name = ckp_path+'/run_{}_iteration_{}_model.pth'.format(args.run_id, iteration)
        class_means_name = ckp_path+'/run_{}_iteration_{}_class_means.pth'.format(args.run_id, iteration)
    else:
        ckp_name = ckp_path+'/run_{}_iteration_{}_CBF_model.pth'.format(args.run_id, iteration)
        class_means_name = ckp_path+'/run_{}_iteration_{}_CBF_class_means.pth'.format(args.run_id, iteration)
    tg_model = torch.load(ckp_name)
    if iteration==start_iter:
        alpha = 0
    elif args.finetune_TDE:
        bias_ckp_name = ckp_path+'/run_{}_iteration_{}_bias_model.pth'.format(args.run_id, iteration)
        bias_model = torch.load(bias_ckp_name)
        alpha = bias_model.alpha.detach()
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    class_means = torch.load(class_means_name)
    current_means = class_means[:, order[:(iteration+1)*nb_cl]]
    causal_embed = np.load(ckp_path+'/causal_embed_run_{}_iteration_{}.npy'.format(args.run_id, iteration))
    causal_embed = torch.from_numpy(causal_embed).to(device)
    causal_embed = causal_embed/(torch.norm(causal_embed, 2, 1, keepdim=True))

    # each task
    for task_i in range(start_iter, iteration+1):
        if task_i == start_iter:
            indices = np.array([i in range(0, (task_i+1)*nb_cl) for i in map_input_labels])
        else:
            indices = np.array([i in range(task_i*nb_cl, (task_i+1)*nb_cl) for i in map_input_labels])
        evalset.test_data = input_data[indices]
        evalset.test_labels = map_input_labels[indices]
        #print('Max and Min of valid labels: {}, {}'.format(min(evalset.test_labels), max(evalset.test_labels)))
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
            shuffle=False, num_workers=2)
        if args.finetune_TDE:
            acc = compute_accuracy_finetune_alpha_TDE(nb_inc, args.nb_cl_fg, args.nb_cl, tg_model, tg_feature_model, current_means, 
                            evalloader, causal_embed, finetune_alpha=alpha, iteration=iteration-start_iter, print_info=False)
        else:
            acc = compute_accuracy_TDE(tg_model, tg_feature_model, current_means, evalloader, causal_embed, iteration=iteration-start_iter, print_info=False)

        all_accu_matrix[step_i, task_i-start_iter] = acc

    # cumul
    indices = np.array([i in range(0, (iteration+1)*nb_cl) for i in map_input_labels])
    evalset.test_data = input_data[indices]
    evalset.test_labels = map_input_labels[indices]
    #print('Max and Min of valid labels: {}, {}'.format(min(evalset.test_labels), max(evalset.test_labels)))
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
        shuffle=False, num_workers=2)
    if args.finetune_TDE:
        acc = compute_accuracy_finetune_alpha_TDE(nb_inc, args.nb_cl_fg, args.nb_cl, tg_model, tg_feature_model, current_means, 
                         evalloader, causal_embed, finetune_alpha=alpha, iteration=iteration-start_iter, print_info=False)
    else:
        acc = compute_accuracy_TDE(tg_model, tg_feature_model, current_means, evalloader, causal_embed, iteration=iteration-start_iter, print_info=False)

    all_accu_matrix[step_i, all_step_num] = acc
    step_i += 1

# the matrix we want
accu_matrix_CNN = all_accu_matrix[:, :-1, 0]
accu_matrix_iCaRL = all_accu_matrix[:, :-1, 1]
accu_matrix_NCM = all_accu_matrix[:, :-1, 2]
accu_matrix_CNN_CBF_TDE = all_accu_matrix[:, :-1, 3]


# culculate cumul and mean
accu_cumul_CNN = all_accu_matrix[:, -1, 0]
accu_cumul_mean = torch.mean(all_accu_matrix[:, -1, 0])
accu_cumul_iCaRL = all_accu_matrix[:, -1, 1]
accu_cumul_iCaRL_mean = torch.mean(all_accu_matrix[:, -1, 1])
accu_cumul_NCM = all_accu_matrix[:, -1, 2]
accu_cumul_NCM_mean = torch.mean(all_accu_matrix[:, -1, 2])
accu_cumul_CNN_CBF_TDE = all_accu_matrix[:, -1, 3]
accu_cumul_CNN_CBF_TDE_mean = torch.mean(all_accu_matrix[:, -1, 3])

# calculate forget
accu_forget_CNN = cal_forget(accu_matrix_CNN)
accu_forget_iCaRL = cal_forget(accu_matrix_iCaRL)
accu_forget_NCM = cal_forget(accu_matrix_NCM)
accu_forget_CNN_CBF_TDE = cal_forget(accu_matrix_CNN_CBF_TDE)

matrix_all_CNN = torch.cat((accu_matrix_CNN, accu_cumul_CNN.unsqueeze(1), accu_forget_CNN.unsqueeze(1)), dim=1)
matrix_all_iCaRL = torch.cat((accu_matrix_iCaRL, accu_cumul_iCaRL.unsqueeze(1), accu_forget_iCaRL.unsqueeze(1)), dim=1)
matrix_all_NCM = torch.cat((accu_matrix_NCM, accu_cumul_NCM.unsqueeze(1), accu_forget_NCM.unsqueeze(1)), dim=1)
matrix_all_CNN_CBF_TDE = torch.cat((accu_matrix_CNN_CBF_TDE, accu_cumul_CNN_CBF_TDE.unsqueeze(1), accu_forget_CNN_CBF_TDE.unsqueeze(1)), dim=1)

## convert your array into a dataframe
matrix_all_CNN = pd.DataFrame(matrix_all_CNN.numpy())
matrix_all_iCaRL = pd.DataFrame(matrix_all_iCaRL.numpy())
matrix_all_NCM = pd.DataFrame(matrix_all_NCM.numpy())
matrix_all_CNN_CBF_TDE = pd.DataFrame(matrix_all_CNN_CBF_TDE.numpy())

excel_name = args.ckp_prefix+'_result.xlsx'
writer = pd.ExcelWriter('./results/'+ excel_name) 
matrix_all_CNN.to_excel(writer, 'CNN', float_format='%.2f')        
matrix_all_iCaRL.to_excel(writer, 'iCaRL', float_format='%.2f')        
matrix_all_NCM.to_excel(writer, 'NCM', float_format='%.2f')        
matrix_all_CNN_CBF_TDE.to_excel(writer, 'CNN_CBF_TDE', float_format='%.2f')        
writer.save()
writer.close()
