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
from utils_incremental.compute_features_FeatCluster_imagenet import compute_features_FeatCluster_imagenet
from utils_incremental.compute_features_FeatCluster import compute_features_FeatCluster
from utils_pytorch import *
from tensorboardX import SummaryWriter
import pdb
import scipy

old_scores = []
new_scores = []


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs


def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs


def incremental_train_and_eval_DCE_MR_LF_TDE(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            dist, K, lw_mr, \
            batch_size, num_samples, dataset='cifar', top_k=10, \
            fix_bn=False, weight_per_class=None, device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        causal_embed = torch.FloatTensor(1, ref_model.fc.in_features).zero_().to(device)
    else:
        causal_embed = torch.FloatTensor(1, tg_model.fc.in_features).zero_().to(device)

    mu = 0.9  # causal embed momentum
    mu_1 = mu_2 = 1.0  # weight assignment hyperparam 

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        num_classes = tg_model.fc.fc1.out_features + tg_model.fc.fc2.out_features
        print(num_classes)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    for epoch in range(epochs):
        # train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    #m.weight.requires_grad = False
                    #m.bias.requires_grad = False
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr())

        if iteration == start_iteration:
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                tg_optimizer.zero_grad()
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss.backward()
                tg_optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            # 1. ref feature model is fixed, get old features z0
            ref_feature_model = nn.Sequential(*list(ref_model.children())[:-1])
            num_features = ref_model.fc.in_features
            input_X_train, old_feat_X_train, targets_X_train = [], [], []
            if dataset=='cifar':
                input_X_train, old_feat_X_train, targets_X_train = compute_features_FeatCluster(
                    ref_feature_model, trainloader, num_samples, num_features, return_data=True)
            else:
                input_X_train, old_feat_X_train, targets_X_train = compute_features_FeatCluster_imagenet(
                    ref_feature_model, trainloader, num_samples, num_features, return_data=True)
            old_feat_X_train = torch.tensor(old_feat_X_train).float()
            targets_X_train = torch.tensor(targets_X_train).long()
            old_feat_X_train = F.normalize(old_feat_X_train, p=2, dim=1) 
            # old feature is normalized :  F.normalize(input, p=2,dim=1)  or np.linalg.norm(features）
            
            # 2. compute the neighbor for each sample
            if num_classes > 500:
                # just for saving time
                dist_z = scipy.spatial.distance.cdist(old_feat_X_train, old_feat_X_train[:10000],
                                                      'euclidean')
                dist_z = torch.tensor(dist_z)
                mask_input = torch.clamp(
                    (targets_X_train[:10000] >= num_old_classes).expand(num_samples, -1).float() -
                    torch.eye(num_samples, 10000).float(),
                    min=0).long()
                dist_z[mask_input == 0] = float("inf")
                match_id = torch.flatten(torch.topk(dist_z, top_k, largest=False, dim=1)[1])

            else:
                old_feat_X_train = torch.tensor(old_feat_X_train).float()
                # 2.1. calculate the L2 distance inside z0
                dist_z = pdist(old_feat_X_train, squared=False)

                # 2.2. calculate the label match matrix following next rules
                mask_old = torch.tensor(targets_X_train) < num_old_classes

                # # v1: not using any sample of the same class when chossing nearest z0 sample
                # mask_cls = []
                # for cls_i in range(num_classes):
                #     mask_cls.append(1 - torch.clamp(torch.tensor(targets_X_train)==cls_i, max=1) )
                # mask_cls = torch.stack(mask_cls)
                # mask_input = mask_cls[targets_X_train]

                # v2: only using samples of the same class when choosing nearest z0 sample (exclude itself)
                # mask_input = (1 - torch.eye(num_samples))

                # # v3: not using any sample of the same class AND old classes when chossing nearest z0 sample
                # mask_cls = []
                # for cls_i in range(num_classes):
                #     mask_cls.append(1 - torch.clamp((torch.tensor(targets_X_train)==cls_i) + mask_old, max=1))
                # mask_cls = torch.stack(mask_cls)
                # mask_input = mask_cls[targets_X_train]

                # v4: not using itself AND old classes when chossing nearest z0 sample
                mask_cls = []
                for cls_i in range(num_classes):
                    mask_cls.append(1 - torch.clamp(mask_old, max=1))
                mask_cls = torch.stack(mask_cls)
                mask_input = torch.clamp(mask_cls[targets_X_train].float() - torch.eye(num_samples),  min=0)

                # 2.3 find the image meets label requirements with nearest old feature
                mask_input = mask_input.float() * dist_z
                mask_input[mask_input == 0] = float("inf")
                match_id = torch.flatten(torch.topk(mask_input, top_k, largest=False, dim=1)[1])

                ## ablation
                ## random
                # match_id = torch.randint(0,mask_input.size(0),(mask_input.size(0)*top_k,)).long()
                ## bottom
                # match_id = torch.flatten(torch.topk(mask_input, top_k, largest=True, dim=1)[1])

            # 3. training data 
            start_idx = 0
            batch_idx = 0
            while start_idx < num_samples:
                if start_idx + batch_size < num_samples:
                    inputs = input_X_train[start_idx:start_idx + batch_size, :, :, :]
                    inputs_match = input_X_train[match_id[start_idx * top_k:(start_idx + batch_size) * top_k], :, :, :]
                    targets = targets_X_train[start_idx:start_idx + batch_size]
                    # targets_match = targets_X_train[match_id[start_idx*top_k:(start_idx+batch_size)*top_k]]
                    ref_features = torch.tensor(
                        old_feat_X_train[start_idx:start_idx + batch_size]).float()
                    ref_match_features = old_feat_X_train[match_id[start_idx * top_k:(start_idx + batch_size) * top_k]]
                else:
                    inputs = input_X_train[start_idx:, :, :, :]
                    inputs_match = input_X_train[match_id[start_idx * top_k:], :, :, :]
                    targets = targets_X_train[start_idx:]
                    # targets_match = targets_X_train[match_id[start_idx*top_k:]]
                    ref_features = old_feat_X_train[start_idx:]
                    ref_match_features = old_feat_X_train[match_id[start_idx * top_k:]]

                inputs = inputs.to(device)
                inputs_match = inputs_match.to(device)
                targets = targets.to(device)
                ref_features = ref_features.to(device)
                start_idx += batch_size
                batch_idx += 1
                tg_optimizer.zero_grad()

                with torch.no_grad():
                    match_features, outputs_match = tg_model(inputs_match, return_feat=True)
                cur_features, outputs = tg_model(inputs, return_feat=True)

                # calculate the softmaxed sum
                outputs_soft = F.softmax(outputs, dim=1)

                outputs_match_soft = F.softmax(outputs_match, dim=1)
                outputs_match_soft = torch.mean(outputs_match_soft.reshape(-1, top_k, outputs_match_soft.size(-1)), dim=1)
                outputs_joint = (mu_1 * outputs_soft + mu_2 * outputs_match_soft) / (mu_1 + mu_2)

                log_outputs_joint = torch.log(outputs_joint)
                loss1 = F.nll_loss(log_outputs_joint, targets)

                ref_match_features = torch.mean(ref_match_features.reshape(
                    -1, top_k, ref_match_features.size(-1)),
                                                dim=1).to(device)
                match_features = torch.mean(match_features.reshape(-1, top_k,
                                                                   match_features.size(-1)),
                                            dim=1)

                # feature distilation
                loss2 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                    torch.ones(inputs.shape[0]).to(device)) * lamda

                # update causal_embed
                with torch.no_grad():
                    cur_features_mean = cur_features.detach().mean(0, keepdim=True)
                    causal_embed = mu * causal_embed + cur_features_mean

                #################################################
                #scores before scale, [-1, 1]
                outputs_bs = torch.cat((old_scores, new_scores), dim=1)
                assert (outputs_bs.size() == outputs.size())
                #get groud truth scores
                gt_index = torch.zeros(outputs_bs.size()).to(device)
                gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
                gt_scores = outputs_bs.masked_select(gt_index)
                #get top-K scores on novel classes
                if outputs_bs.size(1) - num_old_classes == 1:
                    max_novel_scores = outputs_bs[:, num_old_classes:].unsqueeze(0).topk(K, dim=1)[0]
                else:
                    max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
                #the index of hard samples, i.e., samples of old classes
                hard_index = targets.lt(num_old_classes)
                hard_num = torch.nonzero(hard_index).size(0)
                #print("hard examples size: ", hard_num)
                if hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                    max_novel_scores = max_novel_scores[hard_index]
                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    #print("hard example gt scores: ", gt_scores.size(), gt_scores)
                    #print("hard example max novel scores: ", max_novel_scores.size(), max_novel_scores)
                    loss3 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), \
                        max_novel_scores.view(-1, 1), torch.ones(hard_num*K).to(device)) * lw_mr
                else:
                    loss3 = torch.zeros(1).to(device)
                #################################################

                loss = loss1 + loss2 + loss3
                loss.backward()
                tg_optimizer.step()

                train_loss += loss.item()
                if iteration > start_iteration:
                    train_loss1 += loss1.item()
                    train_loss2 += loss2.item()
                    train_loss3 += loss3.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f}, Train Loss3: {:.4f}, \
                Train Loss: {:.4f} Acc: {:.4f}'                                               .format(len(trainloader), \
                train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1),
                train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    if iteration > start_iteration:
        print("Removing register_forward_hook")
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    return tg_model, causal_embed
