#!/usr/bin/env python
# coding=utf-8
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


def compute_features_FeatCluster_imagenet(feature_model,
                                          dataloader,
                                          num_samples,
                                          num_features,
                                          return_data=False,
                                          device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_model.eval()

    targets_all = np.zeros([num_samples]).astype(np.int32)
    features = np.zeros([num_samples, num_features])
    inputs_all = torch.zeros([num_samples, 3, 224, 224])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            features[start_idx:start_idx + inputs.shape[0], :] = np.squeeze(feature_model(inputs))
            inputs_all[start_idx:start_idx + inputs.shape[0], :, :, :] = inputs
            targets_all[start_idx:start_idx + inputs.shape[0]] = targets

            start_idx = start_idx + inputs.shape[0]
    assert (start_idx == num_samples)
    if not return_data:
        return features
    else:
        return inputs_all, features, targets_all
    # features: [500, 64] for training (each class)
