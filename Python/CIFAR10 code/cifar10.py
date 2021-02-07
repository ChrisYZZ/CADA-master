#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:31:55 2020

@author: chris
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import random
from torch.utils.data.sampler import SubsetRandomSampler
""" set random seed """
seed = 1314
torch.manual_seed(seed)
np.random.seed(seed)
""" configuration """
num_workers = 1
num_sample = 50000
num_classes = 10
batch_size = 10000
devices = [torch.device('cuda', 0) for x in range(num_workers)]
device_s = torch.device('cuda', 0)
lr=0.1 



print_iter = 50 
num_epochs = 200
path = './results_CIFAR10/'
comm_t = 0

""" load train and test datasets """


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                      download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=transform_test)

from utils_gpu import sort_idx
sorted_idx = sort_idx(trainset, num_classes, num_sample)
random.shuffle(sorted_idx)


batch_ratio = 32/50000 
print('Batchsize per worker:%f %%' % (batch_size * batch_ratio))

nsample = [ 50000 for i in range(num_workers)]
pointer = 0
subtrainloader = []
weights = []
for i in range(num_workers):
    sampler = SubsetRandomSampler(sorted_idx[pointer:pointer+nsample[i]])
    subtrainloader.append(torch.utils.data.DataLoader(trainset, batch_size=int(batch_ratio*nsample[i]), sampler=sampler, num_workers=1, pin_memory=True))
    del sampler
    pointer = pointer + nsample[i]
    weights.append(nsample[i]/num_sample)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

""" define model """
import torch.nn as nn
import torch.nn.functional as F

""" define loss """
criterion = nn.CrossEntropyLoss()
