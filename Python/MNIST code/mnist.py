import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

""" set random seed """
seed = 9
torch.manual_seed(seed)
np.random.seed(seed)

""" configuration """
num_workers = 10
num_sample = 60000
num_classes = 10
batch_size = 10000
devices = [torch.device('cuda', x%4+0) for x in range(num_workers)]
device_s = torch.device('cpu')
#devices = [torch.device('cuda', 0) for x in range(num_workers)]
#device_s = torch.device('cuda', 0)
lr = 0.0005 #adam
#lr=0.1 # LAG

print_iter = 10 #Initially 50
num_epochs = 20
path = './results_mnist_seed/'
comm_t = 0

""" load train and test datasets """
trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                      download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                     download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)

from utils_gpu import *
alldata = sort_dataset(trainset, num_classes, num_sample)


batch_ratio = 0.002
nsample = [ 6000 for i in range(num_workers)]
pointer = 0
subtrainloader = []
weights = []
for i in range(num_workers):
    subtrainloader.append(torch.utils.data.DataLoader(alldata[pointer:pointer+nsample[i]], batch_size=int(batch_ratio*nsample[i]),
                                          shuffle=True, num_workers=num_workers))
    pointer = pointer + nsample[i]
    weights.append(nsample[i]/num_sample)
alltrainloader = torch.utils.data.DataLoader(alldata, batch_size=batch_size, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

""" define model """
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        #self.fc = nn.Linear(28*28*1, 10)
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x
    
""" define loss """
criterion = nn.CrossEntropyLoss()
