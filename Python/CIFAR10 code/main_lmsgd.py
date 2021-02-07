import torch
#from mnist import *
#from tiny_imagenet import *
from cifar10 import *
from cifar10_resnet import resnet20
from utils_gpu import *
from parallel_apply import *
import time
from copy import deepcopy
#torch.cuda.manual_seed_all(2020)
""" local momentum """
params = {'method':'lmsgd', 'H':8}
H = params['H']
lr = 0.1
momentum = 0.9
weight_decay = 5e-4
""" Initialize models on the server and workers """

pretrain_net = resnet20()


net_s = pretrain_net.to(device_s)
net_w = []
for device in devices:
    pretrain_net = resnet20()
    net_w.append(pretrain_net.to(device))

grads = [ None for i in range(num_workers)]
mt_s = []
mts_w = [[] for i in range(num_workers)]
broadcast_models(net_s, net_w, device_s, devices)



""" record """
comm_count = [0 for i in range(num_workers)]
comm_iter = 0
comm = []
loss = []
test_acc = []
time_l = []

""" start training """
iter = 0
h = H
loss_temp = 0
running_time = 0
zipdata_list = [[] for i in range(num_workers)]
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for zipdata in zip(*subtrainloader):
        for i, data in enumerate(zipdata):
            zipdata_list[i].append(data)
        h = h - 1
        if h == 0:
            start = time.time()
            parallel_apply_LMSGD(net_w, mts_w, zipdata_list, criterion, weights, lr, momentum, weight_decay, devices)
            comm_flag = gather_models(net_w, device_s)
            average_model(net_w, net_s)
            average_momentum(mts_w, mt_s, device_s)
            broadcast_models(net_s, net_w, device_s, devices)
            broadcast_momentum(mt_s, mts_w, devices)
            end = time.time()
            running_time += end - start
            """ keep record of communication rounds """
            comm_iter += sum(comm_flag)
            for i in range(len(comm_flag)):
                comm_count[i] += comm_flag[i]
            zipdata_list = [[] for i in range(num_workers)]
            h = H

        """ keep record of running loss """
        net_s.to(devices[0])
        with torch.no_grad():
            for data in zipdata:
                inputs, labels = data[0].to(devices[0]), data[1].to(devices[0])
                outputs = net_s(inputs)
                loss_temp += criterion(outputs, labels)
        net_s.to(device_s)
        """ print and test """
        if iter%print_iter == 0:
            loss.append(loss_temp/print_iter/num_workers)
            loss_temp = 0
            comm.append(comm_iter)
            time_l.append(running_time)
            print(comm_count)
            print('Epoch %d: training loss %f, communication %d, time %d' %(epoch, loss[-1], comm[-1], time_l[-1]))
        

            correct = 0
            total = 0
            net_s.to(devices[0])
            with torch.no_grad():
                for data in trainloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    # print(type(test_inputs))
                    outputs = net_s(test_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            # net_s.to(device_s)
            print('Accuracy of the network on the 50000 train images: %f %%' % (100 * correct / total))


            correct = 0
            total = 0
            test_loss = 0
            with torch.no_grad():
                for data in testloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    # print(type(test_inputs))
                    outputs = net_s(test_inputs)
                    test_loss += criterion(outputs, test_labels) * test_labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            test_loss = test_loss / 10000
            print('testing loss %.2f' % test_loss)
            print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
            test_acc.append(100*correct/total)
            net_s.to(device_s)
        iter = iter + 1
	

print('Finished Training')
np.savetxt(path+'loss_lmsgd_H8.txt', np.array(loss))
np.savetxt(path+'comm_lmsgd_H8.txt', np.array(comm))
np.savetxt(path+'testacc_lmsgd_H8.txt', np.array(test_acc))
np.savetxt(path+'time_lmsgd_H8.txt', np.array(time_l))
