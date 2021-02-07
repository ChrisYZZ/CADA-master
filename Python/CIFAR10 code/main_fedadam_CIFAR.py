import torch
from cifar10_resnet import resnet20
from cifar10 import *
from utils_gpu import *
from parallel_apply import *
import time

#torch.cuda.manual_seed_all(2020)
""" FED ADAM"""
params = {'method':'lsgd', 'H':8}
H = params['H']

""" Initialize models on the server and workers """
pretrain_net = resnet20()


net_s = pretrain_net.to(device_s)
net_w = [pretrain_net.to(device) for device in devices]
grads = [ None for i in range(num_workers)]
broadcast_models(net_s, net_w, device_s, devices)


""" record """
comm_count = [0 for i in range(num_workers)]
comm_iter = 0
comm = []
loss = []
test_acc = []
time_l = []

""" hyper parameter for fedadam"""
local_lr = 0.1  # 0.005 is smooth 
server_lr = 0.1
fedadam_momentum = {}
for total in range(62 + 62): # 62 theta_diff ; 62 server_momentum for server
    fedadam_momentum[total] = 0  
    
for total in range(62): # Initialize momentum 
    fedadam_momentum[62 + total] = 0
second = False
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
            parallel_apply_fedadam(net_w, zipdata_list, criterion, weights, local_lr, devices)
            comm_flag = gather_models(net_w, device_s)    
            fedadam_momentum = average_fedadam(net_w, net_s, server_lr, fedadam_momentum, second)
            second = True
            broadcast_models(net_s, net_w, device_s, devices)
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
                for data in testloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    outputs = net_s(test_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            net_s.to(device_s)
            print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
            test_acc.append(100*correct/total)
        iter = iter + 1


print('Finished Training')
np.savetxt(path+'loss_fed_adam_H8.txt', np.array(loss))
np.savetxt(path+'comm_fed_adam_H8.txt', np.array(comm))
np.savetxt(path+'testacc_fed_adam_H8.txt', np.array(test_acc))
np.savetxt(path+'time_fed_adam_H8.txt', np.array(time_l))
