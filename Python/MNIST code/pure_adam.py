import torch
from mnist import * 

from utils_gpu import *
from parallel_apply import *
import time



""" Initialize models on the server and workers """
net_s = Net().to(device_s)
#print(net_s.state_dict().keys())
net_w = [Net().to(device) for device in devices]
grads = [ None for i in range(num_workers)]
broadcast_models(net_s, net_w, device_s, devices)




momentum_dict = {}
momentum_dict['weight1_q'] = 0
momentum_dict['bias1_q'] = 0
momentum_dict['weight1_v'] = 0
momentum_dict['bias1_v'] = 0
momentum_dict['weight_v_hat1'] = 0
momentum_dict['bias_v_hat1'] = 0

momentum_dict['weight2_q'] = 0
momentum_dict['bias2_q'] = 0
momentum_dict['weight2_v'] = 0
momentum_dict['bias2_v'] = 0
momentum_dict['weight_v_hat2'] = 0
momentum_dict['bias_v_hat2'] = 0

momentum_dict['weight3_q'] = 0
momentum_dict['bias3_q'] = 0
momentum_dict['weight3_v'] = 0
momentum_dict['bias3_v'] = 0
momentum_dict['weight_v_hat3'] = 0
momentum_dict['bias_v_hat3'] = 0

momentum_dict['weight4_q'] = 0
momentum_dict['bias4_q'] = 0
momentum_dict['weight4_v'] = 0
momentum_dict['bias4_v'] = 0
momentum_dict['weight_v_hat4'] = 0
momentum_dict['bias_v_hat4'] = 0

""" record """
comm_count = [0 for i in range(num_workers)]
comm_iter = 0
comm = []
loss = []
test_acc = []
time_l = []

""" start training """
iter = 0
loss_temp = 0
running_time = 0
first = True
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for zipdata in zip(*subtrainloader):
        start = time.time()
        broadcast_models(net_s, net_w, device_s, devices)
        grads_w= parallel_apply_SGD(net_w, zipdata, criterion, weights, devices)
        comm_flag = gather_grads(grads_w, grads, device_s, comm_t)
        # update_params_sgd(net_s, grads, lr, num_workers)
        diff, momentum_dict= update_params_adam(net_s, grads, lr, num_workers, momentum_dict, first)
        first = False # Decide the tensor shape
        end = time.time()
        """ keep record of communication rounds """
        running_time += end - start
        comm_iter += sum(comm_flag)
        # comm.append(comm_iter)
        for i in range(len(comm_flag)):
            comm_count[i] += comm_flag[i]
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
            net_s.to(devices[0])
            """ evaluate on training data """

            loss.append(loss_temp/print_iter/num_workers)
            comm.append(comm_iter)
            time_l.append(running_time)
            loss_temp = 0
            print(comm_count)
            print('Epoch %d: training loss %.2f, communication %d, time %.2f' %(epoch, loss[-1], comm[-1], time_l[-1]))
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    outputs = net_s(test_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
            test_acc.append(100*correct/total)
            net_s.to(device_s)
        iter = iter + 1

print('Finished Training')
np.savetxt(path+'loss_adam_lr00005.txt', np.array(loss))
np.savetxt(path+'comm_adam_lr00005.txt', np.array(comm))
np.savetxt(path+'testacc_adam_lr00005.txt', np.array(test_acc))
np.savetxt(path+'time_adam_lr00005.txt', np.array(time_l))
