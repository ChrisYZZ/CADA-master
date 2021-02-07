import torch
from cifar10 import *
from utils_gpu import *
from parallel_apply import *
from cifar10_resnet import *
import time
from datetime import datetime



""" CADA-WK """
params = {}
params['method'] = 'wk1' #'wk2'
params['thrd'] = 0
params['delay_bound'] = 50
params['delay'] = [0 for i in range(num_workers)]

original_lr = lr
lr_decay=1

if params['method'] == 'wk1':
    params['delta'] = [None for i in range(num_workers)] 

triggerslot = 2
triggerlist = [0 for i in range(triggerslot)]
thrd_scale = 0.12*5/triggerslot
#thrd_scale = 1.5
thrd_inc=1.0
original_thrd = thrd_scale

print('Start training_'+'CADA_'+params['method']+'_lr_decay('+str(lr_decay)+')_thrd('+str(thrd_inc)+')')
print('Currnet Lr and thrd_scale', lr, thrd_scale)
""" Initialize models on the server and workers """
pretrain_net = resnet20()


net_s = pretrain_net.to(device_s)
net_w = [resnet20().to(device) for device in devices]
net_w_old = [resnet20().to(device) for device in devices]
grads = [ None for i in range(num_workers)]
broadcast_models(net_s, net_w, device_s, devices)
broadcast_models(net_s, net_w_old, device_s, devices)
#print(device_s, devices)


layers = net_s.parameters()
momentum_dict = {}
for layer, param in enumerate(layers):

    momentum_dict['weight_q_' + str(layer)] = 0
    momentum_dict['weight_v_' + str(layer)] = 0
    momentum_dict['weight_v_hat_' + str(layer)] = 0


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
    if epoch != 0 and epoch % 120 == 0:
        lr=lr * 0.5
        thrd_scale = thrd_scale * thrd_inc
    for zipdata in zip(*subtrainloader):
        lr=lr * lr_decay
        start = time.time()
        grads_w = parallel_apply_CADA(net_w, net_w_old, zipdata, criterion, weights, params, devices)
        comm_flag = gather_grads(grads_w, grads, device_s, comm_t)
        diff, momentum_dict = update_params_adam(net_s, grads, lr, num_workers, momentum_dict, first, beta1=0.9, beta2=0.99)
        first = False # Decide the tensor shape
        broadcast_models(net_s, net_w, device_s, devices)
        triggerlist.append(diff)
        triggerlist.pop(0)
        params['thrd'] = sum(triggerlist)*thrd_scale
        end = time.time()
        """ keep record of communication rounds """
        comm_iter += sum(comm_flag)
        running_time += end - start
        for i in range(len(comm_flag)):
            comm_count[i] += comm_flag[i]
        """ keep record of running_loss """
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
            time_l.append(running_time)
            comm.append(comm_iter)
            print(comm_count)
            print('Epoch %d: Iter %d: training loss %f, communication %d, time %d' %(epoch, iter, loss[-1], comm[-1], time_l[-1]))
            
            correct = 0
            total = 0
            net_s.to(devices[0])
            with torch.no_grad():
                for data in trainloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    outputs = net_s(test_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            net_s.to(device_s)
            print('Accuracy of the network on the 50000 train images: %f %%' % (100 * correct / total))

            '''Training accu should store in another variable'''

        
            correct = 0
            total = 0
            net_s.to(devices[0])
            net_s.eval()
            with torch.no_grad():
                for data in testloader:
                    test_inputs, test_labels = data[0].to(devices[0]), data[1].to(devices[0])
                    outputs = net_s(test_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            net_s.to(device_s)
            print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
            print('Currnet Lr:', lr)
            test_acc.append(100*correct/total)
            net_s.train()
        iter = iter + 1


print('Finished_'+'CADA_'+params['method']+'_lr_decay('+str(lr_decay)+')_thrd('+str(thrd_inc)+')')

time_now = datetime.now().strftime("%d-%H-%M-%S")
np.savetxt(path+'loss_cada1.txt', np.array(loss))
np.savetxt(path+'comm_cada1.txt', np.array(comm))
np.savetxt(path+'testacc_cada1.txt', np.array(test_acc))
np.savetxt(path+'time_cada1.txt', np.array(time_l))


