import torch
import torch.cuda.comm as comm
import time
#################################################################################################

def average_model(net_list, net_t):
    params_t = net_t.parameters()
    num_net = len(net_list)   
    params_list = []
    for i, net in enumerate(net_list):
        params_list.append(list(net.parameters()))
    for i, param in enumerate(params_t):
        param.data.mul_(0)
        for j in range(num_net):
            param.data.add_(1/num_net, params_list[j][i].data)

    
def copy_params(net1, net2):
    params1 = list(net1.parameters())
    params2 = list(net2.parameters())
    for i in range(len(params1)):
        params2[i].data.copy_(params1[i].data)

def copy_state(net1, net2):
    net2.load_state_dict(net1.state_dict())

def zero_grad(net):
    for param in net.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
            
def get_grad(net, weight):
    grad = []
    params = list(net.parameters())
    for param in params:
        grad.append(param.grad.data*weight)
    return grad

def check_ps(net_new, net_old, L, thrd):
    diff = get_diff(net_new, net_old)*L
    if diff >= thrd:
        return True
    else:
        return False
    
def check_wk1_lag(grad_new, grad_new_t, delta, thrd):
    diff = 0
    delta_new = []
    if delta is None:
        for i in range(len(grad_new)):
            '''LAG'''
            delta_new.append(grad_new[i])
            diff += torch.norm(delta_new[i])**2
    else:
        for i in range(len(grad_new)):
            delta_new.append(grad_new[i]-grad_new_t[i])
            diff += torch.dist(delta_new[i], delta[i])**2
    if diff >= thrd:
        return delta_new
    else:
        return None    

def check_wk1(grad_new, grad_new_t, delta, thrd):
    diff = 0
    delta_new = []
    if delta is None:
        for i in range(len(grad_new)):
            delta_new.append(grad_new[i]-grad_new_t[i])
            diff += torch.norm(delta_new[i])**2
    else:
        for i in range(len(grad_new)):
            delta_new.append(grad_new[i]-grad_new_t[i])
            diff += torch.dist(delta_new[i], delta[i])**2
    if diff >= thrd:
        return delta_new
    else:
        return None
    
def check_wk2(grad_new, grad_old, thrd):
    diff = 0
    for i in range(len(grad_new)):
        diff += torch.norm(grad_new[i]-grad_old[i])**2
    if diff >= thrd:
        return True
    else:
        return False

def update_params_qsgd(net, grads, lr, num_workers, num_bits):
    params = net.parameters()
    diff = 0
    for i, param in enumerate(params):
        gradsi = 0
        buf = 0
        for j in range(num_workers):
            buf += quantize(grads[j][i], num_bits)
        param.data.add_(-lr, buf)
        diff += (torch.norm(buf)*lr)**2
    return diff

def check_grads(grads, gradt):
    for i, grad in enumerate(gradt):
        grad1 = 0
        for j in range(len(grads)):
            grad1 += grads[j][i]
        #print(torch.norm(grad1)/torch.norm(grad))

def update_params_sgd(net, grads, lr, num_workers):
    params = net.parameters()
    diff = 0
    for i, param in enumerate(params):
        buf = 0      
        for j in range(num_workers):
            buf += grads[j][i]
        param.data.add_(-lr, buf)
        diff += (torch.norm(buf)*lr)**2
    return diff




def update_params_adam(net, grads, lr, num_workers, momentum_dict, first, beta1=0.9, beta2=0.999, eplison=1e-8):
    # Some initialization

    params = net.parameters()
    diff = 0
    flag = 0
    for i, param in enumerate(params):
        buf = 0
        flag = flag + 1 # from 1-8 at the loop below
    
        for j in range(num_workers):

            buf += grads[j][i]

        diff += (torch.norm(buf) * lr) ** 2
       

        """flag from 1-8"""
   
        if flag == 1:
            # dict['weight1_q'] = beta1 * dict['weight1_q'] + (1-beta1)*buf
            momentum_dict['weight1_q'] = beta1 * momentum_dict['weight1_q'] + (1-beta1)*buf
            momentum_dict['weight1_v'] = beta2 * momentum_dict['weight1_v'] + (1-beta2)*(buf**2)
            if first:
                momentum_dict['weight_v_hat1'] = momentum_dict['weight1_v']
            momentum_dict['weight_v_hat1'] = torch.max(momentum_dict['weight_v_hat1'], momentum_dict['weight1_v'])
            param.data.add_(-lr,  momentum_dict['weight1_q']/ (torch.sqrt(momentum_dict['weight_v_hat1']) + eplison))


        if flag == 2:
            momentum_dict['bias1_q'] = beta1 * momentum_dict['bias1_q'] + (1 - beta1) * buf
            momentum_dict['bias1_v'] = beta2 * momentum_dict['bias1_v'] + (1 - beta2) * (buf**2)
            if first:
                momentum_dict['bias_v_hat1'] = momentum_dict['bias1_v']
            
            
            momentum_dict['biast_v_hat1'] = torch.max(momentum_dict['bias_v_hat1'], momentum_dict['bias1_v'])
            param.data.add_(-lr, momentum_dict['bias1_q'] / (torch.sqrt(momentum_dict['bias_v_hat1']) + eplison))
 


        if flag == 3:
            momentum_dict['weight2_q'] = beta1 * momentum_dict['weight2_q'] + (1-beta1) * buf
            momentum_dict['weight2_v'] = beta2 * momentum_dict['weight2_v'] + (1-beta2)*(buf**2)

            if first:
                momentum_dict['weight_v_hat2'] = momentum_dict['weight2_v']
            momentum_dict['weight_v_hat2'] = torch.max(momentum_dict['weight_v_hat2'], momentum_dict['weight2_v'])
            param.data.add_(-lr,momentum_dict['weight2_q']/ (torch.sqrt(momentum_dict['weight_v_hat2']) + eplison))


        if flag == 4:
            momentum_dict['bias2_q'] = beta1 * momentum_dict['bias2_q'] + (1 - beta1) * buf
            momentum_dict['bias2_v'] = beta2 * momentum_dict['bias2_v'] + (1 - beta2) * (buf**2)

            if first:
                momentum_dict['bias_v_hat2'] =  momentum_dict['bias2_v']
            momentum_dict['bias_v_hat2'] = torch.max(momentum_dict['bias_v_hat2'], momentum_dict['bias2_v'])
            param.data.add_(-lr, momentum_dict['bias2_q'] / (torch.sqrt(momentum_dict['bias_v_hat2']) + eplison))


        if flag == 5:
            momentum_dict['weight3_q'] = beta1 * momentum_dict['weight3_q'] + (1-beta1) * buf
            momentum_dict['weight3_v'] = beta2 * momentum_dict['weight3_v'] + (1-beta2)*(buf**2)

            if first:
                momentum_dict['weight_v_hat3'] = momentum_dict['weight3_v']
            momentum_dict['weight_v_hat3'] = torch.max(momentum_dict['weight_v_hat3'], momentum_dict['weight3_v'])
            param.data.add_(-lr, momentum_dict['weight3_q']/ (torch.sqrt(momentum_dict['weight_v_hat3']) + eplison))


        if flag == 6:
            momentum_dict['bias3_q'] = beta1 * momentum_dict['bias3_q'] + (1 - beta1) * buf
            momentum_dict['bias3_v'] = beta2 * momentum_dict['bias3_v'] + (1 - beta2) * (buf**2)

            if first:
                momentum_dict['bias_v_hat3'] =  momentum_dict['bias3_v']
            momentum_dict['bias_v_hat3'] = torch.max(momentum_dict['bias_v_hat3'], momentum_dict['bias3_v'])
            param.data.add_(-lr, momentum_dict['bias3_q'] / (torch.sqrt(momentum_dict['bias_v_hat3']) + eplison))


        if flag == 7:
            momentum_dict['weight4_q'] = beta1 * momentum_dict['weight4_q'] + (1-beta1)* buf
            momentum_dict['weight4_v'] = beta2 * momentum_dict['weight4_v'] + (1-beta2)*(buf**2)

            if first:
                momentum_dict['weight_v_hat4'] = momentum_dict['weight4_v']
            momentum_dict['weight_v_hat4'] = torch.max(momentum_dict['weight_v_hat4'], momentum_dict['weight4_v'])
            param.data.add_(-lr, momentum_dict['weight4_q']/ (torch.sqrt(momentum_dict['weight_v_hat4']) + eplison))


        if flag == 8:
            momentum_dict['bias4_q'] = beta1 * momentum_dict['bias4_q'] + (1 - beta1) *  buf
            momentum_dict['bias4_v'] = beta2 * momentum_dict['bias4_v'] + (1 - beta2) * (buf**2)

            if first:
                momentum_dict['bias_v_hat4'] =  momentum_dict['bias4_v']
            momentum_dict['bias_v_hat4'] = torch.max(momentum_dict['bias_v_hat4'], momentum_dict['bias4_v'])
            param.data.add_(-lr,  momentum_dict['bias4_q'] / (torch.sqrt(momentum_dict['bias_v_hat4']) + eplison))

        

    return diff, momentum_dict










'''Global dictionary that can be checked by every worker and server'''

local_momentum = {}
for total in range(10*8 + 8 + 10): # 10 workers; 8for MNIST 62 for imagenet; 8 stationary glabal momentum and 10 flag
    local_momentum[total] = 0 



def average_localmomentum_model(net_list, net_t, interval, iteration,beta=0.9):
    global local_momentum
    params_t = net_t.parameters()
    num_net = len(net_list)   
    params_list = []
    
    for i, net in enumerate(net_list):
        params_list.append(list(net.parameters()))
    for i, param in enumerate(params_t):
        param.data.mul_(0)
        for j in range(num_net):
            param.data.add_(1/num_net, params_list[j][i].data)
    
     
    if (iteration%interval == 0):
        worker_flag = 0
        for para_id in range (8):  # 8 for MNIST, 62 for imagenet
            flag = para_id
            local_momentum[80 + para_id] = 0 # initialize
            local_momentum[88 + worker_flag] = 1
            worker_flag = worker_flag + 1
            for worker_id in range(10): # Since we have 10 workers
                q_id = flag
                local_momentum[80+para_id] = beta*local_momentum[80+para_id] + local_momentum[q_id]/10
                flag = flag + 8
                
      


def update_params_lmsgd(net, grads, lr, num_workers, worker_id, beta = 0.9):
    global local_momentum
    params = net.parameters()      
    flag = 8*worker_id  # 8 for MNIST
    for i, param in enumerate(params):
        buf = 0
        
            
        for j in range(num_workers):
            buf += grads[j][i]
        local_momentum[flag] = beta * local_momentum[flag] + buf 
        # When the flag denotes updating
        if local_momentum[88+worker_id] == 1:
            local_momentum[flag] = local_momentum[80+i]  # Syn the momentum  
            # Reset theh flag
            local_momentum[80+i] = 0 # Denotes that the server should syn again                    
        param.data.add_(-lr, local_momentum[flag])
        flag = flag + 1
     
     
     
        

        

def update_params_lmsgd(net, grads, lr, num_workers, worker_id, beta = 0.9):
    global local_momentum
    params = net.parameters()      
    flag = 8*worker_id
    for i, param in enumerate(params):
        buf = 0
        
            
        for j in range(num_workers):
            buf += grads[j][i]
        local_momentum[flag] = beta * local_momentum[flag] + buf 
        # When the flag denotes updating
        if local_momentum[88+worker_id] == 1:
            local_momentum[flag] = local_momentum[80+i]  # Syn the momentum  
            # Reset theh flag
            local_momentum[80+i] = 0 # Denotes that the server should syn again                    
        param.data.add_(-lr, local_momentum[flag])
        flag = flag + 1
            




def get_diff(net1, net2):
    params1 = list(net1.parameters())
    params2 = list(net2.parameters())
    diff = 0
    for i in range(len(params1)):
        diff += torch.dist(params1[i], params2[i])**2
    return diff
def get_grad_dist(grad1, grad2):
    dist2 = 0
    for i in range(len(grad1)):
        dist2 += torch.dist(grad1[i], grad2[i])**2
    return dist2

def quantization(vlist, num_bits):
    qv = []
    for i in range(len(vlist)):
        v = vlist[i]
        v_norm = torch.norm(v)
        if v_norm < 1e-10:
            qv.append(0)
        else:
            s = 2**(num_bits-1)
            l = torch.floor(torch.abs(v)/v_norm*s)
            p = torch.abs(v)/v_norm-l
            qv.append(v_norm*torch.sign(v)*(l/s + l/s*(torch.rand(v.shape)<p)))
    return qv
def quantize(v, num_bits):
    v_norm = torch.norm(v)
    if v_norm < 1e-10:
        qv = 0
    else:
        s = 2**(num_bits-1)
        l = torch.floor(torch.abs(v)/v_norm*s)
        p = torch.abs(v)/v_norm-l
        qv = v_norm*torch.sign(v)*(l/s + l/s*(torch.rand_like(v)<p).float())
    return qv

def sort_dataset(dataset, num_classes, num_samples):
    sorted = [[] for i in range(num_classes)]
    for i in range(num_samples):
        sorted[dataset[i][1]].append(dataset[i])
    alldata = []
    for i in range(num_classes):
        for data in sorted[i]:
            alldata.append(data)
    return alldata

def average_model(net_list, net_t):
    params_t = net_t.parameters()
    num_net = len(net_list)   
    params_list = []
    for i, net in enumerate(net_list):
        params_list.append(list(net.parameters()))
    for i, param in enumerate(params_t):
        param.data.mul_(0)
        for j in range(num_net):
            param.data.add_(1/num_net, params_list[j][i].data)

def update_delta(delta, net_wk, net_ps, lr, H):
    params_ps_list = list(net_ps.parameters())
    params_wk_list = list(net_wk.parameters())
    for i in range(len(params_ps_list)):
        p_ps = params_ps_list[i]
        p_wk = params_wk_list[i]
        delta[i] += (p_ps.data-p_wk.data)/lr/H
        
""" cuda communication """
def broadcast_models(net_s, net_w, device_s, devices):
    for i, net in enumerate(net_w):
        net.to(device_s)
        #copy_params(net_s, net)
        copy_state(net_s, net)
        net.to(devices[i])

def gather_models(net_w, device_s, comm_t=0):
    comm_flag = []
    for net in net_w:
        net.to(device_s)
        comm_flag.append(1)
        if comm_t>0:
            time.sleep(comm_t)
    return comm_flag

def broadcast_params(net_s, net_w, device_s, devices):
    # device_s must be difference from devices, gpu to mulple gpus
    params_s = list(net_s.parameters())
    params_w = [list(net.parameters()) for net in net_w]
    params_copies = comm.broadcast_coalesced(params_s, [device_s, *devices])
    for i in range(len(net_w)):
        for j in range(len(params_s)):
            params_w[i][j].data.copy_(params_copies[i][j].data)
    del params_copies

def gather_grads(grads_w, grads, device_s, comm_t=0):
    comm_flag = []
    for i in range(len(grads)):
        if grads_w[i] is not None:
            for j in range(len(grads_w[i])):
                grads_w[i][j] = grads_w[i][j].to(device_s)
            grads[i] = grads_w[i]
            comm_flag.append(1)
            if comm_t>0:
                time.sleep(comm_t)
        else:
            comm_flag.append(0)
    return comm_flag
    
def reduce_average_params(net_s, net_w, device_s):
    params_s = list(net_s.parameters())
    params_w = [list(net.parameters()) for net in net_w]
    num_workers = len(net_w)
    for j, param_s in enumerate(params_s):
        param_w_sum = comm.reduce_add([params_w[i][j] for i in range(num_workers)], device_s)
        param_s.data.mul_(0)
        param_s.data.add_(1/num_workers, param_w_sum.data)
    del param_w_sum
    
    
    


  
def average_fedadam_mnist_model(net_list, net_t, server_lr, fedadam_momentum_mnist, second, beta_fed = 0.999, eplison = 1e-8):      
    params_t = net_t.parameters()
    num_net = len(net_list)   
    params_list = []
    for i, net in enumerate(net_list):
        params_list.append(list(net.parameters()))
    for i, param in enumerate(params_t):
        if second:
            # Reset the difference of theta
            fedadam_momentum_mnist[i].mul_(0)
        for theta_diff in range(num_net):
            fedadam_momentum_mnist[i] = fedadam_momentum_mnist[i] + (params_list[theta_diff][i].data - param.data)/num_net
        # Adding momentum
        fedadam_momentum_mnist[8 + i] = beta_fed*fedadam_momentum_mnist[8 + i] + (1-beta_fed)*fedadam_momentum_mnist[i]**2
        # Updating server param
        param.data.add_(server_lr, fedadam_momentum_mnist[i]/(torch.sqrt(fedadam_momentum_mnist[8 + i])+eplison))
        #for j in range(num_net):
           
            #param.data.add_(1/num_net, params_list[j][i].data)
    return fedadam_momentum_mnist
        
     
