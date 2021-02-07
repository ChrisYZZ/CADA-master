import threading
from torch.cuda._utils import _get_device_index
from utils_gpu import *

""" SGD """
def parallel_apply_SGD(models, zipdata, criterion, weights, devices=None):
    #devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    
    def _worker(i, model, data, weight, device=None):
        try:
            with torch.cuda.device(device):
                zero_grad(model)
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                loss = criterion(output, label)
                loss.backward()
            with lock:
                results[i] = get_grad(model, weight)
        except Exception:
            with lock:
                results[i] = None
    
    threads = [threading.Thread(target=_worker, 
                                args=(i, model, data, weight, device)) 
               for i, (model, data, weight, device) in 
               enumerate(zip(models, zipdata, weights, devices))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    outputs = []
    for i in range(len(zipdata)):
        outputs.append(results[i])
    return outputs




""" CADA"""
def parallel_apply_CADA(models, models_old, zipdata, criterion, weights, params, devices=None):
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    
    method = params['method']
    thrd = params['thrd']
    delay = params['delay']
    delay_bound = params['delay_bound']
    if method == 'wk1':
        delta = params['delta']
        
    def _worker(i, model, model_old, data, weight, device=None):
        try:
            with torch.cuda.device(device):
                # compute model gradient
                zero_grad(model)
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                loss = criterion(output, label)
                loss.backward()
                grad = get_grad(model, weight)
                delay[i] = delay[i] + 1
                if delay[i] >= delay_bound:
                    copy_params(model, model_old)
                    delay[i] = 0
                    if method == 'wk1':
                        delta[i] = None
                else:
                    # compute model_old gradient
                    zero_grad(model_old)
                    output = model_old(input)
                    loss = criterion(output, label)
                    loss.backward()
                    grad_old = get_grad(model_old, weight)
                    if method == 'wk1':
                        delta_new = check_wk1(grad, grad_old, delta[i], thrd)
                        if delta_new is not None:
                            delta[i] = delta_new
                        else:
                            grad = None
                    if method == 'wk2':
                        if check_wk2(grad, grad_old, thrd):
                            copy_params(model, model_old)
                            delay[i] = 0
                        else:
                            grad = None
            with lock:
                results[i] = grad
        except Exception:  
            with lock:
                results[i] = None
    
    threads = [threading.Thread(target=_worker, 
                                args=(i, model, model_old, data, weight, device)) 
               for i, (model, model_old, data, weight, device) in 
               enumerate(zip(models, models_old, zipdata, weights, devices))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    outputs = []
    for i in range(len(zipdata)):
        outputs.append(results[i])
    return outputs

""" LAG """
def parallel_apply_LAG(models, models_old, zipdata, criterion, weights, params, devices=None):
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    
    method = params['method']
    thrd = params['thrd']
    delay = params['delay']
    delay_bound = params['delay_bound']
    if method == 'wk1':
        delta = params['delta']
        
    def _worker(i, model, model_old, data, weight, device=None):
        try:
            with torch.cuda.device(device):
                # compute model gradient
                zero_grad(model)
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                loss = criterion(output, label)
                loss.backward()
                grad = get_grad(model, weight)
                delay[i] = delay[i] + 1
                if delay[i] >= delay_bound:
                    copy_params(model, model_old)
                    delay[i] = 0
                    if method == 'wk1':
                        delta[i] = None
                else:
                    # compute model_old gradient
                    zero_grad(model_old)
                    output = model_old(input)
                    loss = criterion(output, label)
                    loss.backward()
                    grad_old = get_grad(model_old, weight)
                    if method == 'wk1':
                        delta_new = check_wk1_lag(grad, grad_old, delta[i], thrd)
                        if delta_new is not None:
                            delta[i] = delta_new
                        else:
                            grad = None
                    if method == 'wk2':
                        if check_wk2(grad, grad_old, thrd):
                            copy_params(model, model_old)
                            delay[i] = 0
                        else:
                            grad = None
            with lock:
                results[i] = grad
        except Exception:  
            with lock:
                results[i] = None
    
    threads = [threading.Thread(target=_worker, 
                                args=(i, model, model_old, data, weight, device)) 
               for i, (model, model_old, data, weight, device) in 
               enumerate(zip(models, models_old, zipdata, weights, devices))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    outputs = []
    for i in range(len(zipdata)):
        outputs.append(results[i])
    return outputs




""" LOCAL-MOMENTUM """

def parallel_apply_MNIST_LMSGD(models, zipdata_list, criterion, weights, lr, devices=None):
    lock = threading.Lock()
    grad_enabled = torch.is_grad_enabled()
    H = len(zipdata_list[0])
    num_workers = len(zipdata_list)
    
    def _worker(i, model, data_list, weight, device=None):
        '''
        try:
            with torch.cuda.device(device):
                for data in data_list:
                    zero_grad(model)
                    input, label = data[0].to(device), data[1].to(device)
                    output = model(input)
                    loss = criterion(output, label)
                    loss.backward()
                    grad = [get_grad(model, weight)]
                    # update_params_sgd(model, grad, lr*num_workers, 1)
                    update_params_lmsgd(model, grad, lr*num_workers, 1, i, 0.9)
        except Exception:
            print('fail')
        '''
            
        with torch.cuda.device(device):
                for data in data_list:
                    zero_grad(model)
                    input, label = data[0].to(device), data[1].to(device)
                    output = model(input)
                    loss = criterion(output, label)
                    loss.backward()
                    grad = [get_grad(model, weight)]
                    # update_params_sgd(model, grad, lr*num_workers, 1)
                    update_params_lmsgd(model, grad, lr*num_workers, 1, i, 0.9)
    
    threads = [threading.Thread(target=_worker, 
                                args=(i, model, data_list, weight, device)) 
               for i, (model, data_list, weight, device) in 
               enumerate(zip(models, zipdata_list, weights, devices))]
    for thread in threads:
        thread.start()
        thread.join()
    #for thread in threads:
        #thread.join()
    return






""" CADA"""
def parallel_apply_CADA(models, models_old, zipdata, criterion, weights, params, devices=None):
    
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()
    
    method = params['method']
    thrd = params['thrd']
    delay = params['delay']
    delay_bound = params['delay_bound']
    if method == 'wk1':
        delta = params['delta']
        
    def _worker(i, model, model_old, data, weight, device=None):
        global trigger_flag
        try:
            with torch.cuda.device(device):
                # compute model gradient
                zero_grad(model)
                input, label = data[0].to(device), data[1].to(device)
                output = model(input)
                loss = criterion(output, label)
                loss.backward()
                grad = get_grad(model, weight)
                delay[i] = delay[i] + 1
                if delay[i] >= delay_bound:
                    copy_params(model, model_old)
                    delay[i] = 0
                    if method == 'wk1':
                        
                        delta[i] = None
                else:
                    # compute model_old gradient
                    zero_grad(model_old)
                    output = model_old(input)
                    loss = criterion(output, label)
                    loss.backward()
                    grad_old = get_grad(model_old, weight)
                    if method == 'wk1':
                        delta_new = check_wk1(grad, grad_old, delta[i], thrd)
                        if delta_new is not None:
                            delta[i] = delta_new
                        else:
                            grad = None
                    if method == 'wk2':
                        if check_wk2(grad, grad_old, thrd):
                           
                            
                            copy_params(model, model_old)
                            delay[i] = 0
                        else:
                            grad = None
            with lock:
                results[i] = grad
        except Exception:  
            with lock:
                results[i] = None
    
    threads = [threading.Thread(target=_worker, 
                                args=(i, model, model_old, data, weight, device)) 
               for i, (model, model_old, data, weight, device) in 
               enumerate(zip(models, models_old, zipdata, weights, devices))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    outputs = []
    for i in range(len(zipdata)):
        outputs.append(results[i])
    return outputs







""" FED ADAM MNIST """
def parallel_apply_fedadam(models, zipdata_list, criterion, weights, lr, devices=None):
    lock = threading.Lock()
    grad_enabled = torch.is_grad_enabled()
    H = len(zipdata_list[0])
    num_workers = len(zipdata_list)
    
    def _worker(i, model, data_list, weight, device=None):
        
        try:
            with torch.cuda.device(device):
                for data in data_list:
                    zero_grad(model)
                    input, label = data[0].to(device), data[1].to(device)
                    output = model(input)
                    loss = criterion(output, label)
                    loss.backward()
                    grad = [get_grad(model, weight)]
                    update_params_sgd(model, grad, lr, 1)
        except Exception:
            print('fail')
                 
    
    threads = [threading.Thread(target=_worker, 
                                args=(i, model, data_list, weight, device)) 
               for i, (model, data_list, weight, device) in 
               enumerate(zip(models, zipdata_list, weights, devices))]
    for thread in threads:
        thread.start()
        thread.join()
   
        
    return
