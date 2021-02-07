%%Covtype %%
%% CADA - logistic regression real data
clear all
close all

%% data allocation for linear regression\
addpath('./libsvm-3.23/windows/');
addpath('./datasets/');
[ydata_28, Xdata_28] = libsvmread('./datasets/covtype.libsvm.binary.scale');
ydata_28 = 2*ydata_28-3;
Xdata_28 = [Xdata_28(ydata_28==1,:); Xdata_28(ydata_28==-1,:)];
ydata_28 = [ydata_28(ydata_28==1); ydata_28(ydata_28==-1)];
accuracy=1e-4;
num_iter=1000;
num_workers=20;
X=cell(num_workers);
y=cell(num_workers);
print_iter=10;

num_feature=size(Xdata_28,2);
num_sample=size(Xdata_28,1);
per_split=sort(randperm(floor(num_sample/1000),num_workers))*1000;
ratio_batch=0.001;
batch_size = zeros(num_workers, 1);
for n=1:num_workers
    if n==1
        X{n}=Xdata_28(1:per_split(n),1:num_feature);
        y{n}=ydata_28(1:per_split(n));
        batch_size(n) = ratio_batch*length(y{n});
    else
        X{n}=Xdata_28(per_split(n-1)+1:per_split(n),1:num_feature);
        y{n}=ydata_28(per_split(n-1)+1:per_split(n));
        batch_size(n) = ratio_batch*length(y{n});
    end
end

X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end
num_sample=size(y_fede,1);

%% data pre-analysis
lambda=1e-5;


Hmax=zeros(num_workers,1);
for i=1:num_workers
   Hmax(i)=max(abs(eig(X{i}'*X{i})))/num_sample+lambda/num_workers; 

end
Hmax_sum=sum(Hmax);
hfun=Hmax_sum./Hmax;
nonprob=Hmax/Hmax_sum;


X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end




%% parameter initialization for ADAM and CADA
triggerslot=10;
adam_stepsize = 0.005;
thrd=1/(adam_stepsize^2*num_workers^2)/triggerslot;%0.5
num_bits = 4;
H = 20;
beta1 = 0.9;
beta2 = 0.999;
beta_sgd = beta1;
eplison = 0.0000001; % 0.001
thrd_cada = thrd*0.00005;  % 0.01
vr_delay =100;

%% ADAM

theta20=zeros(num_feature,num_iter);
grads20=zeros(num_feature,num_workers);
grads20_tilde=zeros(num_feature,num_workers);
stepsize20=adam_stepsize;
thrd20=thrd;
comm_count20=zeros(num_workers,1);

comm_iter20=1;
comm_index20=zeros(num_workers,num_iter);
comm_error20=[];
comm_grad20=[];
rng('default') 
%vr_delay = 100;


% As recommadation in paper

q = zeros(num_feature,1);
v = zeros(num_feature,1);
v_hat = zeros(num_feature,1);

for iter=1:num_iter
    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        if mod(iter,vr_delay)==1
          theta_tilde=theta20(:,iter);
        end
        % random sample
        c=randperm(size(y{i},1), batch_size(i));
        
        grad_temp=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta20(:,iter))))))/sum(batch_size)+lambda*theta20(:,iter)/num_workers;
        grad_temp_tilde=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta_tilde)))))/sum(batch_size)+lambda*theta_tilde/num_workers;
        
        if iter>0
            trigger=0;
            for n=1:min(triggerslot,iter-1)
                trigger=trigger+norm(theta20(:,iter-(n-1))-theta20(:,iter-n),2)^2;
            end
            % if norm((grad_temp-grad_temp_tilde)-(grads44(:,i)-grads44_tilde(:,i)),2)^2>=thrd44*trigger || mod(iter,vr_delay)==1
            if 1==1
                grads20_tilde(:,i)=grad_temp_tilde;
                grads20(:,i)=grad_temp;
                comm_count20(i)=comm_count20(i)+1;
                comm_index20(i,iter)=1;
                comm_iter20=comm_iter20+1;
                comm_flag=1;
            end
        end       
    end   
    grad_error20(iter)=norm(sum(grads20, 2),2);
    loss20(iter)=lambda*0.5*norm(theta20(:,iter))^2+mean(log(1+exp(-y_fede.*(X_fede*theta20(:,iter)))));
    if  comm_flag == 1%mod(iter,print_iter)==0 
        comm_error20=[comm_error20;comm_iter20,loss20(iter)]; 
        comm_grad20=[comm_grad20;comm_iter20,grad_error20(iter),iter]; 
    end
    % Server nabla recovering
    server_nabla = sum(grads20,2);
    % q^k
    q = beta1*q+(1-beta1)*server_nabla;
    % v^k
    v = beta2.*v+(1-beta2).*(server_nabla).^2;
    % hat v
    v_hat = max(v_hat, v);
    % parameter updates
    theta20(:,iter+1)=theta20(:,iter)-stepsize20*q./(sqrt(v_hat)+eplison);
end

%% CADA1
theta44=zeros(num_feature,num_iter);
grads44=zeros(num_feature,num_workers);
grads44_tilde=zeros(num_feature,num_workers);
stepsize44=adam_stepsize;
thrd44 = thrd_cada;
comm_count44=zeros(num_workers,1);

comm_iter44=1;
comm_index44=zeros(num_workers,num_iter);
comm_error44=[];
comm_grad44=[];
rng('default') 

% As recommadation in paper
q = zeros(num_feature,1);
v = zeros(num_feature,1);
v_hat = zeros(num_feature,1);

for iter=1:num_iter
    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        if mod(iter,vr_delay)==1
          theta_tilde=theta44(:,iter);
        end
        % random sample
        c=randperm(size(y{i},1), batch_size(i));
        
        grad_temp=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta44(:,iter))))))/sum(batch_size)+lambda*theta44(:,iter)/num_workers;
        grad_temp_tilde=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta_tilde)))))/sum(batch_size)+lambda*theta_tilde/num_workers;
        
        if iter>0
            trigger=0;
            for n=1:min(triggerslot,iter-1)
                trigger=trigger+norm(theta44(:,iter-(n-1))-theta44(:,iter-n),2)^2;
            end
            if norm((grad_temp-grad_temp_tilde)-(grads44(:,i)-grads44_tilde(:,i)),2)^2>=thrd44*trigger || mod(iter,vr_delay)==1
                grads44_tilde(:,i)=grad_temp_tilde;
                grads44(:,i)=grad_temp;
                comm_count44(i)=comm_count44(i)+1;
                comm_index44(i,iter)=1;
                comm_iter44=comm_iter44+1;
                comm_flag=1;
            end
        end       
    end   
    grad_error44(iter)=norm(sum(grads44, 2),2);
    loss44(iter)=lambda*0.5*norm(theta44(:,iter))^2+mean(log(1+exp(-y_fede.*(X_fede*theta44(:,iter)))));
    if  comm_flag == 1%mod(iter,print_iter)==0 
        comm_error44=[comm_error44;comm_iter44,loss44(iter)]; 
        comm_grad44=[comm_grad44;comm_iter44,grad_error44(iter),iter]; 
    end
    % Server nabla recovering
    server_nabla = sum(grads44,2);
    % q^k
    q = beta1*q+(1-beta1)*server_nabla;
    % v^k
    v = beta2.*v+(1-beta2).*(server_nabla).^2;
    % hat v
    v_hat = max(v_hat, v);
    % parameter updates
    theta44(:,iter+1)=theta44(:,iter)-stepsize44*q./(sqrt(v_hat)+eplison);
end




%% CADA2
theta66 = zeros(num_feature,num_iter);
theta66_local = zeros(num_feature, num_workers);
grads66 = zeros(num_feature, num_workers);
thrd66= thrd44;
stepsize66=adam_stepsize;
comm_count66=zeros(num_workers,1);

comm_iter66=1;
comm_index66=zeros(num_workers,num_iter);
comm_error66=[];
comm_grad66=[];
rng('default') 
delay_bound = 200;
delay = zeros(num_workers,1);

% As recommadation in paper
q = zeros(num_feature,1);
v = zeros(num_feature,1);
v_hat = zeros(num_feature,1);

for iter=1:num_iter
    comm_flag=0;
    for i=1:num_workers
        %delay(i)=delay(i)+1;
        c=randperm(size(y{i},1),batch_size(i));       
        grad_temp=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta66(:,iter))))))/sum(batch_size)+lambda*theta66(:,iter)/num_workers;
        if iter>0
            trigger=0;
            for n=1:min(triggerslot, iter-1)
                trigger=trigger+norm(theta66(:,iter-(n-1))-theta66(:,iter-n),2)^2;
            end
            grad_tilde = -(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta66_local(:,i))))))/sum(batch_size)+lambda*theta66_local(:,i)/num_workers;
            delay(i) = delay(i) + 1;
            if norm(grad_temp-grad_tilde,2)^2>=thrd66*trigger || delay(i)>delay_bound
                delay(i) = 0;
                grads66(:,i)= grad_temp;
                theta66_local(:,i)=theta66(:,iter);
                comm_count66(i)=comm_count66(i)+1;
                comm_index66(i,iter)=1;
                comm_iter66=comm_iter66+1;
                comm_flag=1;
            end
        end       
    end
    grad_error66(iter)=norm(sum(grads66,2),2);
    loss66(iter)=lambda*0.5*norm(theta66(:,iter))^2+mean(log(1+exp(-y_fede.*(X_fede*theta66(:,iter)))));
    if  comm_flag == 1%mod(iter,print_iter)==0
        comm_error66=[comm_error66;comm_iter66,loss66(iter)]; 
        comm_grad66=[comm_grad66;comm_iter66,grad_error66(iter),iter]; 
    end
    % Server nabla recovering
    server_nabla = sum(grads66,2);
    % q^k
    q = beta1*q+(1-beta1)*server_nabla;
    % v^k
    v = beta2.*v+(1-beta2).*(server_nabla).^2;
    % hat v
    v_hat = max(v_hat, v);
    % parameter updates
    theta66(:,iter+1)=theta66(:,iter)-stepsize66*q./(sqrt(v_hat)+eplison);
end





%% parameter initialization for FED-ADAM
stepsize=1000;
triggerslot=10;
thrd=1/(stepsize^2*num_workers^2)/triggerslot;
num_bits = 4;
H = 10;
adam_stepsize = 0.02;
beta1 = 0.9;
beta2 = 0.9;
eplison = 0.0001;
thrd_adam = thrd; 
v = ones(num_feature,1);
v = v * 0.0001;

%% FED-ADAM
loss13 = zeros(1, num_iter);
theta13=zeros(num_feature,num_iter);
grads13=zeros(num_feature,num_workers);
stepsize13_local = stepsize/H;
stepsize13 = adam_stepsize;
theta13_local = zeros(num_feature, num_workers);
local_thetadiff = zeros(num_feature, num_workers);

comm_error13=[];
comm_grad13=[];


rng('default') 
comm_iter13 = 1;
for iter=1:num_iter
    for i=1:num_workers  % worker in parallel
        % random sample
        c=randperm(size(y{i},1),batch_size(i));  
        % K inner loop for each gradient
      
%         for k = 1:H
            grads13(:,i)=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta13_local(:,i))))))/sum(batch_size)+lambda*theta13_local(:,i)/num_workers; 
            theta13_local(:,i) = theta13_local(:,i) - stepsize13_local*grads13(:,i);
%         end
        % find the theta diff for each worker
      
    end

    if mod(iter, H) == 1
        for i=1:num_workers
            if iter>=H
                local_thetadiff(:,i) = theta13_local(:,i) - theta13(:,iter-H);
            else
                local_thetadiff(:,i) = theta13_local(:,i);
            end
            
            server_nabla = mean(local_thetadiff,2);
            
            v = beta2.*v + (1-beta2).*server_nabla.^2;
            
            % parameter updates
            
            theta13(:,iter+1) = theta13(:,iter)+ stepsize13 .* (server_nabla./(sqrt(v) + eplison));
            
            theta13_local(:, i) = theta13(:, iter+1);
            
            comm_iter13 = comm_iter13 + 1;
            
        end
        grad_error13(iter)=norm(sum(grads13, 2),2);
        % Server Updating
        
        loss13(iter)=lambda*0.5*norm(theta13(:,iter+1))^2+mean(log(1+exp(-y_fede.*(X_fede*theta13(:,iter)))));
        comm_error13=[comm_error13;comm_iter13,loss13(iter)];
        comm_grad13=[comm_grad13;comm_iter13,grad_error13(iter),iter];
        
    else
        theta13(:,iter+1) = theta13(:,iter);
        loss13(iter)=lambda*0.5*norm(theta13(:,iter+1))^2+mean(log(1+exp(-y_fede.*(X_fede*theta13(:,iter)))));
    end
end


%% parameter initialization for LAG
stepsize=0.1;
triggerslot=10;
thrd=1/(stepsize^2*num_workers^2)/triggerslot;
num_bits=4;
H = 20;

%% LAG-WK
theta3=zeros(num_feature,num_iter);
grads3=zeros(num_feature,num_workers);
stepsize3=stepsize;
thrd3=thrd;
comm_count3=zeros(num_workers,1);

comm_iter3=1;
comm_index3=zeros(num_workers,num_iter);
comm_error3=[];
comm_grad3=[];
rng('default') 

for iter=1:num_iter
    comm_flag=0;
    for i=1:num_workers
        c=randperm(size(y{i},1), batch_size(i));       
        grad_temp=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta3(:,iter))))))/sum(batch_size)+lambda*theta3(:,iter)/num_workers;
        if iter>0
            trigger=0;
            for n=1:min(triggerslot, iter-1)
                trigger=trigger+norm(theta3(:,iter-(n-1))-theta3(:,iter-n),2)^2;
            end
            if norm(grad_temp-grads3(:,i),2)^2>=thrd3*trigger
                grads3(:,i)=grad_temp;
                comm_count3(i)=comm_count3(i)+1;
                comm_index3(i,iter)=1;
                comm_iter3=comm_iter3+1;
                comm_flag=1;
            end
        end       
    end
    grad_error3(iter)=norm(sum(grads3,2),2);
    loss3(iter)=lambda*0.5*norm(theta3(:,iter))^2+mean(log(1+exp(-y_fede.*(X_fede*theta3(:,iter)))));
    if  comm_flag == 1%mod(iter,print_iter)==0
        comm_error3=[comm_error3;comm_iter3,loss3(iter)]; 
        comm_grad3=[comm_grad3;comm_iter3,grad_error3(iter),iter]; 
    end
    theta3(:,iter+1)=theta3(:,iter)-stepsize3*sum(grads3,2);
end

%% local momentum
theta11=zeros(num_feature,num_iter);
grads11=zeros(num_feature,num_workers);
stepsize11=stepsize;
theta11_local = zeros(num_feature, num_workers);
comm_error11=[];
comm_grad11=[];
momentum = zeros(num_feature,num_workers);
rng('default') 
comm_iter11 = 1;
for iter=1:num_iter
    for i=1:num_workers
        % random sample
        c=randperm(size(y{i},1),batch_size(i));  
        % central server computation
        if iter>0
            grads11(:,i)=-(X{i}(c,:)'*(y{i}(c,:)./(1+exp(y{i}(c,:).*(X{i}(c,:)*theta11_local(:,i))))))/sum(batch_size)+lambda*theta11_local(:,i)/num_workers;
        end
        momentum(:,i) = beta_sgd*momentum(:,i) + grads11(:,i);
        theta11_local(:,i) = theta11_local(:,i)-stepsize11*momentum(:,i);
        theta11(:,iter+1) = mean(theta11_local, 2);
    end
    if mod(iter, H) == 0
        stand_momentum = sum(momentum, 2)/num_workers;
        for i=1:num_workers
            theta11_local(:,i) = theta11(:,iter+1);
            momentum(:,i) = stand_momentum;
        end
                       
    end
    grad_error11(iter)=norm(sum(grads11,2),2);
    loss11(iter)=lambda*0.5*norm(theta11(:,iter))^2+mean(log(1+exp(-y_fede.*(X_fede*theta11(:,iter)))));
    if mod(iter, H) == 0
        comm_iter11 = comm_iter11 + num_workers;
    end
    if mod(iter, H) == 0%mod(iter, print_iter) == 0
        comm_error11=[comm_error11;comm_iter11,loss11(iter)]; 
        comm_grad11=[comm_grad11;comm_iter11,grad_error11(iter),iter]; 
    end
end




%% figure
set(0,'DefaultFigureWindowStyle','normal');
set(0,'DefaultAxesFontSize',12);
set(0,'DefaultLineLineWidth',2);
set(0,'DefaultAxesLineWidth',2);
figure

semilogx(1:1:num_iter,loss20,'m--'); %ADAM
hold on
semilogx(1:1:num_iter,loss44,'m'); %CADA1
hold on
semilogx(1:1:num_iter,loss66,'k');%CADA2
hold on
semilogx(1:1:num_iter,loss11,'b--'); %local momentum
hold on 
semilogx(1:H:length(loss13),loss13(1:H:length(loss13)),'c--') %FED-ADAM
semilogx(1:H:length(loss3),loss3(1:H:length(loss3)),'c') %LAG



xlabel('Number of iteration','fontsize',16,'fontname','Times New Roman')
ylabel('Objective error','fontsize',16,'fontname','Times New Roman')
legend('ADAM','CADA1','CADA2','local momentum','FED-ADAM','LAG');

title('covtype');
saveas(gcf,'covtype_iter.pdf');



figure;
semilogx(comm_error20(:,1),abs(comm_error20(:,2)),'m--'); %ADAM
hold on
semilogx(comm_error44(:,1),abs(comm_error44(:,2)),'m');%CADA1
hold on
semilogx(comm_error66(:,1),abs(comm_error66(:,2)),'k');%CADA2
hold on;
semilogx(comm_error11(:,1),abs(comm_error11(:,2)),'b--'); %local momentum
hold on;
semilogx(comm_error13(:,1),abs(comm_error13(:,2)),'c--'); %FED-ADAM
semilogx(comm_error3(:,1),abs(comm_error3(:,2)),'c'); %LAG

xlabel('Number of communications (uploads)','fontsize',16,'fontname','Times New Roman')
ylabel('Objective error','fontsize',16,'fontname','Times New Roman')
legend('ADAM','CADA1','CADA2','local momentum','FED-ADAM','LAG');
axis([0, 1e3, 0.5, 0.7]);
title('covtype');
saveas(gcf,'covtype_comm.pdf');

%% Gradient
figure;
plot(loss20,'m--'); %ADAM
hold on
index = 2:2:length(loss44)*2;
plot(index, loss44,'m'); %CADA1
hold on
index = 2:2:length(loss66)*2;
plot(index, loss66,'k');%CADA2
hold on
plot(loss11,'b--'); %local momentum
hold on
plot(loss13,'c--'); %FED ADAM
hold on
index = 2:2:length(loss3)*2;
plot(index, loss3,'c');%LAG


xlabel('gradient evaluation','fontsize',16,'fontname','Times New Roman')
ylabel('Loss','fontsize',16,'fontname','Times New Roman')
legend('ADAM','CADA1','CADA2','local momentum','FED-ADAM','LAG');
axis([1, 1e3, 0.5, 0.7]);
title('covtyp');
saveas(gcf,'covtyp_gradients.pdf')