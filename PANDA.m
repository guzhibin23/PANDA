
function  [result,iter] = PANDA(data,labels, alpha, beta,m,yita,knn)
% data:each array is num_samp by d_v
% num_clus: number of clusters
% num_view: number of views
% num_samp
% k: number of adaptive neighbours
% labels: groundtruth of the data, num by 1


num_view = length(data);
num_samp = size(labels,1);
num_clus = length(unique(labels));
epsilon = 1/num_view*ones(1,num_view);
phi = 1/num_view*ones(1,num_view);
% lambda = randperm(30,1);
lambda = 1;
NITER = 30;
zr = 1e-10;
rho = 0.1; mu = 2;max_rho = 10e12;
sX = [num_samp, num_samp, num_view];

% =====================   Normalization =====================
for i = 1 :num_view
    for  j = 1:num_samp
        normItem = std(data{i}(j,:));
        if (0 == normItem)
            normItem = eps;
        end
        data{i}(j,:) = (data{i}(j,:) - mean(data{i}(j,:)))/normItem;
    end
end

% %normalized X
% for i=1:num_view
%     data{i} = data{i}./repmat(sqrt(sum(data{i}.^2,1)),size(data{i},1),1);
% end

%  ====== Initialization =======
% claculate G_v for all the views
Z = cell(num_view,1);
A = cell(num_view,1);
sumZ = zeros(num_samp);
for v = 1:num_view
    J{v} = zeros(num_samp,num_samp);
    O{v} = zeros(num_samp,num_samp);  %multiplier
    Zv = constructW_PKN(data{v}',knn);
    Z{v} = Zv;
    sumZ = sumZ + Zv;
    clear Zv
end

% initialize U
U = sumZ/num_view;
% initialize F
U0 = U-diag(diag(U));
w0 = (U0+U0')/2;
D0 = diag(sum(w0));
L = D0 - w0;
[F0,~,~] = eig1(L,num_clus,0);
F = F0(:,1:num_clus);
I = eye(num_samp);
 for v = 1:num_view
            Dz{v} = full(sparse(1:num_samp, 1:num_samp, sum(Z{v}))); %所以此处D为相似度矩阵中一列元素加起来放到对角线上，得到度矩阵D
            Lz{v}= eye(num_samp)-(Dz{v}^(-1/2) * Z{v} * Dz{v}^(-1/2));
        end
% iter = 0;
% Isconverg = 0;epson = 1e-7;
% converge_A=[];
% update ...
for iter=1:NITER
    % while(Isconverg == 0)
    fprintf('----processing iter %d--------\n', iter);

    %% =========================== Upadate A{i} ===========================
    
    for v = 1:num_view
        C = (alpha + epsilon(v))*I + phi(v)*Lz{v};
        for ni = 1:num_samp
            index = find(Z{v}(ni,:)>0);
            Ii = I(ni,index);
            ui = U(ni,index);
            d = 2*alpha*Ii + 2*epsilon(v)*ui;
            % solve a^T*C*a-a^T*d
            [ai, ~] = fun_alm(C(index,index),d);
            A{v}(ni,index) = ai';
        end
    end
    
  
    
    %% =========================== Upadate U ===========================
    dist_h = L2_distance_1(F',F');
    U = zeros(num_samp);
    for ni = 1:num_samp
        sumAi = zeros(1,num_samp);
        sumEpsilon=0;
        for v = 1:num_view
            Av = A{v};
            sumAi = sumAi +epsilon(v)*Av(ni,:);
            sumEpsilon=sumEpsilon+epsilon(v);
        end
        index = find(sumAi>0);
        ad = (sumAi(index) - lambda/2*dist_h(ni,index))/sumEpsilon;
        U(ni,index) = EProjSimplex_new(ad);
    end
    
    %% ====================== Upadate epsilon（i） ======================
    
    for v = 1:num_view
        distUA = norm(U - A{v},'fro')^2;
        if distUA == 0
            distUA = eps;
        end
        epsilon(v) = m/2/((distUA)^(1-m/2));
    end
    epsilon = max(epsilon,eps);
    %% ====================== Upadate phi（i） ======================
    
    for v = 1:num_view
        temp=A{v}'*Lz{v}*A{v};
        phi(v)=0.5/sqrt(2*trace(temp));
    end
    phi = max(phi,eps);
    
    %% =========================== Upadate J{i} ===========================
    
    A_tensor = cat(3, A{:,:});%C = cat(dim, Z, B) 沿 dim 指定的数组维度串联数组 Z 和 B
    O_tensor = cat(3, O{:,:});
    a = A_tensor(:);
    o = O_tensor(:);
    %     [j, ~] = wshrinkObj(a+1/rho*o,beta/rho,sX,0,3,omega);
    %     [j, ~] = wshrinkObj_weight_lp(a+1/rho*o,beta/rho*omega,sX,0,3,p);
    %     J_tensor = reshape(j, sX);
    J_tensor=solve_G(a+1/rho*o,rho/beta,sX,yita);
    for v=1:num_view
        J{v} = J_tensor(:,:,v);
    end
    
    
    %% =========================== Upadate F ===========================
    U = (U+U')/2;
    D = diag(sum(U));
    L = D-U;
    F_old = F;
    [F, ~, ev]=eig1(L, num_clus, 0);
    
    fn1 = sum(ev(1:num_clus));
    fn2 = sum(ev(1:num_clus+1));
    if fn1 > zr
        lambda = 2*lambda;
    elseif fn2 < zr
        lambda = lambda/2;
        F = F_old;
    else
        fprintf('the %d -th iteration -> end ...\n',iter)
        break;
    end
    
    %% =========================== Upadate O{i} ===========================
    for v=1:num_view
        O{v} = O{v}+rho*(A{v}-J{v});
    end
    
    %% =========================== Upadate rho ===========================
    rho = min(rho*mu, max_rho);
    %% ====================== Checking Coverge Condition ======================
  
    
    %     Isconverg = 1;
    %     sum_coverge_A=0;
    %     for i=1:num_view
    %         if(norm(A{i}-J{i},inf)>epson)
    %             history.norm_A=norm(A{i}-J{i},inf);
    %             fprintf('    norm_A %7.10f   \n ', history.norm_A);
    %             Isconverg = 0;
    %             sum_coverge_A=sum_coverge_A+history.norm_A;
    %         end
    %     end
    %
    %
    %     converge_A=[converge_A sum_coverge_A];
    %     if sum_coverge_A==inf
    %         break
    %     end;
    %     if (iter>40)
    %         Isconverg  = 1;
    %
    %     end
    % %     if(iter==20)
    % %         break;
    % %     end
    %     iter = iter + 1;
    
    
end

% =====================  result =====================
[clusternum, y]=graphconncomp(sparse(U));
% C_hat=sparse(sparse(S));
% pmax = sum(sum(sparse(S)))/sum(sum(sparse(S)~=0))
% mmax = max(max(sparse(S)))
% pic = imagesc(sparse(S),[pmax,mmax])
% colorbar
y = y';
if clusternum ~= num_clus
    sprintf('Can not find the correct cluster number: %d', num_clus)
end
[ACC,NMI,PUR] = ClusteringMeasure(labels, y);
[ARI,~,~,~] = valid_RandIndex(labels, y);
[F,P,R] = compute_f(labels,y);
result =[ACC NMI PUR F  R ARI];
end


function [v, obj] = fun_alm(A,b)
if size(b,1) == 1
    b = b';
end

% initialize
rho = 1.5;
mu = 30;
n = size(A,1);
alpha = ones(n,1);
v = ones(n,1)/n;
% obj_old = v'*A*v-v'*b;

obj = [v'*A*v-v'*b];
iter = 0;
while iter < 10
    % update z
    z = v-A'*v/mu+alpha/mu;
    
    % update v
    c = A*z-b;
    d = alpha/mu-z;
    mm = d+c/mu;
    v = EProjSimplex_new(-mm);
    
    % update alpha and mu
    alpha = alpha+mu*(v-z);
    mu = rho*mu;
    iter = iter+1;
    obj = [obj;v'*A*v-v'*b];
end
end


function [x] = EProjSimplex_new(v, k)
%
% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%
if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);
    
else
    x = v0;
end
end
