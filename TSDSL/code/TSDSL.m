function [P,Q] = TSDSL(X_src,X_tar,label,options)

lambda1 = options.lambda1;
mu=0.1;
dim=options.dim;
lambda2=options.lambda2;
lambda3=options.lambda3;
lambda4=options.lambda4;
k=options.k;
rho=1.01;
max_lambda4=10^5;
% dim=options.dim;
T = options.T;
[Sw, Sb] = Scatter(X_src, label);
X=[X_src,X_tar];
[m,n]=size(X);
options.ReducedDim = dim;
[P1,~] = PCA1(X',options);
% 这个PCA只为初始化一个正交矩阵P
%%------------------------------initilzation-------------------------------
Q = ones(m,dim);
v=sqrt(sum(Q.*Q,2)+eps);
D=diag(0.5./(v));                                                                   
[G1,G2]=lle(X_src,X_tar,k);
%%-------------------------end of initilazation----------------------------
  for iter = 1:T
    % P
    if (iter == 1)
        P = P1;
    else
        [U1,S1,V1] = svd(X*X'*Q,'econ');
        P = U1*V1';
    end

    Q1 = 2*((Sw-mu*Sb)+lambda1*D+lambda2*G1+lambda3*G2+lambda4*X*X');
%     Q2 = lambda4*X*M'*P;
    Q2 = 2*lambda4*X*X'*P;
    Q =  Q1\Q2;
    v=sqrt(sum(Q.*Q,2)+eps);
    D=diag(0.5./(v));
    leq = X-P*Q'*X;
    obj(iter) = trace(Q'*(Sw-mu*Sb)*Q)+lambda1*sum(v)+lambda2*trace(Q'*G1*Q)+lambda3*trace(Q'*G2*Q)+norm(leq,'fro');
    if iter >2
        if  norm(leq, Inf) < 10^-7 &&abs(obj(iter)-obj(iter-1))<0.000001
            iter
            break;
        end 
    end
end

%A=obj(iter);


