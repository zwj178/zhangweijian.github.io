%Title: Robust Principal Component Analysis with Adaptive Neighbors
%Rui Zhang & Hanghang Tong
%Thirty-third Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada.

%%Notification:
% Data X Dimention d * Data number n
% Adaptive Neighbors k
% Reduced dimension r (r<d)
function W=RPCA_AN(X,k,r)
[d,N]=size(X);
q=rand(N,1);
q=q./sum(q); 
iter=1;err=1;
while err>1e-3
D=diag(q)-q*q';
[W,~,~]=eig1(X*D*X',r,1);
p=zeros(N,1);
for i=1:N
    p(i)=norm((eye(d)-W*W')*(X(:,i)-X*q),2)^2;
end
[q,lam]=RSWL(p,k);
obj(iter)=q'*p+lam*norm(q,2)^2;
if iter>1
    err=abs(obj(iter)-obj(iter-1));
end
iter=iter+1;
end


function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
clc
if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
[v d] = eig(A);
d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
end;

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);

function [w,lam]=RSWL(f,k) %k<N
[N,~]=size(f);
[P,~]=sort(f,'ascend');
w=zeros(N,1);
for i=1:N
    w(i,1)=max(((P(k+1)-f(i))/(k*P(k+1)-sum(P(1:k)))),0);
end
lam=(k*P(k+1)-sum(P(1:k)))/2;

