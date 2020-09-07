function [Y1,Y2] = lle(X_src,X_tar,K)
[n,m1]=size(X_src);
[n,m2]=size(X_tar);
%目标域到源域
distance = EuDist(X_src,X_tar,[]);
[sorted,index] = sort(distance);
neighborhood = index(1:K,:);  %目标域到源域最近的K个点

p = zeros(K,m2);
t1 =zeros(n,K);
S2=zeros(n,n);
for ii=1:m2
 z = X_src(:,neighborhood(:,ii))-repmat(X_tar(:,ii),1,K); % shift ith pt to origin
 C = z'*z; % local covariance
 p(:,ii) = C\ones(K,1); % solve Cw=1
 p(:,ii) = p(:,ii)/sum(p(:,ii)); % enforce sum(w)=1
end
for i=1:m2
    for j=1:K
       t1(:,j)=p(j,i)*X_src(:,neighborhood(j,i));
    end
    S1=sum(t1,2);
    u1=(X_tar(:,i)-S1)*(X_tar(:,i)-S1)';
    S2=S2+u1;
    t1=zeros(n,K);
end
Y1=S2;


%源域到目标域
distance_new = EuDist(X_tar,X_src,[]);
[sorted,index] = sort(distance_new);
neighborhood_new= index(1:K,:);  %源域到目标域最近的K个点
%

t2=zeros(n,K);
S3=zeros(1,n);
S4=zeros(n,n);
q = zeros(K,m1);
for ii=1:m1
 zz= X_tar(:,neighborhood_new(:,ii))-repmat(X_src(:,ii),1,K); % shift ith pt to origin
 C1 = zz'*zz; % local covariance
 q(:,ii) = C1\ones(K,1); % solve Cw=1
 q(:,ii) = q(:,ii)/sum(q(:,ii)); % enforce sum(w)=1
end

for i=1:m1
    for j=1:K
        t2(:,j)=q(j,i)*X_tar(:,neighborhood_new(j,i));
    end
    S3=sum(t2,2);
    u2=(X_src(:,i)-S3)*(X_src(:,i)-S3)';
    S4=S4+u2;
    t2=zeros(n,K);
end
Y2=S4;

