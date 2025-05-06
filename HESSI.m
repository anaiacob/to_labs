function[h]=HESSI(w,y,x)
    m=size(x,2);
    n=size(x,1);
    z=zeros(m,1);
    for i=1:m
        e=sigmoid(w'*x(:,i));
        z(i)=e*(1-e);
    end
    Q=diag(z);
    h=(x*Q*x')/m;
end