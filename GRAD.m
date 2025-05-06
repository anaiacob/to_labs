function[A]=GRAD(w,y,x)
    m=size(x,2);
    f=zeros(m,1);
    for i=1:m
        f(i)=sigmoid(w'*x(:,i));
    end
    A=((f-y)'*x')/m;
end