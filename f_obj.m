function k= f_obj(w,y,x)
    N=size(x,2);
    h=zeros(N,1);
    for i=1:N
        h(i)=sigmoid(w'*x(:,i));
    end
    k=(-y'*log(h)-(ones(N,1)-y)'*(ones(N,1)-log(h)))/N;
end