clc, clear
n = 70;                              % valoarea de redimensionare a img.
A_bar = imread("bunny.jpg");         % citirea img
A_bar = rgb2gray(A_bar);             % convertire in alb-negru
A_bar =im2double(A_bar);             % convertirea in valori double
A_bar = imresize(A_bar,[n n]);       % redimensionarea  pozei  
figure(1)
imshow(A_bar);                           % afisarea pozei initiale 
title("Imaginea originala")

nrintraricunoscute=3000;             % setam un numar de intrari cunoscute 
rPerm = randperm(n*n);                          %generarea random a indicilor pentru intrarile cunoscuti
omega = sort(rPerm(1 : nrintraricunoscute));    %intrarile care se cunosc
A = nan(n); A(omega) = A_bar(omega); 
figure(2)
title("Imagine cu pixeli lipsa")
imshow(A)

epsilon=1e-3;
c=10;
Y_old=rand(n,n);
Y_old(omega)=A(omega);
Y_new=zeros(n,n);

imax=1000000;
iter=0;
ER=1;
cond_stop = [];

while ER>=epsilon && iter<=imax
    cond_stop=[cond_stop,ER];
    Y_old=Y_new;
    [U,S,V]=svd(Y_old);
    alpha=c/(iter+1);
    Y_new=Y_old-alpha*U*V';
    Y_new(omega)=A(omega);
    ER=norm(Y_new-Y_old);
    iter=iter+1;
end
figure(3)
imshow(Y_new)
figure(4)
title("Metoda Gradient Proiectat");
xlabel("Iteratii");
ylabel("cond_stop");
set(gcf,'NumberTitle','off');
set(gcf,'Name','Evolutie criteriu de oprire');
semilogy(1:iter,cond_stop(1:iter),'b','LineWidth',2);
%hold off;