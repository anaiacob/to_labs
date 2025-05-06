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

cvx_begin
variable Y(n,n);
minimize(norm_nuc(Y));
subject to:
Y(omega) == A(omega);
cvx_end

figure(3);
imshow(Y);


%% Metoda gradient conditional

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

ER=1;
Y_old=randn(n,n);
Y_new=zeros(n,n);
T=zeros(n,n);
cond_stop = [];
dr=70;

epsilon1=1e-6;
epsilon2=1e-3;
imax=1000000;
iter=0;

while ER>=epsilon2 && iter<=imax
    cond_stop = [cond_stop, ER];
    Y_old=Y_new;
    T(omega)=Y_old(omega)-A(omega);
    v1=randn(n,1);
    v1=v1/norm(v1);
    u1=v1;
    u2=u1;
    err=1;
    alfa1=1;
    Tt=T'*T;

    while err>=epsilon1
        v2= Tt*v1/norm(Tt*v1);
        alfa2=norm(T*v1);
        u2=T*v2/alfa2;
        err = abs(u2'*T*v2-alfa1);
        alfa1=alfa2;
        v1=v2;
        u1=u2;
    end
    S=-dr*u1*v1';
    a=2/(iter+2);
    Y_new=Y_old+a*(S-Y_old);
    iter=iter+1;
    ER=norm(Y_new-Y_old);
end

figure(3);
imshow(Y_new);
figure(4);
title("Metoda Gradient Conditional");
xlabel("Iteratii");
ylabel("cond_stop");
set(gcf,'NumberTitle','off');
set(gcf,'Name','Evolutie criteriu de oprire');
semilogy(1:iter,cond_stop(1:iter),'r','LineWidth',2);
hold on;

%% Metoda Gradient Proiectat

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

while ER>=epsilon && iter<imax
    cond_stop=[cond_stop,ER];
    Y_old=Y_new;
    [U,~,V]=svd(Y_old);
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
hold off;