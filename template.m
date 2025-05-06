clc
clear
%% Citirea si preprocesare datelor
dataFolder = 'train';
categories = {'cat', 'dog'};

imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

dataFolder = 'test';
imds_t = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

data=[];                            % date de antrenare
data_t =[];                         % date de test
while hasdata(imds) 
    img = read(imds) ;              % citeste o imagine din datastore
    img = imresize(img, [227 227]);
    %figure, imshow(img); % decomentati pentru a vizualiza pozele din
    %pause                        %baza de dat 
    img = double(rgb2gray(img));
    data =[data, reshape(img, [], 1)];

end
eticheta = double(imds.Labels == 'cat'); % eticheta 1 - pisica, 0 - caine
while hasdata(imds_t) 
    img = read(imds_t) ;              % citeste o imagine din datastore
    img = imresize(img, [227 227]); 
    %figure, imshow(img);    % decomentati pentru a vizualiza pozele din
    % pause                  %baza de date
    img = double(rgb2gray(img));
    data_t =[data_t, reshape(img, [], 1)];
    
end
eticheta_t = double(imds_t.Labels == 'cat'); % eticheta 1 - pisica, 0 - caine
%Reducerea dimensiuni
data_pca = pca(data, 'NumComponents', 45)';
data_pca_t = pca(data_t, 'NumComponents', 45)';

clear categories imds imds_t img dataFolder data data_t

[n, N] = size(data_pca); % n - nr de caracteristici; N- nr de poze
%%
pas_constant = [];
pas_ideal = [];

y=eticheta;
x=data_pca;
w=randn(n,1);
wcopie=w;

%alpha=1;
alpha=0.75;
iter=0;
epsilon=0.0001;
maxiter=10000;

while (norm(GRAD(w,y,x)) > epsilon && iter < maxiter)
    w=w-alpha*(inv(HESSI(w,y,x)))*(GRAD(w,y,x))';
    pas_constant=[pas_constant,norm(GRAD(w,y,x))];
    iter=iter+1;
end
%pas ideal
iter=0;

while (norm(GRAD(wcopie,y,x))>epsilon && iter<maxiter)
    cost=@(alpha) f_obj(wcopie-alpha*inv(HESSI(wcopie,y,x))*(GRAD(wcopie,y,x))',y,x);
    alpha=fminbnd(cost,0,1);
    wcopie=wcopie-alpha*inv(HESSI(wcopie,y,x))*(GRAD(wcopie,y,x))';
    pas_ideal=[pas_ideal,norm(GRAD(wcopie,y,x))];
    iter=iter+1;
end

figure;
grid on;
hold on;
plot(pas_constant, 'b','LineWidth',2);
hold on;
plot(pas_ideal,'r','LineWidth',2);
title('Metoda Newton');
%xlabel('Nr iter');
legend('pas const', 'pas ideal');

%Mat confuzie
N2=size(data_pca_t,2);
A=zeros(2,2);
ft=zeros(N2,1);

for i=1:N2
    ft(i)=sigmoid(w'*data_pca_t(:,i));
end

for i=1:N2
    if ft(i)>=0.5 && eticheta_t(i)==1
        A(1,1)=A(1,1)+1;
    elseif ft(i)>=0.5 && eticheta_t(i)==0
        A(1,2)=A(1,2)+1;
    elseif ft(i)<0.5 && eticheta_t(i)==0
        A(2,2)=A(2,2)+1;
    else A(2,1)=A(2,1)+1;
    end
end
disp((A(1,1)+A(1,2))/(A(1,2)+A(2,1)+A(1,1)+A(2,2))); %acuratete