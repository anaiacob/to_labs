load('data.mat')
epsilon=0.001;
alpha=0.003;
n=500;
k=100;
N=3650;
U=randn(n,k);
M=randn(N,k);
nr_max_iter=2000;
val_obiect=zeros(nr_max_iter,1);
stop=zeros(nr_max_iter,1);
matrice_erorr=(R_train-U*M').^2;
val_obiect(1)=0.5*sum(matrice_erorr,"all")
for l=1:nr_max_iter
    gradientU=zeros(n,k);
    gradientM=zeros(N,k);
    for i=1:n
      for j=find(R_train(i,:))
            q=R_train(i,j)-dot(U(i,:),M(j,:)');
            gradientU(i, :)=gradientU(i, :)-q*M(j, :);
            gradientM(j,:)=gradientM(j,:)-q*U(i,:);
      end
      U=U-alpha*gradientU;
      M=M-alpha*gradientM;
      matrice_erorr=(R_train-U*M').^2;
      val_obiect(l+1)=0.5*sum(matrice_erorr,"all")
      stop(l)=abs(val_obiect(l+1)-val_obiect(l));
      stop(l)
      if stop(l)<epsilon
          break;
      end
    end
end

%Afisare
figure;
plot(1:nr_max_iter,val_obiect(1:nr_max_iter),'LineWidth',2);
xlabel('Iteratii');
ylabel('Functia obiectiv');
grid on;

figure;
semilogy(1:nr_max_iter,stop(1:nr_max_iter),'LineWidth',2);
xlabel('Iteratii');
ylabel('Criteriu oprire');
grid on;