clc, clear, close all

% Descrierea sistemului pendulul inversat pe un carucior
M = .5;   % Masa cartului
m = 0.2;  % Masa pendulului
b = 0.1;  % Coeficient de frecare pt carucior
I = 0.006; % Momentul de inertie
g = 9.8;  % Acceleratia gravitationala
l = 0.3;  % Lungimea lantului
p = I*(M+m) + M*m*l^2; % Denominator pentru matricile A și B

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];

B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];

% Matrici de cost
Q = [10 0 0 0;
     0 1 0 0;
     0 0 10 0;
     0 0 0 1];

R = 1;
N = 3; % Orizontul de predictie

% Constrangeri pentru intrare
ub = 3;
lb = -3;

% Starea initiala
z0 = [2; 1; 0.3; 0];

% Referinta
x_ref = 0;
z_ref = [0; 0; 0; 0];

% Parametri algoritm bariera
maxIter = 20;
steps = 0;
u = [];
z = z0;
x_0 = zeros(N,1); % warm start
eps = 0.0001;
sigma = 0.6;
alpha = 0.00001;

while steps < maxIter
    % Formulare QP densa
    [H, q, C, d] = denseMPC(A, B, Q, R, z0, N, ub, lb, z_ref, x_ref);

    %x=quadprog(H,q,C,d);
    % Metoda bariera
    tau = 1;
    x = zeros(N,1);
    while 6*tau >= eps
       cnt = C*x - d;
       grad = H*x + q;
       for i = 1:size(C,1)
           grad = grad - tau * (C(i,:))' / cnt(i);
       end
       while norm(grad) >= 0.5
          x = x - alpha * grad;
           cnt = C*x - d;
           grad = H * x + q;
           for i = 1:size(C,1)
               grad = grad - tau * (C(i,:))' / cnt(i);
           end
       end
       tau = sigma * tau;
    end
    %cvx
    %cvx_begin
    %variable x(N,1)
    % minimize(0.5*x'*H*x+q'*x)
    % subject to
    % C*x<=d
    % cvx_end

    % Salvam controlul si actualizam starea
    u = [u, x(1)];
    z_new = A * z0 + B * x(1);
    z0 = z_new;
    z = [z, z0];
    steps = steps + 1;
end

% Afisare rezultate
figure(1)
subplot(2,2,1)
plot(z(1,:))
legend('x Pozitia caruciorului');
title('Pozitia caruciorului')

subplot(2,2,2)
plot(z(2,:))
legend('Viteza caruciorului');
title('Viteza caruciorului')

subplot(2,2,3)
plot(z(3,:))
legend('\phi Abaterea pendulului');
title('Abaterea pendulului')

subplot(2,2,4)
plot(z(4,:))
legend('Viteza unghiulara');
title('Viteza abaterei unghiulare')
saveas(gcf, 'Figcvx1.png')
figure(2)
plot(u)
legend('u = Forta aplicata');
title('Controlul aplicat (u)')
saveas(gcf, 'Figcvx2.png')
