clear; clc; close all;

% Citire date Wine Quality
data = readmatrix('winequality-red.csv');

A = data(:, 1:end-1); 
quality = data(:, end); 

e = double(quality >= 6); % clasificare binara (0 - slab, 1 - bun)

% Normalizare date
A = normalize(A);

% Impartire in seturi de antrenare si testare
N = size(A,1);
p_train = 0.8;
N_train = round(p_train * N);

idx = randperm(N);

A_train = A(idx(1:N_train), :);
e_train = e(idx(1:N_train), :);

A_test = A(idx(N_train+1:end), :);
e_test = e(idx(N_train+1:end), :);

% Adaugare bias
A_train_bar = [A_train, ones(N_train,1)];
A_test_bar  = [A_test, ones(N-N_train,1)];

n = size(A_train,2);
m = 20; % numarul de neuroni
a_activare = 0.5; % parametrul pentru Sinp

% Initializare parametri
rng(42); % reproducibilitate
X = 0.1 * randn(n+1, m); % ponderi mici
x = 0.1 * randn(m,1);

params = [X(:); x]; % vectorizare

max_iter = 300;
gamma = 0.1;
lambda = 0.1;
toleranta = 1e-6;

% Antrenare Quasi-Newton
params_quasi = params;
fprintf('Incep antrenarea cu Quasi-Newton...\n');
loss_history_quasi = zeros(max_iter,1); % initializare vector loss
grad_norm_history_quasi = zeros(max_iter,1); % initializare vector gradient normalizat

tic
for iter_quasi = 1:max_iter
    [loss, grad] = calc_loss_grad(A_train_bar, e_train, params_quasi, n, m, a_activare);
    loss_history_quasi(iter_quasi) = loss;
    grad_norm_history_quasi(iter_quasi) = norm(grad);

    % Update parametri
    params_quasi = params_quasi - gamma * grad;
    
    % Conditie oprire
    if norm(grad) < toleranta
        fprintf('Convergenta Quasi-Newton atinsa la iteratia %d.\n', iter_quasi);
        break;
    end
end

time_quasi = toc; % timpul total

fprintf('\nTimp de executie Quasi-Newton: %.2f secunde\n', time_quasi);


% Antrenare Levenberg-Marquardt
params_lm = params;
fprintf('Incep antrenarea cu Levenberg-Marquardt...\n');
loss_history_lm = zeros(max_iter,1); % initializare vector loss
grad_norm_history_lm = zeros(max_iter,1); % initializare vector gradient normalizat

tic
for iter_lm = 1:max_iter
    [loss, grad] = calc_loss_grad(A_train_bar, e_train, params_lm, n, m, a_activare);
    loss_history_lm(iter_lm) = loss;
    grad_norm_history_lm(iter_lm) = norm(grad);

    % Aproximare Hessiana
    H_approx = grad * grad';
    
    % Actualizare cu damping
    update = (H_approx + lambda * eye(length(grad))) \ grad;
    
    % Update parametri
    params_lm = params_lm - update;
    
    % Conditie oprire
    if norm(grad) < toleranta
        fprintf('Convergenta Levenberg-Marquardt atinsa la iteratia %d.\n', iter_lm);
        break;
    end
end

time_lm = toc; % timpul total

fprintf('Timp de executie Levenberg-Marquardt: %.2f secunde\n', time_lm);

% Antrenare Metoda de Optimizare Alternativa
params_alt = params; % initializare
loss_history_alt = zeros(max_iter,1);
grad_norm_history_alt = zeros(max_iter,1);

fprintf('Incep antrenarea cu Metoda Alternativa...\n');
tic;
for iter_alt = 1:max_iter
    % Extragem X si x
    X_alt = reshape(params_alt(1:(n+1)*m), n+1, m);
    x_alt = params_alt((n+1)*m+1:end);
    
    % Fixam X si optimizam x (stratul final)
    % predictii cu X curent
    Z = A_train_bar * X_alt;
    H = functie_activare_sinp(Z, a_activare);
    y_pred = 1 ./ (1 + exp(-H * x_alt));
    
    % Loss si gradient fata de x
    error = y_pred - e_train;
    delta = error .* (y_pred .* (1 - y_pred));
    grad_x = (H' * delta) / size(A_train_bar,1);
    
    % Pas de gradient descent pentru x
    pas_x = 0.1; % learning rate mic
    x_alt = x_alt - pas_x * grad_x;
    
    % Fixam x si optimizam X (stratul ascuns)
    % recalculam predictii cu noul x
    y_pred = 1 ./ (1 + exp(-H * x_alt));
    error = y_pred - e_train;
    delta = error .* (y_pred .* (1 - y_pred));
    
    % Derivata activarii Sinp
    dH = cos(Z) - a_activare;
    
    % Gradient fata de X
    grad_X = (A_train_bar' * (delta .* dH)) / size(A_train_bar,1);
    
    % Pas de gradient descent pentru X
    pas_X = 0.01; % learning rate mic
    X_alt = X_alt - pas_X * grad_X;
    
    % Update parametri concatenati
    params_alt = [X_alt(:); x_alt];
    
    % Salvam loss si norma gradientului
    [loss, grad] = calc_loss_grad(A_train_bar, e_train, params_alt, n, m, a_activare);
    loss_history_alt(iter_alt) = loss;
    grad_norm_history_alt(iter_alt) = norm(grad);
    
    % Conditie oprire
    if norm(grad) < toleranta
        fprintf('Convergenta Alternativa atinsa la iteratia %d.\n', iter_alt);
        loss_history_alt = loss_history_alt(1:iter_alt);
        grad_norm_history_alt = grad_norm_history_alt(1:iter_alt);
        break;
    end
end
time_alt = toc;
num_iter_alt = iter_alt;
fprintf('Timp de executie Metoda de Optimizare Alternativa: %.2f secunde\n', time_alt);

% Evaluare
% Predictii pe test
X_quasi = reshape(params_quasi(1:(n+1)*m), n+1, m); % matrice pentru startul ascuns
x_quasi = params_quasi((n+1)*m + 1:end); % vector pentru stratul de iesire
y_pred_quasi = predictie(A_test_bar, X_quasi, x_quasi, a_activare);

X_lm = reshape(params_lm(1:(n+1)*m), n+1, m); % matrice pentru startul ascuns
x_lm = params_lm((n+1)*m + 1:end); % vector pentru stratul de iesire
y_pred_lm = predictie(A_test_bar, X_lm, x_lm, a_activare);

X_alt = reshape(params_alt(1:(n+1)*m), n+1, m); % matrice pentru startul ascuns
x_alt = params_alt((n+1)*m+1:end); % vector pentru stratul de iesire
y_pred_alt = predictie(A_test_bar, X_alt, x_alt, a_activare);

% Clasificare
class_quasi = double(y_pred_quasi >= 0.5);
class_lm = double(y_pred_lm >= 0.5);
class_alt = double(y_pred_alt >= 0.5);

% Matrice de confuzie
fprintf('Matrice confuzie - Quasi-Newton:\n');
confmat_quasi = confusionmat(e_test, class_quasi);
disp(confmat_quasi);

fprintf('Matrice confuzie - Levenberg-Marquardt:\n');
confmat_lm = confusionmat(e_test, class_lm);
disp(confmat_lm);

fprintf('Matrice confuzie - Optimizare Alternativa:\n');
confmat_alt = confusionmat(e_test, class_alt);
disp(confmat_alt);

% Quasi-Newton
[sens_quasi, spec_quasi, prec_quasi, vpn_quasi, f1_quasi, accuracy_quasi] = calculeaza_metrici(confmat_quasi);

fprintf('\nMetrici detaliate Quasi-Newton:\n');
fprintf('Sensibilitate: %.2f%%\n', sens_quasi*100);
fprintf('Specificitate: %.2f%%\n', spec_quasi*100);
fprintf('Precizie: %.2f%%\n', prec_quasi*100);
fprintf('Valoare Predictiva Negativa: %.2f%%\n', vpn_quasi*100);
fprintf('F1 Score: %.2f%%\n', f1_quasi*100);
fprintf('Acuratete Quasi-Newton: %.2f%%\n', accuracy_quasi*100);

% Levenberg-Marquardt
[sens_lm, spec_lm, prec_lm, vpn_lm, f1_lm, accuracy_lm] = calculeaza_metrici(confmat_lm);

fprintf('\nMetrici detaliate Levenberg-Marquardt:\n');
fprintf('Sensibilitate: %.2f%%\n', sens_lm*100);
fprintf('Specificitate: %.2f%%\n', spec_lm*100);
fprintf('Precizie: %.2f%%\n', prec_lm*100);
fprintf('Valoare Predictiva Negativa: %.2f%%\n', vpn_lm*100);
fprintf('F1 Score: %.2f%%\n', f1_lm*100);
fprintf('Acuratete Levenberg-Marquardt: %.2f%%\n', accuracy_lm*100);

% Metoda de Optimizare Alternativa
[sens_alt, spec_alt, prec_alt, vpn_alt, f1_alt, accuracy_alt] = calculeaza_metrici(confmat_alt);

fprintf('\nMetrici detaliate Metoda de Optimizare Alternativa:\n');
fprintf('Sensibilitate: %.2f%%\n', sens_alt*100);
fprintf('Specificitate: %.2f%%\n', spec_alt*100);
fprintf('Precizie: %.2f%%\n', prec_alt*100);
fprintf('Valoare Predictiva Negativa: %.2f%%\n', vpn_alt*100);
fprintf('F1 Score: %.2f%%\n', f1_alt*100);
fprintf('Acuratete Metoda de Optimizare Alternativa: %.2f%%\n', accuracy_alt*100);
% Plotarea evolutiei functiei de cost (loss) pentru fiecare metoda

% Plot loss pentru Quasi-Newton
figure;
plot(1:iter_quasi, loss_history_quasi(1:iter_quasi), 'LineWidth', 2);
title('Evolutia Loss - Quasi-Newton');
xlabel('Iteratie');
ylabel('Loss');
grid on;
saveas(gcf, 'Evolutia Loss - Quasi-Newton.png')

% Plot loss pentru Levenberg-Marquardt
figure;
plot(1:iter_lm, loss_history_lm(1:iter_lm), 'LineWidth', 2);
title('Evolutia Loss - Levenberg-Marquardt');
xlabel('Iteratie');
ylabel('Loss');
grid on;
saveas(gcf, 'Evolutia Loss - Levenberg-Marquardt.png')

% Plot loss pentru Metoda de Optimizare Alternativa
figure;
plot(1:iter_alt, loss_history_alt(1:iter_alt), 'LineWidth', 2);
title('Evolutia Loss - Metoda de Optimizare Alternativa');
xlabel('Iteratie');
ylabel('Loss');
grid on;
saveas(gcf, 'Evolutia Loss - Metoda de Optimizare Alternativa.png')

% Histograma scorurilor predictiei

% Histograma predictii Quasi-Newton
figure;
histogram(y_pred_quasi, 20);
title('Histograma Predictii - Quasi-Newton');
xlabel('Probabilitate');
ylabel('Frecventa');
grid on;
saveas(gcf, 'Histograma Predictii - Quasi-Newton.png')

% Histograma predictii Levenberg-Marquardt
figure;
histogram(y_pred_lm, 20);
title('Histograma Predictii - Levenberg-Marquardt');
xlabel('Probabilitate');
ylabel('Frecventa');
grid on;
saveas(gcf, 'Histograma Predictii - Levenberg-Marquardt.png')

% Histograma predictii Metoda de Optimizare Alternativa
figure;
histogram(y_pred_alt, 20);
title('Histograma Predictii - Metoda de Optimizare Alternativa');
xlabel('Probabilitate');
ylabel('Frecventa');
grid on;
saveas(gcf, 'Histograma Predictii - Metoda de Optimizare Alternativa.png')