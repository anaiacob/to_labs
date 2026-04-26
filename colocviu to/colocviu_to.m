clear; clc;

% -------------------- 1. Încărcare date --------------------
data = readtable('generatoare_5000.csv');
a = data.a_i;
b = data.b_i;
Pmin = data.pmin;
Pmax = data.pmax;
n = height(data);

Q = diag(2 * b);
q = a;
Pd = sum(Pmin) + 0.5 * sum(Pmax - Pmin);

% ------------------------------------------------------------
% ------------ METODA 1: GRADIENT PROIECTAT ------------------
% ------------------------------------------------------------
tic
x_pg = (Pmin + Pmax) / 2;
alpha = 0.001;
max_iter = 5000;
eps = 1e-4;

for k = 1:max_iter
    grad = Q * x_pg + q;
    x_new = x_pg - alpha * grad;

    x_new = min(max(x_new, Pmin), Pmax);  % proiecție pe interval
    delta = Pd - sum(x_new);
    x_new = x_new + delta / n;
    x_new = min(max(x_new, Pmin), Pmax);  % re-proiecție

    if norm(x_new - x_pg, 2) < eps
        break;
    end
    x_pg = x_new;
end

cost_pg = 0.5 * x_pg' * Q * x_pg + q' * x_pg;
toc
% ------------------------------------------------------------
% -------- METODA 2: PUNCT INTERIOR (conform curs) -----------
% ------------------------------------------------------------
tic
x_pi = (Pmin + Pmax) / 2;
tau = 1;
mu = 10;
eps = 1e-6;
max_iter_pi = 50;

% Constrângeri: Aeq x = beq, Cx <= d
Aeq = ones(1, n);
beq = Pd;
C = [eye(n); -eye(n)];
d = [Pmax; -Pmin];

for k = 1:max_iter_pi
    phi = 0.5 * x_pi' * Q * x_pi + q' * x_pi - (1 / tau) * sum(log(d - C * x_pi));
    grad_phi = Q * x_pi + q + (1 / tau) * (C' * (1 ./ (d - C * x_pi)));
    D = diag((1 ./ (d - C * x_pi)).^2);
    H = Q + (1 / tau) * (C' * D * C);

    % Sistemul KKT
    KKT = [H, Aeq'; Aeq, 0];
    rhs = [-grad_phi; -(Aeq * x_pi - beq)];
    delta = KKT \ rhs;
    dx = delta(1:n);
    dl = delta(end);

    % Line search pentru menținere în domeniu
    t = 1;
    while any(C * (x_pi + t * dx) >= d)
        t = 0.9 * t;
    end

    % Actualizare
    x_pi = x_pi + t * dx;
    tau = mu * tau;

    if norm(dx, 2) < eps
        break;
    end
end

cost_pi = 0.5 * x_pi' * Q * x_pi + q' * x_pi;
toc
% -------------------- CVX --------------------
tic
cvx_begin quiet
    variable x(n)
    minimize( 0.5 * quad_form(x, Q) + q' * x )
    subject to
        sum(x) == Pd
        Pmin <= x <= Pmax
cvx_end

cost_cvx = 0.5 * x' * Q * x + q' * x;
toc
% ------------------------------------------------------------
% -------------------- COMPARAȚIE ----------------------------
% ------------------------------------------------------------

fprintf('\n--- Gradient Proiectat ---\n');
fprintf('Cost total: %.2f\n', cost_pg);
fprintf('Putere totală: %.2f\n', sum(x_pg));

fprintf('\n--- Punct Interior ---\n');
fprintf('Cost total: %.2f\n', cost_pi);
fprintf('Putere totală: %.2f\n', sum(x_pi));

fprintf('\n--- Soluție CVX ---\n');
fprintf('Cost total: %.2f\n', cost_cvx);
fprintf('Putere totală generată: %.2f\n', sum(x));
% ------------------------------------------------------------
% -------------------- GRAFICE OPȚIONALE ---------------------
% ------------------------------------------------------------

figure;
bar([cost_pg, cost_pi]);
set(gca, 'XTickLabel', {'Grad. Proiectat', 'Punct Interior'});
ylabel('Cost total');
title('Comparație costuri');

figure;
histogram(x_pg, 50, 'FaceAlpha', 0.5); hold on;
histogram(x_pi, 50, 'FaceAlpha', 0.5);
legend('Gradient Proiectat', 'Punct Interior');
xlabel('Putere generată (MW)');
ylabel('Număr generatoare');
title('Distribuție puteri generate');
