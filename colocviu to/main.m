clear; clc;

% -------------------- 1. Încarcare date --------------------
data = readtable('generatoare_5000.csv');
a = data.a_i;
b = data.b_i;
Pmin = data.pmin;
Pmax = data.pmax;
n = height(data);  % numar generatoare

Q = diag(2 * b);
q = a;

Pd = sum(Pmin) + 0.5 * sum(Pmax - Pmin);  % cerere totala (aleasa în intervalul posibil)

% ============================================================
% -------------------- METODA 1: GRADIENT PROIECTAT ----------
% ============================================================
tic
x_pg = (Pmin + Pmax) / 2;  % Initializare
alpha = 0.001;
max_iter = 5000;
eps = 1e-4;

for k = 1:max_iter
    grad = Q * x_pg + q;
    x_new = x_pg - alpha * grad;
    
    % Proiectie pe domeniul de tip cutie
    x_new = min(max(x_new, Pmin), Pmax);
    
    % Ajustare pentru a respecta cererea totala
    delta = Pd - sum(x_new);
    x_new = x_new + delta / n;
    
    if norm(x_new - x_pg, 2) < eps
        break;
    end
    
    x_pg = x_new;
end

cost_pg = 0.5 * x_pg' * Q * x_pg + q' * x_pg;
time_pg=toc;
% ============================================================
% -------------------- METODA 2: LAGRANGE-NEWTON -------------
% ============================================================
% Initializare
tic
x_ln = (Pmin + Pmax) / 2;
mu = 0;
tol = 1e-6;
max_iter = 50;

for k = 1:max_iter
    grad_f = Q * x_ln + q;
    hess_f = Q;
    h_val = sum(x_ln) - Pd;
    jac_h = ones(1, n);  % Jacobianul lui h(x) = sum(x) - Pd

    grad_L = grad_f + jac_h' * mu;
    KKT = [hess_f, jac_h'; jac_h, 0];
    rhs = [-grad_L; -h_val];

    delta = KKT \ rhs;
    dx = delta(1:n);
    dmu = delta(end);

    x_ln = x_ln + dx;

    % Proiectie pe limite (cutie)
    x_ln = min(max(x_ln, Pmin), Pmax);

    % Corectare cerere (sum x = Pd)
    delta_power = Pd - sum(x_ln);
    x_ln = x_ln + delta_power / n;

    % Reimpunere limite dupa corectare
    x_ln = min(max(x_ln, Pmin), Pmax);

    mu = mu + dmu;

    % Convergenta
    if norm(dx) < tol && abs(h_val) < tol
        break;
    end
end

% Calcul cost final
cost_ln = 0.5 * x_ln' * Q * x_ln + q' * x_ln;
time_ln=toc;
% ============================================================
% -------------------- METODA 3: CVX -------------------------
% ============================================================
tic
cvx_begin quiet
    variable x(n)
    minimize( 0.5 * quad_form(x, Q) + q' * x )
    subject to
        sum(x) == Pd
        Pmin <= x <= Pmax
cvx_end
x_cvx=x;
cost_cvx = 0.5 * x' * Q * x + q' * x;
time_cvx=toc;

% ============================================================
% -------------------- METODA 4: FUNCTIE MATLAB --------------
% ============================================================
tic
Aeq = ones(1, n);
beq = Pd;

opts = optimoptions('quadprog','Display','off');
[x_qp, cost_qp] = quadprog(Q, q, [], [], Aeq, beq, Pmin, Pmax, [], opts);
time_qp=toc;
% ============================================================
% -------------------- COMPARATIE SI GRAFICE -----------------
% ============================================================

fprintf('\n--- Gradient Proiectat ---\n');
fprintf('Cost total: %.2f\n', cost_pg);
fprintf('Putere totala generata: %.2f\n', sum(x_pg));

fprintf('\n--- Lagrange-Newton ---\n');
fprintf('Cost total: %.2f\n', cost_ln);
fprintf('Putere totala generata: %.2f\n', sum(x_ln));

fprintf('\n--- Soluție CVX ---\n');
fprintf('Cost total: %.2f\n', cost_cvx);
fprintf('Putere totală generată: %.2f\n', sum(x));

fprintf('\n--- Rezolvare cu quadprog ---\n');
fprintf('Cost total: %.2f\n', cost_qp);
fprintf('Putere totală generată: %.2f\n', sum(x_qp));

fprintf('\nTimp de executie Metoda Gradient Proiectat: \n: %.2f secunde\n', time_pg);
fprintf('\nTimp de executie Metoda Lagrange-Newton: %.2f secunde\n', time_ln);
fprintf('\nTimp de executie CVX: %.2f secunde\n', time_cvx);
fprintf('\nTimp de executie Functie Matlab: %.2f secunde\n', time_qp);



% Histograme ale puterii generate
figure;
histogram(x_pg);
title('Distributie putere - Gradient Proiectat');
xlabel('Putere generata (MW)'); ylabel('Numar generatoare');
saveas(gcf, 'Distributie putere - Gradient Proiectat.png')

figure;
histogram(x_ln);
title('Distributie putere - Lagrange-Newton');
xlabel('Putere generata (MW)'); ylabel('Numar generatoare');
saveas(gcf, 'Distributie putere - Lagrange-Newton.png')

figure;
histogram(x_cvx);
title('Distributie putere - CVX');
xlabel('Putere generata (MW)'); ylabel('Numar generatoare');
saveas(gcf, 'Distributie putere - CVX.png')

figure;
histogram(x_qp);
title('Distributie putere - Quadprog');
xlabel('Putere generata (MW)'); ylabel('Numar generatoare');
saveas(gcf, 'Distributie putere - Quadprog.png')


% Comparare costuri
figure;
bar([cost_pg, cost_ln]);
set(gca, 'XTickLabel', {'Grad. Proiectat', 'Lagrange-Newton'});
ylabel('Cost total');
title('Comparatie costuri intre metode');
saveas(gcf, 'Comparatie costuri între metode.png')

% 1. Scatter plot: cost individual per generator
cost_pg_i = Q*(x_pg.^2) + q.*x_pg;
cost_ln_i = Q*(x_ln.^2) + q.*x_ln;

figure;
plot(1:n, cost_pg_i, '.', 'DisplayName', 'PG'); hold on;
plot(1:n, cost_ln_i, '.', 'DisplayName', 'LN');
legend; xlabel('Generator'); ylabel('Cost individual');
title('Cost individual per generator');
saveas(gcf, 'Cost individual per generator.png')
% 2. Boxplot: distributia puterii generate
figure;
boxplot([x_pg, x_ln], 'Labels', {'Grad. Proiectat', 'Lagrange-Newton'});
ylabel('Putere generata (MW)');
title('Compararea dispersiei între metode');
saveas(gcf, 'Compararea dispersiei între metode.png')
% 3. Eroare fata de cererea totala
eroare_pg = abs(sum(x_pg) - Pd);
eroare_ln = abs(sum(x_ln) - Pd);
figure;
bar([eroare_pg, eroare_ln]);
set(gca,'XTickLabel',{'Grad. Proiectat','Lagrange-Newton'});
ylabel('Eroare absoluta (MW)');
title('Respectarea cererii totale');
saveas(gcf, 'Respectarea cererii totale.png')
% 4. Diferenta între solutii
figure;
plot(x_pg - x_ln);
xlabel('Generator'); ylabel('Diferenta PG - LN');
title('Diferenta solutii între metode');
saveas(gcf, 'Diferenta solutii între metode.png')
% 5. Cost individual: comparatie PG vs LN (linie)
figure;
plot(1:n, cost_pg_i, 'b-', 'DisplayName', 'PG'); hold on;
plot(1:n, cost_ln_i, 'r--', 'DisplayName', 'LN');
legend; xlabel('Generator'); ylabel('Cost');
title('Comparatie cost individual PG vs LN');
saveas(gcf, 'Comparatie cost individual PG vs LN.png')
% 6. Histograma comparativa (suprapusa)
figure;
histogram(x_pg, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
histogram(x_ln, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
legend('PG', 'LN');
xlabel('Putere generata (MW)'); ylabel('Numar generatoare');
title('Histograme suprapuse - PG vs LN');
saveas(gcf, 'Histograme suprapuse - PG vs LN.png')

