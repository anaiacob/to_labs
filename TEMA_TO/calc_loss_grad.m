function [loss, grad] = calc_loss_grad(A_bias, e, params, n, m, a)
    % calc_loss_grad - Calculeaza loss-ul si gradientul pentru antrenare
    %
    % Input:
    %   A_bias - datele de intrare cu bias
    %   e - etichete reale
    %   params - vector parametri (X si x concatenat)
    %   n - numar caracteristici
    %   m - numar neuroni strat ascuns
    %   a - parametru activare Sinp
    % Output:
    %   loss - valoarea functiei de pierdere
    %   grad - vector gradient
    
    % Extragem X si x
    X = reshape(params(1:(n+1)*m), n+1, m);
    x = params((n+1)*m + 1:end);
    
    Z = A_bias * X;% intrari strat ascuns
    H = functie_activare_sinp(Z, a); % activare strat ascuns
    y = 1 ./ (1 + exp(-H*x)); % iesire strat iesire (sigmoid)
    
    % Loss (entropie incrucisata)
    loss = -mean(e .* log(y + eps) + (1 - e) .* log(1 - y + eps));
    
    error = y - e; % eroare finala
    delta_output = error .* (y .* (1 - y)); % derivata sigmoid
    
    dSinp = cos(Z) - a; % derivata Sinp(z)
    
    % Gradient fata de ponderile stratului ascuns
    delta_hidden = (delta_output * x') .* dSinp;
    
    grad_X = (A_bias' * delta_hidden) / size(A_bias,1);
    grad_x = (H' * delta_output) / size(A_bias,1);
    
    grad = [grad_X(:); grad_x];

end
