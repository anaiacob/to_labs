function y = predictie(A_bias, X, x, a)
    % predictie - Functia de predictie a retelei
    % A_bias - intrari cu bias adaugat
    % X - ponderi strat ascuns
    % x - ponderi iesire
    % a - parametru activare Sinp
    
    Z = A_bias * X;
    H = functie_activare_sinp(Z, a);
    y = 1 ./ (1 + exp(-H*x)); % aplicam sigmoid la iesire

end
