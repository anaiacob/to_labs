function [sens, spec, prec, vpn, f1, accuracy] = calculeaza_metrici(confmat)
    TN = confmat(1,1);
    FP = confmat(1,2);
    FN = confmat(2,1);
    TP = confmat(2,2);
    epsilon =  1.0e-16; % Pentru a evita impartirea la 0
    
    sens = TP / (TP + FN + epsilon); % Sensibilitate
    spec = TN / (TN + FP + epsilon); % Specificitate
    prec = TP / (TP + FP + epsilon); % Precizie
    vpn = TN / (TN + FN + epsilon); % Valoare Predictiva Negativa
    f1 = 2 * (prec * sens) / (prec + sens + epsilon); % Scor F1
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon); % Acuratete
end