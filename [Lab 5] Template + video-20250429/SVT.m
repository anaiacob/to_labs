function [B] = SVT(A, alpha)
    [U, sigma, V] = svd(A, "econ");

    sigma = sigma -alpha*eye(size(sigma));

    i=1;
    while i<=size(sigma,1)
        if sigma(i)<0
            sigma(i)=0;
        end
        i=i+1;
    end

    B=U*sigma*V';
end