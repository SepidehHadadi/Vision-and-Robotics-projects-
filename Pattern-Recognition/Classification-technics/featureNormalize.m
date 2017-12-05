
function [Xn] = featureNormalize(X)

    Xn = zeros(size(X,1), size(X,2));
    
    
    
    for i = 1:size(X,2)
        mu      = mean(X(:,i));
        sigma   = std(X(:,i));
        Xn(:,i) = (X(:,i) - mu) / sigma;
    end