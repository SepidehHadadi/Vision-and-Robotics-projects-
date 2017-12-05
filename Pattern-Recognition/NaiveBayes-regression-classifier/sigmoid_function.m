function [sigma] = sigmoid_function(a)
    sigma = 1 ./ (1 + exp(-a));
end
