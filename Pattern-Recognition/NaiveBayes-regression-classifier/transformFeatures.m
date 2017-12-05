function phi_X = transformFeatures(X)

%-------------------------------------------------------------%
% calculation of Phi to transform X into polynomial terms     %
%-------------------------------------------------------------%


% We regrouped the similar features together: the features 1 to 49, the
% features 49 to 54, the feature 55 and the feature 56. 


% Phi_X = [ 1                                                  1                          ...     1 
%          ||X_test1(1)||  + ||X_test2(1)||  + ... + ||X_test48(1)||      X_test1(2)     ...     X_test1(3065)
%          ||X_test49(1)|| + ||X_test50(1)|| + ... + ||X_test54(1)||         ...
%            X_test55(1)                                                  X_test55(2)    ...     X_test55(3065)
%            X_test56(1)                                                  X_test56(2)    ...     X_test56(3065)    
%            X_test57(1)                                                  X_test57(2)    ...     X_test57(3065)  
       

X = X';
phi_X = ones(6, size(X,2));



for i = 1:48
    phi_X(2, :) = phi_X(2, :) + sqrt(X(i, :).^2);
end


for i = 49:54
    phi_X(3, :) = phi_X(3, :) + sqrt(X(i, :).^2);
end


 phi_X(4, :) = X(55, :);

 phi_X(5, :) = X(56, :);
 
 phi_X(6, :) = X(57, :);
 