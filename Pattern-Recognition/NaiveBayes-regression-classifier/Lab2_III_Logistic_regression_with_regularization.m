
%-----------------------------------------------%
%                                               %
%  III. Logistic regression with regularization %
%                                               %
%-----------------------------------------------%



%---------------------%
%  Visualize the data %
%---------------------%


data2 =  importdata('spamTestLabels.txt');
X = data2(:,1:2);
y = data2(:,3);

X_Accepted = zeros(sum(y),2);
X_Refused = zeros(size(X,1) - sum(y),2);


index_X_Accepted = 1;
index_Y_Accepted = 1;

for i = 1:size(X,1)
   if y(i) == 1
       X_Accepted(index_X_Accepted,:) = X(i,:);
       index_X_Accepted = index_X_Accepted + 1;
   else
       X_Refused(index_Y_Accepted,:) = X(i,:);
       index_Y_Accepted = index_Y_Accepted + 1;
   end
end

figure(1);
plot(X_Accepted(:,1), X_Accepted(:,2), '+');
hold on;
plot(X_Refused(:,1), X_Refused(:,2), 'o');
title('Our data, + for microchips accepted, o for microchips refused');
xlabel('First test');
ylabel('Second test');



%-----------------------------------------------------------------------------------%
%  Map the features into all polynomial terms of X1 and X2 up to the sixth power.   %
%-----------------------------------------------------------------------------------%



phi_X = transformFeatures(X);



%-----------------------------------------------------------------------------------%
%   Use regularization to ovoid overfitting.                                         %
%-----------------------------------------------------------------------------------%


% Note that while the feature mapping allows us to build a more expressive classifier, it is also more
% susceptible to overfitting. So we use regularization to ovoid overfitting.


Order = 5;
w_init = zeros((Order+1)*3,1);
%Lambda = 1;
Lambda = 0.000417;
% Take the smaller Lambda such as there is only one circle,
% More Lambda is small, more the circle is small,
% and more we avoid outliers, more we have a boundary more secure. 

% Question: What values of ? help prevent from overfitting: small or large values?
% Answer:   Large values. 

options = optimset('GradObj', 'on', 'MaxIter', 400);
[w_poly, cost] = fminunc( @(w)(costFunction_plus_Regularization(X, phi_X,y,w,Lambda)), w_init, options ); 






%------------------------------------------------------%
%  Plot the polynome                                   %
%------------------------------------------------------%



step1 = ceil(max(X(:,1)))/100;
step2 = ceil(max(X(:,2)))/100;
u =  min(X(:,1));
v =  min(X(:,2));

P_array = zeros(100);
Save_coordinate = [];
terms = zeros(Order + 1, 1);
k = 1;
for i = 1 : 100
    for j = 1 : 100
        terms(1) = w_poly(1)        + w_poly(2)  * u       + w_poly(3)  * v ;
        terms(2) = w_poly(4)  * u^2 + w_poly(5)  * u * v   + w_poly(6)  * v^2;
        terms(3) = w_poly(7)  * u^3 + w_poly(8)  * u * v^2 + w_poly(9)  * v^3;
        terms(4) = w_poly(10) * u^4 + w_poly(11) * u * v^3 + w_poly(12) * v^4; 
        terms(5) = w_poly(13) * u^5 + w_poly(14) * u * v^4 + w_poly(15) * v^5; 
        terms(6) = w_poly(16) * u^6 + w_poly(17) * u * v^5 + w_poly(18) * v^6; 
        P_array(i,j) = sum(terms); 
        u = u + step1;
        if ( ( -1 < P_array(i,j) ) && ( P_array(i,j) < 0) )
            Save_coordinate(k,:) = [u v];
            k = k + 1;
        end
    end
    v = v + step2;
    u = min(X(:,1));
end



X1 = Save_coordinate(:,1);
X2 = Save_coordinate(:,2);


figure(2);
plot(X_Accepted(:,1), X_Accepted(:,2), '+');
hold on;
plot(X_Refused(:,1), X_Refused(:,2), 'o');

hold on;
plot(X1, X2, '*');

title('Polynomial Regression with Regularization');
xlabel('First test');
ylabel('Second test');








