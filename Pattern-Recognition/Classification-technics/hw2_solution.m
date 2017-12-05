clear all
close all
clc

load('spamData.mat')

%% feature identification
X1=Xtrain(:,55);%average length
X2=Xtrain(:,56);%length

max_ave_length = max(X1);
mean_ave_length = mean(X1);

max_long_length = max(X2);
mean_long_length = mean(X2);

%% preprossing of data
% feature normalization
meanT=mean(Xtrain);
stdT=std(Xtrain);
Xtrain_Normalize = [];
y=[]
for i=1:size(Xtrain,2)
    y=[y meanT(i)*ones(size(Xtrain,1),1)];
end
z=Xtrain-y;
for i=1:size(Xtrain,2)
    Xtrain_Normalize=[Xtrain_Normalize z(:,i)/stdT(i)];
end
%feature transformation
Xtrain_log=[];
Xtrain_log = log(Xtrain+0.1);

%binary feature
Xtrain_binary = Xtrain;
Xtrain_binary(Xtrain_binary>0)=1;
Xtrain_binary(Xtrain_binary<=0)=0;

%% classifier and classification
betaHatNom = mnrfit(ytrain,Xtrain,'model','nominal','interactions','on');
figure
xx = linspace(0,1)';
pHatNom = mnrval(betaHatNom,xx,'model','nominal','interactions','on');
line(xx,cumsum(25*pHatNom,2),'LineWidth',2);
%binary
betaHatNom = mnrfit(ytrain,Xtrain_binary,'model','nominal','interactions','on');
figure
xx = linspace(0,1)';
pHatNom = mnrval(betaHatNom,xx,'model','nominal','interactions','on');
line(xx,cumsum(25*pHatNom,2),'LineWidth',2);
%log
betaHatNom = mnrfit(ytrain,Xtrain_log,'model','nominal','interactions','on');
figure
xx = linspace(0,1)';
pHatNom = mnrval(betaHatNom,xx,'model','nominal','interactions','on');
line(xx,cumsum(25*pHatNom,2),'LineWidth',2);
%normalized
betaHatNom = mnrfit(ytrain,Xtrain_Normalize,'model','nominal','interactions','on');
figure
xx = linspace(0,1)';
pHatNom = mnrval(betaHatNom,xx,'model','nominal','interactions','on');
line(xx,cumsum(25*pHatNom,2),'LineWidth',2);

%% selection of the regularization parameter
load('regularazation_para.mat');

s_regression_lambdatest

w_init = zeros(6,1);
  
% model for training data
transformed_Xtest = transformFeatures(Xtest);
options = optimset('GradObj', 'on', 'MaxIter', 400);
t = cputime;   % This is to compute the time to get the model. 
[w_Xtrain, ~] = fminunc( @(w)(costFunction_plus_Regularization(Xtest, transformed_Xtest, ytest, w, 12)), w_init, options );
tcal= cputime-t; % e is the time to get the model w_Xtrain. 
y_est_Xtrain = sqrt(((w_Xtrain(1:size(w_init,1))' * transformed_Xtest)' - ytest).^2);
y_est_Xtrain = OneOrZero(y_est_Xtrain);
diffAndtime_Xtrain = [sum(sqrt((y_est_Xtrain - ytest).^2)), tcal]; 
                                      


% model for normalized data
transformed_X_norm = transformFeatures(Xtrain_Normalize);
t = cputime;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[w_X_norm, ~] = fminunc( @(w)(costFunction_plus_Regularization(Xtrain_Normalize, transformed_X_norm, ytest, w, 17)), w_init, options );
tcal = cputime-t;

y_X_norm = sqrt(((w_X_norm(1:size(w_init,1))' * transformed_X_norm)' - ytest).^2);
y_X_norm = OneOrZero(y_X_norm);
diffAndTime_X_norm = [sum(sqrt((y_X_norm - ytest).^2)), tcal];


% model for log feature
transformed_X_log = transformFeatures(Xtrain_log);
t = cputime;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[w_X_log, ~] = fminunc( @(w)(costFunction_plus_Regularization(Xtrain_log, transformed_X_log, ytest, w, 7)), w_init, options );
tcal = cputime-t;

y_X_log = sqrt(((w_X_log(1:size(w_init,1))' * transformed_X_log)' - ytest).^2);
y_X_log = OneOrZero(y_X_log);
diffAndTime_X_log = [sum(sqrt((y_X_log - ytest).^2)), tcal]; 

% model for binary feature
transfered_X_bin = transformFeatures(Xtrain_binary);
t = cputime;
options = optimset('GradObj', 'on', 'MaxIter', 400);
[w_X_bin, ~] = fminunc( @(w)(costFunction_plus_Regularization(Xtrain_binary, transfered_X_bin, ytest, w, 3)), w_init, options );
tcal = cputime-t;

y_X_bin = sqrt(((w_X_bin(1:size(w_init,1))' * transfered_X_bin)' - ytest).^2);
y_X_bin = OneOrZero(y_X_bin);
diffAndTime_X_bin = [sum(sqrt((y_X_bin - ytest).^2)), tcal];


diffAndTimeAndErr_All = [diffAndtime_Xtrain Err_train_Xtrain Err_test_Xtrain; diffAndTime_X_norm Err_train_X_norm Err_test_X_norm; diffAndTime_X_log Err_train_X_log Err_test_X_log; diffAndTime_X_bin Err_train_X_bin Err_test_X_bin];

%% naive baysian
[W, W0] = Naive_Bayes(Xtrain, ytrain);
  
 
 y_est_NaiveBayes = W' * Xtest' + W0';
 y_est_NaiveBayes = OneOrZero(y_est_NaiveBayes);
 y_est_NaiveBayes = y_est_NaiveBayes';
 diff_NaiveBaye   = sum(sqrt((y_est_NaiveBayes - ytest).^2));
 
 
