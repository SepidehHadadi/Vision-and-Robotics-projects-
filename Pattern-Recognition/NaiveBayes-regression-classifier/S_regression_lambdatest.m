
%% initialize and load the original data
load('spamData.mat');


%% proprocessing step; calculate features
X_norm = featureNormalize(Xtrain);
X_log = log10(Xtrain + 0.1); 
X_bin = Xtrain > 0;
Y_bin_train = ytrain;

X_t_norm = featureNormalize(Xtest);
X_t_log = log10(Xtest + 0.1); 
X_t_bin = Xtest > 0;
Y_t_bin_test = ytest;
Lambda_vector = 0:1:30;
%% calcuate lambda for a given set of train data and plot the results
% follwoing steps are performed:
% the feature are selected and the x and y for the regression are assigned
% for a given lambda the values of the error is calcuated
% the result is ploted 
% value of the lambda is returned
% [Lambda_Xtrain_n, Err_train_Xtrain_n, Err_test_Xtrain_n] = calculateLambda(X_norm, ytrain, Lambda_vector);
% [Lambda_Xtrain_l, Err_train_Xtrain_l, Err_test_Xtrain_l] = calculateLambda(X_log, ytrain, Lambda_vector);
% [Lambda_Xtrain_b, Err_train_Xtrain_b, Err_test_Xtrain_b] = calculateLambda(X_bin, ytrain, Lambda_vector);
% [Lambda_Xtrain, Err_train_Xtrain, Err_test_Xtrain] = ChooseLambda(Xtrain, ytrain, Lambda_vector);

%% calcuate lambda for a given set of test data and plot the results
% follwoing steps are performed:
% the feature are selected and the x and y for the regression are assigned
% for a given lambda the values of the error is calcuated
% the result is ploted 
% value of the lambda is returned
[Lambda_Xtest_n, Err_train_Xtrain_t_n, Err_test_Xtrain_t_n] = calculateLambda(X_t_norm, ytest, Lambda_vector);
[Lambda_Xtest_l, Err_train_Xtrain_t_l, Err_test_Xtrain_t_l] = calculateLambda(X_t_log, ytest, Lambda_vector);
[Lambda_Xtest_b, Err_train_Xtrain_t_b, Err_test_Xtrain_t_b] = calculateLambda(X_t_bin, ytest, Lambda_vector);
[Lambda_Xtest, Err_train_Xtrain_t, Err_test_Xtrain_t] = calculateLambda(Xtest, ytest, Lambda_vector);

