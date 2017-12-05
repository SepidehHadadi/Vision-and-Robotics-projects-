
%~~~~~~~~~~~~~~~~~~%
% READ THE DATA    %
%~~~~~~~~~~~~~~~~~~%


load('spamData.mat');


Feature_55 = Xtrain(:,55);

Feature_55_max  = max(Feature_55);
Feature_55_mean = mean(Feature_55);


Feature_56 = Xtrain(:,56);

Feature_56_max  = max(Feature_56);
Feature_56_mean = mean(Feature_56);





%~~~~~~~~~~~~~~~~~~%
% PREPROCESSING    %
%~~~~~~~~~~~~~~~~~~%


%---------------------------------------------------------------------
% a) Standardize the columns so they all have mean 0 and unit variance
%---------------------------------------------------------------------


X_norm = featureNormalize(Xtest); % t like transformed


%---------------------------------------------------------------------
% b) Transform the features using log
%---------------------------------------------------------------------

% Differentiate more small values and less big values 

X_log = log10(Xtest + 0.1); % Matlab function log: ln
                             % Matlab function log10: ln/ln(10)

                                  
                                  
%---------------------------------------------------------------------
% (c) Binarize the features 
%---------------------------------------------------------------------

X_bin = Xtest > 0;





%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% CHOOSE THE PREPROCESSING METHOD AND CHOOSE LAMBDA  %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                          %
% We want to estimate y knowing x by using a logistic regression model.    %
% A logistic regression model is also called a discriminative model.       %
% We optimize the function costFunction_plus_Regularization to get this    %
% model, or W, such as: y = W'phi; phi is an organization of x.            %
%                                                                          %
% This model has a regularization parameter called Lambda. We used the     %
% cross validation method to find the "best" Lambda by using the function  %
% ChooseLambda.                                                            %
%                                                                          %
% We got 4 different models: we got 3 models each of them from a different %
%                                                                          %
%                                                                          %
% preprocessing method and 1 model directly from the training data.        %
% We find the best preprocessing method by testing our models              %
% on the test data. We used the training data to compute the models.       %
%                                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%---------------------------------------------------------------------
% Choose Lambda 
%---------------------------------------------------------------------


% Load Lambda_file.mat instead to win time ( around 40 min)
load('Lambda_file.mat');
% The figures plotted during this code are shown in the report. 
% Lambda_vector = 0:1:30;
% [Lambda_Xtrain, Err_train_Xtrain, Err_test_Xtrain] = ChooseLambda(Xtrain, ytrain, Lambda_vector); 
% [Lambda_X_norm, Err_train_X_norm, Err_test_X_norm] = ChooseLambda(X_norm, ytrain, Lambda_vector);
% [Lambda_X_log, Err_train_X_log, Err_test_X_log]    = ChooseLambda(X_log, ytrain, Lambda_vector);
% [Lambda_X_bin, Err_train_X_bin, Err_test_X_bin]    = ChooseLambda(X_bin, ytrain, Lambda_vector);





%---------------------------------------------------------------------
% For each method, compute the error for the "best" Lambda found
%---------------------------------------------------------------------


% Err_train_something and Err_test_something are the RMS error for each 
% of the 57 features of x, for the training and test sets.

% ** Error for the training set and for all the methods **:
N = size(Err_train_Xtrain,2); 
Err_train_Xtrain = sum(Err_train_Xtrain)/N;

N = size(Err_train_X_norm,2);
Err_train_X_norm = sum(Err_train_X_norm)/N;

N = size(Err_train_X_log,2);
Err_train_X_log = sum(Err_train_X_log)/N;

N = size(Err_train_X_bin,2);
Err_train_X_bin = sum(Err_train_X_bin)/N;


% ** Error for the test set and for all the methods **:
N = size(Err_test_Xtrain,2);
Err_test_Xtrain = sum(Err_test_Xtrain)/N;

N = size(Err_test_X_norm,2);
Err_test_X_norm = sum(Err_test_X_norm)/N;

N = size(Err_test_X_log,2);
Err_test_X_log = sum(Err_test_X_log)/N;

N = size(Err_test_X_bin,2);
Err_test_X_bin = sum(Err_test_X_bin)/N;




%---------------------------------------------------------------------
% Compute the 4 models with the "best" Lambda found
%---------------------------------------------------------------------

% We will compute the time comsumed to compute each models, and the
% number of errors for each models (Example: if it's 0 and it should be 
% one, then the number of errors = +1). 

  w_init = zeros(6,1);
  
  % ** Xtrain model ** 
  phi = transformFeatures(Xtest);
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  t = cputime;   % This is to compute the time to get the model. 
  [w_Xtrain, ~] = fminunc( @(w)(costFunction_plus_Regularization(Xtest, phi, ytest, w, 12)), w_init, options );
  e = cputime-t; % e is the time to get the model w_Xtrain. 
  
  y_est_Xtrain = sqrt(((w_Xtrain(1:size(w_init,1))' * phi)' - ytest).^2);
  y_est_Xtrain = OneOrZero(y_est_Xtrain); % OneOrZero(X): For each number of the vector X: 
                                          % it gives 0 if the number is nearer from 0 than to 1
                                          % and it gives 1 if the number is nearer from 1 than to 0. 
                                          
  diffAndtime_Xtrain = [sum(sqrt((y_est_Xtrain - ytest).^2)), e]; 
                                         % diffAndtime_Xtrain is an array composed of: 
                                         % - sum(sqrt((y_est_Xtrain - ytest).^2)), the number of errors.
                                         % - e, the time to get the model w_Xtrain. 
  
                                         
   % ** X_norm model ** 
  phi = transformFeatures(X_norm);
  t = cputime;
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [w_X_norm, ~] = fminunc( @(w)(costFunction_plus_Regularization(X_norm, phi, ytest, w, 17)), w_init, options );
  e = cputime-t;
  
  y_est_X_norm = sqrt(((w_X_norm(1:size(w_init,1))' * phi)' - ytest).^2);
  y_est_X_norm = OneOrZero(y_est_X_norm);
  diffAndTime_X_norm = [sum(sqrt((y_est_X_norm - ytest).^2)), e];




  % ** X_log model **
  phi = transformFeatures(X_log);
  t = cputime;
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [w_X_log, ~] = fminunc( @(w)(costFunction_plus_Regularization(X_log, phi, ytest, w, 7)), w_init, options );
  e = cputime-t;
  
  y_est_X_log = sqrt(((w_X_log(1:size(w_init,1))' * phi)' - ytest).^2);
  y_est_X_log = OneOrZero(y_est_X_log);
  diffAndTime_X_log = [sum(sqrt((y_est_X_log - ytest).^2)), e]; 




  % ** X_bin model **
  phi = transformFeatures(X_bin);
  t = cputime;
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  [w_X_bin, ~] = fminunc( @(w)(costFunction_plus_Regularization(X_bin, phi, ytest, w, 3)), w_init, options );
  e = cputime-t;
  
  y_est_X_bin = sqrt(((w_X_log(1:size(w_init,1))' * phi)' - ytest).^2);
  y_est_X_bin = OneOrZero(y_est_X_bin);
  diffAndTime_X_bin = [sum(sqrt((y_est_X_bin - ytest).^2)), e];
  
  
  
  
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% RESULTS OF THE COMPARAISON BETWEEN THE 4 MODELS WITH THE "BEST LAMBDA"  %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
  

  diffAndTimeAndErr_All = [diffAndtime_Xtrain Err_train_Xtrain Err_test_Xtrain; diffAndTime_X_norm Err_train_X_norm Err_test_X_norm; diffAndTime_X_log Err_train_X_log Err_test_X_log; diffAndTime_X_bin Err_train_X_bin Err_test_X_bin];
  % diffAndTimeAndErr_All is an array which summarizes all the work. 
  
  % Each row = one preprocessing method, in this order: Xtrain, X_norm, X_log, X_bin.
  % Columns: in order: 
  % Number of errors, time comsumed, RMS mean error on the training set, RMS mean error on the test set.

  
  
  
  
 %________________________________________________________________________%
  
  
  
 %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
 % COMPUTE THE MODEL WITH THE NAIVE BAYES METHOD                          %
 %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
 
 % The Naive Bayes classifier is a generative model.
 
 [W, W0] = Naive_Bayes(Xtrain, ytrain);
 
 %-------------------------------------%
 %      Prediction                     %
 %-------------------------------------%

 
 % C1 if y(x, w) >= 0, C2 otherwise. 
 
 y_est_NaiveBayes = W' * Xtest' + W0';
 y_est_NaiveBayes = OneOrZero(y_est_NaiveBayes);
 y_est_NaiveBayes = y_est_NaiveBayes';
 diff_NaiveBaye   = sum(sqrt((y_est_NaiveBayes - ytest).^2));
 
 
 