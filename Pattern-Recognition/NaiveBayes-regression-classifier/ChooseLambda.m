function [Lambda, E_RMS_training_mean, E_RMS_test_mean]  = ChooseLambda(X, Y, Lambda_vector)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% This function implements cross validation method to choose              %
% the "best model", the largest gap between training and test errors      %
% is the "best model".                                                    %
%                                                                         %
% The different models are a logistic regression with different Lambda,   %
% Lambda is the regularization parameter.                                 %
%                                                                         %
% We are using the RMS error to compute the error.                        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 


  %------------------------
  %         Data
  %------------------------
  
  Nb_Lambda = size(Lambda_vector,2);
  Nb_samples = size(X,1);
  X = X(1 : round( Nb_samples * 20/100 ) * 100/20, :); % Resize X to have groups of same size. 
                                                       % X was of size 4601  57, Now size(X) = 4600  57
                                                       % and 20% of 4600 = 920 == elements in each group.
  Nb_samples = size(X,1); % Update Nb_samples.                                                 
                                        
  Nb_elmts_Onegroup  =  Nb_samples * 20/100;            % Nb_elmts_Onegroup == 920.
  Nb_groups          =  Nb_samples / Nb_elmts_Onegroup; % Nb_groups == 5 here, indeed: 5 * 920 = 4600
  
  Nb_test         =  Nb_elmts_Onegroup;              % Nb_test == Nb elements in one group == 920.
  E_RMS_test      =  zeros(1, Nb_samples / Nb_test); % size of E_RMS_test ==  1  5
  E_RMS_test_mean =  zeros(1, Nb_Lambda);            % size of E_RMS_test_mean ==  1  Nb_Lambda
 
  Nb_training         =  Nb_samples - Nb_test;            % Nb_training == 4600 - 920 == 5 groups - 1 group.
  E_RMS_training      =  zeros(1, Nb_samples/Nb_test);    % size of E_RMS_training == 1  5
  E_RMS_training_mean =  zeros(1,  Nb_Lambda);            % size of E_RMS_training_mean ==  1  Nb_Lambda
  

  % Data to compute w for all different Lambda.
  w_init = zeros(6,1);
  phi = transformFeatures(X);
 
  
  %------------------------
  %         Loop
  %------------------------
  for L = 1 : Nb_Lambda
      
        % Compute w for all different Lambda:
        options = optimset('GradObj', 'on', 'MaxIter', 400);
        [w, ~] = fminunc( @(w)(costFunction_plus_Regularization(X, phi, Y, w, Lambda_vector(L))), w_init, options );
      
      
      for G = 1 : Nb_groups 

          
           % Take the test and training set and compute phi.
      
           training_x  =  X(Nb_test+1 : size(X,1), :);
           training_y  =  Y(Nb_test+1 : size(X,1), :);
           
           test_x      =  X(1:Nb_test, :);
           test_y      =  Y(1:Nb_test, :);
           
           phi_training = transformFeatures(training_x);
           phi_test = transformFeatures(test_x);
          
           
          % Calculate E_RMS_training for each w
          Sum = 0;

         for i = 1 : Nb_training
            Sum = Sum + (transpose(w) * phi_training(:,i) - training_y(i))^2 + Lambda_vector(L)/2 * (w') * w ;
         end 
            E_ofW_training = 1/2 * Sum;

            E_RMS_training(G) = sqrt(2 * E_ofW_training / Nb_training);


            % Calculate E_RMS_test for each w
            Sum = 0;
        
          for i = 1 : Nb_test 
            Sum = Sum + (transpose(w) * phi_test(:,i) - test_y(i))^2 + Lambda_vector(L)/2 * (w') * w;
          end 
            E_ofW_test = 1/2 * Sum;

            E_RMS_test(G) = sqrt(2 * E_ofW_test / Nb_test); 

            
            % Change the test and training test
            X =  circshift(X,[-Nb_elmts_Onegroup, -Nb_elmts_Onegroup]);  
            Y =  circshift(Y,[-Nb_elmts_Onegroup, -Nb_elmts_Onegroup]);      
      end
      % Calculate the mean of all the E_RMS_training and E_RMS_test for different Lambda
      E_RMS_training_mean(L) = mean(E_RMS_training);
      E_RMS_test_mean(L)     = mean(E_RMS_test);
      
  end
  
  
  %------------------------
  %         Plotting
  %------------------------
  
  
 
  x_axis = Lambda_vector;
  figure;
  plot(x_axis, E_RMS_training_mean,'--go');
  hold on; 
  plot(x_axis, E_RMS_test_mean,':r*');
  xlabel('Lambda')
  ylabel('E_R_M_S mean')
  str=sprintf('~~ChooseLambda~~ In green: training error, in red: test error');
  title(str);
  
 Vector = sqrt((E_RMS_training_mean - E_RMS_test_mean).^2);
 indexe = find(Vector == max(sqrt((E_RMS_training_mean - E_RMS_test_mean).^2)));
 Lambda = Lambda_vector(indexe);
  