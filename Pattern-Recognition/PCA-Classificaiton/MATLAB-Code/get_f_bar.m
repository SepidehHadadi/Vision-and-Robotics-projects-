function [ f_bar ] = get_f_bar( handles )

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Step 1: 
    % initialize f_bar to the first image that will be trained
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    f_bar = load(['images\train_images\' handles.txt_train_file_array(1).name]);
    f_pre = [13 20; 50 20; 34 34; 16 50; 48 50];

    threshold = 9.912;
    counter   = 0;
    error     = 20;
 
    while(counter < 30)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step 2:
        % Get A and b from Fpre = A * f_bar + b
        % Get Fbar = A * f_bar + b
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [A, b]  = get_Ab(f_pre, f_bar);
        f_bar   = feature_transformation(A, b, f_bar);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step 3:
        % Find fi_dash of all fi
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        f_sum = zeros(5, 2);
        
        for i = 1: length(handles.txt_train_file_array)
            
            fi      = load(['images\train_images\' handles.txt_train_file_array(i).name]);
            
            [A, b]  = get_Ab(f_bar, fi);
            
            fi_dash = feature_transformation(A, b, fi);
            
            f_sum   = f_sum + fi_dash;
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step 4:
        % Update new f_bar by averaging fi_dash
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        f_bar_t = f_sum / length(handles.txt_train_file_array);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Step 5:
        % error between f_bar_t and f_bar_(t - 1)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        error = sum(sqrt(sum(((f_bar_t' - f_bar').^2))));
        
        f_bar = f_bar_t;
        
        if(error < threshold)
            
            break;
            
        end
        
        counter = counter + 1;
        
    end

end

