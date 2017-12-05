function [related_image_1, related_image_2, related_image_3, related_image_name_1, related_image_name_2, related_image_name_3, error_1, error_2, error_3] = recognize_face(handles, original_image_name, f_bar)

%{
    %% WE DON'T NORMALIZE IMAGES AGAIN, THEY HAVE ALREADY NORMALIZED
	for index = 1: length(handles.jpg_train_image_array)
        
        original_image   = imread(['images\train_images\' handles.jpg_train_image_array(index).name()]);
        f_image          = load(['images\train_images\' handles.txt_train_file_array(index).name()]);

        %% LOOK CAREFULLY
        normalized_image = image_normalization(original_image, f_bar, f_image);
        
        imwrite(normalized_image, ['images\normalized_images\' handles.txt_train_file_array(index).name(1:end-4) '_normalized.jpg']);
        d_matrix(index, :) = reshape(normalized_image, 1, 4096);

	end

}







	d_matrix = [];

	for index = 1: length(handles.jpg_normalized_file_array)
        
       normalized_image = imread(['images\normalized_images\' handles.jpg_normalized_file_array(index).name()]);
        
       d_matrix = [d_matrix; reshape(normalized_image', 1, 4096)];
        
    end

    d_matrix = double(d_matrix);
    
    % compute the mean of the columns of d_matrix, this returns a row vector of means
    d_matrix_mean = mean(d_matrix);

    [p d] = size(d_matrix);

    d_matrix_norm = d_matrix;

    % subtract the mean from each row vector in d_matrix, call the new matrix d_matrix_norm
    for index = 1: p
        
        d_matrix_norm(index, :) = d_matrix(index, :) - d_matrix_mean;
        
    end
    
    % compute E'
    sigma_prime = (1 / (p - 1)) * (d_matrix_norm * d_matrix_norm');

    % choose proper k number
    k = 50;    
    
    % find its eigenvectors
    [ eigen_vectors eigen_values ] = eigs(sigma_prime, k);

    % phi' is equal to eigenvectors of E'
    phi_prime = eigen_vectors;

    % We can obtain phi, which consists of the eigenvalues of E, the covariance matrix, by multiplying phi' with d_matrix_norm' 
    phi   = d_matrix_norm' * phi_prime;
    phi_i = d_matrix * phi;
    phi_i = phi_i';

    
    
    
    
    
    
    original_image   = imread(['images\test_images\' original_image_name '.jpg']);
    f_image          = load(['images\test_images\' original_image_name '.txt']);
	normalized_image = image_normalization(original_image, f_bar, f_image);
	Xj               = double(reshape(normalized_image', 1, 4096));

    % find its feature vector by multiplying with Phi
    phi_j    = Xj * phi;
    phi_j    = phi_j';
    distance = [];

	for i = 1: length(handles.jpg_train_image_array)
        
        distance(i, 1) = i;
        
        distance(i, 2) = sqrt(sum((phi_j - phi_i(: , i)).^2));

    end
    
	distance = sortrows(distance, 2);

    
    
    
    
    
    
    related_image_1      = ['images\train_images\' handles.jpg_train_image_array(distance(1, 1)).name];
    related_image_2      = ['images\train_images\' handles.jpg_train_image_array(distance(2, 1)).name];
    related_image_3      = ['images\train_images\' handles.jpg_train_image_array(distance(3, 1)).name];
    related_image_name_1 = handles.jpg_train_image_array(distance(1, 1)).name(1: end - 6);
    related_image_name_2 = handles.jpg_train_image_array(distance(2, 1)).name(1: end - 6);
    related_image_name_3 = handles.jpg_train_image_array(distance(3, 1)).name(1: end - 6);


    if(~strcmp(original_image_name(1: end - 2), related_image_name_3))
        error_3 = 1;
    else
        error_3 = 0;
    end

    
    if(~strcmp(original_image_name(1: end - 2), related_image_name_2))
        error_2 = 1;
    else
        error_2 = 0;
        error_3 = 0;
    end
    

    if(~strcmp(original_image_name(1: end - 2), related_image_name_1))
        error_1 = 1;
    else
        error_1 = 0;
        error_2 = 0;
        error_3 = 0;
    end

    return;
    
end

