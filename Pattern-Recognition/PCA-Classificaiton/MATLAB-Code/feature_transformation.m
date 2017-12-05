function [f_of_norm] = feature_transformation(A, b, f_of_img)

    Ab = [A'; b'];
    
    f_of_norm = [f_of_img, [1; 1; 1; 1; 1]] * Ab;
    
    return;
    
end

