function [A, b] = get_Ab(f_bar, f_image)

    Ab = pinv([f_image, [1; 1; 1; 1; 1]]) * f_bar;

    A = Ab(1: 2, :);
    b = Ab(3, :);

    A = A';
    b = b';
    
    return;
    
end