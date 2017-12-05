function [normalized_image] = image_normalization(original_image, f_bar, f_image)

    original_image_grayscale = rgb2gray(original_image);

    [A, b] = get_Ab(f_bar, f_image);
    
    normalized_image = uint8(zeros(64, 64));
    
    for i = 1: 64
        
        for j = 1: 64
            
            % Solve the equation xy64 = A * xy + b to obtain the pixel
            % Positions in the bigger image
            xy = (pinv(A) * ( [ i; j ] - b ));
            
            % Extract the x and y coordinate
            x240 = int32(xy(1, :));
            y320 = int32(xy(2, :));
            
            % Although very rare, these values can fall down to negative values. 
            % So if it happens, just make it zero.
            if(x240 <= 0)
                
                x240 = 1;
                
            end
            
            if(y320 <= 0)
                
                y320 = 1;
                
            end
            
            if(x240 > 240)
                
                x240 = 240;
                
            end
            
            if(y320 >320)
                
                y320 = 320;
                
            end
            
            % Copy the value of the pixel in the bigger image to the normalized image
            normalized_image(i, j) = uint8(original_image_grayscale(y320, x240));
            
        end
        
    end
    
    normalized_image = normalized_image';
    
    return;
    
end

