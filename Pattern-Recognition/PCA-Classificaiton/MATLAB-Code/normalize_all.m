clc
clear all
close all
txt_files = dir('images\train_images\*.txt');
JPG_files = dir('images\train_images\*.jpg');
m_txt=size(txt_files,1);
m_jpg=size(JPG_files,1);
j=0;
f_bar = get_f_bar_n(txt_files(1:3));
for i=1:m_jpg
    if(j<2) 
        j=j+1;
    else
        j=0;
        if(i<m_jpg-2)
            f_bar = get_f_bar_n(txt_files(i:i+2));
        end
    end
        original_image   = imread(['images\train_images\' JPG_files(i).name]);
        f_image          = load(['images\train_images\' txt_files(i).name]);
        
        %% LOOK CAREFULLY
        normalized_image = image_normalization(original_image, f_bar, f_image);
        
        imwrite(normalized_image, ['images\normalized_images\' txt_files(i).name(1:size(txt_files(i).name,2)-4) '_normalized.jpg']);
        d_matrix(i, :) = reshape(normalized_image, 1, 4096);

	end

% for index = 1: length(handles.jpg_train_image_array)
%         
%         original_image   = imread(['images\train_images\' handles.jpg_train_image_array(index).name()]);
%         f_image          = load(['images\train_images\' handles.txt_train_file_array(index).name()]);
% 
%         %% LOOK CAREFULLY
%         normalized_image = image_normalization(original_image, f_bar, f_image);
%         
%         imwrite(normalized_image, ['images\normalized_images\' handles.txt_train_file_array(index).name(1:end-4) '_normalized.jpg']);
%         d_matrix(index, :) = reshape(normalized_image, 1, 4096);
% 
%         end