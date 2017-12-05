function [file_array] = read_files(path, extention)

    % Check if the directory exists or not
    if isdir(path) == 0
        
        error('ERROR: The faces directory does not exist!');
        
        return;
        
    end

    % Get all files whose type is 'extention'
    file_array = dir([path filesep '*.' extention]);

    % Get the number of feature files
    N = size(file_array, 1);

    % Check if there exists any face files or not
    if N < 1
        
        error('ERROR: There is no any face file in the directory!');
        
        return;
        
    end
    
end

