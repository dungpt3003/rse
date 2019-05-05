classdef FileReader < handle
    % Utility class
    % Written by Dang Manh Truong (dangmanhtruong@gmail.com)
    
    properties
        fid;
    end
    
    methods
        function obj = FileReader(file_name)
            obj.fid = fopen(file_name, 'rt');
        end
        
        function prob_array = read_prob(obj)
            prob_array_as_string = fgetl(obj.fid);
            prob_array = str2num(prob_array_as_string);            
        end
        
        function delete(obj)
            fclose(obj.fid);
        end
    end
    
end

