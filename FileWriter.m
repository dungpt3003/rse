classdef FileWriter < handle
    % Wrapper class to help write data to file
    % Written by Dang Manh Truong (dangmanhtruong@gmail.com)
    properties (Access = private)
        fid; 
    end
    
    methods
        function obj = FileWriter(fileName)
            try
                obj.fid = fopen(fileName, 'wt');
            catch e                
                error(e.message());                
            end
            if obj.fid == -1
                error('In class FileWriter: Unable to read file (file identifier == -1)');
            end
            disp('');
        end
        
        function write(obj, st)            
            fprintf(obj.fid, st);            
        end
        
        function writeNumber(obj, number)
            fprintf(obj.fid, '%d\n', number);
        end
        
        function delete(obj)           
            if obj.fid ~= -1
                fclose(obj.fid);
            end           
        end    
    end    
    
    methods (Access = protected)
        function result = isInvalidFile(obj)
            result = (obj.fid == -1);
        end
    end
end

