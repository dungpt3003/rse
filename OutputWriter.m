classdef OutputWriter < FileWriter
    % Wrapper class to write formatted output
    % Written by Dang Manh Truong (dangmanhtruong@gmail.com)
    % Here, the output is formatted as follows: 
    % 9.333333e-02 
    % 7.333333e-02 
    % 8.000000e-02 
    % 1.066667e-01 
    % 9.333333e-02 
    % 7.333333e-02 
    % 1.000000e-01 
    % 6.000000e-02 
    % 9.333333e-02 
    % 7.333333e-02 
    % ----------
    % Mean:
    % 8.466667e-02 
    % Variance:
    % 1.960000e-04    
    properties (Access = private)
        dataList;
        nIters;
        counter;
    end
    
    methods
        function obj = OutputWriter(fileName, nIters)
            obj@FileWriter(fileName); 
            obj.dataList = zeros(nIters, 1);
            obj.nIters = nIters;
            obj.counter = 1;
        end
        
        function addData(obj, data)
            obj.dataList(obj.counter) = data;
            obj.counter = obj.counter + 1;
        end 
        
        function delete(obj)
            % MAGIC: The output is automatically dumped to file right
            % before the object is deleted. This way we won't have to call
            % a separte method just to dump the results to file !            
            if ~obj.isInvalidFile() 
                for i = 1 : obj.nIters                
                    obj.write(sprintf('%d \n', obj.dataList(i)));
                end            
                obj.write('----------\n');            
                obj.write('Mean:\n');                    
                obj.write(sprintf('%d \n', mean(obj.dataList)));           
                obj.write('Variance:\n');            
                var = sum((obj.dataList - mean(obj.dataList)).^2) / obj.nIters;
                obj.write(sprintf('%d \n', var));
            end
        end
    end    
end

