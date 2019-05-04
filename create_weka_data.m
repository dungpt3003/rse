function [ wekaData ] = create_weka_data( D )
% Convert matlab matrix to weka instances

    X = D(:, 1 : end - 1);
    Y = D(:, end);
    
    % Converting to nominal variables (Weka cannot classify numerical classes)
    YNom = cell(size(Y));
    classes = unique(Y);
    tmpCell = cell(1, 1);
    for i = 1 : length(classes)
        tmpCell{1, 1} = strcat('class_', num2str(i - 1));
        YNom(Y == classes(i), :) = repmat(tmpCell, sum(Y == classes(i)), 1);
    end
    
    nFeatures = size(X, 2);
    
    featName = cell(1, nFeatures);
    for i = 1 : nFeatures
        featName{i} = num2str(i);
    end
    
    % disp('Converting Data into WEKA format...');
    convert2wekaObj = RgStEb.Base.ConvertToWeka('data', featName, X', YNom, true); 
    wekaData = convert2wekaObj.getInstances();
    wekaData.setClassIndex(wekaData.numAttributes() - 1); 
    
    clear convert2wekaObj;
    % disp('Converting Completed!');

end