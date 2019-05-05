function [ confusionMatrix ] = create_confusion_matrix( predictY, trueY, classes)
M = length(classes);
confusionMatrix = zeros(M);
for i=1:M
    trueIdxOfClass_i = trueY == classes(i);
    truePositiveIdxOfClass_Mi = find(predictY(trueIdxOfClass_i) == classes(i)); 
    confusionMatrix(i,i) = length(truePositiveIdxOfClass_Mi);
    for j = 1:M
        if(j ~= i)
            falseNegativeIdxOfClass_Mi = find(predictY(trueIdxOfClass_i) == classes(j));
            confusionMatrix(i,j) = length(falseNegativeIdxOfClass_Mi);
        end
    end
end
end