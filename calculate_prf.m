function [PAvg, RAvg, F1Avg] = calculate_prf( confusionMatrix )
M = size(confusionMatrix,1);
P = zeros(M,1);
R = zeros(M,1);
F1 = zeros(M,1);
for i=1:M
    col = sum(confusionMatrix(:,i));
    if(col ~= 0)
        P(i) = confusionMatrix(i,i)/col;
    end
    
    row = sum(confusionMatrix(i,:));
    if(row ~= 0 )
    	R(i) = confusionMatrix(i,i)/row;
    end
    
    total = P(i)+R(i);
    if(total ~= 0)
        F1(i) = (2*P(i)*R(i))/total;
    end
end
PAvg = mean(P);
RAvg = mean(R);
F1Avg = mean(F1);
end

