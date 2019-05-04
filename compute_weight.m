function weight = compute_weight(Cfr, lambda)
% compute the weights for the base classifiers
% 
% Input: 
%       Cfr: the ensemble classifier, which contains multiple base
%       classifiers
% Output: 
%       weight: the weights for the base classifiers in the ensemble
%
% NOTE: It is recommanded that the QP problem be solve using MOSEK, since
% that its speed is much faster than the default solver of Matlab. You
% can add MOSEK toolbox into Matlab path without modifying codes.
% For more information, please refer the documents of MOSEK or 
% < http://www.mosek.com/index.php?id=38 >
%
% Copyright: Nan Li and Zhi-Hua Zhou, 2009
% Contact: Nan Li (lin@lamda.nju.edu.cn)

Y = Cfr.getTrueLabels(); % get the true labels of the labeled data
Prd = Cfr.getPredictions(); % get the predictions of the base classifiers
Wlink = Cfr.getKernelLinkMatrix(); % get the kernel matrix link matrix
Lap = LaplacianMatrix(Wlink);
PLP = Prd*Lap*Prd';
Q = PLP + PLP';

M = Cfr.getNumBaseClassifiers();
N = Cfr.getNumTrainingData();
NumVar = M + N;

% ---- Prepare QP ----
%equal constaint
Aeq = [ones(1,M),zeros(1,NumVar - M)];
beq = 1;
% lower and upper bounds
lb = zeros(NumVar,1);
% Ax <= b
AP = Prd';
for id = 1:N
    AP(id, :) = AP(id, :) * Y(id);
end
AP = -1 .* AP;
A = [ AP, -1 * eye(N) ];
b = -1 * ones(N ,1);

% H
H = zeros(NumVar,NumVar);
H(1:M, 1:M) = Q;
% f
f = lambda * [zeros(M,1);ones(N, 1)];

% Optimization Using QP - MOSEK
options = optimset('Display','off','LargeScale', 'on');
x0 = quadprog(H,f,A,b,Aeq,beq,lb,[],[],options);
weight = x0(1:M);
weight((weight <= 1e-5)) = 0;

% end of function 
