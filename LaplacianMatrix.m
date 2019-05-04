function Lap = LaplacianMatrix(W)
% compute the laplacian matrix 
% 
% Input: 
%       W: the line matrix
% Output: 
%       Lap: the laplacian matrix
%
% Copyright: Nan Li and Zhi-Hua Zhou, 2009
% Contact: Nan Li (lin@lamda.nju.edu.cn)


D = diag(sum(W,2));
N = D^(-0.5);
Lap = N *(D-W)* N;

