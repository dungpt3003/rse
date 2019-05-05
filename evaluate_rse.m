function [error_rate, num_selected, test_result] = evaluate_rse(Train, Test,num_base_cfr, lambda, is_binary)

% This is an example code of evaluating RSE. 
%
% Input:
%   train_data: the file name of the training data (in .arff format)
%   test_data: the file name of the test data (in .arff format)
%   num_base_cfr: num of base classifier in the ensemble
%   lambda: the regularizer parameter
%   is_binary: whether to used the binary combination method
%
% Output:
%   error_rate: the error rate of RSE (trained using train data) on the test data
%   num_selected: the size of RSE, i.e., the number of classifiers selected
%
% Copyright: Nan Li and Zhi-Hua Zhou, 2009
% Contact: Nan Li (lin@lamda.nju.edu.cn)

% read data (train & test)
% Train = javaObject('weka.core.Instances', javaObject('java.io.FileReader',train_data));
% Train.setClassIndex(Train.numAttributes() - 1);

% Test = javaObject('weka.core.Instances', javaObject('java.io.FileReader',test_data));
% Test.setClassIndex(Test.numAttributes() - 1);

% Ensemble Parameters
CfrName = 'com.vu.BinaryRSE';     % ensemble class name
BaseCfr = 'weka.classifiers.trees.J48';     % base clasifier class in weka
RndSeed = 1;      % random seed used in the ensemble
UseRndSubSpace = false;    % whether to use random subspace method when generating base classifiers

Cfr = javaObject(CfrName);
Cfr.setBaseClassifier(javaObject(BaseCfr));
Cfr.setNumBaseClassifier(num_base_cfr);
Rrd = javaObject('java.util.Random', RndSeed);
Cfr.setR(Rrd);
Cfr.setUseRandomSubspace(UseRndSubSpace);

% build the ensemble classifier
Cfr.buildClassifier(Train);
weight = compute_weight(Cfr, lambda);
if is_binary   % whether to use the binary weights
    weight(weight > 0) = 1;
end
Cfr.setWeights(weight); % set the weights

% --- evaluate error rate 
N = Test.numInstances;
test_result = zeros(N,1);
for id = 1: N
    test_result(id) = Cfr.classifyInstance(Test.instance(id - 1));
end
error_rate = get_error_rate(Cfr, Test);
num_selected = sum(weight > 0);

% clear 
clear Cfr;

%% end of function