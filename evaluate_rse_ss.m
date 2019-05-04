function [error_rate, num_selected] = evaluate_rse_ss(train_data, unlabeled_data, test_data,num_base_cfr, lambda, is_binary)

% This is an example code of evaluating semi-supervised RSE (RSE_ss). 
%
% Input:
%   train_data: the file name of the labeled training data (in .arff format)
%   unlabeled_data: the file name of the unlabeled data (in .arff format)
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


% add the java classpath
ClsPath = './Java/RgStEb/classes';
javaaddpath(ClsPath);


% read data (train & test)
Train = javaObject('weka.core.Instances', javaObject('java.io.FileReader',train_data));
Train.setClassIndex(Train.numAttributes() - 1);

Unlabeled = javaObject('weka.core.Instances', javaObject('java.io.FileReader',unlabeled_data));
Unlabeled.setClassIndex(Unlabeled.numAttributes() - 1);

Test = javaObject('weka.core.Instances', javaObject('java.io.FileReader',test_data));
Test.setClassIndex(Test.numAttributes() - 1);

% Ensemble Parameters
CfrName = 'RgStEb.Base.SemiBinaryRSE';     % ensemble class name
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
numLabeled = Train.numInstances();
Data = javaMethod('mergeInstances','weka.core.Instances',Train, Unlabeled); % merge labeled and unlabeled data

Cfr.setNumLabeled(numLabeled);
Cfr.buildClassifier(Data);

weight = compute_weight(Cfr, lambda);
if is_binary   % whether to use the binary weights
    weight(weight > 0) = 1;
end
Cfr.setWeights(weight); % set the weights

% --- evaluate error rate 
error_rate = get_error_rate(Cfr, Test);
num_selected = sum(weight > 0);

% clear 
clear Cfr;
javarmpath(ClsPath);

%% end of function