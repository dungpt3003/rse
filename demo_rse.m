% Initialise environment
clear;
clc;

% Add data paths
mainPath = [pwd '/'];
resultPath = [mainPath 'results'];
if (~exist(resultPath, 'dir'))
    mkdir(resultPath);
end

addpath([mainPath 'cv']);
addpath([mainPath 'data']);

% For RSE (new code)
% javaaddpath([mainPath 'jar' filesep 'rse.jar']);

% javaaddpath([mainPath 'jar' filesep 'matlab2weka.jar']);
% javaaddpath([mainPath 'jar' filesep 'weka.jar']);
% javaaddpath([mainPath 'jar' filesep 'moa-2017.06.jar']);
javaaddpath([mainPath 'jar' filesep 'rgsteb.jar']);

niters = 3;
nfolds = 10;

fileList = {'abalone', 'Agrawal', 'AssetNegotiation-F2', 'AssetNegotiation-F3', ...
    'AssetNegotiation-F4', 'BayesianNetworkGenerator_bridges_version1', 'BNG_zoo', ...
    'DowJones_dj30-1985-2003', 'electricity-normalized', 'Hyperplane', 'RandomTree', ...
    'RBF', 'shuttle_full', 'Sine', 'STAGGER', 'Waveform'};

for i = 1 : numel(fileList)
    datasetName = fileList{i};
    fprintf('%s\n', datasetName);
    
    allOutput = struct(...
        'err',      OutputWriter([resultPath filesep datasetName '_err.dat'], nfolds * niters), ...
        'prec',     OutputWriter([resultPath filesep datasetName '_prec.dat'], nfolds * niters), ...
        'recall',   OutputWriter([resultPath filesep datasetName '_recall.dat'], nfolds * niters), ...
        'f1',       OutputWriter([resultPath filesep datasetName '_f1.dat'], nfolds * niters)...
    );
    
    D = load([datasetName '.csv']);
    % D = importdata([datasetName '.dat']);
    % wekaD = create_weka_data(D);
    
    tempholder = load([datasetName '.mat']);
    cv = tempholder.cv;
    
    labels = D(:, end); % class
    n0 = length(labels);
    uniq_labels = unique(labels);
    n_labels = length(uniq_labels);
    allIdx = (1:n0)';
   
    for loop = 1 : niters
        for j = 1 :  nfolds
            current = (loop - 1) * nfolds + j;
            fprintf('At the %2d iteration \n', current);
            teIdx = cv{current};
            trIdx = allIdx(~ismember(allIdx, teIdx));
            
            testData = D(teIdx, :);
            l_tests = length(testData);
            % Initiate predicted possibility
            p = zeros(l_tests, n_labels);
            
            % One-vs-All classifications
            for l = 1 : n_labels
                % Set positve label
                pos = uniq_labels(l);
                fprintf('Positive label %d \n', pos);
                
                % Copy dataset
                tempData = D;
                
                % Set new label value (postive/negative)
                for k = 1 : length(D)
                    tempData(k, end) = (tempData(k, end) == pos);
                end
                
                % Split data
                L0 = tempData(trIdx, :);       % current training set
                LTest = tempData(teIdx, :);    % current testing set
                
                % Convert data to weka (need to optimize here)
                wekaTrain = create_weka_data(L0);
                wekaTest = create_weka_data(LTest);
                
                num_base_classifiers = 100;  % number of base classifiers
                lambda = 1; % regularized parameter
                use_binary = true; % whether to use binary combination
                
                [err, num, testResult] = evaluate_rse(wekaTrain, wekaTest, use_binary, lambda, use_binary);
                fprintf(' :: RSE   Error rate with positive class %d = %f \n', pos, err);
                for k = 1 : length(testResult)
                    if testResult(k) == 1
                        p(k, pos) = p(k, pos) + 1;
                    else
                        for x = 1 : n_labels
                            if x ~= pos
                                p(k, x) = p(k, x) + 1 / (n_labels - 1);
                            end
                        end
                    end
                end
            end
            
            % Combine one-vs-all error
            
            realClass = zeros(l_tests, 1);
            predictClass = zeros(l_tests, 1);
            for id = 1: l_tests
                [maxValue, maxPosition] = max(p(id, :));
                predictClass(id) = maxPosition;
                realClass(id) = testData(id, end);
            end
            df = (realClass ~= predictClass);
            E = sum(df) / l_tests;
            err = sum(predictClass ~= realClass) / l_tests;
            confmat = create_confusion_matrix(predictClass, realClass, uniq_labels);
            [prec, recall, f1] = calculate_prf(confmat);
            fprintf('Error rate of one-vs-all in the iteration %d = %f %f %f %f \n', current, err, prec, recall, f1);
            allOutput.err.addData(err);
            allOutput.prec.addData(prec);
            allOutput.recall.addData(recall);
            allOutput.f1.addData(f1);     
        end
    end 
end

