function E = get_error_rate(Cfr, Test)
% compute the error rate on Test data
% 
% Input: 
%       Cfr: the  classifier (java object)
%       Test: the test data (java object: weka.core.instances )
% 
% Output: 
%       E: error rate
%
% Copyright: Nan Li and Zhi-Hua Zhou, 2009
% Contact: Nan Li (lin@lamda.nju.edu.cn)

N = Test.numInstances;
Pt = zeros(N,1);
TL = zeros(N,1);
for id = 1: N
    Pt(id) = Cfr.classifyInstance(Test.instance(id - 1));
    TL(id) = Test.instance(id - 1).classValue();
end
df = (TL ~= Pt);
E = sum(df)/N;

% end of function