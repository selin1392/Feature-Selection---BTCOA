function [TrainData, TestData, R] = ManageData(data, Method)

% Inputs = data(:, 1:end-1);
% Targets = data(:, end);


Inputs = data(:,2:end);
Targets = data(:,1);

%% Finding Nan elements
Inputs = knnimpute(Inputs);

%% Normalaization
LB = 0;
UB = 1;
Inputs = Normalaization(Inputs, LB, UB);

%% Test and Train Data
[nSamples, nInputs] = size(Inputs);
TrPercent = 80;
TrNum = round(nSamples*TrPercent/100);
% TsNum = nSamples - TrNum;

R = randperm(nSamples);
trIndex = R(1:TrNum);
tsIndex = R(1+TrNum:end);

TrainData.Inputs = Inputs(trIndex, :);
TrainData.Targets = Targets(trIndex, :);
TrainData.nInputs = nInputs;
TrainData.nSamples = numel(trIndex);
TrainData.Method = Method;

TestData.Inputs = Inputs(tsIndex, :);
TestData.Targets = Targets(tsIndex, :);

%% Apply K fold Ceoss Validation
K_fold = 10;
CVI = crossvalind('kfold', TrainData.nSamples, K_fold);
TrainData.CVI = CVI;
TrainData.K = K_fold;


end

function X = Normalaization(X, LB, UB)

Min = min(X);
Max = max(X);
if nargin < 2
    LB = 0;
    UB = 1;
end

for i = 1:numel(Min)
    X(:, i) = (X(:, i) - Min(i)) / (Max(i) - Min(i));
end
X = (UB - LB) * X + LB;


end
