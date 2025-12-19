function [OuterFolds, R] = ManageData_NestedCV(data, Method)
% ============================================================
%   Nested Cross-Validation (10-fold outer, 5-fold inner)
%   Stratified splitting + normalization + KNN imputation
% ============================================================

% -----------------------------------------
% Separate Inputs / Targets
% -----------------------------------------
Inputs  = data(:,2:end);
Targets = data(:,1);


% Inputs = data(:, 1:end-1);
% Targets = data(:, end);



nSamples = size(Inputs,1);

% -----------------------------------------
% Fix missing values via KNN imputation
% -----------------------------------------
Inputs = knnimpute(Inputs);

% -----------------------------------------
% Normalize data to [0, 1]
% -----------------------------------------
Inputs = Normalaization(Inputs, 0, 1);

% -----------------------------------------
% OUTER 10-FOLD (STRATIFIED)
% -----------------------------------------
K_outer = 10;
OuterIdx = crossvalind('Kfold', Targets, K_outer);

OuterFolds = struct;

for k = 1:K_outer
    
    % --- outer train/test split ---
    test_idx  = (OuterIdx == k);
    train_idx = ~test_idx;

    % --- store outer test data ---
    OuterFolds(k).Test.Inputs   = Inputs(test_idx, :);
    OuterFolds(k).Test.Targets  = Targets(test_idx, :);

    % --- store outer train data ---
    OuterFolds(k).Train.Inputs  = Inputs(train_idx, :);
    OuterFolds(k).Train.Targets = Targets(train_idx, :);
    OuterFolds(k).Train.Method  = Method;

    % -----------------------------------------
    % INNER 5-FOLD (STRATIFIED)
  
    % -----------------------------------------
    K_inner = 5;
    InnerIdx = crossvalind('Kfold', Targets(train_idx), K_inner);

    OuterFolds(k).Train.InnerIdx = InnerIdx;
    OuterFolds(k).Train.K_inner  = K_inner;
end

% Random permutation (optional, used by older main codes)
R = randperm(nSamples);

end



% ============================================================
% Normalization Function
% ============================================================
function X = Normalaization(X, LB, UB)

MinVals = min(X);
MaxVals = max(X);

for i = 1:numel(MinVals)
    X(:,i) = (X(:,i) - MinVals(i)) ./ (MaxVals(i) - MinVals(i));
end

X = (UB - LB).*X + LB;

end
