function [z, Out] = EvaluateFeatures(s, Data)
warning off

% =============================================================
% 1) Feature check
% =============================================================
ns = sum(s);
if ns == 0
    z   = inf;
    Out = [];
    return;
end

Predaucrion = ns / size(Data.Inputs,2);

% =============================================================
% 2) Data
% =============================================================
Inputs  = Data.Inputs(:, s);
Targets = Data.Targets;
classes = unique(Targets);
nClass  = numel(classes);

InnerIdx = Data.InnerIdx;
K_inner  = Data.K_inner;

% =============================================================
% 3) Classifier template
% =============================================================
switch Data.Method
    case 'SVM'
        template = templateSVM( ...
            'KernelFunction','rbf', ...
            'KernelScale','auto', ...
            'Standardize',true);

    case 'KNN'
        template = templateKNN( ...
            'NumNeighbors',5, ...
            'Distance','euclidean', ...
            'Standardize',true);
end

% =============================================================
% 4) Metrics initialization
% =============================================================
Accuracy    = zeros(K_inner,1);
Sensitivity = zeros(K_inner,1);
Specificity = zeros(K_inner,1);
Precision   = zeros(K_inner,1);
F1score     = zeros(K_inner,1);
BalancedAcc = zeros(K_inner,1);
Gmean       = zeros(K_inner,1);
AUROC       = zeros(K_inner,1);

% =============================================================
% 5) Inner Cross-Validation
% =============================================================
for i = 1:K_inner

    train_idx = InnerIdx ~= i;
    val_idx   = InnerIdx == i;

    Xtrain = Inputs(train_idx,:);
    Ttrain = Targets(train_idx);
    Xval   = Inputs(val_idx,:);
    Tval   = Targets(val_idx);

    % -------------------------------
    % Train classifier (binary & multiclass)
    % -------------------------------
    Model = fitcecoc(Xtrain, Ttrain, 'Learners', template);

    % -------------------------------
    % Prediction
    % -------------------------------
    [Pred, score] = predict(Model, Xval);

    % -------------------------------
    % Confusion Matrix
    % -------------------------------
    CM = confusionmat(Tval, Pred);

    TP = diag(CM);
    FN = sum(CM,2) - TP;
    FP = sum(CM,1)' - TP;
    TN = sum(CM(:)) - (TP + FN + FP);

    sen = mean(TP ./ (TP + FN + eps));
    spe = mean(TN ./ (TN + FP + eps));
    pre = mean(TP ./ (TP + FP + eps));

    acc = sum(TP) / sum(CM(:));
    f1  = mean(2 * (pre .* sen) ./ (pre + sen + eps));
    bal = 0.5 * (sen + spe);
    gmn = sqrt(sen * spe);

    Accuracy(i)    = acc;
    Sensitivity(i) = sen;
    Specificity(i) = spe;
    Precision(i)   = pre;
    F1score(i)     = f1;
    BalancedAcc(i) = bal;
    Gmean(i)       = gmn;

    % -------------------------------
    % ✅ Multiclass AUROC (One-vs-All)
    % -------------------------------
    auc_all = zeros(nClass,1);
    for c = 1:nClass
        [~,~,~,auc_all(c)] = perfcurve(Tval, score(:,c), classes(c));
    end
    AUROC(i) = mean(auc_all);

end

% =============================================================
% 6) Mean performance
% =============================================================
Acc_mean = mean(Accuracy);

% =============================================================
% 7) Feature Selection Cost Function
% =============================================================
switch Data.MethodFS

    case 'MinCost'
        z = 1 - Acc_mean;

    case 'MinRedouction'
        w1 = 0.80; w2 = 0.20;
        z = (1 - w1 * Acc_mean) + w2 * Predaucrion;

    case 'Exact'
        Beta   = 0.1;
        Corner = abs(round(Data.nExact*size(Data.Inputs,2)) - ns);
        z = (1 - Acc_mean) + Beta*Corner;
        Out.Corner = Corner;

    case 'NewFit1'
        Fr = unifrnd(0.02,0.08);
        z = (1 - Fr)*(1 - Acc_mean) + Fr*Predaucrion;

    case 'NewFit2'
        a = 0.2; b = 0.8;
        z = a*(1 - Acc_mean) + ...
            b*(abs(size(Data.Inputs,2) - ns)/size(Data.Inputs,2));
end

% =============================================================
% 8) Output
% =============================================================
Out.ClassificationMethod = Data.Method;
Out.CostValue   = z;
Out.Predaucrion = Predaucrion;

Out.Accuracy    = 100 * mean(Accuracy);
Out.Sensitivity = 100 * mean(Sensitivity);
Out.Specificity = 100 * mean(Specificity);
Out.Precision   = 100 * mean(Precision);
Out.F1score     = 100 * mean(F1score);
Out.BalancedAcc = 100 * mean(BalancedAcc);
Out.Gmean       = 100 * mean(Gmean);
Out.AUROC       = 100 * mean(AUROC);

Out.FeatureSelected = s;
Out.nS = ns;

end
