function ResultsOut = EvaluatePlot(Data, Results, NameTag)

    % ---------------------------
    % 1) استخراج مدل و فیچرهای انتخابی
    % ---------------------------
    if isfield(Results, 'FeatureSelected')
        FeatureSelect = Results.FeatureSelected;
    else
        error('Results.FeatureSelected not found');
    end

    if isfield(Results, 'Classify')
        Classify = Results.Classify;
    else
        error('Results.Classify not found. Train classifier before calling EvaluatePlot.');
    end

    % کنترل اندازه برای جلوگیری از خطای اندیس
    nFeat = size(Data.Inputs, 2);
    FeatureSelect = FeatureSelect(FeatureSelect >= 1 & FeatureSelect <= nFeat);

    if isempty(FeatureSelect)
        error('FeatureSelect is empty after index adjustment.');
    end

    X = Data.Inputs(:, FeatureSelect);
    T = Data.Targets;

    % ---------------------------
    % 2) پیش‌بینی
    % ---------------------------
    try
        [Gpred, Score] = predict(Classify, X);
    catch ME
        disp('--- ERROR in classifier prediction ---');
        disp(ME.message);
        keyboard
    end

    % ---------------------------
    % 3) ساخت Confusion Matrix
    % ---------------------------
    CM = confusionmat(T, Gpred);

    % ---------------------------
    % 4) متریک‌ها
    % ---------------------------
    Metrics = PrecisionRecall(CM);

    ResultsOut = Metrics;
    ResultsOut.ConfMat = CM;

    % ---------------------------
    % 5) ROC / AUC
    % ---------------------------
    try
        if size(CM,1) == 2
            [~,~,~,AUC] = perfcurve(T, Score(:,1), 1);
            ResultsOut.AUC = AUC;
        else
            % اگر چندکلاسه است، میانگین AUC
            AUCs = zeros(1,size(CM,1));
            for c = 1:size(CM,1)
                label = (T == c);
                if numel(unique(label)) < 2
                    AUCs(c) = NaN;
                else
                    [~,~,~,AUCs(c)] = perfcurve(label, Score(:,c), 1);
                end
            end
            ResultsOut.AUC = nanmean(AUCs);
        end
    catch
        ResultsOut.AUC = NaN;
    end

    ResultsOut.Name = NameTag;
end
