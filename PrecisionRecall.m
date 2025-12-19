function Metrics = PrecisionRecall(CM)

    % تعداد کلاس‌ها
    K = size(CM,1);

    % محاسبه برای هر کلاس
    TP = diag(CM);
    FP = sum(CM,1)' - TP;
    FN = sum(CM,2)  - TP;
    TN = sum(CM(:)) - (TP + FP + FN);

    % تبدیل به double برای جلوگیری از Inf
    TP = double(TP);
    FP = double(FP);
    FN = double(FN);
    TN = double(TN);

    % محاسبه متریک‌ها
    Precision  = TP ./ (TP + FP);
    Recall     = TP ./ (TP + FN);
    Specificity = TN ./ (TN + FP);
    F1 = 2 * (Precision .* Recall) ./ (Precision + Recall);

    % جای‌گذاری NaN به‌جای Inf یا 0/0
    Precision(isnan(Precision)) = 0;
    Recall(isnan(Recall))       = 0;
    Specificity(isnan(Specificity)) = 0;
    F1(isnan(F1)) = 0;

    % متریک‌های کلی
    Accuracy  = sum(TP) / sum(CM(:));
    Gmean     = sqrt(mean(Recall .* Specificity));
    BA        = mean((Recall + Specificity) / 2);

    % ذخیره در ساختار خروجی
    Metrics.Accuracy     = Accuracy;
    Metrics.Precision    = mean(Precision);
    Metrics.Recall       = mean(Recall);
    Metrics.Specificity  = mean(Specificity);
    Metrics.F1           = mean(F1);
    Metrics.Gmean        = Gmean;
    Metrics.BA           = BA;
end
