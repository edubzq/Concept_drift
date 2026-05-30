import numpy as np


def compute_ensemble_diversity(model, X=None, base_predictions=None):
    """
    Diversidad media por pares basada en disagreement.
    Devuelve 0 si el ensemble tiene menos de 2 modelos.
    """
    if not hasattr(model, "models") or len(model.models) < 2:
        return 0.0

    if base_predictions is None:
        if X is None:
            raise ValueError("Debe pasarse X o base_predictions.")

        if hasattr(model, "predict_with_base_predictions"):
            _, base_predictions = model.predict_with_base_predictions(X)
        else:
            X_dict = X.to_dict(orient="records")
            base_predictions = np.array([
                [base_model.predict_one(xi) for xi in X_dict]
                for base_model in model.models
            ], dtype=object)

    preds = np.asarray(base_predictions, dtype=object)
    n_models = preds.shape[0]

    if n_models < 2:
        return 0.0

    total_disagreement = 0.0
    total_pairs = 0

    for i in range(n_models):
        for j in range(i + 1, n_models):
            total_disagreement += np.mean(preds[i] != preds[j])
            total_pairs += 1

    return float(total_disagreement / total_pairs)


def compute_multiple_recoveries(
    curve,
    drop_threshold=0.15,
    pre_window=3,
    recovery_ratio=0.60,
    local_min_window=5,
):
    curve = np.array(curve, dtype=float)

    if len(curve) == 0:
        return np.nan, np.nan, [], 0

    min_acc = np.min(curve)
    recoveries = []

    i = 1
    while i < len(curve):
        drop = curve[i] - curve[i - 1]

        if drop <= -drop_threshold:
            start_pre = max(0, i - pre_window)
            baseline_segment = curve[start_pre:i]

            if len(baseline_segment) == 0:
                baseline = curve[i - 1]
            else:
                baseline = np.mean(baseline_segment)

            end_local = min(i + local_min_window, len(curve))
            local_segment = curve[i:end_local]

            if len(local_segment) == 0:
                min_point = curve[i]
            else:
                min_point = np.min(local_segment)

            recovery_target = min_point + recovery_ratio * (baseline - min_point)

            recovered = False
            for j in range(i, len(curve)):
                if curve[j] >= recovery_target:
                    recoveries.append(j - i)
                    i = j + 1
                    recovered = True
                    break

            if not recovered:
                recoveries.append(len(curve) - i)
                break
        else:
            i += 1

    if len(recoveries) == 0:
        mean_recovery = np.nan
    else:
        mean_recovery = float(np.mean(recoveries))

    return min_acc, mean_recovery, recoveries, len(recoveries)


def _safe_nanmean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan


def _safe_nanstd(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan