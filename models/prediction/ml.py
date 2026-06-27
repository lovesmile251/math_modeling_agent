from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_KEYWORDS = ("target", "label", "y", "demand", "sales", "profit", "revenue", "price", "score", "value", "response", "目标", "需求", "销量", "销售", "价格", "得分")
FEATURE_EXCLUDE_KEYWORDS = ("id", "index", "编号", "序号")
MAX_GRADIENT_BOOSTING_ROWS = 500
MAX_GRADIENT_BOOSTING_FEATURES = 24
MAX_TREE_SPLIT_CANDIDATES = 24


def gradient_boosting_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic gradient boosted regression trees for larger samples."""
    prepared = _prepare_supervised(df, min_rows=12)
    if prepared is None:
        return pd.DataFrame()
    x, y, feature_names, target_column = prepared
    x, y, feature_names = _bound_supervised_problem(
        x,
        y,
        feature_names,
        max_rows=MAX_GRADIENT_BOOSTING_ROWS,
        max_features=MAX_GRADIENT_BOOSTING_FEATURES,
    )

    split = max(int(len(x) * 0.8), len(x) - max(2, len(x) // 5))
    split = min(max(split, len(feature_names) + 2), len(x) - 1)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    if len(x_train) < max(8, len(feature_names) + 2) or len(x_test) < 1:
        return pd.DataFrame()

    learning_rate = 0.1
    n_estimators = 24
    max_depth = 2
    base = float(np.mean(y_train))
    predictions_train = np.full(len(y_train), base, dtype=float)
    predictions_test = np.full(len(y_test), base, dtype=float)
    importances = np.zeros(len(feature_names), dtype=float)

    for _ in range(n_estimators):
        residual = y_train - predictions_train
        tree = _build_tree(x_train, residual, depth=0, max_depth=max_depth, min_samples=8, importances=importances)
        predictions_train += learning_rate * _predict_tree(tree, x_train)
        predictions_test += learning_rate * _predict_tree(tree, x_test)

    importance_sum = float(importances.sum())
    if importance_sum > 0:
        importances = importances / importance_sum

    train_rmse = _rmse(y_train, predictions_train)
    test_rmse = _rmse(y_test, predictions_test)
    train_r2 = _r_squared(y_train, predictions_train)
    test_r2 = _r_squared(y_test, predictions_test)

    rows: list[dict[str, float | str | int]] = []
    for name, importance in sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "target": str(target_column),
                "feature": str(name),
                "importance": float(importance),
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r_squared": train_r2,
                "test_r_squared": test_r2,
                "train_size": int(len(y_train)),
                "test_size": int(len(y_test)),
                "method": "gradient_boosted_regression_trees",
            }
        )
    return pd.DataFrame(rows)


def ridge_regression_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Ridge regression with k-fold cross-validated penalty selection."""
    prepared = _prepare_supervised(df, min_rows=6)
    if prepared is None:
        return pd.DataFrame()
    x_raw, y, feature_names, target_column = prepared

    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std[std == 0] = 1.0
    x = (x_raw - mean) / std

    alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
    folds = min(5, len(x))
    if folds < 2:
        return pd.DataFrame()
    best_alpha = 0.0
    best_cv_rmse = float("inf")
    for alpha in alphas:
        cv_rmse = _cross_val_rmse(x, y, alpha, folds)
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_alpha = alpha

    coefficients = _ridge_fit(x, y, best_alpha)
    design = np.column_stack([np.ones(len(x)), x])
    fitted = design @ coefficients
    r2 = _r_squared(y, fitted)
    rmse = _rmse(y, fitted)

    rows: list[dict[str, float | str | int]] = [
        {
            "target": str(target_column),
            "term": "intercept",
            "coefficient": float(coefficients[0]),
            "best_alpha": float(best_alpha),
            "cv_rmse": float(best_cv_rmse),
            "fit_rmse": rmse,
            "r_squared": r2,
            "sample_size": int(len(y)),
            "method": "ridge_regression_cross_validated",
        }
    ]
    for name, coefficient, scale in zip(feature_names, coefficients[1:], std):
        rows.append(
            {
                "target": str(target_column),
                "term": str(name),
                "coefficient": float(coefficient),
                "coefficient_original_scale": float(coefficient / scale) if scale != 0 else float(coefficient),
                "best_alpha": float(best_alpha),
                "cv_rmse": float(best_cv_rmse),
                "fit_rmse": rmse,
                "r_squared": r2,
                "sample_size": int(len(y)),
                "method": "ridge_regression_cross_validated",
            }
        )
    return pd.DataFrame(rows)


def _prepare_supervised(df: pd.DataFrame, min_rows: int) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < min_rows or numeric.shape[1] < 2:
        return None
    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    target_column = _choose_target(numeric)
    feature_columns = [
        column
        for column in numeric.columns
        if column != target_column and not _looks_like(str(column), FEATURE_EXCLUDE_KEYWORDS)
    ]
    if not feature_columns:
        return None
    x = numeric[feature_columns].to_numpy(dtype=float)
    y = numeric[target_column].to_numpy(dtype=float)
    if np.std(y) == 0 or (x.std(axis=0) > 0).sum() < 1:
        return None
    return x, y, [str(column) for column in feature_columns], str(target_column)


def _bound_supervised_problem(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    max_rows: int,
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Bound tree fitting cost for wide or long contest attachments."""
    if x.shape[1] > max_features:
        variances = np.var(x, axis=0)
        selected_features = np.sort(np.argsort(variances)[-max_features:])
        x = x[:, selected_features]
        feature_names = [feature_names[index] for index in selected_features]

    if len(x) > max_rows:
        selected_rows = np.linspace(0, len(x) - 1, max_rows, dtype=int)
        x = x[selected_rows]
        y = y[selected_rows]

    return x, y, feature_names


def _choose_target(df: pd.DataFrame) -> str:
    for column in df.columns:
        if _looks_like(str(column), TARGET_KEYWORDS):
            return str(column)
    return str(df.columns[-1])


def _looks_like(name: str, keywords: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(keyword == lowered or keyword in lowered for keyword in keywords)


def _build_tree(x: np.ndarray, residual: np.ndarray, depth: int, max_depth: int, min_samples: int, importances: np.ndarray) -> dict:
    node_value = float(np.mean(residual)) if len(residual) else 0.0
    if depth >= max_depth or len(residual) < min_samples or np.allclose(residual, residual[0]):
        return {"leaf": True, "value": node_value}

    best_feature = -1
    best_threshold = 0.0
    best_gain = 0.0
    parent_sse = float(np.sum((residual - residual.mean()) ** 2))
    for feature in range(x.shape[1]):
        values = x[:, feature]
        unique_values = np.unique(values)
        if len(unique_values) < 2:
            continue
        thresholds = _candidate_thresholds(unique_values)
        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask
            if left_mask.sum() < 1 or right_mask.sum() < 1:
                continue
            left = residual[left_mask]
            right = residual[right_mask]
            sse = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
            gain = parent_sse - sse
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feature = feature
                best_threshold = float(threshold)

    if best_feature < 0:
        return {"leaf": True, "value": node_value}

    importances[best_feature] += best_gain
    mask = x[:, best_feature] <= best_threshold
    left_tree = _build_tree(x[mask], residual[mask], depth + 1, max_depth, min_samples, importances)
    right_tree = _build_tree(x[~mask], residual[~mask], depth + 1, max_depth, min_samples, importances)
    return {"leaf": False, "feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}


def _candidate_thresholds(unique_values: np.ndarray) -> np.ndarray:
    raw_thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
    if len(raw_thresholds) <= MAX_TREE_SPLIT_CANDIDATES:
        return raw_thresholds
    quantiles = np.linspace(0.05, 0.95, MAX_TREE_SPLIT_CANDIDATES)
    return np.unique(np.quantile(unique_values, quantiles))


def _predict_tree(tree: dict, x: np.ndarray) -> np.ndarray:
    predictions = np.zeros(len(x), dtype=float)
    for idx in range(len(x)):
        node = tree
        while not node["leaf"]:
            if x[idx, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        predictions[idx] = node["value"]
    return predictions


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def _cross_val_rmse(x: np.ndarray, y: np.ndarray, alpha: float, folds: int) -> float:
    n = len(x)
    indices = np.arange(n)
    fold_sizes = np.full(folds, n // folds, dtype=int)
    fold_sizes[: n % folds] += 1
    errors: list[float] = []
    start = 0
    for size in fold_sizes:
        test_idx = indices[start : start + size]
        start += size
        if len(test_idx) == 0:
            continue
        train_idx = np.setdiff1d(indices, test_idx)
        if len(train_idx) < 2:
            continue
        coefficients = _ridge_fit(x[train_idx], y[train_idx], alpha)
        design_test = np.column_stack([np.ones(len(test_idx)), x[test_idx]])
        predictions = design_test @ coefficients
        errors.append(float(np.mean((y[test_idx] - predictions) ** 2)))
    if not errors:
        return float("inf")
    return float(np.sqrt(np.mean(errors)))


def _rmse(observed: np.ndarray, fitted: np.ndarray) -> float:
    if len(observed) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((observed - fitted) ** 2)))


def _r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    if len(observed) == 0:
        return float("nan")
    total = float(np.sum((observed - observed.mean()) ** 2))
    if total == 0:
        return 1.0
    residual = float(np.sum((observed - fitted) ** 2))
    return float(1.0 - residual / total)
