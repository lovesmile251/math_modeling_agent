from __future__ import annotations

import numpy as np
import pandas as pd


def logistic_binary_classifier(df: pd.DataFrame, max_iter: int = 800, learning_rate: float = 0.1) -> pd.DataFrame:
    prepared = _prepare_supervised(df)
    if prepared is None:
        return pd.DataFrame()
    features, labels, row_index, feature_names, label_column = prepared
    classes = list(dict.fromkeys(labels.astype(str)))
    if len(classes) < 2 or len(labels) < 4:
        return pd.DataFrame()

    if len(classes) == 2:
        positive_class = classes[1]
        binary_target = f"{positive_class}_vs_{classes[0]}"
    else:
        class_counts = pd.Series(labels.astype(str)).value_counts()
        positive_class = str(class_counts.index[0])
        binary_target = f"{positive_class}_vs_rest"

    y = (labels.astype(str) == positive_class).astype(float)
    x, mean, std = _standardize(features)
    x_design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(x_design.shape[1], dtype=float)
    for _ in range(max_iter):
        logits = np.clip(x_design @ weights, -35, 35)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = x_design.T @ (probabilities - y) / len(y)
        weights -= learning_rate * gradient

    probabilities = 1.0 / (1.0 + np.exp(-np.clip(x_design @ weights, -35, 35)))
    pred_binary = (probabilities >= 0.5).astype(int)
    negative_class = classes[0] if classes[0] != positive_class else "rest"
    predicted = np.where(pred_binary == 1, positive_class, negative_class)
    binary_actual = np.where(labels.astype(str) == positive_class, positive_class, negative_class)
    accuracy = float((predicted == binary_actual).mean())

    rows = []
    for idx, actual, pred, probability in zip(row_index, labels.astype(str), predicted, probabilities):
        rows.append(
            {
                "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "label_column": label_column,
                "actual": actual,
                "predicted": str(pred),
                "probability_positive": float(probability),
                "accuracy": accuracy,
                "positive_class": positive_class,
                "binary_target": binary_target,
                "class_count": len(classes),
                "method": "logistic_binary_gradient_descent",
            }
        )
    for name, coefficient in zip(["intercept", *feature_names], weights):
        rows.append(
            {
                "row_index": "coefficient",
                "label_column": label_column,
                "actual": "",
                "predicted": str(name),
                "probability_positive": float(coefficient),
                "accuracy": accuracy,
                "positive_class": positive_class,
                "binary_target": binary_target,
                "class_count": len(classes),
                "method": "coefficient",
            }
        )
    return pd.DataFrame(rows)


def naive_bayes_classifier(df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_supervised(df)
    if prepared is None:
        return pd.DataFrame()
    features, labels, row_index, _, label_column = prepared
    classes = list(dict.fromkeys(labels.astype(str)))
    if len(classes) < 2 or len(labels) < 4:
        return pd.DataFrame()

    priors: dict[str, float] = {}
    means: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    for cls in classes:
        mask = labels.astype(str) == cls
        class_values = features[mask]
        priors[cls] = float(mask.mean())
        means[cls] = class_values.mean(axis=0)
        variances[cls] = np.maximum(class_values.var(axis=0), 1e-6)

    predictions = []
    confidences = []
    for row in features:
        log_scores = {}
        for cls in classes:
            log_prior = np.log(max(priors[cls], 1e-12))
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variances[cls]) + ((row - means[cls]) ** 2) / variances[cls])
            log_scores[cls] = float(log_prior + log_likelihood)
        best = max(log_scores, key=log_scores.get)
        scores = np.array(list(log_scores.values()), dtype=float)
        scores = scores - scores.max()
        confidence = float(np.exp(scores).max() / np.exp(scores).sum())
        predictions.append(best)
        confidences.append(confidence)

    actual = labels.astype(str)
    accuracy = float((np.array(predictions) == actual).mean())
    return pd.DataFrame(
        {
            "row_index": row_index,
            "label_column": label_column,
            "actual": actual,
            "predicted": predictions,
            "confidence": confidences,
            "accuracy": accuracy,
            "class_count": len(classes),
            "method": "gaussian_naive_bayes",
        }
    )


def knn_classifier(df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    prepared = _prepare_supervised(df)
    if prepared is None:
        return pd.DataFrame()
    features, labels, row_index, _, label_column = prepared
    classes = list(dict.fromkeys(labels.astype(str)))
    if len(classes) < 2 or len(labels) < 4:
        return pd.DataFrame()

    x, _, _ = _standardize(features)
    k = k or min(5, max(1, int(np.sqrt(len(x)))))
    predictions = []
    for i, row in enumerate(x):
        distances = np.sqrt(((x - row) ** 2).sum(axis=1))
        distances[i] = np.inf
        nearest = np.argsort(distances)[:k]
        votes = pd.Series(labels.astype(str)[nearest]).value_counts()
        predictions.append(str(votes.index[0]))

    actual = labels.astype(str)
    accuracy = float((np.array(predictions) == actual).mean())
    return pd.DataFrame(
        {
            "row_index": row_index,
            "label_column": label_column,
            "actual": actual,
            "predicted": predictions,
            "accuracy": accuracy,
            "k": k,
            "class_count": len(classes),
            "method": "leave_one_out_knn",
        }
    )


def _prepare_supervised(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index, list[str], str] | None:
    label_column = _find_label_column(df)
    if label_column is None:
        return None

    numeric_features = df.drop(columns=[label_column]).select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric_features.empty:
        return None
    work = pd.concat([numeric_features, df[label_column]], axis=1).dropna()
    if work.shape[0] < 4:
        return None

    features = work[numeric_features.columns].to_numpy(dtype=float)
    labels = work[label_column].astype(str).to_numpy()
    if len(set(labels)) < 2:
        return None
    return features, labels, work.index, [str(column) for column in numeric_features.columns], str(label_column)


def _find_label_column(df: pd.DataFrame) -> str | None:
    priority = (
        "label", "class", "target", "category", "type", "status", "healthy", "abnormal", "y",
        "标签", "类别", "分类", "目标", "类型", "是否", "健康", "异常", "风化", "纹饰",
    )
    for column in df.columns:
        name = str(column).lower()
        if _looks_like_identifier(name):
            continue
        if any(keyword == name or keyword in name for keyword in priority):
            if 2 <= df[column].dropna().nunique() <= max(20, int(len(df) * 0.5)):
                return str(column)

    non_numeric = df.select_dtypes(exclude="number")
    for column in non_numeric.columns:
        if _looks_like_identifier(str(column).lower()):
            continue
        unique_count = df[column].dropna().nunique()
        if 2 <= unique_count <= max(20, int(len(df) * 0.5)):
            return str(column)

    numeric = df.select_dtypes(include="number")
    for column in reversed(numeric.columns):
        if _looks_like_identifier(str(column).lower()):
            continue
        unique_count = numeric[column].dropna().nunique()
        if 2 <= unique_count <= min(10, max(2, int(len(df) * 0.3))):
            return str(column)
    return None


def _looks_like_identifier(name: str) -> bool:
    identifier_keywords = ("id", "code", "编号", "代码", "序号", "日期", "时间", "date", "time")
    return any(keyword == name or keyword in name for keyword in identifier_keywords)


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (values - mean) / std, mean, std
