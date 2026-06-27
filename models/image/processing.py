from __future__ import annotations

import math

import numpy as np
import pandas as pd


def histogram_equalization(df: pd.DataFrame) -> pd.DataFrame:
    image = _extract_image(df)
    if image is None:
        return pd.DataFrame()

    matrix, _ = image
    normalized = _normalize_image(matrix)
    if normalized is None:
        return pd.DataFrame()

    flat = normalized.ravel()
    histogram, edges = np.histogram(flat, bins=256, range=(0.0, 1.0))
    cdf = histogram.cumsum().astype(float)
    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0 or cdf[-1] == nonzero[0]:
        return pd.DataFrame()
    cdf = (cdf - nonzero[0]) / (cdf[-1] - nonzero[0])
    equalized = np.interp(flat, edges[:-1], cdf).reshape(normalized.shape)

    rows: list[dict[str, float | str | int]] = []
    for row, col in np.ndindex(normalized.shape):
        rows.append(
            {
                "row": int(row),
                "col": int(col),
                "intensity": float(normalized[row, col]),
                "equalized_intensity": float(equalized[row, col]),
                "height": int(normalized.shape[0]),
                "width": int(normalized.shape[1]),
                "method": "histogram_equalization",
            }
        )
    return pd.DataFrame(rows)


def edge_detection_sobel(df: pd.DataFrame) -> pd.DataFrame:
    image = _extract_image(df)
    if image is None:
        return pd.DataFrame()

    matrix, _ = image
    normalized = _normalize_image(matrix)
    if normalized is None or min(normalized.shape) < 2:
        return pd.DataFrame()

    padded = np.pad(normalized, 1, mode="edge")
    kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    grad_x = np.zeros_like(normalized, dtype=float)
    grad_y = np.zeros_like(normalized, dtype=float)
    for row, col in np.ndindex(normalized.shape):
        window = padded[row : row + 3, col : col + 3]
        grad_x[row, col] = float(np.sum(window * kernel_x))
        grad_y[row, col] = float(np.sum(window * kernel_y))
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_magnitude = float(magnitude.max())
    edge_strength = magnitude / max_magnitude if max_magnitude > 0 else magnitude

    rows: list[dict[str, float | str | int]] = []
    for row, col in np.ndindex(normalized.shape):
        rows.append(
            {
                "row": int(row),
                "col": int(col),
                "gradient_x": float(grad_x[row, col]),
                "gradient_y": float(grad_y[row, col]),
                "edge_magnitude": float(magnitude[row, col]),
                "edge_strength": float(edge_strength[row, col]),
                "method": "sobel_edge_detection",
            }
        )
    return pd.DataFrame(rows)


def threshold_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    image = _extract_image(df)
    if image is None:
        return pd.DataFrame()

    matrix, _ = image
    normalized = _normalize_image(matrix)
    if normalized is None:
        return pd.DataFrame()

    threshold = _otsu_threshold(normalized)
    if threshold is None:
        return pd.DataFrame()
    mask = normalized >= threshold
    foreground_ratio = float(mask.mean())

    rows: list[dict[str, float | str | int]] = []
    for row, col in np.ndindex(normalized.shape):
        rows.append(
            {
                "row": int(row),
                "col": int(col),
                "intensity": float(normalized[row, col]),
                "threshold": float(threshold),
                "segment": int(mask[row, col]),
                "foreground_ratio": foreground_ratio,
                "method": "otsu_threshold_segmentation",
            }
        )
    return pd.DataFrame(rows)


def image_feature_extraction(df: pd.DataFrame) -> pd.DataFrame:
    image = _extract_image(df)
    if image is None:
        return pd.DataFrame()

    matrix, _ = image
    normalized = _normalize_image(matrix)
    if normalized is None or min(normalized.shape) < 3:
        return pd.DataFrame()

    grad_y, grad_x = np.gradient(normalized)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = (np.degrees(np.arctan2(grad_y, grad_x)) + 180.0) % 180.0
    bins = np.linspace(0.0, 180.0, 10)
    hog, _ = np.histogram(angle, bins=bins, weights=magnitude)
    hog_total = float(hog.sum())
    hog = hog / hog_total if hog_total > 0 else hog

    lbp_codes = []
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    for row in range(1, normalized.shape[0] - 1):
        for col in range(1, normalized.shape[1] - 1):
            center = normalized[row, col]
            code = 0
            transitions = 0
            previous = None
            bits = []
            for bit, (dy, dx) in enumerate(offsets):
                value = int(normalized[row + dy, col + dx] >= center)
                code |= value << bit
                bits.append(value)
                if previous is not None and previous != value:
                    transitions += 1
                previous = value
            if bits and bits[-1] != bits[0]:
                transitions += 1
            lbp_codes.append((code, transitions))

    codes = np.array([item[0] for item in lbp_codes], dtype=float)
    transitions = np.array([item[1] for item in lbp_codes], dtype=float)
    if len(codes) == 0:
        return pd.DataFrame()
    lbp_hist, _ = np.histogram(codes, bins=16, range=(0, 256))
    lbp_hist = lbp_hist.astype(float) / max(float(lbp_hist.sum()), 1.0)

    rows: list[dict[str, float | str | int]] = []
    for idx, value in enumerate(hog):
        rows.append(
            {
                "feature": f"hog_orientation_bin_{idx + 1}",
                "value": float(value),
                "height": int(normalized.shape[0]),
                "width": int(normalized.shape[1]),
                "method": "hog_lbp_statistical_features",
            }
        )
    for idx, value in enumerate(lbp_hist):
        rows.append(
            {
                "feature": f"lbp_code_bin_{idx + 1}",
                "value": float(value),
                "height": int(normalized.shape[0]),
                "width": int(normalized.shape[1]),
                "method": "hog_lbp_statistical_features",
            }
        )
    rows.extend(
        [
            {
                "feature": "gradient_mean",
                "value": float(magnitude.mean()),
                "height": int(normalized.shape[0]),
                "width": int(normalized.shape[1]),
                "method": "hog_lbp_statistical_features",
            },
            {
                "feature": "gradient_std",
                "value": float(magnitude.std(ddof=0)),
                "height": int(normalized.shape[0]),
                "width": int(normalized.shape[1]),
                "method": "hog_lbp_statistical_features",
            },
            {
                "feature": "lbp_uniform_ratio",
                "value": float((transitions <= 2).mean()),
                "height": int(normalized.shape[0]),
                "width": int(normalized.shape[1]),
                "method": "hog_lbp_statistical_features",
            },
        ]
    )
    return pd.DataFrame(rows)


def image_registration_shift(df: pd.DataFrame) -> pd.DataFrame:
    pair = _extract_image_pair(df)
    if pair is None:
        return pd.DataFrame()

    reference, moving = pair
    reference_norm = _normalize_image(reference)
    moving_norm = _normalize_image(moving)
    if reference_norm is None or moving_norm is None or reference_norm.shape != moving_norm.shape:
        return pd.DataFrame()
    if min(reference_norm.shape) < 2:
        return pd.DataFrame()

    max_shift_y = min(8, max(1, reference_norm.shape[0] // 2))
    max_shift_x = min(8, max(1, reference_norm.shape[1] // 2))
    best: tuple[float, int, int, int] | None = None
    for dy in range(-max_shift_y, max_shift_y + 1):
        for dx in range(-max_shift_x, max_shift_x + 1):
            ref_slice, mov_slice = _overlap_slices(reference_norm, moving_norm, dy, dx)
            if ref_slice.size < 4:
                continue
            ref_centered = ref_slice.ravel() - float(ref_slice.mean())
            mov_centered = mov_slice.ravel() - float(mov_slice.mean())
            denom = float(np.linalg.norm(ref_centered) * np.linalg.norm(mov_centered))
            score = float(ref_centered @ mov_centered / denom) if denom > 0 else -1.0
            overlap = int(ref_slice.size)
            if best is None or score > best[0]:
                best = (score, dy, dx, overlap)

    if best is None:
        return pd.DataFrame()
    score, dy, dx, overlap = best
    return pd.DataFrame(
        [
            {
                "shift_y": int(dy),
                "shift_x": int(dx),
                "correlation": float(score),
                "overlap_pixels": int(overlap),
                "height": int(reference_norm.shape[0]),
                "width": int(reference_norm.shape[1]),
                "method": "brute_force_translation_registration",
            }
        ]
    )


def _extract_image(df: pd.DataFrame) -> tuple[np.ndarray, str] | None:
    if df.empty:
        return None

    long_form = _extract_long_form_image(df)
    if long_form is not None:
        return long_form, "long_form"

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return None
    matrix = numeric.fillna(numeric.mean(numeric_only=True)).to_numpy(dtype=float)
    if matrix.size < 4:
        return None
    return matrix, "matrix"


def _extract_long_form_image(df: pd.DataFrame) -> np.ndarray | None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[0] < 4 or numeric.shape[1] < 3:
        return None

    row_column = _find_named_column(numeric, ("row", "y", "height", "r"))
    col_column = _find_named_column(numeric, ("col", "column", "x", "width", "c"))
    value_column = _find_named_column(numeric, ("value", "intensity", "pixel", "gray", "grey", "brightness"))
    if row_column is None or col_column is None or value_column is None:
        return None

    work = numeric[[row_column, col_column, value_column]].dropna()
    if work.empty:
        return None
    rows = {value: idx for idx, value in enumerate(sorted(work[row_column].unique()))}
    cols = {value: idx for idx, value in enumerate(sorted(work[col_column].unique()))}
    if len(rows) < 2 or len(cols) < 2:
        return None
    matrix = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for _, item in work.iterrows():
        matrix[rows[item[row_column]], cols[item[col_column]]] = float(item[value_column])
    if np.isnan(matrix).all():
        return None
    fill = float(np.nanmean(matrix))
    return np.where(np.isnan(matrix), fill, matrix)


def _extract_image_pair(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.empty:
        return None

    row_column = _find_named_column(numeric, ("row", "y", "height", "r"))
    col_column = _find_named_column(numeric, ("col", "column", "x", "width", "c"))
    if row_column is not None and col_column is not None:
        value_columns = [column for column in numeric.columns if column not in {row_column, col_column}]
        if len(value_columns) >= 2:
            first = _pivot_pair_image(numeric, row_column, col_column, value_columns[0])
            second = _pivot_pair_image(numeric, row_column, col_column, value_columns[1])
            if first is not None and second is not None and first.shape == second.shape:
                return first, second

    numeric = numeric.fillna(numeric.mean(numeric_only=True))
    if numeric.shape[1] == 2 and numeric.shape[0] >= 4:
        first = _reshape_vector(numeric.iloc[:, 0].to_numpy(dtype=float))
        second = _reshape_vector(numeric.iloc[:, 1].to_numpy(dtype=float))
        if first is not None and second is not None and first.shape == second.shape:
            return first, second

    matrix = numeric.to_numpy(dtype=float)
    if matrix.shape[0] % 2 == 0 and matrix.shape[0] >= 4 and matrix.shape[1] >= 2:
        midpoint = matrix.shape[0] // 2
        return matrix[:midpoint, :], matrix[midpoint:, :]
    if matrix.shape[1] % 2 == 0 and matrix.shape[1] >= 4 and matrix.shape[0] >= 2:
        midpoint = matrix.shape[1] // 2
        return matrix[:, :midpoint], matrix[:, midpoint:]
    return None


def _pivot_pair_image(df: pd.DataFrame, row_column: str, col_column: str, value_column: str) -> np.ndarray | None:
    work = df[[row_column, col_column, value_column]].dropna()
    if work.empty:
        return None
    rows = {value: idx for idx, value in enumerate(sorted(work[row_column].unique()))}
    cols = {value: idx for idx, value in enumerate(sorted(work[col_column].unique()))}
    if len(rows) < 2 or len(cols) < 2:
        return None
    matrix = np.full((len(rows), len(cols)), np.nan, dtype=float)
    for _, item in work.iterrows():
        matrix[rows[item[row_column]], cols[item[col_column]]] = float(item[value_column])
    if np.isnan(matrix).all():
        return None
    return np.where(np.isnan(matrix), float(np.nanmean(matrix)), matrix)


def _reshape_vector(values: np.ndarray) -> np.ndarray | None:
    length = len(values)
    if length < 4:
        return None
    root = int(math.sqrt(length))
    for height in range(root, 1, -1):
        if length % height == 0:
            width = length // height
            if width >= 2:
                return values.reshape(height, width)
    return None


def _normalize_image(matrix: np.ndarray) -> np.ndarray | None:
    if matrix.size < 4:
        return None
    values = np.asarray(matrix, dtype=float)
    if not np.isfinite(values).all():
        finite_mean = float(np.nanmean(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.0
        values = np.where(np.isfinite(values), values, finite_mean)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return None
    return (values - minimum) / (maximum - minimum)


def _otsu_threshold(image: np.ndarray) -> float | None:
    flat = image.ravel()
    if flat.size < 4 or np.allclose(flat, flat[0]):
        return None
    histogram, edges = np.histogram(flat, bins=256, range=(0.0, 1.0))
    total = float(flat.size)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_background = np.cumsum(histogram).astype(float)
    weight_foreground = total - weight_background
    mean_background = np.cumsum(histogram * centers) / np.maximum(weight_background, 1.0)
    reverse_mean = np.cumsum((histogram * centers)[::-1]) / np.maximum(np.cumsum(histogram[::-1]), 1.0)
    mean_foreground = reverse_mean[::-1]
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    variance[(weight_background == 0) | (weight_foreground == 0)] = -1.0
    idx = int(np.argmax(variance))
    if variance[idx] < 0:
        return None
    return float(centers[idx])


def _overlap_slices(reference: np.ndarray, moving: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference.shape
    ref_y0 = max(0, dy)
    ref_y1 = min(height, height + dy)
    mov_y0 = max(0, -dy)
    mov_y1 = min(height, height - dy)
    ref_x0 = max(0, dx)
    ref_x1 = min(width, width + dx)
    mov_x0 = max(0, -dx)
    mov_x1 = min(width, width - dx)
    return reference[ref_y0:ref_y1, ref_x0:ref_x1], moving[mov_y0:mov_y1, mov_x0:mov_x1]


def _find_named_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for column in df.columns:
        lower = str(column).lower()
        if any(name == lower or name in lower for name in names):
            return str(column)
    return None
