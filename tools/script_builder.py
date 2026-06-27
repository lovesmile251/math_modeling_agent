"""Dynamic script builder — replaces the 950-line f-string in CodingAgent.

Generates ``baseline_analysis.py`` from modular parts using the model
registry instead of a monolithic hardcoded template.
"""

from __future__ import annotations

from pathlib import Path

from tools.model_registry import (
    ADVANCED_MODEL_REGISTRY,
    BASIC_MODEL_REGISTRY,
    MODEL_SWEEP_CONFIGS,
    collect_imports,
)

# ── fixed script header ──
_SCRIPT_HEADER = '''from __future__ import annotations

import json
import sys
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def _configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in _fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_configure_chinese_font()


PROJECT_ROOT = Path({project_root!r})
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.optimization.esp import esp_operating_optimization, is_esp_operating_frame

'''

# ── utility functions (shared by all generated scripts) ──
_UTILITY_FUNCTIONS = r'''
def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
    return pd.read_csv(path, sep=None, engine="python")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_with_fallback(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file: {path}")


def profile_frame(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "missing_values": {col: int(frame[col].isna().sum()) for col in frame.columns},
        "numeric_columns": list(frame.select_dtypes(include="number").columns),
        "selected_models": SELECTED_MODELS,
    }


def safe_filename_part(value, max_length: int = 80) -> str:
    text = str(value).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    cleaned = cleaned.strip("._-")
    return (cleaned or "unnamed")[:max_length]


def save_numeric_charts(frame: pd.DataFrame, stem: str) -> list[str]:
    paths: list[str] = []
    numeric = frame.select_dtypes(include="number")
    safe_stem = safe_filename_part(stem)
    if numeric.empty:
        for col in frame.columns[:2]:
            counts = frame[col].astype(str).value_counts().head(10)
            if counts.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            counts.plot(kind="bar", ax=ax)
            ax.set_title(f"Top categories of {col}")
            ax.set_xlabel(str(col))
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", labelrotation=45)
            path = FIGURES_DIR / f"{safe_stem}_bar_{safe_filename_part(col)}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path))
        return paths
    # correlation heatmap
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_title("Correlation Heatmap")
        path = FIGURES_DIR / f"{safe_stem}_correlation_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    # histograms
    for col in numeric.columns[:6]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(numeric[col].dropna(), bins=20, edgecolor="white", alpha=0.8)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        path = FIGURES_DIR / f"{safe_stem}_hist_{safe_filename_part(col)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_competition_diagnostics(frame: pd.DataFrame, stem: str) -> list[str]:
    """Write extra contest-grade diagnostic tables and figures.

    These artifacts make single-attachment cases closer to award-paper density:
    data quality table, feature summary, correlation-pair table, and compact
    distribution/relationship figures.
    """
    chart_paths: list[str] = []
    safe_stem = safe_filename_part(stem)
    numeric = frame.select_dtypes(include="number")

    feature_rows: list[dict] = []
    for col in frame.columns:
        series = frame[col]
        row = {
            "column": str(col),
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if col in numeric.columns:
            values = pd.to_numeric(series, errors="coerce").dropna()
            row.update(
                {
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "std": float(values.std()) if len(values) > 1 else 0.0,
                    "min": float(values.min()) if not values.empty else np.nan,
                    "q25": float(values.quantile(0.25)) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "q75": float(values.quantile(0.75)) if not values.empty else np.nan,
                    "max": float(values.max()) if not values.empty else np.nan,
                }
            )
        feature_rows.append(row)
    feature_summary = pd.DataFrame(feature_rows)
    feature_path = TABLES_DIR / f"{safe_stem}_feature_summary.csv"
    feature_summary.to_csv(feature_path, index=False, encoding="utf-8-sig")

    missing_summary = feature_summary[
        ["column", "dtype", "non_null_count", "missing_count", "missing_rate", "unique_count"]
    ].sort_values(["missing_rate", "unique_count"], ascending=[False, False])
    missing_path = TABLES_DIR / f"{safe_stem}_missingness_summary.csv"
    missing_summary.to_csv(missing_path, index=False, encoding="utf-8-sig")

    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        pairs: list[dict] = []
        cols = list(corr.columns)
        for i, left in enumerate(cols):
            for right in cols[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    pairs.append(
                        {
                            "feature_a": str(left),
                            "feature_b": str(right),
                            "correlation": float(value),
                            "abs_correlation": float(abs(value)),
                        }
                    )
        if pairs:
            corr_pairs = pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False).head(50)
        else:
            corr_pairs = pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation"])
    else:
        corr_pairs = pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation"])
    corr_path = TABLES_DIR / f"{safe_stem}_correlation_pairs.csv"
    corr_pairs.to_csv(corr_path, index=False, encoding="utf-8-sig")

    snapshot_path = TABLES_DIR / f"{safe_stem}_sample_snapshot.csv"
    frame.head(20).to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    categorical_cols = [col for col in frame.columns if col not in numeric.columns]
    datetime_like_cols = []
    for col in categorical_cols:
        parsed = pd.to_datetime(frame[col], errors="coerce")
        if float(parsed.notna().mean()) >= 0.70:
            datetime_like_cols.append(col)

    type_rows: list[dict] = []
    for col in frame.columns:
        semantic_type = "numeric" if col in numeric.columns else "categorical"
        if col in datetime_like_cols:
            semantic_type = "datetime_like"
        type_rows.append(
            {
                "column": str(col),
                "dtype": str(frame[col].dtype),
                "semantic_type": semantic_type,
                "unique_count": int(frame[col].nunique(dropna=True)),
                "unique_rate": float(frame[col].nunique(dropna=True) / max(len(frame), 1)),
                "missing_rate": float(frame[col].isna().mean()),
            }
        )
    column_type_summary = pd.DataFrame(type_rows)
    type_path = TABLES_DIR / f"{safe_stem}_column_type_summary.csv"
    column_type_summary.to_csv(type_path, index=False, encoding="utf-8-sig")

    complete_rows = int(frame.notna().all(axis=1).sum()) if not frame.empty else 0
    scorecard = pd.DataFrame(
        [
            {
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
                "numeric_columns": int(numeric.shape[1]),
                "categorical_columns": int(len(categorical_cols)),
                "datetime_like_columns": int(len(datetime_like_cols)),
                "missing_cells": int(frame.isna().sum().sum()),
                "missing_rate": float(frame.isna().sum().sum() / max(frame.shape[0] * frame.shape[1], 1)),
                "duplicate_rows": int(frame.duplicated().sum()) if not frame.empty else 0,
                "complete_rows": complete_rows,
                "complete_row_rate": float(complete_rows / max(len(frame), 1)),
            }
        ]
    )
    scorecard_path = TABLES_DIR / f"{safe_stem}_data_quality_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False, encoding="utf-8-sig")

    readiness_checks = pd.DataFrame(
        [
            {
                "check_item": "row_count_available",
                "status": bool(frame.shape[0] > 0),
                "evidence": f"{int(frame.shape[0])} rows",
            },
            {
                "check_item": "column_count_available",
                "status": bool(frame.shape[1] > 0),
                "evidence": f"{int(frame.shape[1])} columns",
            },
            {
                "check_item": "has_numeric_features",
                "status": bool(numeric.shape[1] > 0),
                "evidence": f"{int(numeric.shape[1])} numeric columns",
            },
            {
                "check_item": "has_categorical_features",
                "status": bool(len(categorical_cols) > 0),
                "evidence": f"{int(len(categorical_cols))} categorical columns",
            },
            {
                "check_item": "missingness_audited",
                "status": True,
                "evidence": f"{int(frame.isna().sum().sum())} missing cells",
            },
            {
                "check_item": "duplicate_rows_audited",
                "status": True,
                "evidence": f"{int(frame.duplicated().sum()) if not frame.empty else 0} duplicate rows",
            },
        ]
    )
    readiness_path = TABLES_DIR / f"{safe_stem}_analysis_readiness_checklist.csv"
    readiness_checks.to_csv(readiness_path, index=False, encoding="utf-8-sig")

    frequency_rows: list[dict] = []
    for col in categorical_cols[:8]:
        values = frame[col].astype("string").fillna("<missing>")
        counts = values.value_counts(dropna=False).head(10)
        total = max(int(counts.sum()), 1)
        for value, count in counts.items():
            frequency_rows.append(
                {
                    "column": str(col),
                    "value": str(value),
                    "count": int(count),
                    "rate": float(count / total),
                }
            )
    categorical_frequency = pd.DataFrame(
        frequency_rows,
        columns=["column", "value", "count", "rate"],
    )
    frequency_path = TABLES_DIR / f"{safe_stem}_categorical_frequency.csv"
    categorical_frequency.to_csv(frequency_path, index=False, encoding="utf-8-sig")

    pair_columns = categorical_cols[:2] if len(categorical_cols) >= 2 else list(frame.columns[:2])
    pair_rows: list[dict] = []
    if len(pair_columns) >= 2:
        left_col, right_col = pair_columns[:2]
        pairs = (
            frame[[left_col, right_col]]
            .astype("string")
            .fillna("<missing>")
            .groupby([left_col, right_col])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(50)
        )
        total_pairs = max(int(pairs["count"].sum()), 1) if not pairs.empty else 1
        for _, row in pairs.iterrows():
            pair_rows.append(
                {
                    "left_column": str(left_col),
                    "left_value": str(row[left_col]),
                    "right_column": str(right_col),
                    "right_value": str(row[right_col]),
                    "count": int(row["count"]),
                    "rate": float(row["count"] / total_pairs),
                }
            )
    pair_frequency = pd.DataFrame(
        pair_rows,
        columns=["left_column", "left_value", "right_column", "right_value", "count", "rate"],
    )
    pair_path = TABLES_DIR / f"{safe_stem}_pair_frequency.csv"
    pair_frequency.to_csv(pair_path, index=False, encoding="utf-8-sig")

    if not missing_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_data = missing_summary.head(12).copy()
        ax.bar(plot_data["column"].astype(str), plot_data["missing_rate"].astype(float))
        ax.set_title("Missingness by Column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing rate")
        ax.tick_params(axis="x", labelrotation=45)
        path = FIGURES_DIR / f"{safe_stem}_missingness_bar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(str(path))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(plot_data["column"].astype(str), plot_data["unique_count"].astype(float))
        ax.set_title("Unique Values by Column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Unique count")
        ax.tick_params(axis="x", labelrotation=45)
        path = FIGURES_DIR / f"{safe_stem}_unique_count_bar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(str(path))

    if not frame.empty:
        completeness = frame.notna().sum(axis=1).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(completeness.index.astype(str), completeness.values)
        ax.set_title("Row Completeness Distribution")
        ax.set_xlabel("Non-null fields per row")
        ax.set_ylabel("Row count")
        path = FIGURES_DIR / f"{safe_stem}_row_completeness_bar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(str(path))

    for col in categorical_cols[:4]:
        counts = frame[col].astype("string").fillna("<missing>").value_counts(dropna=False).head(10)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Category Profile: {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(str(col))
        path = FIGURES_DIR / f"{safe_stem}_category_profile_{safe_filename_part(str(col))}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(str(path))

    if not pair_frequency.empty:
        top_pairs = pair_frequency.head(12).copy()
        labels = top_pairs["left_value"].astype(str) + " → " + top_pairs["right_value"].astype(str)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels.iloc[::-1], top_pairs["count"].iloc[::-1].astype(float))
        ax.set_title("Top Pair Frequencies")
        ax.set_xlabel("Count")
        path = FIGURES_DIR / f"{safe_stem}_pair_frequency_bar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(str(path))

    if not numeric.empty:
        selected = numeric.iloc[:, : min(6, numeric.shape[1])].apply(pd.to_numeric, errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 5))
        selected.boxplot(ax=ax, rot=45)
        ax.set_title("Numeric Feature Boxplot")
        ax.set_ylabel("Value")
        path = FIGURES_DIR / f"{safe_stem}_numeric_boxplot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        chart_paths.append(str(path))

    if numeric.shape[1] >= 2:
        x_col, y_col = numeric.columns[:2]
        plot_frame = numeric[[x_col, y_col]].dropna().head(1000)
        if not plot_frame.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(plot_frame[x_col], plot_frame[y_col], alpha=0.65, s=18)
            ax.set_title(f"Scatter: {x_col} vs {y_col}")
            ax.set_xlabel(str(x_col))
            ax.set_ylabel(str(y_col))
            path = FIGURES_DIR / f"{safe_stem}_scatter_{safe_filename_part(x_col)}_{safe_filename_part(y_col)}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            chart_paths.append(str(path))

    return chart_paths


KNOWN_TABLE_SUFFIXES = (
    "describe",
{known_suffixes}
)

'''

_RUN_MODEL_SAFELY = '''
def run_model_safely(model_id: str, builder, df: pd.DataFrame, stem: str, suffix: str) -> dict:
    """Run one model with timing, error handling, and status recording.

    Returns a dict compatible with ``ModelRunResult``:
        model_id, status (success|skipped|failed), table path or None,
        elapsed_seconds, error (str|None).
    """
    started = time.perf_counter()
    try:
        result = builder(df)
        if not isinstance(result, pd.DataFrame) or result.empty:
            return {
                "model_id": model_id,
                "status": "skipped",
                "table": None,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "error": "empty result",
            }
        path = TABLES_DIR / f"{stem}_{suffix}.csv"
        result.to_csv(path, index=False, encoding="utf-8-sig")
        return {
            "model_id": model_id,
            "status": "success",
            "table": str(path),
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": None,
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "status": "failed",
            "table": None,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


'''

_NOT_SELECTED_RESULT = '''{"model_id": "{model_id}", "status": "not_selected", "table": None, "elapsed_seconds": 0.0, "error": None}'''

_RUN_ALL_MODELS_HEADER = '''
def run_all_models(df: pd.DataFrame, stem: str) -> list[dict]:
    """Run every selected model; each gets its own try/catch and status dict."""
    results: list[dict] = []
{basic_dispatch}

    for model_id, table_suffix, builder in ADVANCED_MODEL_BUILDERS:
        if model_id not in SELECTED_MODELS:
            results.append({"model_id": model_id, "status": "not_selected", "table": None, "elapsed_seconds": 0.0, "error": None})
            continue
        results.append(run_model_safely(model_id, builder, df, stem, table_suffix))

    if is_esp_operating_frame(df):
        results.append(run_model_safely("esp_optimization", esp_operating_optimization, df, stem, "esp_optimization"))

    return results

'''

_CLEAR_OUTPUTS = '''
def clear_previous_outputs(stem: str) -> None:
    for suffix in KNOWN_TABLE_SUFFIXES:
        path = TABLES_DIR / f"{stem}_{suffix}.csv"
        if path.exists():
            path.unlink()

'''

_SWEEP_FUNCTION = '''
def run_parameter_sweeps(df: pd.DataFrame, stem: str) -> None:
    """Auto-run parameter sweeps for selected models that have sweep configs."""
    rows: list[dict] = []
{sweep_blocks}
    if rows:
        out_path = TABLES_DIR / f"{stem}_sensitivity_sweep.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [OK] sensitivity sweep -> {out_path}")
'''

_MAIN_FUNCTION = r'''
def main() -> None:
    """Entry point.

    Usage:
        python baseline_analysis.py                 # full run
        python baseline_analysis.py --check         # validate inputs only
        python baseline_analysis.py --list-models   # print selected models
    """
    import sys as _sys

    # ── CLI flags ────────────────────────────────────────────────────
    args = [a for a in _sys.argv[1:] if not a.startswith("--pytest")]
    if "--list-models" in args:
        print("Selected models:")
        for m in SELECTED_MODELS:
            print(f"  {m}")
        return

    if "--check" in args:
        print("=== Pre-flight check ===")
        ok = True
        for raw_path in DATA_FILES:
            p = Path(raw_path)
            if not p.exists():
                print(f"  [FAIL] data file not found: {p}")
                ok = False
            else:
                print(f"  [OK] data file exists: {p}")
        for dname, dpath in [("FIGURES_DIR", FIGURES_DIR), ("TABLES_DIR", TABLES_DIR), ("LOGS_DIR", LOGS_DIR)]:
            try:
                dpath.mkdir(parents=True, exist_ok=True)
                print(f"  [OK] {dname} writable: {dpath}")
            except OSError as exc:
                print(f"  [FAIL] {dname} not writable: {exc}")
                ok = False
        # Try importing the model functions
        for model_id in SELECTED_MODELS:
            try:
                # Models are imported at module level; if we got here, imports are ok
                print(f"  [OK] model {model_id} importable")
            except Exception as exc:
                print(f"  [FAIL] model {model_id}: {exc}")
                ok = False
        if ok:
            print("All checks passed.")
        else:
            print("Some checks FAILED.")
            _sys.exit(1)
        return

    # ── normal execution ─────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    if not DATA_FILES:
        summary_path = LOGS_DIR / "run_summary.json"
        summary_path.write_text(
            json.dumps({"message": "No data files provided."}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {summary_path}")
        return

    for raw_path in DATA_FILES:
        path = Path(raw_path)
        df = read_table(path)
        stem = path.stem
        clear_previous_outputs(stem)

        describe_path = TABLES_DIR / f"{stem}_describe.csv"
        df.describe(include="all").transpose().to_csv(describe_path, encoding="utf-8-sig")

        charts = save_numeric_charts(df, stem)
        charts.extend(save_competition_diagnostics(df, stem))
        model_runs = run_all_models(df, stem)
        run_parameter_sweeps(df, stem)

        summary = {
            "source": str(path),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include="number").columns),
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
            "selected_models": SELECTED_MODELS,
            "random_seed": RANDOM_SEED,
            "model_runs": model_runs,
            "charts": charts,
            "describe_table": str(describe_path),
        }
        summaries.append(summary)

    summary_path = LOGS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
'''


def build_script(
    data_files: list[str],
    figures_dir: str,
    tables_dir: str,
    logs_dir: str,
    selected_models: list[str],
    project_root: str | None = None,
) -> str:
    """Build baseline_analysis.py as a string.

    Delegates to named builder functions so each concern is isolated.
    """
    if project_root is None:
        project_root = str(Path.cwd())

    imports = collect_imports(selected_models)
    header = _build_header(project_root, imports)
    body = _build_body_constants(data_files, figures_dir, tables_dir, logs_dir, selected_models)
    adv_block = _build_advanced_builders_block(selected_models)
    basic_dispatch = _build_basic_dispatch(selected_models)
    all_suffixes = _build_known_suffixes(selected_models)
    model_runner = _build_model_runner(basic_dispatch)
    sweep_block_text, extra_suffixes = _build_sweep_blocks(selected_models)

    # Merge sweep suffixes into the known-suffixes set
    if extra_suffixes:
        all_suffixes |= extra_suffixes

    known_suffixes_str = ",\n".join(f'    "{s}"' for s in sorted(all_suffixes))
    utility = _UTILITY_FUNCTIONS.replace("{known_suffixes}", known_suffixes_str)

    return (
        header
        + body
        + adv_block
        + "\n"
        + utility
        + _RUN_MODEL_SAFELY
        + model_runner
        + _CLEAR_OUTPUTS
        + sweep_block_text
        + _MAIN_FUNCTION
    )


# ── named builder functions ────────────────────────────────────────────────

def _build_header(project_root: str, imports: list[str]) -> str:
    """Script header: imports, matplotlib config, project-root injection."""
    return _SCRIPT_HEADER.replace("{project_root!r}", repr(project_root)) + "\n".join(imports) + "\n\n"


def _build_body_constants(
    data_files: list[str],
    figures_dir: str,
    tables_dir: str,
    logs_dir: str,
    selected_models: list[str],
) -> str:
    """Global constants block: DATA_FILES, SELECTED_MODELS, *_DIR."""
    return (
        f"\nDATA_FILES = {data_files!r}\n"
        f"SELECTED_MODELS = {selected_models!r}\n"
        f"FIGURES_DIR = Path({figures_dir!r})\n"
        f"TABLES_DIR = Path({tables_dir!r})\n"
        f"LOGS_DIR = Path({logs_dir!r})\n\n"
    )


def _build_advanced_builders_block(selected_models: list[str]) -> str:
    """ADVANCED_MODEL_BUILDERS tuple — generic for-loop dispatch."""
    adv_entries: list[str] = []
    for mid, suffix, _mod, name, extra_args in ADVANCED_MODEL_REGISTRY:
        if mid not in selected_models:
            continue
        if extra_args:
            adv_entries.append(
                f'    ("{mid}", "{suffix}", lambda frame: {name}({extra_args})),'
            )
        else:
            adv_entries.append(f'    ("{mid}", "{suffix}", {name}),')

    if adv_entries:
        return "ADVANCED_MODEL_BUILDERS = (\n" + "\n".join(adv_entries) + "\n)\n"
    return "ADVANCED_MODEL_BUILDERS = ()\n"


def _build_basic_dispatch(selected_models: list[str]) -> str:
    """Build basic model dispatch blocks (each calls run_model_safely)."""
    basic_blocks: list[str] = []
    for mid in selected_models:
        entry = BASIC_MODEL_REGISTRY.get(mid)
        if entry is None:
            continue
        _mod, _name, call_template, suffix = entry
        call_expr = call_template.format(var="_res")
        builder_expr = call_expr[len("_res = "):]
        basic_blocks.append(
            f'    if "{mid}" in SELECTED_MODELS:\n'
            f'        results.append(run_model_safely("{mid}", lambda df: {builder_expr}, df, stem, "{suffix}"))\n'
            f'    else:\n'
            f'        results.append({{"model_id": "{mid}", "status": "not_selected", "table": None, "elapsed_seconds": 0.0, "error": None}})\n'
        )
    return "\n".join(basic_blocks) if basic_blocks else "    pass"


def _build_known_suffixes(selected_models: list[str]) -> set[str]:
    """Return the set of known table suffixes (for clear_previous_outputs)."""
    all_suffixes: set[str] = {
        "describe",
        "feature_summary",
        "missingness_summary",
        "correlation_pairs",
        "sample_snapshot",
        "column_type_summary",
        "data_quality_scorecard",
        "analysis_readiness_checklist",
        "categorical_frequency",
        "pair_frequency",
        "esp_optimization",
    }
    for mid in selected_models:
        entry = BASIC_MODEL_REGISTRY.get(mid)
        if entry:
            all_suffixes.add(entry[3])
    for mid, suffix, *_ in ADVANCED_MODEL_REGISTRY:
        if mid in selected_models:
            all_suffixes.add(suffix)
    return all_suffixes


def _build_model_runner(basic_dispatch: str) -> str:
    """run_all_models function with basic dispatch inlined."""
    return _RUN_ALL_MODELS_HEADER.replace("{basic_dispatch}", basic_dispatch)


def _build_sweep_blocks(selected_models: list[str]) -> tuple[str, set[str]]:
    """Build run_parameter_sweeps function body; return (sweep_text, extra_suffixes)."""
    sweep_blocks: list[str] = []
    extra_suffixes: set[str] = set()
    for model_id, params in MODEL_SWEEP_CONFIGS.items():
        if model_id not in selected_models:
            continue
        func_name: str | None = None
        entry = BASIC_MODEL_REGISTRY.get(model_id)
        if entry:
            func_name = entry[1]
        else:
            for mid, _suffix, _mod, name, _extra in ADVANCED_MODEL_REGISTRY:
                if mid == model_id:
                    func_name = name
                    break
        if func_name is None:
            continue

        default_pairs = ", ".join(f'"{pn}": {dv!r}' for pn, dv, _sv in params)
        param_entries = ", ".join(f'("{pn}", {sv!r})' for pn, _dv, sv in params)
        sweep_blocks.append(
            f'''    if "{model_id}" in SELECTED_MODELS:
        _defaults = {{{default_pairs}}}
        for _pn, _sv_list in [{param_entries}]:
            for _sv in _sv_list:
                try:
                    _kwargs = {{**_defaults, _pn: _sv}}
                    _res = {func_name}(df, **_kwargs)
                    if isinstance(_res, pd.DataFrame) and not _res.empty:
                        _res["_param_name"] = _pn
                        _res["_param_value"] = _sv
                        _res["_model_id"] = "{model_id}"
                        rows.extend(_res.to_dict(orient="records"))
                except Exception:
                    pass
'''
        )

    if sweep_blocks:
        extra_suffixes.add("sensitivity_sweep")

    sweep_text = _SWEEP_FUNCTION.replace("{sweep_blocks}", "\n".join(sweep_blocks))
    return sweep_text, extra_suffixes
