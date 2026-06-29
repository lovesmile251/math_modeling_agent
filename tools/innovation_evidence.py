from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.file_tool import write_text


_INNOVATION_PATTERNS: dict[str, tuple[str, ...]] = {
    "residual_correction_grey_markov": (
        "grey-markov",
        "grey markov",
        "gm(1,1)",
        "markov residual",
        "\u7070\u8272-\u9a6c\u5c14\u53ef\u592b",
        "\u6b8b\u5dee\u4fee\u6b63",
    ),
    "ahp_entropy_combined_weights": (
        "ahp-entropy",
        "ahp entropy",
        "\u5c42\u6b21\u5206\u6790-\u71b5\u6743",
        "\u7ec4\u5408\u8d4b\u6743",
    ),
    "stacking_ensemble": (
        "stacking",
        "ensemble",
        "\u96c6\u6210\u5b66\u4e60",
        "\u591a\u6a21\u578b\u96c6\u6210",
    ),
    "robust_optimization": (
        "robust optimization",
        "soyster",
        "ben-tal",
        "\u9c81\u68d2\u4f18\u5316",
        "\u7a33\u5065\u4f18\u5316",
    ),
    "global_sensitivity_analysis": (
        "global sensitivity",
        "sensitivity analysis",
        "spearman",
        "tornado",
        "\u5168\u5c40\u654f\u611f\u6027",
        "\u654f\u611f\u6027\u5206\u6790",
    ),
    "dynamic_weight_evaluation": (
        "dynamic weight",
        "sliding window weight",
        "\u52a8\u6001\u6743\u91cd",
        "\u6eda\u52a8\u7a97\u53e3",
    ),
    "graph_propagation_mechanism": (
        "propagation mechanism",
        "information propagation",
        "independent cascade",
        "sir",
        "\u4f20\u64ad\u673a\u7406",
        "\u4f20\u64ad\u4eff\u771f",
    ),
    "mechanism_data_fusion": (
        "mechanism-data",
        "mechanism data",
        "data fusion",
        "\u673a\u7406-\u6570\u636e",
        "\u878d\u5408\u5efa\u6a21",
    ),
    "monte_carlo_uncertainty": (
        "monte carlo",
        "uncertainty quantification",
        "\u8499\u7279\u5361\u6d1b",
        "\u4e0d\u786e\u5b9a\u6027\u91cf\u5316",
    ),
}


_REQUIREMENT_LABELS: dict[str, tuple[str, ...]] = {
    "residual_correction_grey_markov": (
        "rolling backtest evidence",
        "baseline comparison table",
    ),
    "ahp_entropy_combined_weights": (
        "combined weight output table",
        "weight stability or sensitivity evidence",
    ),
    "stacking_ensemble": (
        "model comparison table",
        "passed strong baseline audit",
    ),
    "robust_optimization": (
        "perturbation robustness evidence",
        "constraint violation or feasibility table",
    ),
    "global_sensitivity_analysis": (
        "sensitivity, ablation, or robustness table",
    ),
    "dynamic_weight_evaluation": (
        "dynamic weight table",
        "weight stability evidence",
    ),
    "graph_propagation_mechanism": (
        "propagation simulation table or figure",
    ),
    "mechanism_data_fusion": (
        "model comparison table",
        "feature ablation evidence",
    ),
    "monte_carlo_uncertainty": (
        "monte carlo or uncertainty output table",
    ),
}


def build_innovation_evidence_report(
    workspace,
    *,
    paper_text: str = "",
    model_selection_report: Path | None = None,
    experiment_report: Path | None = None,
) -> dict[str, Any]:
    """Audit whether innovation claims are backed by executed artifacts."""

    text_claims = _detect_claimed_innovations(paper_text)
    report_recommendations = _load_report_recommendations(model_selection_report)
    claimed = sorted(set(text_claims) | set(_explicit_report_claims(report_recommendations, paper_text)))
    evidence = _collect_evidence(workspace, experiment_report)
    audits = [_audit_one(innovation_id, evidence) for innovation_id in claimed]
    issues = [
        f"{item['innovation_id']}: {issue}"
        for item in audits
        for issue in item.get("issues", [])
        if issue
    ]
    return {
        "passed": not issues,
        "issues": issues,
        "claimed_innovations": claimed,
        "paper_claims": text_claims,
        "model_selection_recommendations": report_recommendations,
        "audits": audits,
        "evidence": evidence,
    }


def innovation_evidence_blocking_issues(report: dict[str, Any]) -> list[str]:
    if report.get("passed") is True:
        return []
    return [str(item) for item in report.get("issues", []) if str(item)]


def write_innovation_evidence_report(workspace, report: dict[str, Any]) -> Path:
    return write_text(
        workspace.logs_dir / "innovation_evidence_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def _detect_claimed_innovations(paper_text: str) -> list[str]:
    text = paper_text.lower()
    has_innovation_section = any(
        marker in text
        for marker in (
            "innovation",
            "\u521b\u65b0",
            "\u6a21\u578b\u521b\u65b0",
            "\u6539\u8fdb",
        )
    )
    claims: list[str] = []
    for innovation_id, patterns in _INNOVATION_PATTERNS.items():
        if any(pattern.lower() in text for pattern in patterns):
            claims.append(innovation_id)

    if not has_innovation_section:
        return claims
    return claims


def _load_report_recommendations(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [{"parse_error": str(path)}]
    raw = payload.get("innovation_recommendations", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _explicit_report_claims(recommendations: list[dict[str, Any]], paper_text: str) -> list[str]:
    if not paper_text:
        return []
    text = paper_text.lower()
    claims: list[str] = []
    for item in recommendations:
        haystack = " ".join(
            str(value)
            for key, value in item.items()
            if key in {"model_id", "label", "tier", "innovation_extensions", "reason"}
        ).lower()
        for innovation_id, patterns in _INNOVATION_PATTERNS.items():
            if any(pattern.lower() in haystack and pattern.lower() in text for pattern in patterns):
                claims.append(innovation_id)
    return claims


def _collect_evidence(workspace, experiment_report: Path | None) -> dict[str, Any]:
    files = [
        str(path.relative_to(workspace.root))
        for folder in (workspace.tables_dir, workspace.figures_dir, workspace.logs_dir)
        if Path(folder).exists()
        for path in Path(folder).glob("*")
        if path.is_file()
    ]
    file_text = " ".join(files).lower()
    report = _load_experiment_report(workspace, experiment_report)
    executed = report.get("executed_validation") if isinstance(report.get("executed_validation"), dict) else {}
    strong = report.get("strong_baseline_audit") if isinstance(report.get("strong_baseline_audit"), dict) else {}
    models = report.get("models") if isinstance(report.get("models"), list) else []
    field_proofs = _load_field_level_proofs(report)

    return {
        "files": files,
        "field_level_proofs": field_proofs,
        "has_model_comparison": "model_experiment_comparison" in file_text or bool(report.get("comparison_table")),
        "has_strong_baseline": strong.get("passed") is True,
        "has_rolling_backtest": bool(executed.get("rolling_backtest")) or "backtest" in file_text,
        "has_robustness": bool(executed.get("robustness")) or "robust" in file_text,
        "has_ablation": bool(executed.get("ablation")) or "ablation" in file_text,
        "has_sensitivity": any(token in file_text for token in ("sensitivity", "elasticity", "tornado")),
        "has_monte_carlo": any(token in file_text for token in ("monte", "uncertainty", "simulation")),
        "has_weight_output": any(token in file_text for token in ("weight", "entropy", "ahp")),
        "has_dynamic_weight_output": any(token in file_text for token in ("dynamic_weight", "rolling_weight", "window_weight")),
        "has_propagation_output": any(token in file_text for token in ("propagation", "cascade", "sir", "information_propagation")),
        "successful_model_count": sum(
            1
            for row in models
            if isinstance(row, dict)
            and row.get("status") == "success"
            and int(row.get("table_rows") or 0) > 0
        ),
    }


def _load_experiment_report(workspace, explicit_path: Path | None) -> dict[str, Any]:
    path = explicit_path or workspace.logs_dir / "experiment_report.json"
    if not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _audit_one(innovation_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
    proof = evidence.get("field_level_proofs", {}).get(innovation_id)
    if proof is not None:
        passed = _field_proof_passed(proof)
        issues = [] if passed else [
            "field-level innovation proof failed; required evidence: "
            + ", ".join(_REQUIREMENT_LABELS.get(innovation_id, ("executed artifact",)))
        ]
        return {
            "innovation_id": innovation_id,
            "passed": passed,
            "issues": issues,
            "required": list(_REQUIREMENT_LABELS.get(innovation_id, ("executed artifact",))),
            "proof_source": "experiment_report",
            "proof": proof,
        }

    checks = {
        "residual_correction_grey_markov": evidence["has_rolling_backtest"] and evidence["has_model_comparison"],
        "ahp_entropy_combined_weights": evidence["has_weight_output"] and (
            evidence["has_sensitivity"] or evidence["has_ablation"]
        ),
        "stacking_ensemble": evidence["has_model_comparison"]
        and evidence["has_strong_baseline"]
        and evidence["successful_model_count"] >= 2,
        "robust_optimization": evidence["has_robustness"],
        "global_sensitivity_analysis": evidence["has_sensitivity"]
        or evidence["has_ablation"]
        or evidence["has_robustness"],
        "dynamic_weight_evaluation": evidence["has_dynamic_weight_output"]
        and (evidence["has_sensitivity"] or evidence["has_ablation"]),
        "graph_propagation_mechanism": evidence["has_propagation_output"],
        "mechanism_data_fusion": evidence["has_model_comparison"] and evidence["has_ablation"],
        "monte_carlo_uncertainty": evidence["has_monte_carlo"],
    }
    passed = checks.get(innovation_id, False)
    issues = [] if passed else [
        "unsupported innovation claim; required evidence: "
        + ", ".join(_REQUIREMENT_LABELS.get(innovation_id, ("executed artifact",)))
    ]
    return {
        "innovation_id": innovation_id,
        "passed": passed,
        "issues": issues,
        "required": list(_REQUIREMENT_LABELS.get(innovation_id, ("executed artifact",))),
        "proof_source": "artifact_heuristic",
    }


def _load_field_level_proofs(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = (
        report.get("innovation_evidence")
        or report.get("innovation_proofs")
        or report.get("field_level_innovation_evidence")
        or {}
    )
    items: list[dict[str, Any]]
    if isinstance(raw, dict):
        items = [
            {"innovation_id": key, **value}
            for key, value in raw.items()
            if isinstance(value, dict)
        ]
    elif isinstance(raw, list):
        items = [item for item in raw if isinstance(item, dict)]
    else:
        items = []

    proofs: dict[str, dict[str, Any]] = {}
    for item in items:
        innovation_id = str(item.get("innovation_id") or item.get("id") or "").strip()
        if not innovation_id:
            continue
        proofs[innovation_id] = {
            "passed": bool(item.get("passed", False)),
            "artifacts": [str(value) for value in _as_list(item.get("artifacts", [])) if str(value)],
            "metrics": item.get("metrics", {}) if isinstance(item.get("metrics", {}), dict) else {},
            "checks": [str(value) for value in _as_list(item.get("checks", [])) if str(value)],
            "fields": [str(value) for value in _as_list(item.get("fields", [])) if str(value)],
            "notes": str(item.get("notes", "")),
        }
    return proofs


def _field_proof_passed(proof: dict[str, Any]) -> bool:
    if proof.get("passed") is not True:
        return False
    return bool(proof.get("artifacts")) and bool(proof.get("checks") or proof.get("metrics") or proof.get("fields"))


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
