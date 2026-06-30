from __future__ import annotations

from tools.paper_structure_audit import audit_national_award_structure


def test_paper_structure_audit_flags_missing_award_sections_and_answer_closure():
    text = """# Paper

## Abstract
This paper studies Problem 1 and Problem 2 with a general model.

## Model
We describe the approach briefly.

## Results
Problem 1 has a table-backed score of 88.5.
"""

    audit = audit_national_award_structure(text)

    assert any("Award structure weak" in issue for issue in audit.issues)
    assert any("Problem-answer closure weak" in issue for issue in audit.issues)
    assert any("Model formulation weak" in issue for issue in audit.issues)
    assert audit.metrics["problem_answer_closure_missing"] == 1
    assert audit.metrics["award_sections_missing"] > 0


def test_paper_structure_audit_accepts_complete_award_structure():
    text = """# Paper

## Abstract
For Problem 1, Problem 2, and Problem 3, the model obtains objective 12.5, error 0.31, stability 0.91, coverage 0.83, and score 88.0.

## Keywords
optimization; validation; sensitivity

## Problem Restatement
Problem 1 asks for evaluation. Problem 2 asks for optimization. Problem 3 asks for validation.

## Problem Analysis
We decompose Problem 1, Problem 2, and Problem 3 into data, model, and validation tasks.

## Assumptions
1. The data are representative.
2. Parameters are stable.

## Notation
| symbol | meaning | unit |
| --- | --- | --- |
| x | decision variable | unit |
| y | predicted value | unit |

## Model Formulation
The decision variable x maximizes the objective under constraints.
\\[
\\max f(x)=12.5 \\quad s.t.\\quad x \\le 10
\\]

## Results
Problem 1 obtains score 88.0. Problem 2 obtains objective 12.5. Problem 3 obtains error 0.31.

## Validation
The validation compares a baseline, reports error 0.31, and includes sensitivity analysis.

## Sensitivity
The sensitivity test perturbs the main parameter by 10%.

## Evaluation
The model advantage is reproducibility, the limitation is data scope, and the improvement is richer validation.

## Conclusion
Problem 1 answer is 88.0. Problem 2 answer is 12.5. Problem 3 answer is 0.31.

## References
[1] Zhang. Model validation. Journal, 2024.

## Appendix
The appendix lists code and result tables.
"""

    audit = audit_national_award_structure(text)

    assert audit.issues == []
    assert audit.metrics["award_sections_missing"] == 0
    assert audit.metrics["problem_answer_closure_missing"] == 0


def test_paper_structure_audit_keeps_nested_subsections_inside_parent():
    text = """# Paper

## Abstract
For Problem 1, Problem 2, and Problem 3, the model obtains objective 12.5, error 0.31, stability 0.91, coverage 0.83, and score 88.0.

## Keywords
optimization; validation; sensitivity

## Problem Restatement
Problem 1 asks for evaluation. Problem 2 asks for optimization. Problem 3 asks for validation.

## Problem Analysis
### Task split
Problem 1, Problem 2, and Problem 3 are decomposed into data, model, and validation tasks.

## Assumptions
1. The data are representative.
2. Parameters are stable.

## Notation
| symbol | meaning | unit |
| --- | --- | --- |
| x | decision variable | unit |
| y | predicted value | unit |

## Model Formulation
### Core model
The decision variable x maximizes the objective under constraints.
\\[
\\max f(x)=12.5 \\quad s.t.\\quad x \\le 10
\\]

## Results
### Answer closure
Problem 1 obtains score 88.0. Problem 2 obtains objective 12.5. Problem 3 obtains error 0.31.

## Validation
### Checks
The validation compares a baseline, reports error 0.31, and includes sensitivity analysis.

## Sensitivity
The sensitivity test perturbs the main parameter by 10%.

## Evaluation
The model advantage is reproducibility, the limitation is data scope, and the improvement is richer validation.

## Conclusion
Problem 1 answer is 88.0. Problem 2 answer is 12.5. Problem 3 answer is 0.31.

## References
[1] Zhang. Model validation. Journal, 2024.

## Appendix
The appendix lists code and result tables.
"""

    audit = audit_national_award_structure(text)

    assert audit.issues == []
    assert audit.metrics["award_sections_missing"] == 0
    assert audit.metrics["problem_answer_closure_missing"] == 0
