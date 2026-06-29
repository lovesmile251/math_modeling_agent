from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tools.exporters import SUPPORTED_FORMATS, export_document
from tools.report_builder import Document, parse_markdown

NATIONAL_CONTEST_TEMPLATE_STEM = "national_contest_template"
NATIONAL_CONTEST_REQUIRED_SECTIONS = (
    "摘要",
    "关键词",
    "一、问题重述",
    "二、问题分析",
    "三、模型假设",
    "四、符号说明",
    "五、数据预处理",
    "六、模型建立与求解",
    "七、结果分析",
    "八、模型检验",
    "九、灵敏度分析",
    "十、模型评价与推广",
    "十一、结论",
    "参考文献",
    "附录",
)


def build_national_contest_markdown(title: str = "全国大学生数学建模竞赛论文") -> str:
    """Return a fillable national-contest paper template in Markdown."""

    return "\n".join(
        [
            f"# {title}",
            "",
            "## 摘要",
            "",
            "本文围绕题目所给实际问题，依次完成问题分析、模型假设、模型建立、求解检验和结论解释。摘要应集中呈现问题目标、核心方法、关键数值结果、模型检验结论和可执行建议，避免只描述过程而不写结果。",
            "",
            "## 关键词",
            "",
            "数学建模；模型检验；灵敏度分析；优化决策；结果可追溯",
            "",
            "## 一、问题重述",
            "",
            "在不改变原题含义的前提下，概括背景、已知条件、约束条件和需要回答的具体问题。",
            "",
            "## 二、问题分析",
            "",
            "拆解子问题之间的逻辑关系，说明变量、数据、约束和评价指标如何支撑后续建模。",
            "",
            "## 三、模型假设",
            "",
            "1. 假设一：说明简化依据及其对结果的影响范围。",
            "2. 假设二：说明数据、环境或行为条件的稳定性。",
            "3. 假设三：说明异常情形的处理边界。",
            "",
            "## 四、符号说明",
            "",
            "| 符号 | 含义 | 单位/说明 |",
            "|---|---|---|",
            "| \\(x_i\\) | 第 \\(i\\) 个决策变量 | 按题意填写 |",
            "| \\(y_i\\) | 第 \\(i\\) 个观测或预测结果 | 按题意填写 |",
            "| \\(n\\) | 样本量或对象数量 | 个 |",
            "| \\(f(x)\\) | 目标函数 | 与评价指标一致 |",
            "",
            "## 五、数据预处理",
            "",
            "说明数据来源、字段含义、缺失值处理、异常值识别、量纲统一和可复现实验输入。",
            "",
            "## 六、模型建立与求解",
            "",
            "### 6.1 模型建立",
            "",
            "给出目标函数、约束条件、参数估计方法和必要的数学推导。",
            "",
            "\\[",
            "\\min f(x) \\quad \\text{s.t.} \\quad g_j(x) \\le 0,\\ j=1,2,\\ldots,m",
            "\\]",
            "",
            "### 6.2 求解方法",
            "",
            "说明算法流程、参数设置、收敛或复杂度依据，并标注代码和结果表位置。",
            "",
            "## 七、结果分析",
            "",
            "结合表格和图形给出关键结果，所有数值结论应能追溯到实验输出或结果表。",
            "",
            "## 八、模型检验",
            "",
            "从误差、稳定性、对照基线、残差或交叉验证等角度验证模型可靠性。",
            "",
            "## 九、灵敏度分析",
            "",
            "选择关键参数进行扰动，比较目标值、约束满足情况和最终决策变化。",
            "",
            "## 十、模型评价与推广",
            "",
            "概括模型优点、局限、适用边界和推广场景，说明进一步改进方向。",
            "",
            "## 十一、结论",
            "",
            "按子问题逐条给出明确结论、核心数值和实际建议。",
            "",
            "## 参考文献",
            "",
            "[1] 作者. 文献题名[J]. 期刊名, 年份, 卷(期): 起止页码.",
            "[2] 作者. 书名[M]. 出版地: 出版社, 年份.",
            "[3] 作者. 资料题名[EB/OL]. 发布或更新日期. URL.",
            "",
            "## 附录",
            "",
            "### 附录A 主要程序",
            "",
            "列出核心代码文件路径、关键函数和可复现实验命令。",
            "",
            "### 附录B 结果数据表",
            "",
            "附完整结果表、补充图表和未在正文完整展示的中间结果。",
            "",
        ]
    )


def build_national_contest_document(title: str = "全国大学生数学建模竞赛论文") -> Document:
    doc = parse_markdown(build_national_contest_markdown(title))
    doc.title = title
    return doc


def write_national_contest_template(output_dir: Path, title: str = "全国大学生数学建模竞赛论文") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{NATIONAL_CONTEST_TEMPLATE_STEM}.md"
    path.write_text(build_national_contest_markdown(title), encoding="utf-8")
    return path


def export_national_contest_template(
    output_dir: Path,
    formats: Iterable[str] = SUPPORTED_FORMATS,
    *,
    title: str = "全国大学生数学建模竞赛论文",
) -> dict[str, Path]:
    doc = build_national_contest_document(title)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"markdown": write_national_contest_template(output_dir, title)}
    for fmt in formats:
        results[fmt] = export_document(doc, fmt, output_dir, stem=NATIONAL_CONTEST_TEMPLATE_STEM)
    return results
