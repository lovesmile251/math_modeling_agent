# Math Modeling Agent

面向数学建模竞赛的一体化 Python 工作流：从题意解析、模型选择、实验执行、结果验证，到论文生成、质量审阅和文档导出。项目既可以作为命令行工具一次性运行，也提供 Streamlit 工作台用于分阶段检查和返工。

当前分支重点补强了国赛论文交付能力，包括正式 DOCX 模板导出、证据追溯、论文结构审计、压力审计、赛制盲审模拟和最终交付基准。

详细优化路线见 [docs/national_award_agent_optimization.md](docs/national_award_agent_optimization.md)。

## 核心能力

- 分阶段建模工作流：题意分析、候选模型生成、模型裁决、实验设计、代码生成、执行修复、结果分析、证据映射、论文写作、事实/数学/结构/语言审阅和文档导出。
- 107 个已注册模型，覆盖预测、评价、优化、统计、分类、聚类、图论、金融、控制、信号、图像、交通、供应链、社会网络等方向。
- 自动生成 `ProblemSpec`、`formulation_spec.json`、实验报告、结果登记表、证据映射和可追溯性报告。
- 支持 CSV、TSV、XLSX、XLS 数据读取，并生成描述统计、模型结果表和图表。
- 自动生成并执行分析脚本，记录运行状态、错误、哈希、产物清单和模型反馈。
- Markdown 论文可导出为 DOCX、PDF 和 LaTeX；DOCX 默认使用 `tools/paper_templates/assets/national_contest_2025.docx` 国赛模板。
- 论文数值结论需要匹配结果表或证据 ID；证据覆盖不足时会阻止正式导出。
- 提供真实赛题盲测、限时赛制模拟、答案复现审计、论文结构审计、压力审计和最终交付基准。
- 支持 OpenAI、DeepSeek 和 OpenAI-compatible LLM；未配置 LLM 时回退到本地规则与模板。
- 内置上传文件校验、生成代码静态安全检查、依赖漏洞扫描和 CI 配置。

## 环境要求

- Python 3.11 或更高版本
- Windows、Linux 或 macOS
- 建议使用虚拟环境

安装运行依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装测试和安全检查依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

Linux/macOS 激活虚拟环境：

```bash
source .venv/bin/activate
```

## 快速开始

直接传入题目文本：

```powershell
python -m app.main `
  --problem-text "根据附件数据建立模型，分析主要影响因素并预测未来趋势。" `
  --data .\workspace\data\sample.xlsx
```

从题目文件读取：

```powershell
python -m app.main `
  --problem-file .\workspace\input\problem.docx `
  --data .\workspace\data\data.csv
```

为正式任务创建独立运行目录：

```powershell
python -m app.main `
  --problem-file .\problem.md `
  --data .\data.xlsx `
  --run-workspace `
  --export docx pdf
```

结果会写入：

```text
workspace/runs/<UTC 时间或运行 ID>/
```

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `python -m app.main --problem-text TEXT` | 直接运行完整工作流 |
| `python -m app.main --problem-file PATH` | 从 `.txt`、`.md`、`.docx` 或文本型 `.pdf` 读取题目 |
| `python -m app.main --data PATH...` | 指定一个或多个 CSV/TSV/XLSX/XLS 数据文件 |
| `python -m app.main --run-workspace` | 在 `workspace/runs/` 下创建独立运行目录 |
| `python -m app.main --export docx pdf latex` | 导出论文文档 |
| `streamlit run app/streamlit_app.py` | 启动分阶段工作台 |
| `python -m pytest` | 运行测试 |
| `python -m ruff check app agents models tools workflows tests` | 运行 lint |

必须提供 `--problem-text` 或 `--problem-file`。如果不传 `--data`，程序会扫描默认工作区的 `workspace/data/`。

## Streamlit 工作台

启动：

```powershell
streamlit run app/streamlit_app.py
```

工作台支持输入或上传题目、上传多个数据文件、启用或关闭 LLM、选择导出格式、查看阶段进度和日志，并在关键节点进行确认、修改、返工或重新执行。

上传限制：

- 题目文件最大 10 MB
- 单个数据文件最大 50 MB
- DOCX/XLSX 会检查 ZIP 条目数、解压后大小和异常压缩比

## 竞赛评测工具

真实赛题模型选择盲测：

```powershell
python -m tools.real_case_benchmark
```

端到端真实赛题演练：

```powershell
python -m tools.real_case_drill --case cumcm-2025-c --max-rows-per-file 1000
```

限时赛制盲审模拟：

```powershell
python -m tools.contest_simulation `
  --case cumcm-2023-c cumcm-2024-c cumcm-2025-c `
  --max-rows-per-file 1000 `
  --timeout-seconds 120 `
  --time-budget-hours 6 `
  --candidate-profile
```

答案复现审计：

```powershell
python -m tools.answer_reproduction_audit `
  --simulation-report benchmarks\results\contest_simulation.json `
  --output-dir benchmarks\results
```

压力审计：

```powershell
python -m tools.pressure_audit `
  --contest benchmarks\results\contest_simulation.json `
  --output-dir benchmarks\results
```

最终交付基准：

```powershell
python -m tools.final_delivery_benchmark `
  --contest benchmarks\results\contest_simulation.json `
  --pressure benchmarks\results\pressure_audit.json `
  --output-dir benchmarks\results
```

## 输出结构

典型工作区结构：

```text
workspace/
├── input/       题目文本
├── data/        输入数据
├── code/        自动生成的分析代码
├── figures/     PNG 图表
├── tables/      CSV 结果表
├── paper/       论文、审稿报告和导出文档
├── logs/        执行日志、模型报告、证据和诊断信息
└── runs/        独立运行目录
```

常见产物：

```text
logs/problem_spec.json
logs/formulation_spec.json
logs/experiment_report.json
logs/result_registry.json
logs/claim_evidence_map.json
logs/traceability_report.json
paper/paper_draft.md
paper/paper_quality_report.md
paper/review_report.md
paper/paper.docx
paper/paper.pdf
paper/paper.tex
```

## LLM 配置

OpenAI：

```powershell
$env:MMA_LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_MODEL="your-model"
```

DeepSeek：

```powershell
$env:MMA_LLM_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="your-api-key"
$env:DEEPSEEK_MODEL="your-model"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

OpenAI-compatible：

```powershell
$env:MMA_LLM_PROVIDER="openai_compatible"
$env:MMA_LLM_API_KEY="your-api-key"
$env:MMA_LLM_MODEL="your-model"
$env:MMA_LLM_BASE_URL="https://provider.example/v1"
```

常用运行参数：

```powershell
$env:MMA_LLM_TIMEOUT_SECONDS="90"
$env:MMA_LLM_RETRIES="2"
$env:MMA_LLM_TEMPERATURE="0.2"
$env:MMA_LLM_CACHE="1"
$env:MMA_LLM_FAST_MODE="1"
```

LLM 日志只记录 provider、模型、耗时、字符数、成功/失败和缓存命中等元数据，不保存 prompt 正文。未显式传入 `--use-llm`、缺少 API key 或 provider 不受支持时，工作流会继续使用本地能力。

## 安全边界

项目会执行自动生成的 Python 分析代码，因此实现了以下防护：

- 执行前进行语法检查、Ruff 检查和 AST 静态安全检查。
- 只允许预设标准库、数据分析库和项目模型包。
- 拒绝命令执行、动态代码执行、网络访问和高风险文件操作。
- 子进程使用 Python 隔离模式 `-I`，不继承 API key、token 等普通环境变量。
- 执行有超时限制，并记录脚本和输入文件 SHA-256。
- 上传文件会移除路径部分，Office 文件会检查压缩炸弹风险。
- CI 使用 Bandit 和 pip-audit 执行安全扫描。

这些措施不是操作系统级沙箱。不要在拥有高权限、重要凭据或敏感文件的主机上运行不可信生成代码；面向公网或多租户部署时，应进一步使用容器、低权限账户、网络隔离和资源配额。

## 测试与质量检查

```powershell
python -m pytest
python -m ruff check app agents models tools workflows tests
python -m coverage run -m pytest
python -m coverage report
python -m bandit -r app agents models tools workflows -q -ll
python -m pip_audit -r requirements.txt --progress-spinner off
```

构建安装包：

```powershell
python -m pip install build
python -m build
```

GitHub Actions 会在 push 和 pull request 时执行 lint、测试、覆盖率和安全检查。

## 目录说明

```text
agents/       工作流 Agent
app/          CLI、配置和 Streamlit 界面
models/       数学模型实现
tools/        执行、导出、审计、质量、安全和文档工具
workflows/    工作流编排与阶段状态管理
prompts/      LLM 提示词
benchmarks/   竞赛盲测用例与评分要求
tests/        单元测试、集成测试和安全测试
docs/         项目设计和优化路线文档
workspace/    默认输入与运行输出
```

## 已知限制

- 自动模型选择依赖题目关键词、字段名和数据结构，重要任务仍需要人工核验。
- 模型成功运行不代表假设、参数和结论必然适合实际问题。
- LLM 生成内容可能存在事实、公式或表达错误，最终论文必须人工审查。
- 大型数据集可能超过当前单机工作流的时间和内存能力。
- PDF 导出使用 ReportLab，复杂 Markdown、LaTeX 宏和精细排版不会被完整复现。
- `.xls` 兼容性取决于本地 pandas Excel 读取后端。

## 编码约定

项目源码、配置、提示词和文档统一使用无 BOM 的 UTF-8。命令行入口会尽量将标准输入输出配置为 UTF-8。
