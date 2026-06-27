# Math Modeling Agent

一个面向数学建模任务的 Python 工作流：从题目理解、模型选择、代码生成与执行，到结果分析、证据映射、论文写作、审稿和文档导出。

项目既可以作为命令行工具一次性运行，也提供 Streamlit 分阶段工作台，允许用户在关键节点确认、修改或重新执行。

## 当前能力

- 17 个有序工作流阶段，覆盖“建模 → 实验 → 代码 → 结果 → 论文 → 审稿 → 导出”。
- 19 个核心 Agent，分别负责题意分析、模型讨论、决策、数学形式化、实验设计、代码、执行、证据、写作和审稿。
- 95 个已注册模型，覆盖预测、评价、优化、统计、分类、聚类、图论、金融、控制、信号、图像、交通、社会网络等方向。
- 自动生成结构化 `ProblemSpec`，记录子问题、变量角色、约束、尺度、不确定性、数据要求和任务依赖。
- 每个注册模型均有模型契约，包括适用任务、数据要求、假设、指标、诊断、基线和失效条件。
- 自动读取 CSV、TSV、XLSX 数据，并生成描述统计、模型结果表和图表。
- 自动生成并执行 `baseline_analysis.py`，记录执行状态、错误、哈希和运行清单。
- 执行后生成统一实验报告，对主模型、基线模型、结果完整性和指标产物进行比较。
- 自动构建数学模型 DSL，描述变量、参数、目标、约束和跨子问题阶段依赖。
- 对预测模型执行真实滚动回测，并对主模型执行参数扰动和特征消融。
- 自动生成 Markdown 论文，并可导出为 DOCX、PDF 和 LaTeX。
- 对论文中的实质性数值结论执行结果表/证据 ID 追溯；未达到门槛时阻止正式导出。
- 支持 OpenAI、DeepSeek 和 OpenAI-compatible 接口，可在未配置 LLM 时回退到本地规则与模板。
- 支持独立运行目录，避免多次任务互相覆盖。
- 包含静态代码门禁、上传文件校验、依赖漏洞扫描和持续集成配置。
- 已索引 2010—2012、2017—2025 共 54 道历年国赛题。
- 具备 26 道人工金标真题盲测，标签仅在推理完成后用于评分。
- 10 道代表性历年真题端到端限时演练均成功完成，平均赛制分 97.54，平均盲审结构分 95.68，`first_prize_ready` 覆盖率 100%。
- 提供限时赛制盲审模拟工具，可按时间管理、交付完整性、建模深度、结果验证、论文竞争力和可复现性进行 100 分制评估。
- 提供数值答案复现审计，可复算证据映射中的关键数值声明，校验结果表哈希，并对照优秀论文中位指标定位差距；当前 10 题平均复现审计分 93.77，数值声明复现率 100%，哈希通过率 100%，高风险题 0。
- 小样本、类别型和图网络赛题会自动补充样本快照、字段类型摘要、质量记分卡、类别频数、二元关系频数、建模就绪检查表及对应可视化，降低非数值型赛题的证据密度短板。
- CLI 和 Streamlit 均可直接读取文本型 PDF 赛题。

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

Linux 或 macOS 激活方式：

```bash
source .venv/bin/activate
```

如需运行测试和安全检查：

```powershell
python -m pip install -r requirements-dev.txt
```

## 快速开始

### 使用题目文本

```powershell
python -m app.main `
  --problem-text "根据附件数据建立模型，分析主要影响因素并预测未来趋势。" `
  --data .\workspace\data\sample.xlsx
```

### 使用题目文件

题目文件支持 `.txt`、`.md`、`.docx` 和文本型 `.pdf`：

```powershell
python -m app.main `
  --problem-file .\workspace\input\problem.docx `
  --data .\workspace\data\data.csv
```

如果不传 `--data`，程序会扫描默认工作区的 `workspace/data/`。

### 使用独立运行目录

推荐为每次正式任务创建独立目录：

```powershell
python -m app.main `
  --problem-file .\problem.md `
  --data .\data.xlsx `
  --run-workspace `
  --export docx pdf
```

结果将写入：

```text
workspace/runs/<UTC时间戳_随机标识>/
```

也可以指定运行标识：

```powershell
python -m app.main `
  --problem-text "分析区域资源配置效率。" `
  --run-id resource-allocation-v1
```

`--run-id` 会自动启用独立运行目录。`--workspace` 不能与 `--run-workspace` 或 `--run-id` 同时使用。

## 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--problem-text TEXT` | 直接传入题目文本 |
| `--problem-file PATH` | 读取 `.txt`、`.md`、`.docx` 或 `.pdf` 题目文件 |
| `--data PATH...` | 指定一个或多个 CSV、TSV、XLSX、XLS 数据文件 |
| `--use-llm` | 启用已配置的 LLM |
| `--workspace PATH` | 使用指定工作区 |
| `--run-workspace` | 在 `workspace/runs/` 下创建独立运行目录 |
| `--run-id ID` | 使用指定运行标识创建独立工作区 |
| `--export FORMATS...` | 导出 `docx`、`pdf`、`latex` 中的一种或多种格式 |

必须提供 `--problem-text` 或 `--problem-file`。

旧式 `.xls` 文件由 pandas 的可用读取后端决定；如果读取失败，建议先转换为 `.xlsx` 或安装兼容的 Excel 读取器。

## Streamlit 工作台

启动网页界面：

```powershell
streamlit run app/streamlit_app.py
```

工作台提供两种模式：

- 分阶段工作台：在模型决策、实验方案、代码计划、结果分析、论文提纲和语言审稿等节点暂停确认。
- 一键运行：自动执行完整工作流。

网页端支持：

- 输入或上传题目；
- 上传多个数据文件；
- 选择是否启用 LLM；
- 选择论文导出格式；
- 查看阶段进度、日志、模型决策、结果表、图表和审稿意见；
- 下载 Markdown、DOCX、PDF 和 LaTeX 文件；
- 对部分阶段进行修改、返工或重新执行。

上传限制：

- 题目文件最大 10 MB；
- 单个数据文件最大 50 MB；
- DOCX/XLSX 会检查 ZIP 条目数、解压后大小和异常压缩比。

## 工作流

完整工作流按以下顺序执行：

1. 题意分析
2. 候选模型方案
3. 模型批评
4. 模型决策
5. 实验方案
6. 代码计划
7. 代码生成
8. 代码执行与修复
9. 结果分析
10. 证据映射
11. 论文提纲
12. 分章节写作
13. 事实审稿
14. 数学审稿
15. 结构审稿
16. 语言与综合审稿
17. 文档导出

分阶段模式会在以下节点等待用户确认：

- 模型决策
- 实验方案
- 代码计划
- 结果分析
- 论文提纲
- 语言审稿

程序支持从指定阶段重新运行，并自动将依赖该阶段的后续结果标记为需要重新生成。

题意分析阶段同时生成 `logs/problem_spec.json`，其中包含：

- 显式子问题及任务类型；
- 观测变量、决策变量、状态变量和参数；
- 目标、约束、评价指标和期望输出；
- 时间尺度、空间尺度和不确定性来源；
- 数据充分性要求、歧义和子问题依赖。

模型决策之后会生成 `logs/formulation_spec.json`。该文件用机器可读方式描述：

- 变量角色与取值域；
- 参数来源；
- 优化方向和目标；
- 约束类型；
- 每个子问题对应的模型阶段；
- 阶段输入、输出以及上游依赖。

## 模型库

当前注册了 95 个可执行模型。主要类别包括：

| 类别 | 示例 |
| --- | --- |
| 综合评价 | 熵权法、TOPSIS、AHP、VIKOR、灰色关联、DEA、模糊评价 |
| 预测 | 线性趋势、GM(1,1)、平滑预测、季节预测、VAR、非线性预测 |
| 优化 | 资源配置、背包、指派、调度、整数规划、多目标优化、TSP、VRP |
| 统计分析 | 相关分析、回归、参数估计、假设检验、ANOVA、蒙特卡洛 |
| 分类与聚类 | 逻辑回归、朴素贝叶斯、KNN、K-means、DBSCAN、层次聚类、SMOTE |
| 图与网络 | 最短路、最小生成树、最大流、中心性、社区发现 |
| 金融 | VaR/CVaR、GARCH、Black-Scholes、Markowitz |
| 机理与控制 | SIR、Logistic、Lotka-Volterra、Kalman、最优控制、鲁棒控制 |
| 信号与图像 | FFT、信号去噪、边缘检测、图像分割、特征提取、图像配准 |
| 交通与供应链 | 交通流、跟驰模型、库存策略、牛鞭效应、Jackson 网络 |

查看当前实际注册的模型 ID：

```powershell
python -c "from tools.model_registry import registered_model_ids; print(*sorted(registered_model_ids()), sep='\n')"
```

模型会根据题目文本、数据字段和数据结构参与选择。字段条件不满足时，模型通常返回空结果或记录为跳过，不会直接中断整个工作流。

每个模型均有可校验契约：

```powershell
python -c "from models.catalog import validate_model_contracts; print(validate_model_contracts())"
```

选模基准要求首选模型命中预期模型集合，同时完整识别任务类型，不能仅凭题型大致正确通过测试。

## 历年真题盲测

赛题原始压缩包位于 `examples/`，隔离展开目录为
`examples/extracted/`。

重新生成题库索引：

```powershell
python -m tools.competition_corpus examples/extracted
```

运行真题盲测：

```powershell
python -m tools.real_case_benchmark
```

报告输出到：

- `benchmarks/results/real_case_benchmark.json`
- `benchmarks/results/real_case_benchmark.md`

当前 26 道金标真题基准结果：

- 综合均分：98.51
- 任务识别 F1：0.9752
- 首选模型命中率：100%
- 前五候选覆盖率：100%

## 端到端真题演练

第三阶段的 `tools.real_case_benchmark` 主要评估“题目分解与模型选择是否命中”。第四阶段新增 `tools.real_case_drill`，用于评估真实赛题从题面读取到代码执行、结果产出、论文草稿和质量审查的闭环稳定性。第五阶段继续补强了真实零售类赛题的核心模型产出和无 LLM 论文模板质量。第六阶段重点处理批量真题稳定性、慢模型执行、分类标签识别和证据映射鲁棒性。第七阶段新增限时赛制盲审模拟，用于判断工程闭环是否接近真实国赛冲奖交付节奏。第八阶段新增数值答案复现与优秀论文对照审计，用于检查结果是否能被证据表重新计算、论文与产物是否一致。第九阶段进一步提升论文证据覆盖率和优秀论文风格指标，使 3 题审计全部达到 `reproducible`。第十阶段补强竞赛级诊断产物，使 2025-C 从 `competitive` 提升到 `first_prize_ready`。

默认只跑前 3 道题，并且每个附件最多抽样 5000 行到独立工作区，避免一次性运行全部 54 道真题或大附件耗时过长：

```powershell
python -m tools.real_case_drill
```

指定单题演练：

```powershell
python -m tools.real_case_drill --case cumcm-2023-c
```

运行全部索引真题：

```powershell
python -m tools.real_case_drill --limit -1
```

使用全量附件并提高执行超时：

```powershell
python -m tools.real_case_drill `
  --case cumcm-2023-c `
  --full-data `
  --timeout-seconds 600
```

启用 LLM 或论文导出：

```powershell
python -m tools.real_case_drill `
  --case cumcm-2023-c `
  --use-llm `
  --export docx pdf
```

报告输出到：

- `benchmarks/results/real_case_drill.json`
- `benchmarks/results/real_case_drill.md`

每道题的完整工作区默认位于：

```text
workspace/runs/real_case_drill/<case_id>/
```

端到端演练评分由五部分构成：

- 代码执行成功：25 分；
- 关键产物完整性：20 分，包括代码、论文、质量报告、结果表和图片；
- 已选模型实际产出覆盖率：20 分；
- 论文质量分折算：25 分；
- 无工作流错误记录：10 分。

当前第五阶段样例结果（`cumcm-2023-c`，默认 5000 行抽样演练）：

- 端到端得分：99.25；
- 执行成功率：100%；
- 论文质量分：97；
- 结果表数量：25；
- 图片数量：15；
- 题目级模型产出覆盖：`inventory_policy`、`naive_bayes_classifier`、`error_analysis`、`sensitivity_analysis`、`model_comparison` 均有实际产物。

当前第六阶段批量回归结果（10 道代表题，`--max-rows-per-file 1000 --timeout-seconds 120`）：

- 端到端均分：95.85；
- 执行成功率：100%；
- 平均论文质量分：83.4；
- 平均结果表数量：8.0；
- 平均图片数量：6.3；
- 已选核心模型缺失数：0；
- 工作流错误数：0；
- 覆盖案例：`cumcm-2011-a`、`cumcm-2011-b`、`cumcm-2012-a`、`cumcm-2018-b`、`cumcm-2020-c`、`cumcm-2022-c`、`cumcm-2023-c`、`cumcm-2024-c`、`cumcm-2024-e`、`cumcm-2025-c`。

第六阶段修复点：

- 梯度提升树改为有界样本、有界特征、有界候选阈值搜索，`cumcm-2025-c` 单题运行从分钟级降到约 13 秒；
- 逻辑分类器支持多分类 one-vs-rest，并优先识别“是否/健康/异常/类型/风化”等语义标签，避免把编号、代码、日期误当标签；
- 图网络题中由边表误触发的 `scheduling_plan` 不再计为核心缺口；
- 证据映射对混合字符串/数值列执行安全数值转换，避免在论文证据追溯阶段崩溃；
- 真题 drill 支持逐题写入中间报告，长批量任务即使中途失败也保留已完成结果。

## 限时赛制盲审模拟

第七阶段新增 `tools.contest_simulation`。它先运行 `tools.real_case_drill`，再按更接近竞赛交付的六个维度生成赛制分和盲审结构分：

- 时间管理：10 分；
- 交付完整性：15 分；
- 建模深度：20 分；
- 结果验证：20 分；
- 论文竞争力：25 分；
- 可复现性：10 分。

运行单题赛制模拟：

```powershell
python -m tools.contest_simulation `
  --case cumcm-2025-c `
  --max-rows-per-file 1000 `
  --timeout-seconds 120 `
  --time-budget-hours 6
```

运行多题赛制模拟：

```powershell
python -m tools.contest_simulation `
  --case cumcm-2023-c cumcm-2024-c cumcm-2025-c `
  --max-rows-per-file 1000 `
  --timeout-seconds 120 `
  --time-budget-hours 6
```

报告输出到：

- `benchmarks/results/contest_simulation.json`
- `benchmarks/results/contest_simulation.md`

当前第七阶段 3 题赛制模拟结果（`cumcm-2023-c`、`cumcm-2024-c`、`cumcm-2025-c`，6 小时预算）：

- 平均赛制得分：99.34；
- 平均盲审结构分：98.05；
- 一等奖就绪率：66.7%；
- 高风险题数量：0；
- 总体判断：具备一等奖冲刺工程基础。

赛制模拟仍是结构化工程评估，不等价于真实评委结论。

## 数值答案复现与优秀论文对照审计

第八阶段新增 `tools.answer_reproduction_audit`。它读取赛制模拟报告中的工作区，执行以下检查：

- 复算 `claim_evidence_map.json` 中可解析的 `mean/std` 数值声明；
- 校验 `result_registry.json` 中结果表的 SHA-256 哈希；
- 对比 `run_summary.json` 与 `review_report.md` 中的模型产出数量，发现论文审稿与真实产物不一致的问题；
- 读取 `traceability_report.json`，检查论文数值声明证据追溯覆盖；
- 将论文指标与优秀论文中位参考对照，包括字数、公式、表格、图片、参考文献和质量分。

运行复现审计：

```powershell
python -m tools.answer_reproduction_audit `
  --simulation-report benchmarks\results\contest_simulation.json `
  --output-dir benchmarks\results `
  --max-claims-per-case 80
```

报告输出到：

- `benchmarks/results/answer_reproduction_audit.json`
- `benchmarks/results/answer_reproduction_audit.md`

第八阶段初始 3 题审计结果：

- 平均审计分：89.73；
- 数值声明复现率：100%；
- 结果表哈希通过率：100%；
- 优秀论文风格对齐率：72.22%；
- 高风险案例数：0；
- `cumcm-2024-c`：`reproducible`；
- `cumcm-2023-c`、`cumcm-2025-c`：`mostly_reproducible`，主要短板是论文数值证据追溯覆盖不足 85%，以及公式/参考文献等优秀论文风格指标不足。

第八阶段同时修复了审稿报告的模型产出计数兼容问题：当前 `run_summary.json` 使用 `model_runs` 格式，审稿器现在能正确统计成功产出的模型表。

第九阶段补强结果：

- 追溯器现在将结果表列的最小值、最大值、均值、标准差、中位数和四分位数纳入可复核证据；
- Markdown 统计表头不再被误判为数值结论；
- 追溯器兼容布尔结果列，避免分位数计算崩溃；
- 通用论文模板补充分类、交叉验证和模型选择公式；
- 参考文献生成最低数量提高到 10 篇，对齐优秀论文中位参考。

当前第九阶段 3 题审计结果：

- 平均审计分：98.84；
- 数值声明复现率：100%；
- 结果表哈希通过率：100%；
- 优秀论文风格对齐率：100%；
- 高风险案例数：0；
- `cumcm-2023-c`、`cumcm-2024-c`、`cumcm-2025-c` 均达到 `reproducible`；
- 证据追溯覆盖率分别为 86.5%、99.4%、96.7%。

第十阶段题型专项补强结果：

- 生成脚本新增竞赛级诊断产物：`feature_summary`、`missingness_summary`、`correlation_pairs` 三类结果表；
- 新增缺失率柱状图、数值特征箱线图、前两数值字段散点图；
- 诊断产物兼容全常数数值列，相关系数全 NaN 时仍安全输出空结构表；
- `cumcm-2025-c` 当前产物提升到 12 张结果表、10 张图片；
- `cumcm-2025-c` 赛制分：100.0，分层：`first_prize_ready`；
- 3 题赛制模拟结果：平均赛制分 100.0，平均盲审结构分 100.0，一等奖就绪率 100%，高风险题数量 0；
- 3 题复现审计仍保持：平均审计分 98.84，数值声明复现率 100%，结果表哈希通过率 100%，优秀论文风格对齐率 100%。

## LLM 配置

### OpenAI

```powershell
$env:MMA_LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_MODEL="your-model"

python -m app.main `
  --use-llm `
  --problem-text "建立预测模型并完成论文。"
```

### DeepSeek

```powershell
$env:MMA_LLM_PROVIDER="deepseek"
$env:DEEPSEEK_API_KEY="your-api-key"
$env:DEEPSEEK_MODEL="your-model"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"

python -m app.main `
  --use-llm `
  --problem-text "建立评价与优化模型。"
```

### OpenAI-compatible

适用于兼容 `/chat/completions` 的供应商或自托管服务：

```powershell
$env:MMA_LLM_PROVIDER="openai_compatible"
$env:MMA_LLM_API_KEY="your-api-key"
$env:MMA_LLM_MODEL="your-model"
$env:MMA_LLM_BASE_URL="https://provider.example/v1"

python -m app.main `
  --use-llm `
  --problem-text "建立预测、评价与优化模型。"
```

通用配置：

```powershell
$env:MMA_LLM_TIMEOUT_SECONDS="90"
$env:MMA_LLM_RETRIES="2"
$env:MMA_LLM_RETRY_BACKOFF_SECONDS="0.5"
$env:MMA_LLM_TEMPERATURE="0.2"
$env:MMA_LLM_TOP_P="0.9"
$env:MMA_LLM_MAX_OUTPUT_TOKENS="4096"
$env:MMA_LLM_CACHE="1"
$env:MMA_LLM_CACHE_DIR=".cache/llm"
```

LLM 调用日志只记录 provider、模型、耗时、字符数、成功/失败、缓存命中等元数据，不保存 prompt 正文；错误信息写入日志前会对常见 key/token/secret 做脱敏。

快速模式会在未显式覆盖时启用更短超时、更少重试、默认输出长度限制和缓存；写作阶段会从逐节多次调用切换为单次完整论文生成：

```powershell
$env:MMA_LLM_FAST_MODE="1"
```

也可以只加速写作阶段：

```powershell
$env:MMA_LLM_WRITING_MODE="single"
```

按任务类型覆盖模型：

```powershell
$env:MMA_LLM_MODEL_REASONING="strong-reasoning-model"
$env:MMA_LLM_MODEL_PLANNING="planning-model"
$env:MMA_LLM_MODEL_WRITING="writing-model"
$env:MMA_LLM_MODEL_REVIEW="review-model"
$env:MMA_LLM_MODEL_CODE_REPAIR="code-repair-model"
```

也可以按具体 agent 文件名精确覆盖，精确配置优先于任务类型配置：

```powershell
$env:MMA_LLM_MODEL_WRITING_AGENT="paper-writing-model"
$env:MMA_LLM_MODEL_MODEL_SELECTION_CREW="model-selection-model"
```

DeepSeek 额外支持：

```powershell
$env:DEEPSEEK_REASONING_EFFORT="medium"
$env:DEEPSEEK_THINKING="enabled"
```

未传入 `--use-llm`、缺少 API Key、缺少模型名或 provider 不受支持时，工作流会记录 LLM 状态并继续使用本地能力。

### LLM 代码修复

LLM 返回的代码默认不会直接替换并执行。只有在受控的本地环境中，才建议显式开启：

```powershell
$env:MMA_ALLOW_LLM_CODE_REPAIR="1"
```

LLM 修复后的代码仍需通过语法、Ruff 和静态安全门禁。

## 工作区结构

每个工作区采用相同结构：

```text
workspace/
├─ input/       题目文本
├─ data/        输入数据
├─ code/        自动生成的分析代码
├─ figures/     PNG 图表
├─ tables/      CSV 结果表
├─ paper/       论文、审稿报告和导出文档
├─ logs/        执行日志、模型报告、证据和诊断信息
└─ runs/        独立运行目录
```

常见输出：

```text
code/baseline_analysis.py
logs/agent.log
logs/execution.log
logs/execution_attempt_<n>.log
logs/execution_manifest.json
logs/run_summary.json
logs/model_selection_report.json
logs/model_execution_feedback.json
logs/problem_spec.json
logs/formulation_spec.json
logs/experiment_report.json
logs/result_registry.json
logs/claim_evidence_map.json
logs/traceability_report.json
tables/model_experiment_comparison.csv
tables/rolling_backtest_metrics.csv
tables/model_robustness.csv
tables/feature_ablation.csv
paper/paper_draft.md
paper/paper_quality_report.md
paper/review_report.md
paper/paper.docx
paper/paper.pdf
paper/paper.tex
```

表格和图表文件名由数据文件名及模型 ID 组合生成。

## 安全边界

项目会执行自动生成的 Python 分析代码，因此实现了以下防护：

- 生成代码执行前进行语法检查、Ruff 检查和 AST 静态安全检查；
- 仅允许预设的标准库、数据分析库和项目模型包；
- 拒绝命令执行、动态代码执行、网络访问及高风险文件操作；
- 子进程使用 Python 隔离模式 `-I`；
- 执行环境不会继承 API Key、Token 等普通环境变量；
- 生成脚本必须位于当前工作区内；
- 执行带超时限制，并记录脚本及输入文件 SHA-256；
- 主模型必须指定可比较基线，实验报告会检查二者是否实际成功运行；
- 论文数值结论必须匹配结果表数值或显式证据 ID；
- 数值追溯覆盖率低于门槛时，导出 Agent 会拒绝生成正式文档；
- 上传文件名会移除路径部分，防止目录穿越；
- Office 文件会检查压缩炸弹风险；
- CI 使用 Bandit 和 pip-audit 执行安全扫描。

这些措施不是操作系统级沙箱。不要在拥有高权限、重要凭据或敏感文件的主机上运行不可信生成代码。面向公网或多租户部署时，应进一步使用容器、独立低权限账户、网络隔离和资源配额。

## 测试与质量检查

运行全部测试：

```powershell
python -m pytest
```

运行 Ruff：

```powershell
python -m ruff check app agents models tools workflows tests
```

运行覆盖率：

```powershell
python -m coverage run -m pytest
python -m coverage report
```

运行安全检查：

```powershell
python -m bandit -r app agents models tools workflows -q -ll
python -m pip_audit -r requirements.txt --progress-spinner off
```

构建安装包：

```powershell
python -m pip install build
python -m build
```

GitHub Actions 会在 push 和 pull request 时使用 Python 3.11、3.12 执行 lint、测试、覆盖率和安全检查。

项目还提供竞赛盲测评分框架：

```python
from tools.competition_benchmark import (
    evaluate_competition_case,
    load_competition_cases,
)
```

评分维度包括题目拆解、主模型、基线与形式化、实际实验、数值追溯、论文质量和产物完整性。基础用例位于 `benchmarks/national_competition_cases.json`；正式评测时应使用不向智能体暴露的隐藏题集。

## 项目目录

```text
agents/       工作流 Agent
app/          CLI、配置和 Streamlit 界面
models/       数学模型实现
tools/        执行、导出、质量、安全和文档工具
workflows/    工作流编排与阶段状态管理
prompts/      LLM 提示词
benchmarks/   竞赛盲测用例与评分要求
tests/        单元测试、集成测试和安全测试
PDF/          项目参考资料
workspace/    默认输入与运行输出
```

## 已知限制

- 自动模型选择依赖题目关键词、字段名称和数据结构，重要任务仍需人工核验。
- 模型返回成功不代表其假设、参数和结论必然适合实际问题。
- LLM 生成内容可能存在事实、公式或表达错误，最终论文必须人工审查。
- 大型数据集可能超过当前单机工作流的时间和内存能力。
- PDF 导出使用 ReportLab；复杂 Markdown、LaTeX 宏和版式不会被完整复现。
- `.xls` 兼容性取决于本地 pandas Excel 读取后端。
- 当前代码执行防护不是强隔离沙箱。

## 编码约定

项目源代码、配置、提示词和文档统一使用无 BOM 的 UTF-8。命令行入口会尽量将标准输入输出配置为 UTF-8。
