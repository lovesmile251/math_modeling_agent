from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.model_registry import registered_model_ids, validate_model_registry


@dataclass(frozen=True)
class AlgorithmEntry:
    category: str
    problem: str
    recommended: tuple[str, ...]
    alternatives: tuple[str, ...]
    keywords: tuple[str, ...]
    executable_model_ids: tuple[str, ...] = ()


def _entry(
    category: str,
    problem: str,
    recommended: tuple[str, ...],
    alternatives: tuple[str, ...],
    keywords: tuple[str, ...],
    executable_model_ids: tuple[str, ...] = (),
) -> AlgorithmEntry:
    return AlgorithmEntry(category, problem, recommended, alternatives, keywords, executable_model_ids)


ALGORITHM_CATALOG: tuple[AlgorithmEntry, ...] = (
    _entry("评价类", "多指标综合评价（客观）", ("熵权法", "TOPSIS"), ("变异系数法", "灰色关联分析", "VIKOR 法"), ("评价", "指标", "综合", "客观", "权重", "排序"), ("entropy_weights", "topsis_rank", "grey_relation", "vikor")),
    _entry("评价类", "多指标综合评价（主观）", ("层次分析法 AHP",), ("德尔菲法", "模糊层次分析法"), ("评价", "指标", "主观", "专家", "层次分析", "ahp"), ("ahp_weights",)),
    _entry("评价类", "多指标综合评价（混合）", ("AHP-熵权法组合",), ("博弈论组合赋权", "主客观集成"), ("评价", "组合赋权", "主客观", "混合"), ("ahp_entropy_combined",)),
    _entry("评价类", "效率评价问题", ("数据包络分析 DEA",), ("超效率 DEA", "Malmquist 指数"), ("效率", "投入产出", "dea", "malmquist"), ("dea_efficiency",)),
    _entry("评价类", "风险评价问题", ("模糊综合评价",), ("蒙特卡洛模拟", "VaR 模型", "贝叶斯网络"), ("风险", "不确定", "var", "贝叶斯"), ("var_cvar_risk", "fuzzy_evaluation")),
    _entry("评价类", "质量评价问题", ("模糊综合评价",), ("灰色评价法", "云模型评价"), ("质量", "等级", "模糊", "云模型"), ("fuzzy_evaluation",)),
    _entry("优化类", "线性规划问题", ("单纯形法", "线性规划"), ("内点法", "对偶单纯形法"), ("线性规划", "资源配置", "约束", "最优", "成本", "收益"), ("resource_allocation",)),
    _entry("优化类", "非线性规划问题", ("梯度下降法", "拟牛顿法"), ("遗传算法", "粒子群算法", "模拟退火"), ("非线性", "梯度", "拟牛顿", "优化"), ("nonlinear_optimization",)),
    _entry("优化类", "整数规划问题", ("分支定界法", "整数规划"), ("动态规划", "割平面法"), ("整数规划", "整数", "分支定界", "割平面"), ("integer_programming",)),
    _entry("优化类", "0-1 规划问题", ("分支定界法", "动态规划"), ("贪心算法", "启发式算法"), ("0-1", "01", "选择", "背包", "二元变量"), ("knapsack_01",)),
    _entry("优化类", "多目标优化问题", ("加权求和法", "NSGA-II"), ("约束法", "MOEA/D", "理想点法"), ("多目标", "帕累托", "pareto", "nsga"), ("multiobjective_optimization",)),
    _entry("优化类", "路径规划问题", ("Dijkstra 算法", "A* 算法"), ("Floyd 算法", "遗传算法", "蚁群算法"), ("路径", "最短路", "导航", "dijkstra", "a*", "floyd"), ("graph_shortest_paths", "astar_path")),
    _entry("优化类", "分配问题", ("匈牙利算法", "KM 算法"), ("线性规划", "贪心算法"), ("分配", "匹配", "指派", "匈牙利", "km"), ("assignment_plan",)),
    _entry("优化类", "背包问题", ("动态规划", "0-1 规划"), ("贪心算法", "分支定界法"), ("背包", "容量", "选择", "预算"), ("knapsack_01",)),
    _entry("优化类", "装箱问题", ("首次适应算法", "最佳适应算法"), ("遗传算法", "模拟退火"), ("装箱", "bin packing", "装载"), ("bin_packing",)),
    _entry("优化类", "调度问题", ("贪心算法", "动态规划"), ("遗传算法", "粒子群算法"), ("调度", "排程", "工序", "任务分配"), ("scheduling_plan",)),
    _entry("预测类", "时间序列预测（趋势型）", ("ARIMA 模型", "指数平滑法"), ("Holt-Winters 法", "线性回归"), ("时间序列", "趋势", "预测", "未来", "arima", "指数平滑"), ("trend_forecast", "smoothing_forecast")),
    _entry("预测类", "时间序列预测（周期性）", ("季节分解", "SARIMA"), ("傅里叶变换", "小波分析"), ("周期", "季节", "sarima", "seasonal"), ("seasonal_forecast",)),
    _entry("预测类", "数据很少的预测（<10 个）", ("灰色预测 GM(1,1)",), ("指数平滑法", "移动平均法"), ("小样本", "灰色预测", "gm(1,1)", "少数据"), ("grey_gm11", "smoothing_forecast")),
    _entry("预测类", "多变量预测", ("VAR 模型", "多元回归"), ("神经网络", "支持向量回归"), ("多变量", "多元", "var", "回归"), ("var_forecast",)),
    _entry("预测类", "非线性预测", ("多项式回归", "SVR"), ("随机森林", "XGBoost", "神经网络"), ("非线性", "svr", "多项式", "随机森林", "xgboost"), ("nonlinear_forecast",)),
    _entry("预测类", "大数据预测（>1000 样本）", ("XGBoost", "LightGBM"), ("随机森林", "深度神经网络"), ("大数据", "大样本", "lightgbm", "xgboost", "梯度提升", "boosting"), ("gradient_boosting",)),
    _entry("预测类", "中小数据预测", ("多元线性回归", "SVR"), ("随机森林", "K 近邻"), ("中小数据", "回归", "svr", "knn", "岭回归", "ridge"), ("ridge_regression",)),
    _entry("分类聚类", "无标签聚类（球形）", ("K-means 聚类",), ("高斯混合模型", "K-medoids"), ("聚类", "无标签", "球形", "kmeans", "k-means"), ("kmeans_cluster",)),
    _entry("分类聚类", "无标签聚类（任意形状）", ("DBSCAN 聚类",), ("OPTICS 聚类", "谱聚类"), ("聚类", "密度", "任意形状", "dbscan", "optics"), ("dbscan_cluster",)),
    _entry("分类聚类", "层次聚类", ("凝聚层次聚类",), ("分裂层次聚类", "BIRCH"), ("层次聚类", "凝聚", "birch"), ("hierarchical_cluster",)),
    _entry("分类聚类", "二分类问题", ("逻辑回归", "SVM"), ("朴素贝叶斯", "决策树", "随机森林"), ("二分类", "分类", "逻辑回归", "svm"), ("logistic_classifier", "naive_bayes_classifier", "knn_classifier")),
    _entry("分类聚类", "多分类问题", ("随机森林", "XGBoost"), ("决策树", "K 近邻", "神经网络"), ("多分类", "分类", "随机森林", "xgboost", "knn"), ("naive_bayes_classifier", "knn_classifier")),
    _entry("分类聚类", "文本分类", ("朴素贝叶斯", "TF-IDF+SVM"), ("词向量+深度学习",), ("文本", "分类", "tf-idf", "词向量"), ("naive_bayes_classifier",)),
    _entry("分类聚类", "不平衡分类", ("SMOTE+随机森林",), ("代价敏感学习", "集成方法"), ("不平衡", "smote", "少数类"), ("smote_balance",)),
    _entry("机理建模", "人口增长问题", ("Logistic 模型",), ("指数增长模型", "Gompertz 模型"), ("人口", "增长", "logistic", "gompertz"), ("logistic_growth",)),
    _entry("机理建模", "传染病传播", ("SIR 模型", "SEIR 模型"), ("SIS 模型", "SIRD 模型"), ("传染病", "疫情", "传播", "sir", "seir"), ("sir_model",)),
    _entry("机理建模", "生态系统模型", ("Lotka-Volterra 模型",), ("Leslie 矩阵模型", "竞争模型"), ("生态", "种群", "捕食", "lotka"), ("lotka_volterra",)),
    _entry("机理建模", "经济增长模型", ("Solow 模型", "生产函数"), ("Harrod-Domar 模型", "内生增长"), ("经济增长", "solow", "生产函数"), ("solow_growth",)),
    _entry("机理建模", "热传导问题", ("热传导方程", "有限差分"), ("有限元法", "边界元法"), ("热传导", "温度", "有限差分", "有限元"), ("heat_conduction",)),
    _entry("机理建模", "振动问题", ("简谐振动方程",), ("阻尼振动", "受迫振动方程"), ("振动", "阻尼", "受迫"), ("harmonic_oscillator",)),
    _entry("机理建模", "流体问题", ("Navier-Stokes 方程",), ("欧拉方程", "Bernoulli 方程"), ("流体", "流速", "navier", "伯努利", "bernoulli", "管道", "压强"), ("bernoulli_flow",)),
    _entry("机理建模", "化学反应动力学", ("质量作用定律",), ("Michaelis-Menten 方程",), ("化学反应", "动力学", "酶", "michaelis"), ("michaelis_menten",)),
    _entry("图论网络", "最短路径问题", ("Dijkstra 算法", "Floyd 算法"), ("A* 算法", "Bellman-Ford 算法"), ("最短路径", "路径选择", "dijkstra", "floyd"), ("graph_shortest_paths",)),
    _entry("图论网络", "最小生成树", ("Kruskal 算法", "Prim 算法"), ("Boruvka 算法",), ("最小生成树", "mst", "kruskal", "prim"), ("graph_mst",)),
    _entry("图论网络", "网络流问题", ("Ford-Fulkerson 算法",), ("Edmonds-Karp 算法", "Dinic 算法"), ("网络流", "最大流", "ford", "dinic"), ("graph_max_flow",)),
    _entry("图论网络", "社区发现", ("Louvain 算法",), ("Girvan-Newman 算法", "标签传播"), ("社区发现", "社群", "louvain", "模块度", "community"), ("community_detection",)),
    _entry("图论网络", "网络中心性分析", ("度中心性", "介数中心性"), ("接近中心性", "特征向量中心性"), ("中心性", "网络节点", "介数", "度中心"), ("graph_centrality",)),
    _entry("图论网络", "旅行商问题 TSP", ("遗传算法", "模拟退火"), ("蚁群算法", "动态规划"), ("旅行商", "tsp", "巡回", "路径"), ("tsp_route",)),
    _entry("排队论", "单服务台排队", ("M/M/1 排队模型",), ("M/M/1/K 模型", "M/G/1 模型"), ("排队", "单服务台", "等待", "mm1"), ("queue_metrics",)),
    _entry("排队论", "多服务台排队", ("M/M/c 排队模型",), ("M/M/c/K 模型", "M/G/c 模型"), ("排队", "多服务台", "mmc"), ("queue_metrics",)),
    _entry("排队论", "网络排队", ("Jackson 网络",), ("BCMP 网络", "排队网络"), ("网络排队", "jackson", "bcmp"), ("jackson_network",)),
    _entry("库存管理", "确定性库存", ("EOQ 模型", "EPQ 模型"), ("批量折扣模型",), ("库存", "确定性", "eoq", "epq"), ("inventory_policy",)),
    _entry("库存管理", "随机库存", ("报童模型", "(s,S) 策略"), ("(r,Q) 策略", "连续审查策略"), ("随机库存", "报童", "库存策略"), ("inventory_policy",)),
    _entry("库存管理", "多级库存", ("供应链优化",), ("牛鞭效应模型", "博弈论"), ("多级库存", "供应链", "牛鞭"), ("multi_echelon_inventory", "bullwhip_effect")),
    _entry("概率统计", "参数估计", ("最大似然估计", "最小二乘"), ("贝叶斯估计", "矩估计"), ("参数估计", "最大似然", "最小二乘", "mle"), ("parameter_estimation",)),
    _entry("概率统计", "假设检验", ("t 检验", "卡方检验"), ("F 检验", "非参数检验"), ("假设检验", "t检验", "卡方", "f检验"), ("hypothesis_tests",)),
    _entry("概率统计", "方差分析", ("双因素 ANOVA",), ("协方差分析", "多元方差分析"), ("方差分析", "anova", "协方差"), ("anova_analysis",)),
    _entry("概率统计", "回归分析", ("一元/多元线性回归",), ("逻辑回归", "非线性回归"), ("回归", "线性回归", "逻辑回归"), ("linear_regression",)),
    _entry("概率统计", "蒙特卡洛模拟", ("随机采样", "重要性采样"), ("马尔科夫链蒙特卡洛",), ("蒙特卡洛", "随机模拟", "mcmc"), ("monte_carlo",)),
    _entry("博弈论", "静态博弈", ("纳什均衡", "占优策略"), ("混合策略均衡",), ("博弈", "纳什", "均衡"), ("nash_equilibrium",)),
    _entry("博弈论", "动态博弈", ("子博弈完美均衡",), ("完美贝叶斯均衡",), ("动态博弈", "子博弈", "贝叶斯均衡", "stackelberg", "序贯", "领导者"), ("stackelberg_equilibrium",)),
    _entry("博弈论", "合作博弈", ("Shapley 值", "核心"), ("谈判集", "稳定集"), ("合作博弈", "shapley", "核心"), ("shapley_value",)),
    _entry("博弈论", "拍卖理论", ("一价密封拍卖", "二价拍卖"), ("英式拍卖", "荷式拍卖"), ("拍卖", "竞价", "二价"), ("auction_pricing",)),
    _entry("图像处理", "图像增强", ("直方图均衡化", "滤波"), ("小波变换", "形态学处理"), ("图像", "增强", "直方图", "滤波"), ("histogram_equalization",)),
    _entry("图像处理", "边缘检测", ("Canny 算子", "Sobel 算子"), ("Laplacian 算子", "Roberts 算子"), ("边缘检测", "canny", "sobel"), ("edge_detection",)),
    _entry("图像处理", "图像分割", ("阈值分割", "区域增长"), ("分水岭算法", "聚类分割"), ("图像分割", "阈值", "分水岭"), ("image_segmentation",)),
    _entry("图像处理", "特征提取", ("SIFT", "SURF"), ("HOG", "LBP", "ORB"), ("图像特征", "sift", "surf", "hog", "orb"), ("image_features",)),
    _entry("图像处理", "图像配准", ("基于特征点配准",), ("基于区域配准", "基于变换配准"), ("图像配准", "特征点"), ("image_registration",)),
    _entry("信号处理", "信号去噪", ("小波去噪", "滤波器"), ("经验模态分解", "独立成分分析"), ("信号", "去噪", "滤波", "小波"), ("signal_denoising",)),
    _entry("信号处理", "频域分析", ("快速傅里叶变换 FFT",), ("短时傅里叶变换", "小波变换"), ("频域", "fft", "傅里叶"), ("fft_frequency_analysis",)),
    _entry("信号处理", "信号检测", ("匹配滤波", "能量检测"), ("循环平稳检测",), ("信号检测", "匹配滤波", "能量检测"), ("energy_detection",)),
    _entry("数据拟合", "线性拟合", ("最小二乘法",), ("加权最小二乘", "总体最小二乘"), ("拟合", "线性拟合", "最小二乘"), ("weighted_least_squares",)),
    _entry("数据拟合", "非线性拟合", ("Levenberg-Marquardt 算法",), ("Gauss-Newton 法", "梯度下降法"), ("非线性拟合", "levenberg", "gauss-newton"), ("nonlinear_fit",)),
    _entry("数据拟合", "曲线拟合", ("多项式拟合", "样条插值"), ("贝塞尔曲线", "B 样条"), ("曲线拟合", "多项式", "样条"), ("polynomial_fit",)),
    _entry("数据拟合", "参数辨识", ("最小二乘", "极大似然"), ("贝叶斯方法", "卡尔曼滤波"), ("参数辨识", "系统辨识", "卡尔曼"), ("parameter_identification",)),
    _entry("控制论", "最优控制", ("动态规划", "变分法"), ("庞特里亚金最小值原理",), ("最优控制", "控制", "动态规划"), ("optimal_control",)),
    _entry("控制论", "鲁棒控制", ("H∞控制", "滑模控制"), ("自适应控制", "预测控制"), ("鲁棒控制", "滑模", "自适应"), ("robust_control",)),
    _entry("控制论", "状态估计", ("卡尔曼滤波",), ("扩展卡尔曼滤波", "粒子滤波"), ("状态估计", "卡尔曼", "粒子滤波"), ("kalman_filter",)),
    _entry("金融建模", "期权定价", ("Black-Scholes 模型",), ("二叉树模型", "蒙特卡洛模拟"), ("期权", "定价", "black-scholes"), ("black_scholes_pricing",)),
    _entry("金融建模", "风险管理", ("VaR 模型", "CVaR 模型"), ("GARCH 模型", "Copula 模型"), ("金融风险", "var", "cvar", "garch"), ("var_cvar_risk", "garch_volatility")),
    _entry("金融建模", "投资组合优化", ("马科维茨模型",), ("单指数模型", "Black-Litterman"), ("投资组合", "资产配置", "markowitz"), ("markowitz_portfolio",)),
    _entry("交通运输", "交通流模型", ("元胞自动机", "跟驰模型"), ("流体力学模型", "微观仿真"), ("交通流", "元胞自动机", "跟驰"), ("traffic_flow", "car_following")),
    _entry("交通运输", "路径选择", ("Dijkstra 算法", "A* 算法"), ("动态规划", "启发式算法"), ("路径选择", "交通", "导航"), ("graph_shortest_paths", "astar_path")),
    _entry("交通运输", "车辆路径问题 VRP", ("遗传算法", "蚁群算法"), ("模拟退火", "禁忌搜索"), ("车辆路径", "vrp", "配送"), ("vrp_route",)),
    _entry("降维分析", "线性降维", ("主成分分析 PCA",), ("线性判别分析 LDA", "因子分析"), ("降维", "pca", "主成分", "lda"), ("pca",)),
    _entry("降维分析", "非线性降维", ("t-SNE", "UMAP"), ("等距映射", "局部线性嵌入"), ("非线性降维", "tsne", "umap"), ("nonlinear_embedding",)),
    _entry("降维分析", "特征选择", ("方差选择", "相关分析"), ("递归特征消除", "LASSO"), ("特征选择", "方差", "相关", "lasso"), ("feature_selection",)),
    _entry("关联分析", "相关性分析", ("Pearson 相关", "Spearman 相关"), ("偏相关", "典型相关分析"), ("相关性", "pearson", "spearman"), ("correlation_analysis",)),
    _entry("关联分析", "因果分析", ("Granger 因果检验",), ("结构方程模型", "贝叶斯网络"), ("因果", "granger", "结构方程"), ("granger_causality",)),
    _entry("关联分析", "关联规则挖掘", ("Apriori 算法", "FP-Growth"), ("关联规则", "序列模式挖掘"), ("关联规则", "apriori", "fp-growth", "频繁项集"), ("apriori_rules",)),
)


EXECUTABLE_MODEL_LABELS = {
    "trend_forecast": "线性趋势预测",
    "entropy_weights": "熵权法指标权重",
    "topsis_rank": "TOPSIS 综合排名",
    "capacity_gap": "需求容量缺口分析",
    "resource_allocation": "资源配置优化",
    "knapsack_01": "0-1 背包/选择优化",
    "assignment_plan": "任务分配/指派优化",
    "bin_packing": "装箱优化",
    "scheduling_plan": "调度排程优化",
    "correlation_analysis": "Pearson/Spearman 相关性分析",
    "linear_regression": "线性回归分析",
    "parameter_estimation": "参数估计",
    "hypothesis_tests": "假设检验",
    "anova_analysis": "方差分析 ANOVA",
    "polynomial_fit": "多项式曲线拟合",
    "pca": "主成分分析 PCA",
    "feature_selection": "特征选择",
    "kmeans_cluster": "K-means 聚类",
    "dbscan_cluster": "DBSCAN 密度聚类",
    "hierarchical_cluster": "凝聚层次聚类",
    "logistic_classifier": "逻辑回归二分类",
    "naive_bayes_classifier": "朴素贝叶斯分类",
    "knn_classifier": "K 近邻分类",
    "ahp_weights": "层次分析法 AHP 权重",
    "grey_relation": "灰色关联分析",
    "vikor": "VIKOR 综合评价",
    "grey_gm11": "灰色预测 GM(1,1)",
    "smoothing_forecast": "指数平滑/移动平均预测",
    "graph_shortest_paths": "图论最短路径",
    "graph_mst": "最小生成树",
    "graph_max_flow": "网络最大流",
    "graph_centrality": "网络中心性分析",
    "queue_metrics": "M/M/c 排队指标",
    "inventory_policy": "EOQ 库存策略",
    "var_cvar_risk": "VaR/CVaR 风险度量",
    "garch_volatility": "GARCH 波动率估计",
    "black_scholes_pricing": "Black-Scholes 期权定价",
    "markowitz_portfolio": "Markowitz 投资组合优化",
    "apriori_rules": "Apriori 关联规则",
    "granger_causality": "Granger 因果检验",
    "fft_frequency_analysis": "FFT 频域分析",
    "signal_denoising": "信号去噪",
    "energy_detection": "能量检测",
    "ahp_entropy_combined": "AHP-熵权组合赋权",
    "dea_efficiency": "DEA 效率评价",
    "fuzzy_evaluation": "模糊综合评价",
    "nonlinear_optimization": "非线性规划（梯度法）",
    "integer_programming": "整数规划（分支定界）",
    "multiobjective_optimization": "多目标加权求和优化",
    "astar_path": "A* 路径规划",
    "tsp_route": "TSP 启发式路径",
    "vrp_route": "VRP 车辆路径（节约算法）",
    "seasonal_forecast": "季节分解预测",
    "var_forecast": "VAR 多变量预测",
    "nonlinear_forecast": "非线性回归预测",
    "smote_balance": "SMOTE 不平衡分析",
    "nonlinear_embedding": "非线性降维（MDS）",
    "monte_carlo": "蒙特卡洛模拟",
    "jackson_network": "Jackson 排队网络",
    "multi_echelon_inventory": "多级库存策略",
    "bullwhip_effect": "牛鞭效应分析",
    "logistic_growth": "Logistic 人口增长拟合",
    "sir_model": "SIR 传染病参数估计",
    "lotka_volterra": "Lotka-Volterra 生态模型",
    "solow_growth": "Solow 经济增长模型",
    "heat_conduction": "一维热传导扩散系数",
    "harmonic_oscillator": "简谐振动拟合",
    "michaelis_menten": "Michaelis-Menten 动力学",
    "nash_equilibrium": "纳什均衡（2x2）",
    "shapley_value": "Shapley 值分配",
    "auction_pricing": "拍卖定价（一价/二价）",
    "kalman_filter": "卡尔曼滤波状态估计",
    "optimal_control": "动态规划最优控制",
    "robust_control": "鲁棒控制评估",
    "weighted_least_squares": "加权最小二乘拟合",
    "nonlinear_fit": "非线性最小二乘拟合",
    "parameter_identification": "动态系统参数辨识",
    "histogram_equalization": "直方图均衡化",
    "edge_detection": "Sobel 边缘检测",
    "image_segmentation": "Otsu 阈值分割",
    "image_features": "HOG/LBP 特征提取",
    "image_registration": "图像配准平移估计",
    "traffic_flow": "元胞自动机交通流",
    "car_following": "智能驾驶跟驰模型",
    "gradient_boosting": "梯度提升树回归预测",
    "ridge_regression": "岭回归交叉验证预测",
    "bernoulli_flow": "Bernoulli 流体能量分析",
    "community_detection": "贪心模块度社区发现",
    "friend_recommendation": "Campus friend recommendation",
    "information_propagation": "Campus information propagation",
    "influence_maximization": "Campus influence maximization",
    "stackelberg_equilibrium": "Stackelberg 子博弈完美均衡",
    "error_analysis": "误差分析（回归残差与拟合优度）",
    "sensitivity_analysis": "灵敏度分析（参数扰动弹性）",
    "model_comparison": "模型对比（交叉验证 RMSE/R²）",
}


def executable_model_ids() -> set[str]:
    return set(EXECUTABLE_MODEL_LABELS)


def catalog_size() -> int:
    return len(ALGORITHM_CATALOG)


def catalog_executable_model_ids() -> set[str]:
    """Return every executable model ID referenced by catalog entries."""
    return {
        model_id
        for entry in ALGORITHM_CATALOG
        for model_id in entry.executable_model_ids
    }


def validate_catalog_registry_integrity(*, import_symbols: bool = False) -> list[str]:
    """Return consistency errors across catalog labels and codegen registry.

    The catalog is allowed to omit always-on or fallback models, but every ID it
    references must be labeled and registered for code generation. Conversely,
    every registered executable model must have a user-facing label.
    """
    errors = validate_model_registry(import_symbols=import_symbols)
    catalog_ids = catalog_executable_model_ids()
    label_ids = executable_model_ids()
    registry_ids = registered_model_ids()
    applicability_ids = set(MODEL_APPLICABILITY)

    errors.extend(
        _missing_model_errors(
            sorted(catalog_ids - label_ids),
            "catalog references model_id without label",
        )
    )
    errors.extend(
        _missing_model_errors(
            sorted(catalog_ids - registry_ids),
            "catalog references model_id not registered for code generation",
        )
    )
    errors.extend(
        _missing_model_errors(
            sorted(label_ids - registry_ids),
            "label exists for model_id not registered for code generation",
        )
    )
    errors.extend(
        _missing_model_errors(
            sorted(registry_ids - label_ids),
            "codegen registry model_id has no label",
        )
    )
    errors.extend(
        _missing_model_errors(
            sorted(applicability_ids - label_ids),
            "applicability rule exists for model_id without label",
        )
    )
    errors.extend(
        _missing_model_errors(
            sorted(applicability_ids - registry_ids),
            "applicability rule exists for model_id not registered for code generation",
        )
    )
    return errors


def _missing_model_errors(model_ids: list[str], message: str) -> list[str]:
    return [f"{message}: {model_id}" for model_id in model_ids]


@dataclass(frozen=True)
class ApplicabilityRule:
    task_types: tuple[str, ...]
    min_rows: int = 0
    max_rows_preferred: int = 0  # 0 = no upper bound; preferred ceiling for the model
    min_numeric_cols: int = 0
    min_categorical_cols: int = 0
    needs_time: bool = False
    needs_target: bool = False
    needs_label: bool = False
    needs_demand: bool = False
    needs_capacity: bool = False
    needs_relation: bool = False
    needs_objective_or_constraint: bool = False
    role: str = "candidate"
    validation: str = ""
    not_recommended_when: tuple[str, ...] = ()  # conditions that make this model a poor fit
    innovation_extensions: tuple[str, ...] = ()  # innovative ways to enhance this model
    tier: str = "baseline"  # baseline | improved | innovation


@dataclass(frozen=True)
class ApplicabilityResult:
    model_id: str
    can_run: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    required_fields: tuple[str, ...]
    role: str = "candidate"
    task_types: tuple[str, ...] = ()
    not_recommended_when: tuple[str, ...] = ()
    innovation_extensions: tuple[str, ...] = ()
    tier: str = "baseline"
    validation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "can_run": self.can_run,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "required_fields": list(self.required_fields),
            "role": self.role,
            "task_types": list(self.task_types),
            "not_recommended_when": list(self.not_recommended_when),
            "innovation_extensions": list(self.innovation_extensions),
            "tier": self.tier,
            "validation": self.validation,
        }


@dataclass(frozen=True)
class ModelContract:
    """Auditable modeling contract shared by selection, execution and review."""

    model_id: str
    label: str
    task_types: tuple[str, ...]
    role: str
    tier: str
    min_rows: int
    required_data: tuple[str, ...]
    assumptions: tuple[str, ...]
    metrics: tuple[str, ...]
    diagnostics: tuple[str, ...]
    baseline_models: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    innovation_extensions: tuple[str, ...]
    explicit: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "task_types": list(self.task_types),
            "role": self.role,
            "tier": self.tier,
            "min_rows": self.min_rows,
            "required_data": list(self.required_data),
            "assumptions": list(self.assumptions),
            "metrics": list(self.metrics),
            "diagnostics": list(self.diagnostics),
            "baseline_models": list(self.baseline_models),
            "failure_conditions": list(self.failure_conditions),
            "innovation_extensions": list(self.innovation_extensions),
            "explicit": self.explicit,
        }


MODEL_APPLICABILITY: dict[str, ApplicabilityRule] = {
    # Prediction
    "trend_forecast": ApplicabilityRule(("forecast", "exploration"), min_rows=3, min_numeric_cols=1, role="primary", tier="baseline", validation="check trend residuals and extrapolation range", not_recommended_when=("strong nonlinearity", "multiple seasonality"), innovation_extensions=("residual_correction", "piecewise_trend")),
    "smoothing_forecast": ApplicabilityRule(("forecast",), min_rows=4, min_numeric_cols=1, role="comparison", tier="baseline", validation="compare MAE/RMSE with trend forecast", not_recommended_when=("long forecast horizon", "abrupt level shifts"), innovation_extensions=("holt_winters_extension",)),
    "grey_gm11": ApplicabilityRule(("forecast",), min_rows=4, max_rows_preferred=30, min_numeric_cols=1, role="comparison", tier="baseline", validation="check posterior error ratio", not_recommended_when=("strong seasonality", "many explanatory variables", "large sample >100"), innovation_extensions=("grey_markov_residual_correction", "metabolic_gm11")),
    "seasonal_forecast": ApplicabilityRule(("forecast",), min_rows=8, min_numeric_cols=1, needs_time=True, role="comparison", tier="improved", validation="confirm stable seasonality", not_recommended_when=("no visible seasonal pattern", "irregular time intervals"), innovation_extensions=("stl_decomposition", "fourier_seasonal")),
    "var_forecast": ApplicabilityRule(("forecast", "statistics"), min_rows=8, min_numeric_cols=2, needs_time=True, role="comparison", tier="improved", validation="check stationarity and lag order", not_recommended_when=("fewer than 3 interrelated variables", "short time series"), innovation_extensions=("structual_var", "cointegration_vecm")),
    "nonlinear_forecast": ApplicabilityRule(("forecast",), min_rows=6, min_numeric_cols=2, role="comparison", tier="improved", validation="compare against linear baseline", not_recommended_when=("near-linear data", "very small sample"), innovation_extensions=("kernel_ridge", "gaussian_process")),
    "gradient_boosting": ApplicabilityRule(("forecast", "classification"), min_rows=20, min_numeric_cols=2, needs_target=True, role="comparison", tier="improved", validation="use holdout or cross validation", not_recommended_when=("need high interpretability", "very small sample <50"), innovation_extensions=("stacking_ensemble", "shap_explanation")),
    "ridge_regression": ApplicabilityRule(("forecast", "statistics"), min_rows=6, min_numeric_cols=2, needs_target=True, role="comparison", tier="baseline", validation="cross-validate regularization", not_recommended_when=("high-dimensional sparse data", "strong multicollinearity already handled"), innovation_extensions=("elastic_net", "lasso_lars")),
    # Evaluation
    "entropy_weights": ApplicabilityRule(("evaluation", "exploration"), min_rows=3, min_numeric_cols=2, role="primary", tier="baseline", validation="check weight stability", not_recommended_when=("need subjective input", "indicators are highly correlated"), innovation_extensions=("entropy_topsis_combined", "critic_weights")),
    "topsis_rank": ApplicabilityRule(("evaluation", "exploration"), min_rows=3, min_numeric_cols=2, role="primary", tier="baseline", validation="run sensitivity analysis on indicator direction", not_recommended_when=("indicators are not independent", "no clear benefit/cost classification"), innovation_extensions=("entropy_topsis", "grey_topsis")),
    "grey_relation": ApplicabilityRule(("evaluation",), min_rows=3, min_numeric_cols=2, role="comparison", tier="baseline", validation="inspect relation coefficient separation", not_recommended_when=("need absolute ranking", "indicators have strong collinearity"), innovation_extensions=("grey_ahp", "grey_clustering")),
    "vikor": ApplicabilityRule(("evaluation",), min_rows=3, min_numeric_cols=2, role="comparison", tier="improved", validation="check compromise ranking stability", not_recommended_when=("decision maker prefers simple ranking", "indicators are homogeneous"), innovation_extensions=("fuzzy_vikor", "entropy_vikor")),
    "ahp_weights": ApplicabilityRule(("evaluation",), min_rows=2, min_numeric_cols=2, role="comparison", tier="baseline", validation="check pairwise matrix consistency", not_recommended_when=("no expert judgment available", "too many indicators >15"), innovation_extensions=("fuzzy_ahp", "ahp_entropy_combined")),
    "ahp_entropy_combined": ApplicabilityRule(("evaluation",), min_rows=3, min_numeric_cols=2, role="comparison", tier="innovation", validation="compare subjective and objective weights", not_recommended_when=("no subjective preference", "purely objective problem"), innovation_extensions=("game_theory_combination", "deviation_maximization")),
    "dea_efficiency": ApplicabilityRule(("evaluation",), min_rows=3, min_numeric_cols=2, role="comparison", tier="improved", validation="confirm input/output indicator split", not_recommended_when=("fewer than 2 inputs and 1 output", "need ranking beyond efficiency frontier"), innovation_extensions=("super_efficiency_dea", "malmquist_index")),
    "fuzzy_evaluation": ApplicabilityRule(("evaluation",), min_rows=3, min_numeric_cols=1, min_categorical_cols=1, role="comparison", tier="improved", validation="confirm membership grades", not_recommended_when=("data is crisp with clear boundaries", "membership function hard to define"), innovation_extensions=("cloud_model", "intuitionistic_fuzzy")),
    # Capacity gap
    "capacity_gap": ApplicabilityRule(("evaluation", "optimization"), min_rows=1, min_numeric_cols=2, needs_demand=True, needs_capacity=True, role="primary", tier="baseline", validation="match demand and capacity fields", not_recommended_when=("no paired demand-capacity columns", "data is purely qualitative"), innovation_extensions=("dynamic_capacity_planning", "stochastic_gap_analysis")),
    # Optimization
    "resource_allocation": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=2, needs_objective_or_constraint=True, role="primary", tier="baseline", validation="check objective and constraints", not_recommended_when=("problem is nonlinear", "integer decisions required"), innovation_extensions=("robust_optimization", "stochastic_programming")),
    "knapsack_01": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=2, needs_objective_or_constraint=True, role="comparison", tier="baseline", validation="check capacity feasibility", not_recommended_when=("continuous decisions", "multiple knapsacks with coupling"), innovation_extensions=("multi_knapsack", "branch_and_price")),
    "assignment_plan": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=2, role="comparison", tier="baseline", validation="check one-to-one assignment constraints", not_recommended_when=("many-to-many assignment", "dynamic reassignment needed"), innovation_extensions=("generalized_assignment", "stable_marriage")),
    "bin_packing": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=1, needs_capacity=True, role="comparison", tier="baseline", validation="check bin capacity", not_recommended_when=("items are splittable", "online packing scenario"), innovation_extensions=("column_generation", "heuristic_meta")),
    "scheduling_plan": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=2, needs_objective_or_constraint=True, role="comparison", tier="improved", validation="check schedule conflicts", not_recommended_when=("preemptive scheduling", "stochastic processing times"), innovation_extensions=("genetic_scheduling", "constraint_programming")),
    "nonlinear_optimization": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=2, needs_objective_or_constraint=True, role="comparison", tier="improved", validation="check convergence from multiple starts", not_recommended_when=("problem is convex and linear methods suffice", "many local optima without global solver"), innovation_extensions=("global_optimization", "surrogate_model")),
    "integer_programming": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=2, needs_objective_or_constraint=True, role="comparison", tier="improved", validation="check integrality and feasibility", not_recommended_when=("continuous relaxation gives good enough answer", "problem too large for exact solver"), innovation_extensions=("branch_and_cut", "lagrangian_relaxation")),
    "multiobjective_optimization": ApplicabilityRule(("optimization",), min_rows=2, min_numeric_cols=2, needs_objective_or_constraint=True, role="comparison", tier="innovation", validation="inspect Pareto tradeoffs", not_recommended_when=("objectives can be merged into single", "decision maker has clear preference"), innovation_extensions=("nsga2_genetic", "interactive_method")),
    # Classification
    "logistic_classifier": ApplicabilityRule(("classification",), min_rows=8, min_numeric_cols=1, needs_label=True, role="primary", tier="baseline", validation="check accuracy, recall, and confusion matrix", not_recommended_when=("nonlinear decision boundary", "high-dimensional sparse features"), innovation_extensions=("regularized_logistic", "ordinal_logistic")),
    "naive_bayes_classifier": ApplicabilityRule(("classification",), min_rows=8, min_numeric_cols=1, needs_label=True, role="comparison", tier="baseline", validation="compare classification metrics", not_recommended_when=("features are highly correlated", "continuous features not normal"), innovation_extensions=("gaussian_nb", "complement_nb")),
    "knn_classifier": ApplicabilityRule(("classification",), min_rows=8, min_numeric_cols=1, needs_label=True, role="comparison", tier="baseline", validation="use cross validation", not_recommended_when=("high-dimensional data", "large dataset >10000"), innovation_extensions=("weighted_knn", "distance_metric_learning")),
    "smote_balance": ApplicabilityRule(("classification",), min_rows=8, min_numeric_cols=1, needs_label=True, role="validation", tier="improved", validation="check minority class ratio", not_recommended_when=("classes are balanced", "minority class too rare <5 samples"), innovation_extensions=("adasyn", "borderline_smote")),
    # Clustering and dimensionality reduction
    "kmeans_cluster": ApplicabilityRule(("clustering",), min_rows=4, min_numeric_cols=2, role="primary", tier="baseline", validation="inspect silhouette score", not_recommended_when=("non-spherical clusters", "unknown cluster count", "outliers present"), innovation_extensions=("kmeans_plusplus", "gap_statistic")),
    "dbscan_cluster": ApplicabilityRule(("clustering",), min_rows=6, min_numeric_cols=2, role="comparison", tier="improved", validation="inspect noise point ratio", not_recommended_when=("varying density clusters", "high-dimensional data"), innovation_extensions=("optics", "hdbscan")),
    "hierarchical_cluster": ApplicabilityRule(("clustering",), min_rows=4, min_numeric_cols=2, role="comparison", tier="baseline", validation="check dendrogram stability", not_recommended_when=("large dataset >5000", "need flat clustering only"), innovation_extensions=("birch", "two_stage_clustering")),
    "pca": ApplicabilityRule(("clustering", "statistics"), min_rows=4, min_numeric_cols=2, role="validation", tier="baseline", validation="check cumulative explained variance", not_recommended_when=("features are not continuous", "nonlinear relationships dominate"), innovation_extensions=("kernel_pca", "sparse_pca")),
    "feature_selection": ApplicabilityRule(("statistics", "classification"), min_rows=4, min_numeric_cols=2, role="validation", tier="improved", validation="check selected feature stability", not_recommended_when=("fewer than 5 features", "features are already orthogonal"), innovation_extensions=("recursive_feature_elimination", "boruta")),
    "nonlinear_embedding": ApplicabilityRule(("clustering",), min_rows=6, min_numeric_cols=2, role="comparison", tier="improved", validation="check neighborhood stability", not_recommended_when=("data is already low-dimensional", "global structure matters more"), innovation_extensions=("umap", "autoencoder")),
    # Graph/network
    "graph_shortest_paths": ApplicabilityRule(("network", "optimization"), min_rows=2, needs_relation=True, role="primary", tier="baseline", validation="check path connectivity", not_recommended_when=("negative edge weights", "dynamic edge weights"), innovation_extensions=("bidirectional_dijkstra", "contraction_hierarchies")),
    "graph_mst": ApplicabilityRule(("network", "optimization"), min_rows=2, needs_relation=True, role="comparison", tier="baseline", validation="check graph connectivity", not_recommended_when=("directed graph with asymmetric weights", "need steiner tree"), innovation_extensions=("degree_constrained_mst", "steiner_tree")),
    "graph_max_flow": ApplicabilityRule(("network", "optimization"), min_rows=2, needs_relation=True, role="comparison", tier="baseline", validation="check capacity conservation", not_recommended_when=("multi-commodity flow", "node capacities dominate"), innovation_extensions=("min_cost_flow", "multi_commodity_flow")),
    "graph_centrality": ApplicabilityRule(("network",), min_rows=2, needs_relation=True, role="primary", tier="baseline", validation="interpret centrality ranking", not_recommended_when=("disconnected graph", "need dynamic centrality"), innovation_extensions=("pagerank", "eigenvector_centrality")),
    "community_detection": ApplicabilityRule(("network", "clustering"), min_rows=2, needs_relation=True, role="primary", tier="improved", validation="check modularity and community size", not_recommended_when=("overlapping communities needed", "network is too small <10 nodes"), innovation_extensions=("overlapping_community", "label_propagation")),
    "friend_recommendation": ApplicabilityRule(("network", "forecast"), min_rows=2, needs_relation=True, role="primary", tier="improved", validation="compare link-prediction scores and inspect top recommendations", not_recommended_when=("no friend edge list", "target user cannot be identified"), innovation_extensions=("supervised_link_prediction", "embedding_similarity")),
    "information_propagation": ApplicabilityRule(("network", "simulation"), min_rows=2, needs_relation=True, role="primary", tier="improved", validation="run repeated diffusion simulations and compare key-user rankings", not_recommended_when=("network is too sparse", "no propagation assumptions can be justified"), innovation_extensions=("time_aware_ic_model", "topic_sensitive_diffusion")),
    "influence_maximization": ApplicabilityRule(("network", "optimization"), min_rows=2, needs_relation=True, role="primary", tier="innovation", validation="compare greedy seeds with degree, PageRank, and random baselines", not_recommended_when=("push quota is undefined", "network is too small for seed selection"), innovation_extensions=("budgeted_influence_maximization", "community_diverse_seeding")),
    "astar_path": ApplicabilityRule(("network", "optimization"), min_rows=2, needs_relation=True, role="comparison", tier="improved", validation="check heuristic admissibility", not_recommended_when=("no admissible heuristic", "need all-pairs shortest path"), innovation_extensions=("theta_star", "any_angle_path")),
    "tsp_route": ApplicabilityRule(("network", "optimization"), min_rows=3, needs_relation=True, role="comparison", tier="improved", validation="compare route length baseline", not_recommended_when=("fewer than 5 cities", "asymmetric distances not handled"), innovation_extensions=("lin_kernighan", "ant_colony_tsp")),
    "vrp_route": ApplicabilityRule(("network", "optimization"), min_rows=3, needs_relation=True, needs_capacity=True, role="comparison", tier="improved", validation="check vehicle capacity feasibility", not_recommended_when=("time windows dominate", "single vehicle suffices"), innovation_extensions=("vrptw", "sweep_algorithm")),
    # Statistics
    "correlation_analysis": ApplicabilityRule(("statistics",), min_rows=3, min_numeric_cols=2, role="validation", tier="baseline", validation="check Pearson/Spearman significance", not_recommended_when=("nonlinear relationships only", "categorical variables dominate"), innovation_extensions=("partial_correlation", "distance_correlation")),
    "linear_regression": ApplicabilityRule(("statistics", "forecast"), min_rows=4, min_numeric_cols=2, needs_target=True, role="comparison", tier="baseline", validation="check R2, residuals, and collinearity", not_recommended_when=("nonlinear relationship", "heteroscedasticity severe"), innovation_extensions=("robust_regression", "quantile_regression")),
    "parameter_estimation": ApplicabilityRule(("statistics",), min_rows=3, min_numeric_cols=1, role="validation", tier="baseline", validation="report confidence intervals", not_recommended_when=("distribution is unknown", "sample too small for asymptotic normality"), innovation_extensions=("bootstrap_ci", "bayesian_estimation")),
    "hypothesis_tests": ApplicabilityRule(("statistics",), min_rows=4, min_numeric_cols=1, role="validation", tier="baseline", validation="report p-value and effect size", not_recommended_when=("multiple testing without correction", "distribution assumptions violated"), innovation_extensions=("permutation_test", "bootstrap_test")),
    "anova_analysis": ApplicabilityRule(("statistics",), min_rows=4, min_numeric_cols=1, min_categorical_cols=1, role="validation", tier="improved", validation="check group sizes", not_recommended_when=("groups are unbalanced and small", "normality assumption violated"), innovation_extensions=("welch_anova", "kruskal_wallis")),
    "monte_carlo": ApplicabilityRule(("statistics", "simulation"), min_rows=1, min_numeric_cols=1, role="comparison", tier="improved", validation="check random seed sensitivity", not_recommended_when=("analytic solution is tractable", "computational budget too low"), innovation_extensions=("latin_hypercube", "importance_sampling")),
}


_TASK_METRICS: dict[str, tuple[str, ...]] = {
    "forecast": ("MAE", "RMSE", "MAPE", "R²", "rolling_backtest_error"),
    "evaluation": ("ranking_stability", "weight_sensitivity", "rank_correlation"),
    "optimization": ("objective_value", "constraint_violation", "runtime", "robustness"),
    "classification": ("accuracy", "precision", "recall", "F1", "confusion_matrix"),
    "clustering": ("silhouette_score", "davies_bouldin_index", "cluster_stability"),
    "network": ("connectivity", "path_or_flow_validity", "network_stability"),
    "statistics": ("effect_size", "confidence_interval", "p_value", "residual_diagnostics"),
    "simulation": ("replication_variance", "confidence_interval", "seed_sensitivity"),
    "exploration": ("coverage", "missing_rate", "distribution_summary"),
}

_TASK_ASSUMPTIONS: dict[str, tuple[str, ...]] = {
    "forecast": ("observations are ordered in time", "historical patterns remain informative"),
    "evaluation": ("indicator directions and scales are correctly defined",),
    "optimization": ("objective and constraints approximate the real decision problem",),
    "classification": ("labels are reliable and train/evaluation data are leakage-free",),
    "clustering": ("chosen features and distance scale represent meaningful similarity",),
    "network": ("nodes and edges represent the intended relation without duplicate semantics",),
    "statistics": ("samples are representative and inferential assumptions are checked",),
    "simulation": ("random mechanism and parameter distributions are defensible",),
    "exploration": ("data definitions are consistent across files",),
}

_TASK_BASELINES: dict[str, tuple[str, ...]] = {
    "forecast": ("trend_forecast", "smoothing_forecast"),
    "evaluation": ("entropy_weights", "topsis_rank"),
    "optimization": ("resource_allocation",),
    "classification": ("logistic_classifier",),
    "clustering": ("kmeans_cluster",),
    "network": ("graph_centrality", "graph_shortest_paths"),
    "statistics": ("correlation_analysis", "linear_regression"),
    "simulation": ("monte_carlo",),
    "exploration": ("trend_forecast", "entropy_weights"),
}


def get_model_contract(model_id: str) -> ModelContract:
    """Return a complete contract for every registered executable model."""
    if model_id not in registered_model_ids():
        raise KeyError(f"unregistered model_id: {model_id}")
    rule = MODEL_APPLICABILITY.get(model_id)
    explicit = rule is not None
    if rule is None:
        task_types = _infer_contract_task_types(model_id)
        rule = ApplicabilityRule(
            task_types=task_types,
            min_rows=2,
            min_numeric_cols=1,
            validation="compare against a task-appropriate baseline and inspect output validity",
        )

    required_data = _required_data_from_rule(rule)
    assumptions = tuple(
        dict.fromkeys(
            assumption
            for task_type in rule.task_types
            for assumption in _TASK_ASSUMPTIONS.get(task_type, ())
        )
    )
    metrics = tuple(
        dict.fromkeys(
            metric
            for task_type in rule.task_types
            for metric in _TASK_METRICS.get(task_type, ())
        )
    )
    baseline_models = tuple(
        model
        for model in dict.fromkeys(
            baseline
            for task_type in rule.task_types
            for baseline in _TASK_BASELINES.get(task_type, ())
        )
        if model != model_id and model in registered_model_ids()
    )
    diagnostics = tuple(
        dict.fromkeys(
            [
                rule.validation or "inspect result validity",
                *("check data sufficiency", "compare with baseline"),
            ]
        )
    )
    return ModelContract(
        model_id=model_id,
        label=EXECUTABLE_MODEL_LABELS[model_id],
        task_types=rule.task_types,
        role=rule.role,
        tier=rule.tier,
        min_rows=rule.min_rows,
        required_data=required_data,
        assumptions=assumptions or _TASK_ASSUMPTIONS["exploration"],
        metrics=metrics or _TASK_METRICS["exploration"],
        diagnostics=diagnostics,
        baseline_models=baseline_models,
        failure_conditions=rule.not_recommended_when,
        innovation_extensions=rule.innovation_extensions,
        explicit=explicit,
    )


def model_contracts() -> dict[str, ModelContract]:
    return {model_id: get_model_contract(model_id) for model_id in sorted(registered_model_ids())}


def validate_model_contracts() -> list[str]:
    errors: list[str] = []
    for model_id, contract in model_contracts().items():
        if not contract.task_types:
            errors.append(f"{model_id}: contract has no task_types")
        if not contract.required_data:
            errors.append(f"{model_id}: contract has no required_data")
        if not contract.assumptions:
            errors.append(f"{model_id}: contract has no assumptions")
        if not contract.metrics:
            errors.append(f"{model_id}: contract has no metrics")
        if not contract.diagnostics:
            errors.append(f"{model_id}: contract has no diagnostics")
    return errors


def _required_data_from_rule(rule: ApplicabilityRule) -> tuple[str, ...]:
    required = [f"rows>={rule.min_rows}"]
    if rule.min_numeric_cols:
        required.append(f"numeric_columns>={rule.min_numeric_cols}")
    if rule.min_categorical_cols:
        required.append(f"categorical_columns>={rule.min_categorical_cols}")
    for enabled, label in (
        (rule.needs_time, "time_column"),
        (rule.needs_target, "target_column"),
        (rule.needs_label, "label_column"),
        (rule.needs_demand, "demand_column"),
        (rule.needs_capacity, "capacity_column"),
        (rule.needs_relation, "source_target_relation"),
        (rule.needs_objective_or_constraint, "objective_or_constraint_columns"),
    ):
        if enabled:
            required.append(label)
    return tuple(required)


def _infer_contract_task_types(model_id: str) -> tuple[str, ...]:
    groups = {
        "forecast": (
            "forecast", "growth", "garch", "regression", "fit", "identification",
            "kalman", "smoothing", "grey_gm",
        ),
        "evaluation": ("weight", "topsis", "vikor", "relation", "efficiency", "risk"),
        "optimization": (
            "optimization", "allocation", "knapsack", "assignment", "packing",
            "scheduling", "route", "control", "portfolio", "pricing",
        ),
        "classification": ("classifier", "smote"),
        "clustering": ("cluster", "embedding", "pca", "feature_selection", "segmentation"),
        "network": ("graph", "community", "network", "path", "flow"),
        "statistics": (
            "correlation", "estimation", "hypothesis", "anova", "causality",
            "monte_carlo", "error_analysis", "sensitivity", "comparison",
        ),
        "simulation": (
            "sir", "lotka", "traffic", "oscillator", "heat", "signal",
            "energy", "inventory", "bullwhip", "queue", "jackson",
        ),
    }
    matched = [
        task_type
        for task_type, tokens in groups.items()
        if any(token in model_id for token in tokens)
    ]
    return tuple(dict.fromkeys(matched or ["exploration"]))


def check_applicable(model_id: str, profile: Any, task_type: str | None = None) -> ApplicabilityResult:
    """Return a structured applicability decision for a runnable model.

    ``profile`` can be a DataProfile-like object or a dict with the same keys.
    The result is intentionally serializable so model selection and code
    generation can share the same gate without importing agent internals.
    """

    rule = MODEL_APPLICABILITY.get(model_id)
    if rule is None:
        return _default_applicability(model_id, profile, task_type)

    reasons: list[str] = []
    warnings: list[str] = []
    required_fields: list[str] = []
    can_run = True

    rows = int(_profile_value(profile, "rows", 0) or 0)
    numeric = _profile_tuple(profile, "numeric_columns")
    categorical = _profile_tuple(profile, "categorical_columns")
    datetime_cols = _profile_tuple(profile, "datetime_columns")
    target = _profile_tuple(profile, "target_columns")
    demand = _profile_tuple(profile, "demand_columns")
    capacity = _profile_tuple(profile, "capacity_columns")
    relation = _profile_tuple(profile, "relation_columns")
    id_like = _profile_tuple(profile, "id_like_columns")
    cost = _profile_tuple(profile, "cost_columns")
    benefit = _profile_tuple(profile, "benefit_columns")
    constraint = _profile_tuple(profile, "constraint_columns")
    monotonic_time = _profile_tuple(profile, "monotonic_time_columns")
    binary_labels = _profile_tuple(profile, "binary_label_columns")
    multiclass_labels = _profile_tuple(profile, "multiclass_label_columns")
    resource = _profile_tuple(profile, "resource_columns")
    objective_constraint = _profile_tuple(profile, "objective_constraint_columns")
    has_edge_table = bool(_profile_value(profile, "has_edge_table", False))
    has_objective_constraint_combo = bool(_profile_value(profile, "has_objective_constraint_combo", False))

    if task_type and task_type not in rule.task_types and task_type != "exploration":
        warnings.append(f"model is tuned for {', '.join(rule.task_types)}, current task is {task_type}")
    elif task_type:
        reasons.append(f"matches task type {task_type}")

    if rows < rule.min_rows:
        can_run = False
        required_fields.append(f"min_rows:{rule.min_rows}")
        warnings.append(f"needs at least {rule.min_rows} rows, got {rows}")
    elif rule.min_rows:
        reasons.append(f"has {rows} rows")

    if len(numeric) < rule.min_numeric_cols:
        can_run = False
        required_fields.append(f"numeric_columns:{rule.min_numeric_cols}")
        warnings.append(f"needs at least {rule.min_numeric_cols} numeric columns")
    elif rule.min_numeric_cols:
        reasons.append(f"has {len(numeric)} numeric columns")

    if len(categorical) < rule.min_categorical_cols:
        can_run = False
        required_fields.append(f"categorical_columns:{rule.min_categorical_cols}")
        warnings.append(f"needs at least {rule.min_categorical_cols} categorical columns")
    elif rule.min_categorical_cols:
        reasons.append(f"has {len(categorical)} categorical columns")

    if rule.needs_time:
        time_cols = monotonic_time or datetime_cols
        _require_any(time_cols, "datetime_columns", "time field", required_fields, warnings, reasons)
        if monotonic_time:
            reasons.append("has monotonic time field")
        can_run = can_run and bool(time_cols)
    if rule.needs_target:
        has_target = bool(target) or len(numeric) >= max(2, rule.min_numeric_cols)
        if has_target:
            reasons.append("has target-like field or enough numeric columns")
        else:
            can_run = False
            required_fields.append("target_columns")
            warnings.append("needs a target column or dependent variable")
    if rule.needs_label:
        has_label = bool(binary_labels or multiclass_labels or target) or _has_named_column(profile, ("label", "class"))
        if has_label:
            reasons.append("has binary/multiclass label field")
        else:
            can_run = False
            required_fields.append("target_columns")
            warnings.append("needs a label/class target column")
    if rule.needs_demand:
        _require_any(demand, "demand_columns", "demand field", required_fields, warnings, reasons)
        can_run = can_run and bool(demand)
    if rule.needs_capacity:
        _require_any(capacity, "capacity_columns", "capacity field", required_fields, warnings, reasons)
        can_run = can_run and bool(capacity)
    if rule.needs_relation:
        has_relation = has_edge_table or len(relation) >= 2 or len(id_like) >= 2
        if has_relation:
            reasons.append("has edge/node relation fields")
            if has_edge_table:
                reasons.append("has source-target edge table")
        else:
            can_run = False
            required_fields.append("relation_columns:2")
            warnings.append("needs edge, node, source/target, or id relation fields")
    if rule.needs_objective_or_constraint:
        has_objective = has_objective_constraint_combo or bool(cost or benefit or constraint or resource or capacity or objective_constraint)
        if has_objective:
            reasons.append("has objective or constraint fields")
            if has_objective_constraint_combo:
                reasons.append("has objective-resource constraint combination")
        else:
            can_run = False
            required_fields.append("objective_or_constraint_fields")
            warnings.append("needs cost, benefit, capacity, resource, or constraint fields")

    if rule.validation:
        reasons.append(f"validation: {rule.validation}")

    if rule.max_rows_preferred and rows > rule.max_rows_preferred:
        warnings.append(f"row count {rows} exceeds preferred max {rule.max_rows_preferred}; model may degrade")

    return ApplicabilityResult(
        model_id=model_id,
        can_run=can_run,
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        required_fields=tuple(dict.fromkeys(required_fields)),
        role=rule.role,
        task_types=rule.task_types,
        not_recommended_when=rule.not_recommended_when,
        innovation_extensions=rule.innovation_extensions,
        tier=rule.tier,
        validation=rule.validation,
    )


def _default_applicability(model_id: str, profile: Any, task_type: str | None) -> ApplicabilityResult:
    numeric = _profile_tuple(profile, "numeric_columns")
    columns = _profile_tuple(profile, "columns")
    if not columns:
        return ApplicabilityResult(
            model_id=model_id,
            can_run=False,
            reasons=(),
            warnings=("no structured data profile is available",),
            required_fields=("columns",),
            task_types=(task_type,) if task_type else (),
        )
    if numeric:
        return ApplicabilityResult(
            model_id=model_id,
            can_run=True,
            reasons=(f"generic runnable model has {len(numeric)} numeric columns",),
            warnings=("no dedicated applicability rule is registered",),
            required_fields=(),
            task_types=(task_type,) if task_type else (),
        )
    return ApplicabilityResult(
        model_id=model_id,
        can_run=False,
        reasons=(),
        warnings=("generic model needs at least one numeric column", "no dedicated applicability rule is registered"),
        required_fields=("numeric_columns:1",),
        task_types=(task_type,) if task_type else (),
    )


def _profile_value(profile: Any, key: str, default: Any = None) -> Any:
    if isinstance(profile, dict):
        return profile.get(key, default)
    return getattr(profile, key, default)


def _profile_tuple(profile: Any, key: str) -> tuple[str, ...]:
    value = _profile_value(profile, key, ())
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _has_named_column(profile: Any, names: tuple[str, ...]) -> bool:
    columns = _profile_tuple(profile, "columns")
    lowered = tuple(column.lower() for column in columns)
    return any(any(name in column for name in names) for column in lowered)


def _require_any(
    values: tuple[str, ...],
    field_name: str,
    label: str,
    required_fields: list[str],
    warnings: list[str],
    reasons: list[str],
) -> None:
    if values:
        reasons.append(f"has {label}")
        return
    required_fields.append(field_name)
    warnings.append(f"needs {label}")
