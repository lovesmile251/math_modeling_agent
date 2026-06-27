"""校园社交网络分析求解器。

输入真实的“好友关系表”（边列表，至少包含两列用户 ID，可选时间列），
围绕四个核心问题输出真实的结果表与图表：

Q1 社群发现：贪心模块度 / Louvain 社区划分 + 内部连接密度排序 + 社群间关系强度与重叠分析。
Q2 好友推荐：基于共同邻居、Jaccard、Adamic-Adar、资源分配指数的链路预测，为目标用户推荐 Top-3。
Q3 信息传播：独立级联（IC）模型 + 多中心性关键用户筛选 + 48 小时传播过程仿真。
Q4 推送优化：贪心影响力最大化，在每日推送名额约束下最大化 48 小时传播范围，并与基准策略对比。

所有计算均在传入的真实数据上完成；当行为/属性数据缺失时，采用基于网络结构的代理指标，
并在结果表的 method/assumption 字段中明确标注，绝不编造观测值。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import font_manager as _fm


# --------------------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------------------
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

_EDGE_KEYWORDS_A = ("source", "from", "start", "学生1", "用户1", "节点1", "u", "user1")
_EDGE_KEYWORDS_B = ("target", "to", "end", "学生2", "用户2", "节点2", "v", "user2")

_SOCIAL_KEYWORDS = (
    "社交网络",
    "社交平台",
    "好友关系",
    "好友推荐",
    "社群",
    "社区发现",
    "关系网络",
    "信息传播",
    "影响力",
    "关键用户",
    "转发",
    "推送",
    "social network",
    "friend",
    "community",
)


def is_social_network_problem(problem_text: str, columns: list[str]) -> bool:
    """判断是否为社交网络类题目：题面命中社交关键词，且数据为边列表（>=2 个类别列）。"""
    text = (problem_text or "").lower()
    keyword_hits = sum(1 for kw in _SOCIAL_KEYWORDS if kw.lower() in text)
    has_edge_like = _columns_look_like_edge_list(columns)
    return keyword_hits >= 2 and has_edge_like


def _columns_look_like_edge_list(columns: list[str]) -> bool:
    lowered = [str(c).lower() for c in columns]
    pair_hit = any(any(k.lower() in c for k in _EDGE_KEYWORDS_A) for c in lowered) and any(
        any(k.lower() in c for k in _EDGE_KEYWORDS_B) for c in lowered
    )
    return pair_hit or len(columns) >= 2


def build_graph(df: pd.DataFrame) -> nx.Graph | None:
    """从边列表构建无向图。优先识别明确的 source/target 列，否则使用前两个非数值列。"""
    source_col, target_col = _detect_edge_columns(df)
    if source_col is None or target_col is None:
        return None

    graph = nx.Graph()
    for _, row in df.iterrows():
        a = str(row[source_col]).strip()
        b = str(row[target_col]).strip()
        if not a or not b or a.lower() == "nan" or b.lower() == "nan" or a == b:
            continue
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += 1.0
        else:
            graph.add_edge(a, b, weight=1.0)
    if graph.number_of_edges() == 0:
        return None
    return graph


def _detect_edge_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    def find(keywords: tuple[str, ...]) -> str | None:
        for column in df.columns:
            name = str(column).lower()
            if any(keyword.lower() in name for keyword in keywords):
                return str(column)
        return None

    source_col = find(_EDGE_KEYWORDS_A)
    target_col = find(_EDGE_KEYWORDS_B)
    if source_col is not None and target_col is not None and source_col != target_col:
        return source_col, target_col

    candidates = [
        str(c)
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    return None, None


def detect_target_user(problem_text: str, graph: nx.Graph) -> str | None:
    """从题目文本中识别目标用户 ID（如 S11）；找不到则回退为度最高的节点。"""
    nodes = set(graph.nodes)
    if problem_text:
        for token in re.findall(r"[A-Za-z]+\d+", problem_text):
            if token in nodes:
                return token
    if nodes:
        return max(graph.degree, key=lambda kv: kv[1])[0]
    return None


def _save_fig(fig, path: Path) -> str:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------------------
# 问题一：社群发现与高密度社群
# --------------------------------------------------------------------------------------
def community_analysis(graph: nx.Graph, tables_dir: Path, figures_dir: Path, stem: str) -> dict[str, str]:
    outputs: dict[str, str] = {}

    try:
        communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        method = "Louvain 模块度优化"
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
        method = "贪心模块度优化"

    communities = [set(c) for c in communities]
    modularity = nx.community.modularity(graph, communities, weight="weight")

    membership = {}
    for idx, community in enumerate(communities):
        for node in community:
            membership[node] = idx

    rows = []
    for idx, community in enumerate(communities):
        subgraph = graph.subgraph(community)
        n = subgraph.number_of_nodes()
        e = subgraph.number_of_edges()
        density = (2.0 * e) / (n * (n - 1)) if n > 1 else 0.0
        degrees = dict(subgraph.degree())
        core_members = sorted(degrees, key=lambda node: degrees[node], reverse=True)[:5]
        rows.append(
            {
                "社群编号": idx,
                "成员数": n,
                "内部边数": e,
                "内部连接密度": round(density, 4),
                "平均内部度": round((2.0 * e / n) if n else 0.0, 3),
                "核心成员(度Top5)": "、".join(core_members),
                "method": method,
            }
        )

    community_table = pd.DataFrame(rows).sort_values("内部连接密度", ascending=False).reset_index(drop=True)
    community_table["modularity"] = round(float(modularity), 4)
    path = tables_dir / f"{stem}_community_detection.csv"
    community_table.to_csv(path, index=False, encoding="utf-8-sig")
    outputs["community_detection"] = str(path)

    # 取成员数>=3 的高密度社群中的前 5 个
    meaningful = community_table[community_table["成员数"] >= 3].head(5)
    if meaningful.empty:
        meaningful = community_table.head(5)
    top_ids = list(meaningful["社群编号"])

    top_path = tables_dir / f"{stem}_top5_communities.csv"
    meaningful.to_csv(top_path, index=False, encoding="utf-8-sig")
    outputs["top5_communities"] = str(top_path)

    # 社群间关系强度（边数）与成员/功能重叠（基于跨社群连接的 Jaccard 邻接重叠）
    relation_rows = []
    for i in top_ids:
        for j in top_ids:
            if j <= i:
                continue
            ci, cj = communities[i], communities[j]
            cross = sum(1 for u in ci for v in graph.neighbors(u) if v in cj)
            # 桥接节点：在 j 中有邻居的 i 节点数 + 反向
            bridge_i = sum(1 for u in ci if any(v in cj for v in graph.neighbors(u)))
            bridge_j = sum(1 for v in cj if any(u in ci for u in graph.neighbors(v)))
            possible = len(ci) * len(cj)
            strength = cross / possible if possible else 0.0
            relation_rows.append(
                {
                    "社群A": i,
                    "社群B": j,
                    "跨社群边数": cross,
                    "连接强度": round(strength, 5),
                    "A侧桥接成员数": bridge_i,
                    "B侧桥接成员数": bridge_j,
                    "重叠判定": "存在功能重叠" if strength > 0 and (bridge_i + bridge_j) >= 2 else "基本独立",
                }
            )
    if relation_rows:
        rel_path = tables_dir / f"{stem}_community_relation.csv"
        pd.DataFrame(relation_rows).to_csv(rel_path, index=False, encoding="utf-8-sig")
        outputs["community_relation"] = str(rel_path)

    # 图1：社群结构网络图
    outputs["fig_community"] = _plot_communities(graph, membership, top_ids, communities, figures_dir, stem)
    # 图2：高密度社群密度条形图
    outputs["fig_density"] = _plot_density_bar(meaningful, figures_dir, stem)
    return outputs


def _plot_communities(graph, membership, top_ids, communities, figures_dir: Path, stem: str) -> str:
    pos = nx.spring_layout(graph, seed=42, k=0.6 / math.sqrt(max(graph.number_of_nodes(), 1)))
    fig, ax = plt.subplots(figsize=(11, 9))
    num_comm = max(membership.values()) + 1 if membership else 1
    cmap = plt.cm.get_cmap("tab20", max(num_comm, 1))
    node_colors = [cmap(membership.get(node, 0)) for node in graph.nodes]
    degrees = dict(graph.degree())
    node_sizes = [30 + 12 * degrees[node] for node in graph.nodes]
    nx.draw_networkx_edges(graph, pos, alpha=0.15, width=0.5, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    # 标注 5 大社群的核心节点
    top_set = set(top_ids)
    labels = {}
    for idx in top_ids:
        members = sorted(communities[idx], key=lambda node: degrees[node], reverse=True)[:2]
        for node in members:
            labels[node] = node
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
    ax.set_title(f"图1 好友关系网络社群结构（共 {num_comm} 个社群，高亮标注 5 大高密度社群核心成员）", fontsize=13)
    ax.axis("off")
    path = figures_dir / f"{stem}_community_structure.png"
    return _save_fig(fig, path)


def _plot_density_bar(meaningful: pd.DataFrame, figures_dir: Path, stem: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"社群{int(r)}\n(n={int(s)})" for r, s in zip(meaningful["社群编号"], meaningful["成员数"])]
    ax.bar(labels, meaningful["内部连接密度"], color="#4C72B0", edgecolor="white")
    for x, value in enumerate(meaningful["内部连接密度"]):
        ax.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("内部连接密度 ρ")
    ax.set_title("图2 内部连接密度最大的 5 个社群", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    path = figures_dir / f"{stem}_community_density.png"
    return _save_fig(fig, path)


# --------------------------------------------------------------------------------------
# 问题二：基于链路预测的好友推荐
# --------------------------------------------------------------------------------------
def friend_recommendation(
    graph: nx.Graph, target: str, tables_dir: Path, figures_dir: Path, stem: str
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if target not in graph:
        return outputs

    neighbors_target = set(graph.neighbors(target))
    candidates = [n for n in graph.nodes if n != target and n not in neighbors_target]

    rows = []
    for v in candidates:
        nv = set(graph.neighbors(v))
        common = neighbors_target & nv
        cn = len(common)
        union = len(neighbors_target | nv)
        jaccard = cn / union if union else 0.0
        aa = sum(1.0 / math.log(graph.degree(z)) for z in common if graph.degree(z) > 1)
        ra = sum(1.0 / graph.degree(z) for z in common if graph.degree(z) > 0)
        pa = graph.degree(target) * graph.degree(v)
        rows.append(
            {
                "候选用户": v,
                "共同好友数": cn,
                "Jaccard系数": round(jaccard, 4),
                "AdamicAdar指数": round(aa, 4),
                "资源分配指数": round(ra, 4),
                "优先连接指数": pa,
                "候选用户度": graph.degree(v),
            }
        )

    if not rows:
        return outputs

    table = pd.DataFrame(rows)
    # 综合得分：AA 为主，Jaccard 辅助（归一化）
    for col, weight in (("AdamicAdar指数", 0.5), ("资源分配指数", 0.3), ("Jaccard系数", 0.2)):
        max_value = table[col].max()
        table[f"_norm_{col}"] = table[col] / max_value if max_value > 0 else 0.0
    table["综合推荐得分"] = (
        0.5 * table["_norm_AdamicAdar指数"]
        + 0.3 * table["_norm_资源分配指数"]
        + 0.2 * table["_norm_Jaccard系数"]
    ).round(4)
    table = table.drop(columns=[c for c in table.columns if c.startswith("_norm_")])
    table = table.sort_values("综合推荐得分", ascending=False).reset_index(drop=True)
    table.insert(0, "目标用户", target)

    full_path = tables_dir / f"{stem}_friend_recommendation.csv"
    table.head(20).to_csv(full_path, index=False, encoding="utf-8-sig")
    outputs["friend_recommendation"] = str(full_path)

    # Top-3 + 未成为好友原因
    top3 = table.head(3).copy()
    reasons = []
    distances = {}
    for v in top3["候选用户"]:
        try:
            distances[v] = nx.shortest_path_length(graph, target, v)
        except nx.NetworkXNoPath:
            distances[v] = math.inf
    for _, r in top3.iterrows():
        v = r["候选用户"]
        d = distances.get(v, math.inf)
        reason = (
            f"与 {target} 有 {int(r['共同好友数'])} 位共同好友、Adamic-Adar={r['AdamicAdar指数']}；"
            f"二者最短路径距离为 {d if d != math.inf else '不连通'}，"
            f"目前未连边，推测因{'同属密集子群但尚未直接互加' if d == 2 else '跨子群弱关系尚未触达'}。"
        )
        reasons.append({"目标用户": target, "推荐用户": v, "综合推荐得分": r["综合推荐得分"], "推荐理由与未成好友原因": reason})
    reason_path = tables_dir / f"{stem}_recommendation_reason.csv"
    pd.DataFrame(reasons).to_csv(reason_path, index=False, encoding="utf-8-sig")
    outputs["recommendation_reason"] = str(reason_path)

    # 全网络好友特性 + 网络结构指标
    outputs["network_properties"] = _network_properties(graph, tables_dir, stem)

    # 图：目标用户自我网络 + 推荐高亮
    outputs["fig_ego"] = _plot_ego_network(graph, target, list(top3["候选用户"]), figures_dir, stem)
    return outputs


def _network_properties(graph: nx.Graph, tables_dir: Path, stem: str) -> str:
    components = list(nx.connected_components(graph))
    giant = max(components, key=len)
    giant_graph = graph.subgraph(giant)
    degrees = [d for _, d in graph.degree()]
    try:
        assortativity = nx.degree_assortativity_coefficient(graph)
    except Exception:
        assortativity = float("nan")
    rows = [
        {"指标": "节点数", "数值": graph.number_of_nodes()},
        {"指标": "边数", "数值": graph.number_of_edges()},
        {"指标": "平均度", "数值": round(float(np.mean(degrees)), 3)},
        {"指标": "最大度", "数值": int(np.max(degrees))},
        {"指标": "网络密度", "数值": round(nx.density(graph), 5)},
        {"指标": "平均聚类系数", "数值": round(nx.average_clustering(graph), 4)},
        {"指标": "连通分量数", "数值": len(components)},
        {"指标": "最大连通分量规模", "数值": len(giant)},
        {"指标": "最大连通分量直径", "数值": nx.diameter(giant_graph) if len(giant) > 1 else 0},
        {"指标": "度同配系数", "数值": round(float(assortativity), 4) if assortativity == assortativity else "—"},
    ]
    path = tables_dir / f"{stem}_network_properties.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _plot_ego_network(graph, target, recommended, figures_dir: Path, stem: str) -> str:
    nodes = {target}
    nodes |= set(graph.neighbors(target))
    for v in recommended:
        nodes.add(v)
        nodes |= set(graph.neighbors(v))
    ego = graph.subgraph(nodes)
    pos = nx.spring_layout(ego, seed=7, k=0.5)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(ego, pos, alpha=0.25, width=0.6, ax=ax)
    colors, sizes = [], []
    for node in ego.nodes:
        if node == target:
            colors.append("#C44E52")
            sizes.append(420)
        elif node in recommended:
            colors.append("#55A868")
            sizes.append(320)
        elif node in set(graph.neighbors(target)):
            colors.append("#4C72B0")
            sizes.append(140)
        else:
            colors.append("#CCCCCC")
            sizes.append(70)
    nx.draw_networkx_nodes(ego, pos, node_color=colors, node_size=sizes, ax=ax)
    labels = {target: target}
    for v in recommended:
        labels[v] = v
    nx.draw_networkx_labels(ego, pos, labels=labels, font_size=10, ax=ax)
    ax.set_title(
        f"图3 用户 {target} 的自我网络与 Top-3 推荐（红=目标，绿=推荐，蓝=现有好友）", fontsize=12
    )
    ax.axis("off")
    path = figures_dir / f"{stem}_ego_network.png"
    return _save_fig(fig, path)


# --------------------------------------------------------------------------------------
# 问题三：信息传播（独立级联）与关键用户筛选
# --------------------------------------------------------------------------------------
def _activity_score(graph: nx.Graph) -> dict[str, float]:
    """行为数据表缺失时，用归一化度作为活跃度 (α+β) 的结构代理。"""
    degrees = dict(graph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    return {node: degrees[node] / max_deg for node in graph.nodes}


# 传播模型参数（经典独立级联 IC）：
# 每个已转发用户对每位好友仅有一次以概率 p 触发其转发的机会，p 随活跃度上升、随时间衰减。
# 行为数据表缺失，故活跃度 (α+β) 用归一化度代理；p0 为基础转发强度，控制传播规模不饱和。
_P0 = 0.085
_GAMMA = 0.03
_HOP_DELAY = 5.0  # 平均阅读-转发延迟（小时），反映用户按活跃时段（上午/下午/晚上）才看到帖子


def _forward_prob(act: float, t_hours: float, p0: float = _P0, gamma: float = _GAMMA) -> float:
    """单次曝光的转发概率，形式对齐题目公式 P=σ(α+β-γT) 的衰减结构。

    act 为归一化活跃度代理 (α+β 的结构替代)，t_hours 为距发帖小时数。
    """
    return p0 * (0.3 + 0.7 * act) * math.exp(-gamma * t_hours)


def _ic_times(graph, seeds, activity, horizon, hop_delay, rng):
    """经典独立级联仿真，返回 (seen_time, forwarders)。

    seen 表示看到帖子的用户（即传播范围），forwarders 表示发生转发的用户。
    """
    import heapq

    seen_time: dict[str, float] = {s: 0.0 for s in seeds}
    forwarders = set(seeds)
    queue: list[tuple[float, str]] = [(0.0, s) for s in seeds]
    heapq.heapify(queue)
    while queue:
        t, u = heapq.heappop(queue)
        if t > horizon:
            continue
        for v in graph.neighbors(u):
            if v not in seen_time or seen_time[v] > t:
                seen_time.setdefault(v, t)
            if v in forwarders:
                continue
            arrival = t + hop_delay
            if arrival > horizon:
                continue
            if rng.random() < _forward_prob(activity[v], arrival):
                forwarders.add(v)
                seen_time[v] = min(seen_time.get(v, arrival), arrival)
                heapq.heappush(queue, (arrival, v))
    return seen_time, forwarders


def _cascade_curve(graph, seeds, activity, horizon=48, hop_delay=_HOP_DELAY, runs=80):
    """多次仿真的逐小时平均累计触达曲线。"""
    rng = np.random.default_rng(2026)
    hourly = np.zeros(horizon + 1)
    final_reach = []
    for _ in range(runs):
        seen_time, _ = _ic_times(graph, set(seeds), activity, horizon, hop_delay, rng)
        final_reach.append(len(seen_time))
        times_sorted = sorted(seen_time.values())
        cumulative = 0
        ptr = 0
        for hour in range(horizon + 1):
            while ptr < len(times_sorted) and times_sorted[ptr] <= hour:
                cumulative += 1
                ptr += 1
            hourly[hour] += cumulative
    hourly /= runs
    return hourly, float(np.mean(final_reach)), float(np.std(final_reach))


def _avg_reach(graph, seeds, activity, runs=60, horizon=48, hop_delay=_HOP_DELAY, rng=None):
    if rng is None:
        rng = np.random.default_rng(7)
    total = 0
    for _ in range(runs):
        seen_time, _ = _ic_times(graph, set(seeds), activity, horizon, hop_delay, rng)
        total += len(seen_time)
    return total / runs


def information_propagation(
    graph: nx.Graph, tables_dir: Path, figures_dir: Path, stem: str
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    activity = _activity_score(graph)

    degree_c = nx.degree_centrality(graph)
    pagerank = nx.pagerank(graph, weight="weight")
    core = nx.core_number(graph)
    try:
        betweenness = nx.betweenness_centrality(graph, weight=None, k=min(graph.number_of_nodes(), 100), seed=1)
    except Exception:
        betweenness = {n: 0.0 for n in graph.nodes}

    # 候选关键用户：综合中心性 Top12
    composite = {
        n: degree_c[n] + pagerank[n] * 5 + betweenness[n] + core[n] / max(core.values())
        for n in graph.nodes
    }
    candidates = sorted(composite, key=lambda n: composite[n], reverse=True)[:12]

    rng = np.random.default_rng(2026)
    rows = []
    for node in candidates:
        reach = _avg_reach(graph, {node}, activity, runs=60, rng=rng)
        rows.append(
            {
                "候选用户": node,
                "度中心性": round(degree_c[node], 4),
                "PageRank": round(pagerank[node], 5),
                "介数中心性": round(betweenness[node], 4),
                "k核": core[node],
                "活跃度代理": round(activity[node], 4),
                "平均传播范围(人)": round(reach, 2),
            }
        )
    cand_table = pd.DataFrame(rows).sort_values("平均传播范围(人)", ascending=False).reset_index(drop=True)
    cand_path = tables_dir / f"{stem}_key_user_candidates.csv"
    cand_table.to_csv(cand_path, index=False, encoding="utf-8-sig")
    outputs["key_user_candidates"] = str(cand_path)

    key_user = cand_table.iloc[0]["候选用户"]

    # 关键用户 48 小时传播曲线
    hourly, mean_reach, std_reach = _cascade_curve(graph, {key_user}, activity, runs=80)
    curve_rows = [
        {"小时": h, "累计触达人数(均值)": round(float(hourly[h]), 2)} for h in range(len(hourly))
    ]
    curve_path = tables_dir / f"{stem}_propagation_curve.csv"
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False, encoding="utf-8-sig")
    outputs["propagation_curve"] = str(curve_path)

    summary_path = tables_dir / f"{stem}_key_user_summary.csv"
    pd.DataFrame(
        [
            {
                "关键用户": key_user,
                "发帖时刻": "12:00",
                "仿真时长(小时)": 48,
                "48h平均传播范围(人)": round(mean_reach, 2),
                "传播范围标准差": round(std_reach, 2),
                "传播模型": "独立级联(IC)",
                "转发概率": "P=p0·(0.3+0.7·act)·e^(-γT)，act为度归一化活跃度代理(行为数据表缺失)，p0=0.085，γ=0.03",
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    outputs["key_user_summary"] = str(summary_path)

    outputs["fig_centrality"] = _plot_centrality(cand_table, figures_dir, stem)
    outputs["fig_propagation"] = _plot_propagation_curve(hourly, key_user, figures_dir, stem)
    return outputs, key_user, activity


def _plot_centrality(cand_table: pd.DataFrame, figures_dir: Path, stem: str) -> str:
    top = cand_table.head(10)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["候选用户"][::-1], top["平均传播范围(人)"][::-1], color="#DD8452", edgecolor="white")
    for y, value in enumerate(top["平均传播范围(人)"][::-1]):
        ax.text(value, y, f" {value:.1f}", va="center", fontsize=9)
    ax.set_xlabel("48h 平均传播范围（人）")
    ax.set_title("图4 关键用户候选的传播影响力排序", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    path = figures_dir / f"{stem}_key_user_influence.png"
    return _save_fig(fig, path)


def _plot_propagation_curve(hourly, key_user, figures_dir: Path, stem: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(len(hourly)), hourly, marker="o", markersize=3, color="#4C72B0", linewidth=2)
    ax.fill_between(range(len(hourly)), hourly, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("发帖后小时数")
    ax.set_ylabel("累计触达人数（均值）")
    ax.set_title(f"图5 关键用户 {key_user} 正午发帖后 48 小时信息传播曲线", fontsize=12)
    ax.grid(alpha=0.3)
    path = figures_dir / f"{stem}_propagation_curve.png"
    return _save_fig(fig, path)


# --------------------------------------------------------------------------------------
# 问题四：推送名额优化（贪心影响力最大化）
# --------------------------------------------------------------------------------------
def push_optimization(
    graph: nx.Graph, activity: dict[str, float], tables_dir: Path, figures_dir: Path, stem: str,
    daily_quota: int = 10, days: int = 2,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    total_quota = daily_quota * days
    rng = np.random.default_rng(2026)

    # 贪心：每一步选边际传播增益最大的用户
    selected: list[str] = []
    schedule_rows = []
    candidate_pool = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)[: min(40, graph.number_of_nodes())]
    prev_reach = 0.0
    for step in range(total_quota):
        best_node, best_gain, best_reach = None, -1.0, prev_reach
        for node in candidate_pool:
            if node in selected:
                continue
            reach = _avg_reach(graph, set(selected) | {node}, activity, runs=40, rng=rng)
            gain = reach - prev_reach
            if gain > best_gain:
                best_gain, best_node, best_reach = gain, node, reach
        if best_node is None:
            break
        selected.append(best_node)
        prev_reach = best_reach
        day = step // daily_quota + 1
        batch = "上午" if (step % daily_quota) < daily_quota // 2 else "下午"
        schedule_rows.append(
            {
                "推送序号": step + 1,
                "推送用户": best_node,
                "推送日": f"第{day}天",
                "时段": batch,
                "边际传播增益(人)": round(best_gain, 3),
                "累计平均传播范围(人)": round(best_reach, 2),
            }
        )

    schedule_path = tables_dir / f"{stem}_push_schedule.csv"
    pd.DataFrame(schedule_rows).to_csv(schedule_path, index=False, encoding="utf-8-sig")
    outputs["push_schedule"] = str(schedule_path)

    # ── 辅助：多次仿真返回 (均值, 标准差) ──
    def _avg_reach_with_std(seeds, runs=200):
        vals = []
        rng_local = np.random.default_rng(42)
        for _ in range(runs):
            seen, _ = _ic_times(graph, set(seeds), activity, 48, _HOP_DELAY, rng_local)
            vals.append(len(seen))
        return float(np.mean(vals)), float(np.std(vals))

    # ── 构建多种基准策略的种子集 ──
    n_nodes = graph.number_of_nodes()
    pagerank = nx.pagerank(graph, weight="weight")
    try:
        betweenness = nx.betweenness_centrality(graph, weight=None, k=min(n_nodes, 100), seed=1)
    except Exception:
        betweenness = {n: 0.0 for n in graph.nodes}
    core_number = nx.core_number(graph)

    # 社群多样性策略：从 Louvain 划分的每个社群中各取一个度最高节点
    try:
        communities = list(nx.community.louvain_communities(graph, weight="weight", seed=42))
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    community_seeds: list[str] = []
    for comm in sorted(communities, key=len, reverse=True):
        top_in_comm = max(comm, key=lambda n: graph.degree(n))
        if top_in_comm not in community_seeds:
            community_seeds.append(top_in_comm)
        if len(community_seeds) >= total_quota:
            break

    # k-核 + 度组合策略
    k_core_sorted = sorted(core_number, key=lambda n: (core_number[n], graph.degree(n)), reverse=True)

    strategies_def = {
        "贪心影响力最大化(本文)": selected,
        "度中心性Top策略": [n for n, _ in sorted(graph.degree(), key=lambda kv: kv[1], reverse=True)[:total_quota]],
        "PageRank Top策略": sorted(pagerank, key=lambda n: pagerank[n], reverse=True)[:total_quota],
        "介数中心性Top策略": sorted(betweenness, key=lambda n: betweenness[n], reverse=True)[:total_quota],
        "k-核+度Top策略": k_core_sorted[:total_quota],
        "社群多样性策略": community_seeds[:total_quota],
        "随机推送策略": list(rng.choice(list(graph.nodes), size=min(total_quota, n_nodes), replace=False)),
    }

    # ── 单次 20-seed 完整对比（含标准差）──
    comparison_rows = []
    for name, seeds in strategies_def.items():
        mean_v, std_v = _avg_reach_with_std(seeds, runs=200)
        comparison_rows.append({
            "推送策略": name,
            "种子数": len(seeds),
            "48h平均传播范围(人)": round(mean_v, 2),
            "标准差(人)": round(std_v, 2),
        })
    comparison = pd.DataFrame(comparison_rows).sort_values("48h平均传播范围(人)", ascending=False).reset_index(drop=True)
    comp_path = tables_dir / f"{stem}_push_strategy_comparison.csv"
    comparison.to_csv(comp_path, index=False, encoding="utf-8-sig")
    outputs["push_strategy_comparison"] = str(comp_path)

    # ── 多规模对比（5/10/15/20 种子）──
    for seed_k in [5, 10, 15, 20]:
        multi_comparison_rows = []
        for name, seeds in strategies_def.items():
            subset = seeds[:seed_k]
            mean_v, std_v = _avg_reach_with_std(subset, runs=120)
            multi_comparison_rows.append({
                "推送策略": name,
                "种子数": len(subset),
                "48h平均传播范围(人)": round(mean_v, 2),
                "标准差(人)": round(std_v, 2),
            })
        multi_comp = pd.DataFrame(multi_comparison_rows).sort_values("48h平均传播范围(人)", ascending=False)
        multi_path = tables_dir / f"{stem}_push_comparison_k{seed_k}.csv"
        multi_comp.to_csv(multi_path, index=False, encoding="utf-8-sig")
        outputs[f"push_comparison_k{seed_k}"] = str(multi_path)

    # ── 图表 ──
    cumulative = [r["累计平均传播范围(人)"] for r in schedule_rows]
    outputs["fig_push"] = _plot_push_curve(cumulative, comparison, figures_dir, stem)
    return outputs


def _plot_push_curve(cumulative, comparison, figures_dir: Path, stem: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # Left: marginal gain curve
    axes[0].plot(range(1, len(cumulative) + 1), cumulative, marker="o", color="#55A868", linewidth=2)
    axes[0].set_xlabel("已选推送种子数")
    axes[0].set_ylabel("累计平均传播范围（人）")
    axes[0].set_title("图6(a) 贪心推送的边际收益曲线")
    axes[0].grid(alpha=0.3)

    # Right: strategy comparison bar chart with error bars (std)
    comp_plot = comparison.sort_values("48h平均传播范围(人)", ascending=True)
    names = comp_plot["推送策略"]
    values = comp_plot["48h平均传播范围(人)"]
    errors = comp_plot.get("标准差(人)", [0] * len(comp_plot))
    colors = ["#4C72B0"] * len(names)
    # Highlight the greedy strategy
    for i, name in enumerate(names):
        if "贪心" in str(name):
            colors[i] = "#DD8452"
    bars = axes[1].barh(names, values, xerr=errors, color=colors, edgecolor="white", capsize=4)
    for bar, val, err in zip(bars, values, errors):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f" {val:.1f}±{err:.1f}", va="center", fontsize=8)
    axes[1].set_xlabel("48h 平均传播范围（人）± 标准差")
    axes[1].set_title("图6(b) 不同推送策略传播范围对比（含±1σ误差线）", fontsize=11)
    axes[1].grid(axis="x", alpha=0.3)
    path = figures_dir / f"{stem}_push_optimization.png"
    return _save_fig(fig, path)


# --------------------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------------------
def campus_friend_recommendation_model(df: pd.DataFrame) -> pd.DataFrame:
    """Registry-compatible friend recommendation summary."""
    graph = build_graph(df)
    if graph is None:
        return pd.DataFrame()

    target = detect_target_user("", graph)
    if target is None or target not in graph:
        return pd.DataFrame()

    neighbors_target = set(graph.neighbors(target))
    rows = []
    for candidate in graph.nodes:
        if candidate == target or candidate in neighbors_target:
            continue
        candidate_neighbors = set(graph.neighbors(candidate))
        common = neighbors_target & candidate_neighbors
        common_count = len(common)
        union_count = len(neighbors_target | candidate_neighbors)
        jaccard = common_count / union_count if union_count else 0.0
        adamic_adar = sum(1.0 / math.log(graph.degree(node)) for node in common if graph.degree(node) > 1)
        resource_allocation = sum(1.0 / graph.degree(node) for node in common if graph.degree(node) > 0)
        rows.append(
            {
                "target_user": target,
                "recommended_user": candidate,
                "common_neighbors": common_count,
                "jaccard": round(jaccard, 6),
                "adamic_adar": round(adamic_adar, 6),
                "resource_allocation": round(resource_allocation, 6),
                "preferential_attachment": int(graph.degree(target) * graph.degree(candidate)),
                "candidate_degree": int(graph.degree(candidate)),
            }
        )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    for column in ("adamic_adar", "resource_allocation", "jaccard"):
        maximum = table[column].max()
        table[f"_norm_{column}"] = table[column] / maximum if maximum > 0 else 0.0
    table["recommendation_score"] = (
        0.5 * table["_norm_adamic_adar"]
        + 0.3 * table["_norm_resource_allocation"]
        + 0.2 * table["_norm_jaccard"]
    ).round(6)
    return (
        table.drop(columns=[column for column in table.columns if column.startswith("_norm_")])
        .sort_values("recommendation_score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )


def campus_information_propagation_model(df: pd.DataFrame) -> pd.DataFrame:
    """Registry-compatible information propagation and key-user summary."""
    graph = build_graph(df)
    if graph is None or graph.number_of_nodes() == 0:
        return pd.DataFrame()

    activity = _activity_score(graph)
    degree_centrality = nx.degree_centrality(graph)
    pagerank = nx.pagerank(graph, weight="weight")
    core = nx.core_number(graph)
    max_core = max(core.values()) if core else 1
    try:
        betweenness = nx.betweenness_centrality(graph, weight=None, k=min(graph.number_of_nodes(), 100), seed=1)
    except Exception:
        betweenness = {node: 0.0 for node in graph.nodes}

    composite = {
        node: degree_centrality[node]
        + pagerank[node] * 5
        + betweenness[node]
        + (core[node] / max_core if max_core else 0.0)
        for node in graph.nodes
    }
    rng = np.random.default_rng(2026)
    rows = []
    for node in sorted(composite, key=lambda item: composite[item], reverse=True)[:12]:
        rows.append(
            {
                "candidate_user": node,
                "degree_centrality": round(degree_centrality[node], 6),
                "pagerank": round(pagerank[node], 6),
                "betweenness": round(betweenness[node], 6),
                "core_number": int(core[node]),
                "activity_proxy": round(activity[node], 6),
                "estimated_48h_reach": round(_avg_reach(graph, {node}, activity, runs=20, rng=rng), 3),
            }
        )
    return pd.DataFrame(rows)


def campus_influence_maximization_model(df: pd.DataFrame) -> pd.DataFrame:
    """Registry-compatible greedy seed selection summary."""
    graph = build_graph(df)
    if graph is None or graph.number_of_nodes() == 0:
        return pd.DataFrame()

    activity = _activity_score(graph)
    rng = np.random.default_rng(2026)
    quota = min(10, graph.number_of_nodes())
    pool = sorted(graph.nodes, key=lambda node: graph.degree(node), reverse=True)[: min(40, graph.number_of_nodes())]
    selected: list[str] = []
    previous_reach = 0.0
    rows = []
    for step in range(quota):
        best_node = None
        best_gain = -1.0
        best_reach = previous_reach
        for node in pool:
            if node in selected:
                continue
            reach = _avg_reach(graph, set(selected) | {node}, activity, runs=15, rng=rng)
            gain = reach - previous_reach
            if gain > best_gain:
                best_node = node
                best_gain = gain
                best_reach = reach
        if best_node is None:
            break
        selected.append(best_node)
        rows.append(
            {
                "selection_order": step + 1,
                "seed_user": best_node,
                "marginal_reach_gain": round(best_gain, 3),
                "cumulative_estimated_reach": round(best_reach, 3),
                "seed_degree": int(graph.degree(best_node)),
            }
        )
        previous_reach = best_reach
    return pd.DataFrame(rows)


def run_campus_social_analysis(
    df: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    stem: str,
    problem_text: str = "",
) -> dict[str, str]:
    """运行四个核心模型，返回 {输出名: 文件路径}（含表格与图表）。"""
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph(df)
    if graph is None:
        return {}

    outputs: dict[str, str] = {}
    outputs.update(community_analysis(graph, tables_dir, figures_dir, stem))

    target = detect_target_user(problem_text, graph)
    if target:
        outputs.update(friend_recommendation(graph, target, tables_dir, figures_dir, stem))

    prop_outputs, key_user, activity = information_propagation(graph, tables_dir, figures_dir, stem)
    outputs.update(prop_outputs)
    if "key_user_candidates" in prop_outputs:
        outputs["information_propagation"] = prop_outputs["key_user_candidates"]

    push_outputs = push_optimization(graph, activity, tables_dir, figures_dir, stem)
    outputs.update(push_outputs)
    if "push_strategy_comparison" in push_outputs:
        outputs["influence_maximization"] = push_outputs["push_strategy_comparison"]
    return outputs
