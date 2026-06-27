from __future__ import annotations

from collections import defaultdict, deque
import math

import pandas as pd


def graph_shortest_paths(df: pd.DataFrame) -> pd.DataFrame:
    edge_data = _extract_edges(df)
    if edge_data.empty:
        return pd.DataFrame()
    nodes = sorted(set(edge_data["source"]) | set(edge_data["target"]))
    distances = _floyd_warshall(nodes, edge_data)
    rows = []
    for source in nodes:
        for target in nodes:
            if source != target and math.isfinite(distances[source][target]):
                rows.append({"source": source, "target": target, "shortest_distance": distances[source][target]})
    return pd.DataFrame(rows)


def graph_mst(df: pd.DataFrame) -> pd.DataFrame:
    edge_data = _extract_edges(df)
    if edge_data.empty:
        return pd.DataFrame()
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> bool:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return False
        parent[root_right] = root_left
        return True

    rows = []
    for _, row in edge_data.sort_values("weight").iterrows():
        if union(row["source"], row["target"]):
            rows.append({"source": row["source"], "target": row["target"], "weight": float(row["weight"])})
    return pd.DataFrame(rows)


def graph_max_flow(df: pd.DataFrame) -> pd.DataFrame:
    edge_data = _extract_edges(df, capacity_mode=True)
    if edge_data.empty:
        return pd.DataFrame()
    nodes = sorted(set(edge_data["source"]) | set(edge_data["target"]))
    source = _find_named_node(nodes, ("source", "start", "s", "源")) or nodes[0]
    sink = _find_named_node(nodes, ("sink", "target", "end", "t", "汇")) or nodes[-1]
    if source == sink:
        return pd.DataFrame()

    capacity: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, row in edge_data.iterrows():
        capacity[row["source"]][row["target"]] += float(row["weight"])
        capacity[row["target"]].setdefault(row["source"], 0.0)

    max_flow = 0.0
    while True:
        parent = _bfs_augmenting_path(capacity, source, sink)
        if sink not in parent:
            break
        path_flow = float("inf")
        node = sink
        while node != source:
            prev = parent[node]
            path_flow = min(path_flow, capacity[prev][node])
            node = prev
        node = sink
        while node != source:
            prev = parent[node]
            capacity[prev][node] -= path_flow
            capacity[node][prev] += path_flow
            node = prev
        max_flow += path_flow
    return pd.DataFrame([{"source": source, "sink": sink, "max_flow": max_flow, "method": "edmonds_karp"}])


def graph_centrality(df: pd.DataFrame) -> pd.DataFrame:
    edge_data = _extract_edges(df)
    if edge_data.empty:
        return pd.DataFrame()
    nodes = sorted(set(edge_data["source"]) | set(edge_data["target"]))
    adjacency = {node: set() for node in nodes}
    for _, row in edge_data.iterrows():
        adjacency[row["source"]].add(row["target"])
        adjacency[row["target"]].add(row["source"])
    n = max(len(nodes) - 1, 1)
    rows = []
    for node in nodes:
        rows.append({"node": node, "degree": len(adjacency[node]), "degree_centrality": len(adjacency[node]) / n})
    return pd.DataFrame(rows).sort_values("degree_centrality", ascending=False).reset_index(drop=True)


def community_detection(df: pd.DataFrame) -> pd.DataFrame:
    edge_data = _extract_edges(df)
    if edge_data.empty:
        return pd.DataFrame()
    nodes = sorted(set(edge_data["source"]) | set(edge_data["target"]))
    n = len(nodes)
    if n < 2 or n > 300:
        return pd.DataFrame()

    index = {node: position for position, node in enumerate(nodes)}
    adjacency = [[0.0] * n for _ in range(n)]
    for _, row in edge_data.iterrows():
        i = index[row["source"]]
        j = index[row["target"]]
        if i == j:
            continue
        weight = float(row["weight"])
        adjacency[i][j] += weight
        adjacency[j][i] += weight

    degree = [sum(adjacency[i]) for i in range(n)]
    m2 = float(sum(degree))
    if m2 <= 0:
        return pd.DataFrame()

    community_of = list(range(n))
    members: dict[int, set[int]] = {i: {i} for i in range(n)}
    tot: dict[int, float] = {i: float(degree[i]) for i in range(n)}
    links: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            cross = adjacency[i][j] + adjacency[j][i]
            if cross > 0:
                links[i][j] = cross
                links[j][i] = cross

    while True:
        best_gain = 1e-12
        best_pair: tuple[int, int] | None = None
        for a, neighbors in links.items():
            for b, weight in neighbors.items():
                if b <= a:
                    continue
                gain = 2.0 * (weight / m2 - (tot[a] * tot[b]) / (m2 * m2))
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (a, b)
        if best_pair is None:
            break

        a, b = best_pair
        members[a] |= members[b]
        tot[a] += tot[b]
        for c, weight in links[b].items():
            if c == a:
                continue
            links[a][c] = links[a].get(c, 0.0) + weight
            links[c][a] = links[c].get(a, 0.0) + weight
            links[c].pop(b, None)
        links[a].pop(b, None)
        links.pop(b, None)
        for node in members[b]:
            community_of[node] = a
        del members[b]
        del tot[b]

    labels = [0] * n
    for label, community in enumerate(sorted(members)):
        for node in members[community]:
            labels[node] = label

    modularity = 0.0
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                modularity += adjacency[i][j] - degree[i] * degree[j] / m2
    modularity = modularity / m2
    community_sizes = {label: sum(1 for value in labels if value == label) for label in set(labels)}

    rows = []
    for node in nodes:
        label = labels[index[node]]
        rows.append(
            {
                "node": node,
                "community": int(label),
                "community_size": int(community_sizes[label]),
                "num_communities": int(len(community_sizes)),
                "modularity": float(modularity),
                "method": "greedy_modularity_community_detection",
            }
        )
    return pd.DataFrame(rows).sort_values(["community", "node"]).reset_index(drop=True)


def _extract_edges(df: pd.DataFrame, capacity_mode: bool = False) -> pd.DataFrame:
    source_col = _find_column(df, ("source", "from", "start", "origin", "起点", "源"))
    target_col = _find_column(df, ("target", "to", "end", "dest", "destination", "终点", "汇"))
    if source_col is None or target_col is None:
        fallback_source, fallback_target = _fallback_edge_columns(df)
        source_col = source_col or fallback_source
        target_col = target_col or fallback_target
        if source_col is not None and source_col == target_col:
            target_col = fallback_target if fallback_target != source_col else None
    if source_col is None or target_col is None:
        return pd.DataFrame()
    weight_col = _find_column(df, ("capacity", "cap", "流量", "容量")) if capacity_mode else None
    weight_col = weight_col or _find_column(df, ("weight", "distance", "cost", "length", "time", "权重", "距离", "成本", "长度", "时间"))
    result = pd.DataFrame({"source": df[source_col].astype(str), "target": df[target_col].astype(str)})
    result["weight"] = pd.to_numeric(df[weight_col], errors="coerce") if weight_col else 1.0
    result = result.dropna()
    result = result[result["weight"] > 0]
    return result


def _fallback_edge_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """当没有显式的 source/target 列时，把前两个类别型（非数值、非日期）列视为边的两端。

    适用于「学生1, 学生2, 建立时间」这类社交网络边列表数据。
    """
    candidates: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        candidates.append(str(column))
        if len(candidates) == 2:
            break
    if len(candidates) < 2:
        return None, None
    return candidates[0], candidates[1]


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    return None


def _floyd_warshall(nodes: list[str], edges: pd.DataFrame) -> dict[str, dict[str, float]]:
    distances = {left: {right: float("inf") for right in nodes} for left in nodes}
    for node in nodes:
        distances[node][node] = 0.0
    for _, row in edges.iterrows():
        left, right, weight = row["source"], row["target"], float(row["weight"])
        distances[left][right] = min(distances[left][right], weight)
        distances[right][left] = min(distances[right][left], weight)
    for mid in nodes:
        for left in nodes:
            for right in nodes:
                candidate = distances[left][mid] + distances[mid][right]
                if candidate < distances[left][right]:
                    distances[left][right] = candidate
    return distances


def _bfs_augmenting_path(capacity: dict[str, dict[str, float]], source: str, sink: str) -> dict[str, str]:
    parent: dict[str, str] = {}
    visited = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for next_node, residual in capacity[node].items():
            if next_node not in visited and residual > 1e-12:
                visited.add(next_node)
                parent[next_node] = node
                if next_node == sink:
                    return parent
                queue.append(next_node)
    return parent


def _find_named_node(nodes: list[str], keywords: tuple[str, ...]) -> str | None:
    for node in nodes:
        name = node.lower()
        if any(keyword == name or keyword in name for keyword in keywords):
            return node
    return None
