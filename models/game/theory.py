from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pandas as pd


def nash_equilibrium_2x2(df: pd.DataFrame) -> pd.DataFrame:
    game = _payoff_matrices(df)
    if game is None:
        return pd.DataFrame()
    row_actions, col_actions, payoff1, payoff2 = game

    rows: list[dict[str, float | str]] = []
    for i, row_action in enumerate(row_actions):
        for j, col_action in enumerate(col_actions):
            p1_best = payoff1[i, j] >= payoff1[:, j].max() - 1e-12
            p2_best = payoff2[i, j] >= payoff2[i, :].max() - 1e-12
            if p1_best and p2_best:
                rows.append(
                    {
                        "equilibrium_type": "pure",
                        "player1_strategy": row_action,
                        "player2_strategy": col_action,
                        "player1_mixed_probability_first": 1.0 if i == 0 else 0.0,
                        "player2_mixed_probability_first": 1.0 if j == 0 else 0.0,
                        "player1_payoff": float(payoff1[i, j]),
                        "player2_payoff": float(payoff2[i, j]),
                        "method": "nash_equilibrium_2x2",
                    }
                )

    mixed = _mixed_equilibrium(row_actions, col_actions, payoff1, payoff2)
    if mixed is not None:
        rows.append(mixed)
    return pd.DataFrame(rows)


def stackelberg_equilibrium(df: pd.DataFrame) -> pd.DataFrame:
    game = _payoff_matrices(df)
    if game is None:
        return pd.DataFrame()
    row_actions, col_actions, payoff1, payoff2 = game

    rows: list[dict[str, float | str | int]] = []
    leader_best_index = 0
    leader_best_payoff = -np.inf
    follower_responses: list[int] = []
    for i, leader_action in enumerate(row_actions):
        follower_index = int(np.argmax(payoff2[i, :]))
        follower_responses.append(follower_index)
        leader_payoff = float(payoff1[i, follower_index])
        if leader_payoff > leader_best_payoff:
            leader_best_payoff = leader_payoff
            leader_best_index = i

    for i, leader_action in enumerate(row_actions):
        follower_index = follower_responses[i]
        rows.append(
            {
                "leader_action": str(leader_action),
                "follower_best_response": str(col_actions[follower_index]),
                "leader_payoff": float(payoff1[i, follower_index]),
                "follower_payoff": float(payoff2[i, follower_index]),
                "is_subgame_perfect": int(i == leader_best_index),
                "method": "stackelberg_backward_induction",
            }
        )
    return pd.DataFrame(rows)


def shapley_value(df: pd.DataFrame) -> pd.DataFrame:
    coalition_col = _find_column(df, ("coalition", "players", "set", "group"))
    value_col = _find_column(df, ("value", "worth", "payoff", "profit", "utility"))
    if coalition_col is None or value_col is None:
        return pd.DataFrame()

    values: dict[frozenset[str], float] = {frozenset(): 0.0}
    players: set[str] = set()
    for _, row in df.iterrows():
        coalition = frozenset(_parse_coalition(row.get(coalition_col)))
        value = _to_float(row.get(value_col), np.nan)
        if not coalition or pd.isna(value):
            continue
        values[coalition] = float(value)
        players.update(coalition)
    if not players or len(players) > 10:
        return pd.DataFrame()

    ordered_players = sorted(players)
    n = len(ordered_players)
    grand_coalition = frozenset(ordered_players)
    grand_value = values.get(grand_coalition, 0.0)
    factorial_n = math.factorial(n)

    rows = []
    for player in ordered_players:
        others = [item for item in ordered_players if item != player]
        shapley = 0.0
        for size in range(n):
            for subset in itertools.combinations(others, size):
                coalition = frozenset(subset)
                with_player = frozenset((*subset, player))
                weight = math.factorial(size) * math.factorial(n - size - 1) / factorial_n
                shapley += weight * (values.get(with_player, 0.0) - values.get(coalition, 0.0))
        rows.append(
            {
                "player": player,
                "shapley_value": float(shapley),
                "grand_coalition_value": float(grand_value),
                "players": n,
                "method": "shapley_value",
            }
        )
    return pd.DataFrame(rows)


def auction_pricing(df: pd.DataFrame) -> pd.DataFrame:
    bid_col = _find_numeric_column(df, ("bid", "price", "offer", "valuation", "value"))
    if bid_col is None:
        return pd.DataFrame()
    bidder_col = _find_column(df, ("bidder", "player", "buyer", "agent", "name"))
    reserve_col = _find_column(df, ("reserve", "minimum", "floor"))

    bids = []
    for idx, row in df.iterrows():
        bid = _to_float(row.get(bid_col), np.nan)
        if pd.isna(bid):
            continue
        bidder = row.get(bidder_col) if bidder_col else idx
        reserve = _to_float(row.get(reserve_col), 0.0) if reserve_col else 0.0
        bids.append((str(bidder), float(bid), float(reserve)))
    if not bids:
        return pd.DataFrame()

    bids.sort(key=lambda item: item[1], reverse=True)
    winner, winning_bid, reserve = bids[0]
    second_bid = bids[1][1] if len(bids) > 1 else reserve
    second_price = max(reserve, second_bid)
    accepted = winning_bid >= reserve
    return pd.DataFrame(
        [
            {
                "winner": winner if accepted else "",
                "winning_bid": winning_bid,
                "reserve_price": reserve,
                "first_price_revenue": winning_bid if accepted else 0.0,
                "second_price_revenue": second_price if accepted else 0.0,
                "bid_count": len(bids),
                "method": "auction_pricing_first_second_price",
            }
        ]
    )


def _payoff_matrices(df: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray, np.ndarray] | None:
    row_col = _find_column(df, ("player1_strategy", "row_strategy", "row", "strategy1", "action1"))
    col_col = _find_column(df, ("player2_strategy", "column_strategy", "col", "strategy2", "action2"))
    p1_col = _find_column(df, ("payoff1", "player1_payoff", "u1", "utility1"))
    p2_col = _find_column(df, ("payoff2", "player2_payoff", "u2", "utility2"))
    if row_col and col_col and p1_col and p2_col:
        row_actions = list(dict.fromkeys(str(value) for value in df[row_col].dropna()))
        col_actions = list(dict.fromkeys(str(value) for value in df[col_col].dropna()))
        if len(row_actions) != 2 or len(col_actions) != 2:
            return None
        payoff1 = np.zeros((2, 2), dtype=float)
        payoff2 = np.zeros((2, 2), dtype=float)
        seen = set()
        for _, row in df.iterrows():
            r = str(row.get(row_col))
            c = str(row.get(col_col))
            if r in row_actions and c in col_actions:
                i = row_actions.index(r)
                j = col_actions.index(c)
                payoff1[i, j] = _to_float(row.get(p1_col))
                payoff2[i, j] = _to_float(row.get(p2_col))
                seen.add((i, j))
        if len(seen) != 4:
            return None
        return row_actions, col_actions, payoff1, payoff2

    numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if numeric.shape[0] < 4 or numeric.shape[1] < 2:
        return None
    values = numeric.iloc[:4, :2].to_numpy(dtype=float)
    return ["A", "B"], ["X", "Y"], values[:, 0].reshape(2, 2), values[:, 1].reshape(2, 2)


def _mixed_equilibrium(
    row_actions: list[str],
    col_actions: list[str],
    payoff1: np.ndarray,
    payoff2: np.ndarray,
) -> dict[str, float | str] | None:
    a, b, c, d = payoff1[0, 0], payoff1[0, 1], payoff1[1, 0], payoff1[1, 1]
    e, f, g, h = payoff2[0, 0], payoff2[0, 1], payoff2[1, 0], payoff2[1, 1]
    q_denominator = a - b - c + d
    p_denominator = e - f - g + h
    if abs(q_denominator) < 1e-12 or abs(p_denominator) < 1e-12:
        return None
    q = (d - b) / q_denominator
    p = (h - g) / p_denominator
    if q < -1e-12 or q > 1.0 + 1e-12 or p < -1e-12 or p > 1.0 + 1e-12:
        return None
    p = float(np.clip(p, 0.0, 1.0))
    q = float(np.clip(q, 0.0, 1.0))
    p1_payoff = q * a + (1.0 - q) * b
    p2_payoff = p * e + (1.0 - p) * g
    return {
        "equilibrium_type": "mixed",
        "player1_strategy": f"{row_actions[0]}:{p:.6g}, {row_actions[1]}:{1.0 - p:.6g}",
        "player2_strategy": f"{col_actions[0]}:{q:.6g}, {col_actions[1]}:{1.0 - q:.6g}",
        "player1_mixed_probability_first": p,
        "player2_mixed_probability_first": q,
        "player1_payoff": float(p1_payoff),
        "player2_payoff": float(p2_payoff),
        "method": "nash_equilibrium_2x2",
    }


def _find_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    for column in df.columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return str(column)
    return None


def _find_numeric_column(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    numeric_columns = set(str(column) for column in df.select_dtypes(include="number").columns)
    exact_matches = {keyword for keyword in keywords if len(keyword) <= 4}
    for column in df.columns:
        name = str(column).lower()
        if str(column) not in numeric_columns:
            continue
        if name in exact_matches or any(keyword in name for keyword in keywords if keyword not in exact_matches):
            return str(column)
    return None


def _parse_coalition(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if re.fullmatch(r"[01]+", text):
        return [f"p{index + 1}" for index, flag in enumerate(text) if flag == "1"]
    return [item for item in re.split(r"[,;|+\s]+", text) if item]


def _to_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
