import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dash_table, dcc, html


DATA_DIR = Path(__file__).resolve().parent
DYNAMIC_EVENTS_PATH = DATA_DIR / "match_data" / "dynamicevents.csv"
MATCH_PATH = DATA_DIR / "match_data" / "match.json"
TRACKING_PATH = DATA_DIR / "match_data" / "tracking.jsonl"
FLOW_CACHE_PATH = DATA_DIR / "outputs" / "full_game_max_flow_cache.csv"

# Some open event feeds can miss one or more explicit goal markers.
# Use known official finals to backfill missing goal events in the dashboard timeline.
FINAL_SCORE_OVERRIDES: dict[int, tuple[int, int]] = {
    2060235: (3, 4),
}

# Explicit scorer + minute timeline overrides for matches where open event feeds
# mis-attribute or omit goal events.
GOAL_EVENT_OVERRIDES: dict[int, list[dict[str, str]]] = {
    2060235: [
        {"team_side": "home", "player": "Ndoye", "minute": "17"},
        {"team_side": "away", "player": "Tah", "minute": "26"},
        {"team_side": "home", "player": "Embolo", "minute": "41"},
        {"team_side": "away", "player": "Gnabry", "minute": "45+2"},
        {"team_side": "away", "player": "Wirtz", "minute": "61"},
        {"team_side": "home", "player": "Monteiro", "minute": "79"},
        {"team_side": "away", "player": "Wirtz", "minute": "85"},
    ]
}

# Optional display-only event timeline overrides for the Events table.
# These do not alter score progression logic.
EVENT_TABLE_OVERRIDES: dict[int, list[dict[str, str]]] = {
    2060235: [
        {"team_side": "home", "time": "16:00", "player": "Ndoye", "event": "Goal"},
        {"team_side": "home", "time": "18:50", "player": "", "event": "Good Swiss Defence"},
    ]
}


def safe_int(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        if pd.isna(value):
            return int(default)
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_float(value, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    try:
        if pd.isna(value):
            return float(default)
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def format_mmss(seconds_value: float | int) -> str:
    total_seconds = max(0, int(round(float(seconds_value))))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def parse_tracking_timestamp_seconds(timestamp_value) -> float:
    if timestamp_value is None:
        return np.nan
    s = str(timestamp_value).strip()
    if not s or s.lower() == "none":
        return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return float(int(h) * 3600 + int(m) * 60 + float(sec))
        if len(parts) == 2:
            m, sec = parts
            return float(int(m) * 60 + float(sec))
    except (TypeError, ValueError):
        return np.nan
    return np.nan


def parse_mmss_to_seconds(value) -> float:
    if value is None:
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    parts = s.split(":")
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            if minutes < 0 or seconds < 0 or seconds >= 60:
                return np.nan
            return float(minutes * 60 + seconds)
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
                return np.nan
            return float(hours * 3600 + minutes * 60 + seconds)
    except (TypeError, ValueError):
        return np.nan
    return np.nan


def parse_minute_notation_to_game_clock_seconds(value) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace("'", "")
    if not text:
        return np.nan
    try:
        if "+" in text:
            base_text, added_text = text.split("+", 1)
            base_minute = int(base_text.strip())
            added_minute = int(added_text.strip())
            return float((base_minute + added_minute) * 60)
        return float(int(text) * 60)
    except (TypeError, ValueError):
        return np.nan


def get_theme_mode(value) -> str:
    text = str(value).strip().lower() if value is not None else ""
    return "dark" if text == "dark" else "light"


def format_percent(value: float) -> str:
    if np.isfinite(value):
        return f"{100.0 * float(value):.1f}%"
    return "N/A"


def title_case_label(value, default: str = "None") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    return text.replace("_", " ").title()


def team_abbreviation(team_name: str) -> str:
    name = str(team_name).strip()
    known = {
        "switzerland": "SUI",
        "germany": "GER",
    }
    key = name.lower()
    if key in known:
        return known[key]
    letters = "".join(ch for ch in name.upper() if ch.isalpha())
    return letters[:3] if letters else "TEAM"


def parse_json_list(value) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def parse_json_obj(value) -> dict | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def load_tracking_jsonl(file_path: Path) -> list[dict]:
    tracking = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tracking.append(json.loads(line))
    return tracking


def get_home_attacking_direction(match_data: dict, period: int) -> str:
    sides = match_data.get("home_team_side", [])
    if not sides:
        return "left_to_right"
    idx = max(0, min(period - 1, len(sides) - 1))
    return sides[idx]


def get_team_attack_sign(team_id: int, period: int, match_data: dict) -> int:
    home_team_id = int(match_data["home_team"]["id"])
    away_team_id = int(match_data["away_team"]["id"])
    home_dir = get_home_attacking_direction(match_data, period)

    if team_id == home_team_id:
        return 1 if home_dir == "left_to_right" else -1
    if team_id == away_team_id:
        return -1 if home_dir == "left_to_right" else 1
    return 1


def normalize_attacking_coordinates(x: float, y: float, team_id: int, period: int, match_data: dict) -> tuple[float, float]:
    sign = get_team_attack_sign(team_id, period, match_data)
    return sign * x, sign * y


def get_xg(normalized_x: float, normalized_y: float) -> float:
    goal_x = 52.5
    goal_y = 0.0

    distance = math.hypot(goal_x - normalized_x, goal_y - normalized_y)
    dx = goal_x - normalized_x
    if dx <= 0:
        return 0.0

    angle = math.atan2(7.32 * dx, (dx ** 2) + (normalized_y ** 2) - ((7.32 / 2.0) ** 2))
    if angle < 0:
        angle += math.pi

    logit = -1.0 - (0.15 * distance) + (1.5 * angle)
    return float(1.0 / (1.0 + math.exp(-logit)))


def active_passing_options(events_df: pd.DataFrame, frame_number: int, carrier_id: int | None) -> pd.DataFrame:
    options = events_df[
        (events_df["event_type"] == "passing_option")
        & (events_df["frame_start"] <= frame_number)
        & (events_df["frame_end"] >= frame_number)
    ].copy()

    if carrier_id is not None and "player_in_possession_id" in options.columns:
        options = options[options["player_in_possession_id"] == carrier_id].copy()

    return options


def option_utility(option_row: pd.Series) -> float:
    xpass = option_row.get("player_targeted_xpass_completion", np.nan)
    if pd.isna(xpass):
        xpass = option_row.get("xpass_completion", np.nan)

    xthreat = option_row.get("player_targeted_xthreat", np.nan)
    if pd.isna(xthreat):
        xthreat = option_row.get("xthreat", np.nan)

    if pd.isna(xpass) or pd.isna(xthreat):
        return np.nan

    return float(xpass) * float(xthreat)


def frame_possession_team(frame_data: dict, match_data: dict) -> str:
    possession = frame_data.get("possession", None) or {}
    group = possession.get("group", None)
    if group == "home team":
        return "home"
    if group == "away team":
        return "away"
    return "none"


def frame_shot_xg(frame_data: dict, match_data: dict) -> float:
    possession = frame_data.get("possession", None) or {}
    ball = frame_data.get("ball_data", None)
    period = safe_int(frame_data.get("period", 1), 1)

    if ball is None:
        return np.nan

    group = possession.get("group", None)
    if group == "home team":
        team_id = int(match_data["home_team"]["id"])
    elif group == "away team":
        team_id = int(match_data["away_team"]["id"])
    else:
        return np.nan

    nx, ny = normalize_attacking_coordinates(float(ball["x"]), float(ball["y"]), team_id, period, match_data)
    return get_xg(nx, ny)


def frame_metrics(frame_data: dict, events_df: pd.DataFrame, match_data: dict) -> dict:
    frame_number = safe_int(frame_data.get("frame", 0), 0)
    possession = frame_data.get("possession", None) or {}
    carrier_id = possession.get("player_id", None)

    options = active_passing_options(events_df, frame_number, carrier_id)
    if not options.empty:
        options = options.copy()
        options["utility"] = options.apply(option_utility, axis=1)
        options = options[np.isfinite(options["utility"])].copy()
    n_options = int(len(options))

    best_pass_utility = float(options["utility"].max()) if n_options > 0 else np.nan
    pass_capacity_sum = float(np.clip(options["utility"], 0.0, None).sum()) if n_options > 0 else 0.0
    shot_xg = frame_shot_xg(frame_data, match_data)
    shot_capacity = float(max(shot_xg, 0.0)) if pd.notna(shot_xg) else 0.0

    best_action = "none"
    pass_val = best_pass_utility if pd.notna(best_pass_utility) else -np.inf
    shot_val = shot_xg if pd.notna(shot_xg) else -np.inf
    if pass_val >= shot_val and np.isfinite(pass_val):
        best_action = "pass"
    elif np.isfinite(shot_val):
        best_action = "shot"

    frame_flow_capacity = float(pass_capacity_sum + shot_capacity)
    team_side = frame_possession_team(frame_data, match_data)
    signed_flow_proxy = 0.0
    if team_side == "home":
        signed_flow_proxy = frame_flow_capacity
    elif team_side == "away":
        signed_flow_proxy = -frame_flow_capacity

    return {
        "frame": frame_number,
        "timestamp": frame_data.get("timestamp", ""),
        "period": frame_data.get("period", ""),
        "team_side": team_side,
        "carrier_id": carrier_id,
        "n_options": n_options,
        "best_pass_utility": best_pass_utility,
        "pass_capacity_sum": pass_capacity_sum,
        "shot_xg": shot_xg,
        "shot_capacity": shot_capacity,
        "frame_flow_capacity": frame_flow_capacity,
        "best_action": best_action,
        "flow_proxy": float(signed_flow_proxy),
    }


def build_pitch_figure(
    frame_data: dict,
    match_data: dict,
    player_number_map: dict[int, int],
    player_name_map: dict[int, str],
    player_team_map: dict[int, int],
    option_rows: list[dict] | None = None,
    run_rows: list[dict] | None = None,
    shot_info: dict | None = None,
    carrier_id: int | None = None,
    best_pass_offensive_return: float = np.nan,
    theme_mode: str = "light",
) -> go.Figure:
    home_team_id = int(match_data["home_team"]["id"])
    away_team_id = int(match_data["away_team"]["id"])
    home_color = match_data.get("home_team_kit", {}).get("jersey_color", "red")
    away_color = match_data.get("away_team_kit", {}).get("jersey_color", "black")

    pitch_length = float(match_data["pitch_length"])
    pitch_width = float(match_data["pitch_width"])
    half_length = pitch_length / 2.0
    half_width = pitch_width / 2.0
    penalty_depth = 16.5
    penalty_half_width = 20.16
    goal_area_depth = 5.5
    goal_area_half_width = 9.16
    penalty_spot_distance = 11.0
    penalty_arc_radius = 9.15
    penalty_spot_radius = 0.28

    is_dark = get_theme_mode(theme_mode) == "dark"
    field_color = "#a9d98d"
    line_color = "#ffffff"
    paper_bg = "#0f172a" if is_dark else "white"
    text_color = "#e5e7eb" if is_dark else "#111827"
    legend_bg = "rgba(15,23,42,0.76)" if is_dark else "rgba(255,255,255,0.78)"

    players = frame_data.get("player_data", [])
    positions: dict[int, tuple[float, float]] = {}

    hx, hy, htext, hhover = [], [], [], []
    hxu, hyu, htextu, hhoveru = [], [], [], []
    ax, ay, atext, ahover = [], [], [], []
    axu, ayu, atextu, ahoveru = [], [], [], []
    ux, uy, utext, uhover = [], [], [], []

    for p in players:
        pid = safe_int(p.get("player_id", -1), -1)
        if pid < 0:
            continue
        x = safe_float(p.get("x", np.nan), np.nan)
        y = safe_float(p.get("y", np.nan), np.nan)
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        positions[pid] = (x, y)
        team_id = player_team_map.get(pid, None)
        label = f"<b>{player_number_map.get(pid, pid)}</b>"
        hover_label = str(player_name_map.get(pid, f"Player {pid}"))
        is_detected = bool(p.get("is_detected", True))

        if team_id == home_team_id:
            if is_detected:
                hx.append(x)
                hy.append(y)
                htext.append(label)
                hhover.append(hover_label)
            else:
                hxu.append(x)
                hyu.append(y)
                htextu.append(label)
                hhoveru.append(hover_label)
        elif team_id == away_team_id:
            if is_detected:
                ax.append(x)
                ay.append(y)
                atext.append(label)
                ahover.append(hover_label)
            else:
                axu.append(x)
                ayu.append(y)
                atextu.append(label)
                ahoveru.append(hover_label)
        else:
            ux.append(x)
            uy.append(y)
            utext.append(label)
            uhover.append(hover_label)

    fig = go.Figure()
    fig.add_shape(type="rect", x0=-half_length, y0=-half_width, x1=half_length, y1=half_width, line=dict(color=line_color, width=2.2), fillcolor=field_color, layer="below")
    fig.add_shape(type="line", x0=0, y0=-half_width, x1=0, y1=half_width, line=dict(color=line_color, width=1.5), layer="below")
    fig.add_shape(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=dict(color=line_color, width=1.5), layer="below")
    fig.add_shape(type="circle", x0=-0.2, y0=-0.2, x1=0.2, y1=0.2, line=dict(color=line_color, width=1.0), fillcolor=line_color, layer="below")

    fig.add_shape(type="rect", x0=-half_length, y0=-penalty_half_width, x1=-half_length + penalty_depth, y1=penalty_half_width, line=dict(color=line_color, width=1.5), layer="below")
    fig.add_shape(type="rect", x0=half_length - penalty_depth, y0=-penalty_half_width, x1=half_length, y1=penalty_half_width, line=dict(color=line_color, width=1.5), layer="below")
    fig.add_shape(type="rect", x0=-half_length, y0=-goal_area_half_width, x1=-half_length + goal_area_depth, y1=goal_area_half_width, line=dict(color=line_color, width=1.5), layer="below")
    fig.add_shape(type="rect", x0=half_length - goal_area_depth, y0=-goal_area_half_width, x1=half_length, y1=goal_area_half_width, line=dict(color=line_color, width=1.5), layer="below")

    left_spot_x = -half_length + penalty_spot_distance
    right_spot_x = half_length - penalty_spot_distance
    fig.add_shape(type="circle", x0=left_spot_x - penalty_spot_radius, y0=-penalty_spot_radius, x1=left_spot_x + penalty_spot_radius, y1=penalty_spot_radius, line=dict(color=line_color, width=1.0), fillcolor=line_color, layer="below")
    fig.add_shape(type="circle", x0=right_spot_x - penalty_spot_radius, y0=-penalty_spot_radius, x1=right_spot_x + penalty_spot_radius, y1=penalty_spot_radius, line=dict(color=line_color, width=1.0), fillcolor=line_color, layer="below")

    arc_angle = math.acos(goal_area_depth / penalty_arc_radius)
    left_t = np.linspace(-arc_angle, arc_angle, 64)
    right_t = np.linspace(math.pi - arc_angle, math.pi + arc_angle, 64)
    fig.add_trace(
        go.Scatter(
            x=left_spot_x + penalty_arc_radius * np.cos(left_t),
            y=penalty_arc_radius * np.sin(left_t),
            mode="lines",
            line=dict(color=line_color, width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=right_spot_x + penalty_arc_radius * np.cos(right_t),
            y=penalty_arc_radius * np.sin(right_t),
            mode="lines",
            line=dict(color=line_color, width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if hx:
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers+text", text=htext, textposition="middle center", textfont=dict(color="white", size=10), customdata=hhover, hovertemplate="%{customdata}<extra></extra>", name=match_data["home_team"]["name"], marker=dict(size=22, color=home_color, line=dict(color="black", width=1.2), opacity=1.0), showlegend=True))
    if hxu:
        fig.add_trace(go.Scatter(x=hxu, y=hyu, mode="markers+text", text=htextu, textposition="middle center", textfont=dict(color="white", size=10), customdata=hhoveru, hovertemplate="%{customdata}<extra></extra>", marker=dict(size=22, color=home_color, line=dict(color="black", width=1.2), opacity=0.35), name=match_data["home_team"]["name"], showlegend=False))
    if ax:
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers+text", text=atext, textposition="middle center", textfont=dict(color="white", size=10), customdata=ahover, hovertemplate="%{customdata}<extra></extra>", name=match_data["away_team"]["name"], marker=dict(size=22, color=away_color, line=dict(color="black", width=1.2), opacity=1.0), showlegend=True))
    if axu:
        fig.add_trace(go.Scatter(x=axu, y=ayu, mode="markers+text", text=atextu, textposition="middle center", textfont=dict(color="white", size=10), customdata=ahoveru, hovertemplate="%{customdata}<extra></extra>", marker=dict(size=22, color=away_color, line=dict(color="black", width=1.2), opacity=0.35), name=match_data["away_team"]["name"], showlegend=False))
    if ux:
        fig.add_trace(go.Scatter(x=ux, y=uy, mode="markers+text", text=utext, textposition="middle center", textfont=dict(color="black", size=10), customdata=uhover, hovertemplate="%{customdata}<extra></extra>", name="Unknown", marker=dict(size=22, color="gray", line=dict(color="black", width=1.2), opacity=0.7), showlegend=False))

    ball = frame_data.get("ball_data", None)
    if ball is not None:
        fig.add_trace(go.Scatter(x=[safe_float(ball.get("x", np.nan), np.nan)], y=[safe_float(ball.get("y", np.nan), np.nan)], mode="markers", name="Ball", marker=dict(size=14, color="orange", line=dict(color="black", width=1.2)), showlegend=True))

    carrier_xy = positions.get(int(carrier_id), None) if carrier_id is not None else None
    if carrier_xy is not None:
        fig.add_trace(go.Scatter(x=[float(carrier_xy[0])], y=[float(carrier_xy[1])], mode="markers", name="Player in possession", marker=dict(size=34, color="rgba(0,0,0,0)", line=dict(color="gold", width=3)), showlegend=True))

    start_x = np.nan
    start_y = np.nan
    if carrier_xy is not None:
        start_x, start_y = float(carrier_xy[0]), float(carrier_xy[1])
    elif ball is not None:
        start_x = safe_float(ball.get("x", np.nan), np.nan)
        start_y = safe_float(ball.get("y", np.nan), np.nan)

    if option_rows and np.isfinite(start_x) and np.isfinite(start_y):
        rows_sorted = sorted(option_rows, key=lambda r: safe_float(r.get("utility", np.nan), -np.inf), reverse=True)
        for idx, row in enumerate(rows_sorted):
            tx = safe_float(row.get("target_x", row.get("receiver_x", np.nan)), np.nan)
            ty = safe_float(row.get("target_y", row.get("receiver_y", np.nan)), np.nan)
            if not np.isfinite(tx) or not np.isfinite(ty):
                continue

            is_best = bool(row.get("is_best", False)) or idx == 0
            color = "limegreen" if is_best else "gray"
            width = 3.0 if is_best else 1.6
            alpha = 0.60 if is_best else 0.35
            fig.add_annotation(x=tx, y=ty, ax=start_x, ay=start_y, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=width, arrowcolor=color, opacity=alpha)

            util = safe_float(row.get("utility", np.nan), np.nan)
            if np.isfinite(util):
                fig.add_annotation(x=(start_x + tx) / 2.0, y=(start_y + ty) / 2.0, xref="x", yref="y", text=f"{100.0 * util:.1f}%", showarrow=False, font=dict(size=9, color=color), bgcolor="rgba(255,255,255,0.45)", borderwidth=0, borderpad=1)

    if run_rows and np.isfinite(start_x) and np.isfinite(start_y):
        runs_sorted = sorted(run_rows, key=lambda r: safe_float(r.get("run_value", np.nan), -np.inf), reverse=True)
        if runs_sorted:
            run_row = runs_sorted[0]
            tx = safe_float(run_row.get("target_x", np.nan), np.nan)
            ty = safe_float(run_row.get("target_y", np.nan), np.nan)
            run_val = safe_float(run_row.get("run_value", np.nan), np.nan)
            best_pass_util = max([safe_float(r.get("utility", np.nan), -np.inf) for r in option_rows or []], default=np.nan)
            shot_xg = safe_float(shot_info.get("xg", np.nan), np.nan) if shot_info is not None else np.nan

            if np.isfinite(tx) and np.isfinite(ty) and np.isfinite(run_val):
                run_beats = bool(pd.notna(best_pass_util) and pd.notna(shot_xg) and run_val > best_pass_util and run_val > shot_xg)
                run_color = "royalblue" if run_beats else "lightskyblue"
                run_alpha = 0.60 if run_beats else 0.35
                run_width = 3.0 if run_beats else 1.6

                uxv = tx - start_x
                uyv = ty - start_y
                norm = max(math.hypot(uxv, uyv), 1e-9)
                tx_long = max(-half_length, min(half_length, tx + 1.5 * uxv / norm))
                ty_long = max(-half_width, min(half_width, ty + 1.5 * uyv / norm))

                fig.add_shape(type="line", x0=start_x, y0=start_y, x1=tx_long, y1=ty_long, line=dict(color=run_color, width=run_width, dash="dash"), opacity=run_alpha)
                fig.add_annotation(x=tx_long, y=ty_long, ax=tx_long - 0.8 * uxv / norm, ay=ty_long - 0.8 * uyv / norm, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=run_width, arrowcolor=run_color, opacity=run_alpha)

                label_text = f"{100.0 * run_val:.1f}%"
                if run_beats:
                    label_text = f"<b>{label_text}</b>"
                fig.add_annotation(x=(start_x + tx_long) / 2.0, y=(start_y + ty_long) / 2.0, xref="x", yref="y", text=label_text, showarrow=False, font=dict(size=9, color=run_color), bgcolor="rgba(255,255,255,0.45)", borderwidth=0, borderpad=1)

    if shot_info is not None and pd.notna(shot_info.get("xg", np.nan)) and float(shot_info["xg"]) > 0.01:
        sx = safe_float(shot_info.get("start_x", shot_info.get("ball_x", start_x)), np.nan)
        sy = safe_float(shot_info.get("start_y", shot_info.get("ball_y", start_y)), np.nan)
        tx = safe_float(shot_info.get("target_x", shot_info.get("goal_x", np.nan)), np.nan)
        ty = safe_float(shot_info.get("target_y", shot_info.get("goal_y", np.nan)), np.nan)
        xg = safe_float(shot_info.get("xg", np.nan), np.nan)

        if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(tx) and np.isfinite(ty) and np.isfinite(xg):
            shot_color = "gray"
            shot_alpha = 0.35
            shot_lw = 1.6
            if pd.notna(best_pass_offensive_return) and xg > best_pass_offensive_return:
                shot_color = "limegreen"
                shot_alpha = 0.60
                shot_lw = 3.0

            fig.add_annotation(x=tx, y=ty, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=shot_lw, arrowcolor=shot_color, opacity=shot_alpha)
            fig.add_annotation(x=(sx + tx) / 2.0, y=(sy + ty) / 2.0, xref="x", yref="y", text=f"{100.0 * xg:.1f}%", showarrow=False, font=dict(size=9, color=shot_color), bgcolor="rgba(255,255,255,0.45)", borderwidth=0, borderpad=1)

    left_view_pad = 0.5
    right_view_pad = 16.0
    fig.update_layout(
        margin=dict(l=0, r=10, t=16, b=10),
        xaxis=dict(range=[-half_length - left_view_pad, half_length + left_view_pad + right_view_pad], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-half_width - 3, half_width + 3], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        paper_bgcolor=paper_bg,
        plot_bgcolor=paper_bg,
        font=dict(color=text_color),
        title=None,
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor=legend_bg, font=dict(color=text_color)),
    )

    return fig


def build_flow_figure(hist: pd.DataFrame, match_data: dict, current_t: float | None = None, theme_mode: str = "light") -> go.Figure:
    del current_t
    is_dark = get_theme_mode(theme_mode) == "dark"
    text_color = "#e5e7eb" if is_dark else "#111827"
    paper_bg = "#0f172a" if is_dark else "white"
    plot_bg = "#111827" if is_dark else "white"
    grid_color = "rgba(148,163,184,0.25)" if is_dark else "rgba(15,23,42,0.12)"
    home_line = "#ff6b6b" if is_dark else "red"
    home_fill = "rgba(255,107,107,0.26)" if is_dark else "rgba(255,0,0,0.22)"
    away_line = "#e5e7eb" if is_dark else "black"
    away_fill = "rgba(229,231,235,0.22)" if is_dark else "rgba(0,0,0,0.22)"
    legend_bg = "rgba(15,23,42,0.76)" if is_dark else "rgba(255,255,255,0.78)"

    if hist.empty:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=40),
            title="Live Maximum Flow Capacity",
            xaxis_title="Match Time (MM:SS)",
            yaxis_title="Network Flow (Swiss up | Germany down)",
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            font=dict(color=text_color),
        )
        return fig

    times = hist["t_seconds"].to_numpy(dtype=float)
    flow = hist["flow_proxy"].to_numpy(dtype=float)
    sides = hist["team_side"].to_numpy(dtype=object)

    if {"game_clock_s", "half_clock_s", "period"}.issubset(set(hist.columns)):
        game_clock = hist["game_clock_s"].to_numpy(dtype=float)
        half_clock = hist["half_clock_s"].to_numpy(dtype=float)
        period_vals = hist["period"].to_numpy(dtype=float)
        hover_labels = np.array(
            [
                f"Game {format_mmss(gc)} | {int(pv)}H {format_mmss(hc)}"
                if np.isfinite(gc) and np.isfinite(hc) and np.isfinite(pv)
                else "Time N/A"
                for gc, hc, pv in zip(game_clock, half_clock, period_vals)
            ],
            dtype=object,
        )
    else:
        hover_labels = np.array([f"Time {format_mmss(t)}" for t in times], dtype=object)

    home_flow = np.where(sides == "home", np.abs(flow), np.nan)
    away_flow = np.where(sides == "away", -np.abs(flow), np.nan)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=home_flow,
            mode="lines",
            name=match_data["home_team"]["name"],
            line=dict(color=home_line, width=3),
            fill="tozeroy",
            fillcolor=home_fill,
            customdata=hover_labels,
            hovertemplate="%{customdata}<br>Flow %{y:.3f}<extra>%{fullData.name}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=away_flow,
            mode="lines",
            name=match_data["away_team"]["name"],
            line=dict(color=away_line, width=3),
            fill="tozeroy",
            fillcolor=away_fill,
            customdata=hover_labels,
            hovertemplate="%{customdata}<br>Flow %{y:.3f}<extra>%{fullData.name}</extra>",
        )
    )

    peak_abs = float(np.nanmax(np.abs(flow))) if np.isfinite(flow).any() else 0.0
    # Keep a tighter default y-range, expanding only when needed.
    max_abs = 0.4 if peak_abs <= 0.4 else (peak_abs * 1.08)

    if len(times) > 0:
        t_min = float(np.nanmin(times))
        t_max = float(np.nanmax(times))
    else:
        t_min, t_max = 0.0, 1.0
    if not np.isfinite(t_min) or not np.isfinite(t_max) or abs(t_max - t_min) < 1e-9:
        t_min, t_max = 0.0, 1.0

    x_tick_step = 5.0
    x_tick_start = int(math.floor(t_min / x_tick_step) * x_tick_step)
    x_tick_vals = np.arange(float(x_tick_start), t_max + x_tick_step, x_tick_step, dtype=float)
    x_tick_vals = x_tick_vals[x_tick_vals >= (t_min - 1e-6)]
    x_tick_text = [format_mmss(v) for v in x_tick_vals]

    tick_vals = np.linspace(-max_abs, max_abs, 7)
    tick_text = [f"{v:.2f}" for v in tick_vals]

    fig.add_hline(y=0.0, line_color=("#94a3b8" if is_dark else "gray"), line_width=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=40),
        title="Live Maximum Flow Capacity",
        xaxis=dict(title="Match Time (MM:SS)", range=[t_min, t_max], tickvals=x_tick_vals, ticktext=x_tick_text, gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(title="Network Flow (Swiss up | Germany down)", range=[-max_abs, max_abs], tickvals=tick_vals, ticktext=tick_text, gridcolor=grid_color, zerolinecolor=grid_color),
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=text_color),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor=legend_bg, font=dict(color=text_color)),
    )

    return fig


def top_options_for_frame(frame_data: dict, events_df: pd.DataFrame, player_name_map: dict[int, str]) -> pd.DataFrame:
    frame_number = safe_int(frame_data.get("frame", 0), 0)
    possession = frame_data.get("possession", None) or {}
    carrier_id = possession.get("player_id", None)

    player_positions = {}
    for p in frame_data.get("player_data", []):
        pid = safe_int(p.get("player_id", -1), -1)
        if pid < 0:
            continue
        player_positions[pid] = (float(p.get("x", 0.0)), float(p.get("y", 0.0)))

    carrier_xy = player_positions.get(carrier_id, None)
    if carrier_xy is None:
        return pd.DataFrame(columns=["receiver_id", "receiver_name", "utility", "start_x", "start_y", "target_x", "target_y", "is_best"])

    options = active_passing_options(events_df, frame_number, carrier_id)
    if options.empty:
        return pd.DataFrame(columns=["receiver_id", "receiver_name", "utility", "start_x", "start_y", "target_x", "target_y", "is_best"])

    options = options.copy()
    options["utility"] = options.apply(option_utility, axis=1)
    options = options[np.isfinite(options["utility"])].copy()
    if options.empty:
        return pd.DataFrame(columns=["receiver_id", "receiver_name", "utility", "start_x", "start_y", "target_x", "target_y", "is_best"])

    options = options.sort_values("utility", ascending=False).head(8).copy()
    options["receiver_id"] = options["player_id"].astype(int)
    options["receiver_name"] = options["receiver_id"].map(lambda pid: player_name_map.get(pid, str(pid)))
    options["utility"] = options["utility"].astype(float)
    options["target_x"] = options["receiver_id"].map(lambda pid: player_positions.get(pid, (np.nan, np.nan))[0])
    options["target_y"] = options["receiver_id"].map(lambda pid: player_positions.get(pid, (np.nan, np.nan))[1])
    options = options[np.isfinite(options["target_x"]) & np.isfinite(options["target_y"])].copy()
    options["start_x"] = float(carrier_xy[0])
    options["start_y"] = float(carrier_xy[1])
    options["is_best"] = False
    if not options.empty:
        options.iloc[0, options.columns.get_loc("is_best")] = True

    return options[["receiver_id", "receiver_name", "utility", "start_x", "start_y", "target_x", "target_y", "is_best"]]


def shot_option_for_frame(frame_data: dict, match_data: dict) -> dict | None:
    xg = frame_shot_xg(frame_data, match_data)
    if not pd.notna(xg):
        return None

    ball = frame_data.get("ball_data", None)
    possession = frame_data.get("possession", None) or {}
    if ball is None:
        return None

    period = safe_int(frame_data.get("period", 1), 1)
    group = possession.get("group", None)
    if group == "home team":
        team_id = int(match_data["home_team"]["id"])
    elif group == "away team":
        team_id = int(match_data["away_team"]["id"])
    else:
        return None

    goal_sign = get_team_attack_sign(team_id, period, match_data)
    return {
        "start_x": float(ball.get("x", 0.0)),
        "start_y": float(ball.get("y", 0.0)),
        "target_x": float(52.5 * goal_sign),
        "target_y": 0.0,
        "xg": float(xg),
    }


def build_dashboard() -> Dash:
    print("[dashboard] Loading events...")
    events_df = pd.read_csv(DYNAMIC_EVENTS_PATH)
    print("[dashboard] Loading tracking data...")
    tracking_data = load_tracking_jsonl(TRACKING_PATH)

    with open(MATCH_PATH, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    player_name_map = {int(p["id"]): p.get("short_name", str(p["id"])) for p in match_data["players"]}
    player_number_map = {int(p["id"]): int(p.get("number", p["id"])) for p in match_data["players"]}
    player_team_map = {int(p["id"]): int(p["team_id"]) for p in match_data["players"]}

    frame_map = {}
    for fr in tracking_data:
        frame_value = fr.get("frame", None)
        if frame_value is None:
            continue
        frame_key = safe_int(frame_value, -1)
        if frame_key >= 0:
            frame_map[frame_key] = fr
    all_frame_numbers = sorted(frame_map.keys())

    precomputed_by_frame: dict[int, dict] = {}
    if FLOW_CACHE_PATH.exists():
        try:
            needed_cols = {
                "frame", "timestamp", "period", "team_side", "carrier_id", "carrier_x", "carrier_y", "n_options",
                "best_pass_utility", "best_pass_offensive_return", "best_run_value", "pass_capacity_sum", "shot_xg",
                "shot_capacity", "frame_flow_capacity", "network_flow", "best_action", "flow_proxy", "t_seconds",
                "pass_options_json", "run_options_json", "shot_option_json",
            }
            cache_df = pd.read_csv(FLOW_CACHE_PATH, usecols=lambda c: c in needed_cols)
            if "frame" in cache_df.columns:
                cache_df = cache_df.drop_duplicates(subset=["frame"], keep="last")
                for _, r in cache_df.iterrows():
                    frame_key = safe_int(r.get("frame", -1), -1)
                    if frame_key < 0:
                        continue

                    team_side = str(r.get("team_side", "none"))
                    shot_xg = safe_float(r.get("shot_xg", np.nan), np.nan)
                    pass_capacity_sum = safe_float(r.get("pass_capacity_sum", np.nan), np.nan)
                    shot_capacity = safe_float(r.get("shot_capacity", np.nan), np.nan)
                    frame_flow_capacity = safe_float(r.get("frame_flow_capacity", np.nan), np.nan)
                    network_flow = safe_float(r.get("network_flow", np.nan), np.nan)
                    signed_flow = safe_float(r.get("flow_proxy", np.nan), np.nan)

                    if not np.isfinite(frame_flow_capacity) and np.isfinite(network_flow):
                        frame_flow_capacity = network_flow
                    if not np.isfinite(signed_flow) and np.isfinite(network_flow):
                        if team_side == "home":
                            signed_flow = network_flow
                        elif team_side == "away":
                            signed_flow = -network_flow
                        else:
                            signed_flow = 0.0

                    precomputed_by_frame[frame_key] = {
                        "frame": frame_key,
                        "timestamp": r.get("timestamp", ""),
                        "period": r.get("period", ""),
                        "team_side": team_side,
                        "carrier_id": safe_int(r.get("carrier_id", -1), -1),
                        "carrier_x": safe_float(r.get("carrier_x", np.nan), np.nan),
                        "carrier_y": safe_float(r.get("carrier_y", np.nan), np.nan),
                        "n_options": safe_int(r.get("n_options", 0), 0),
                        "best_pass_utility": safe_float(r.get("best_pass_utility", np.nan), np.nan),
                        "best_pass_offensive_return": safe_float(r.get("best_pass_offensive_return", np.nan), np.nan),
                        "best_run_value": safe_float(r.get("best_run_value", np.nan), np.nan),
                        "pass_capacity_sum": pass_capacity_sum if np.isfinite(pass_capacity_sum) else 0.0,
                        "shot_xg": shot_xg,
                        "shot_capacity": shot_capacity if np.isfinite(shot_capacity) else (max(shot_xg, 0.0) if pd.notna(shot_xg) else 0.0),
                        "frame_flow_capacity": frame_flow_capacity if np.isfinite(frame_flow_capacity) else 0.0,
                        "best_action": str(r.get("best_action", "none")),
                        "flow_proxy": signed_flow if np.isfinite(signed_flow) else 0.0,
                        "t_seconds": safe_float(r.get("t_seconds", np.nan), np.nan),
                        "pass_options_json": r.get("pass_options_json", "[]"),
                        "run_options_json": r.get("run_options_json", "[]"),
                        "shot_option_json": r.get("shot_option_json", ""),
                    }
            print(f"[dashboard] Loaded precomputed cache rows: {len(precomputed_by_frame)} from {FLOW_CACHE_PATH}")
        except Exception as exc:
            print(f"[dashboard] Warning: failed to load precomputed cache at {FLOW_CACHE_PATH}: {exc}")

    frame_records: list[dict] = []
    for frame_number in all_frame_numbers:
        fr = frame_map[frame_number]
        period = safe_int(fr.get("period", -1), -1)
        if period not in (1, 2):
            continue

        game_clock_s = parse_tracking_timestamp_seconds(fr.get("timestamp", None))
        if not np.isfinite(game_clock_s):
            continue

        if period == 1:
            half_clock_s = game_clock_s
        else:
            half_clock_s = game_clock_s - 2700.0
            if not np.isfinite(half_clock_s) or half_clock_s < 0:
                half_clock_s = game_clock_s

        frame_records.append(
            {
                "frame": int(frame_number),
                "period": int(period),
                "game_clock_s": float(game_clock_s),
                "half_clock_s": float(max(0.0, half_clock_s)),
            }
        )

    if not frame_records:
        raise ValueError("No playable period 1/2 tracking frames with valid timestamps were found.")

    first_half_end = max((r["half_clock_s"] for r in frame_records if r["period"] == 1), default=2700.0)
    for r in frame_records:
        if r["period"] == 1:
            r["timeline_s"] = float(r["half_clock_s"])
        else:
            r["timeline_s"] = float(first_half_end + r["half_clock_s"])

    frame_numbers = [int(r["frame"]) for r in frame_records]
    frame_times = np.array([float(r["timeline_s"]) for r in frame_records], dtype=float)
    frame_periods = np.array([int(r["period"]) for r in frame_records], dtype=int)
    frame_game_clocks = np.array([float(r["game_clock_s"]) for r in frame_records], dtype=float)
    frame_half_clocks = np.array([float(r["half_clock_s"]) for r in frame_records], dtype=float)

    frame_num_array = np.array(frame_numbers, dtype=int)

    print(f"[dashboard] Loaded {len(frame_numbers)} active match frames (periods 1/2). Building lazy metric cache...")

    start_frame = frame_numbers[0] if frame_numbers else 0
    metrics_cache: dict[int, dict] = {}

    if len(frame_times) >= 2:
        diffs = np.diff(frame_times)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        frame_step_seconds = float(np.median(diffs)) if len(diffs) > 0 else 0.1
    else:
        frame_step_seconds = 0.1
    max_time_seconds = float(frame_times[-1]) if len(frame_times) > 0 else 0.0

    def clamp_time_seconds(t_value: float) -> float:
        if not np.isfinite(t_value):
            return 0.0
        return float(min(max(t_value, 0.0), max_time_seconds))

    def time_to_idx(t_value: float) -> int:
        if len(frame_times) == 0:
            return 0
        t_value = clamp_time_seconds(float(t_value))
        pos = int(np.searchsorted(frame_times, t_value))
        if pos <= 0:
            return 0
        if pos >= len(frame_times):
            return len(frame_times) - 1
        left_dist = abs(t_value - float(frame_times[pos - 1]))
        right_dist = abs(float(frame_times[pos]) - t_value)
        return pos if right_dist < left_dist else (pos - 1)

    mark_step = 900
    first_half_end_int = int(max(0, round(first_half_end)))
    second_half_span = max(0.0, max_time_seconds - float(first_half_end_int))
    second_half_span_int = int(round(second_half_span))

    def build_time_slider_marks(label_color: str) -> dict[int, dict]:
        marks: dict[int, dict] = {}
        mark_style = {"color": label_color, "fontSize": "11px"}

        def minutes_label(seconds_value: float) -> str:
            return f"{int(round(float(seconds_value) / 60.0))}m"

        for local_t in range(0, first_half_end_int + 1, mark_step):
            marks[int(local_t)] = {"label": f"1H {minutes_label(local_t)}", "style": mark_style}
        marks[first_half_end_int] = {"label": f"1H {minutes_label(first_half_end_int)}", "style": mark_style}

        for local_t in range(0, second_half_span_int + 1, mark_step):
            marks[int(round(first_half_end_int + local_t))] = {"label": f"2H {minutes_label(local_t)}", "style": mark_style}
        if second_half_span_int > 0:
            marks[int(round(max_time_seconds))] = {"label": f"2H {minutes_label(second_half_span_int)}", "style": mark_style}
        return marks

    time_slider_marks_light = build_time_slider_marks("#334155")
    time_slider_marks_dark = build_time_slider_marks("#cbd5e1")

    home_team_id = int(match_data["home_team"]["id"])
    away_team_id = int(match_data["away_team"]["id"])
    match_id = safe_int(match_data.get("id", -1), -1)

    def nearest_idx_for_frame(frame_value: int) -> int:
        if len(frame_num_array) == 0:
            return 0
        pos = int(np.searchsorted(frame_num_array, frame_value))
        if pos <= 0:
            return 0
        if pos >= len(frame_num_array):
            return len(frame_num_array) - 1
        left = abs(int(frame_num_array[pos - 1]) - int(frame_value))
        right = abs(int(frame_num_array[pos]) - int(frame_value))
        return pos if right < left else (pos - 1)

    def timeline_from_game_clock(game_clock_s: float) -> float:
        if len(frame_game_clocks) == 0 or not np.isfinite(game_clock_s):
            return 0.0
        idx = int(np.nanargmin(np.abs(frame_game_clocks - float(game_clock_s))))
        return float(frame_times[idx]) if len(frame_times) > idx else 0.0

    home_abbr = team_abbreviation(str(match_data["home_team"]["name"]))
    away_abbr = team_abbreviation(str(match_data["away_team"]["name"]))
    home_team_name = str(match_data["home_team"]["name"])
    away_team_name = str(match_data["away_team"]["name"])

    event_rows: list[dict] = []
    goal_events_for_score: list[tuple[float, int]] = []
    if not events_df.empty:
        event_candidates = events_df.copy()
        if "event_type" in event_candidates.columns:
            event_candidates["event_type_l"] = event_candidates["event_type"].astype(str).str.lower()
        else:
            event_candidates["event_type_l"] = ""
        if "end_type" in event_candidates.columns:
            event_candidates["end_type_l"] = event_candidates["end_type"].astype(str).str.lower()
        else:
            event_candidates["end_type_l"] = ""

        lead_to_goal = pd.Series([False] * len(event_candidates), index=event_candidates.index)
        if "lead_to_goal" in event_candidates.columns:
            lead_to_goal = (
                event_candidates["lead_to_goal"]
                .fillna(False)
                .map(lambda v: str(v).strip().lower() in {"1", "true", "t", "yes", "y"})
            )

        # Preferred path: one goal per lead_to_goal phase.
        if "phase_index" in event_candidates.columns and bool(lead_to_goal.any()):
            phase_rows = event_candidates.loc[lead_to_goal].copy()
            phase_rows = phase_rows[phase_rows["phase_index"].notna()].copy()

            if "frame_start" in phase_rows.columns:
                phase_rows["frame_start"] = pd.to_numeric(phase_rows["frame_start"], errors="coerce")
            else:
                phase_rows["frame_start"] = np.nan
            if "frame_end" in phase_rows.columns:
                phase_rows["frame_end"] = pd.to_numeric(phase_rows["frame_end"], errors="coerce")
            else:
                phase_rows["frame_end"] = np.nan

            phase_rows["goal_frame"] = phase_rows["frame_end"]
            missing_goal_frame = phase_rows["goal_frame"].isna()
            phase_rows.loc[missing_goal_frame, "goal_frame"] = phase_rows.loc[missing_goal_frame, "frame_start"]
            phase_rows = phase_rows[phase_rows["goal_frame"].notna()].copy()
            phase_rows["goal_frame"] = phase_rows["goal_frame"].astype(int)

            for _, phase_group in phase_rows.sort_values(["phase_index", "goal_frame", "frame_start"]).groupby("phase_index", dropna=True, sort=True):
                ev = phase_group.iloc[-1]
                jump_frame = safe_int(ev.get("goal_frame", -1), -1)
                if jump_frame < 0:
                    continue

                idx = nearest_idx_for_frame(jump_frame)
                period = int(frame_periods[idx]) if len(frame_periods) > idx else safe_int(ev.get("period", 0), 0)
                if period not in (1, 2):
                    continue

                team_id = safe_int(ev.get("team_id", -1), -1)
                if team_id not in (home_team_id, away_team_id) and "team_id" in phase_group.columns:
                    phase_team_ids = [safe_int(v, -1) for v in phase_group["team_id"].tolist()]
                    valid_team_ids = [tid for tid in phase_team_ids if tid in (home_team_id, away_team_id)]
                    if valid_team_ids:
                        team_id = valid_team_ids[-1]

                if team_id == home_team_id:
                    team_name = str(match_data["home_team"]["name"])
                elif team_id == away_team_id:
                    team_name = str(match_data["away_team"]["name"])
                else:
                    team_name = "Unknown"

                player_id = safe_int(ev.get("player_id", -1), -1)
                if player_id < 0 and "player_id" in phase_group.columns:
                    phase_player_ids = pd.to_numeric(phase_group["player_id"], errors="coerce").dropna().astype(int).tolist()
                    if phase_player_ids:
                        player_id = phase_player_ids[-1]
                player_name = player_name_map.get(player_id, "")

                goal_t = float(frame_times[idx])
                jump_t = clamp_time_seconds(goal_t - 20.0)

                event_rows.append(
                    {
                        "time": f"{period}H {format_mmss(frame_half_clocks[idx])}",
                        "team": team_name,
                        "player": str(player_name),
                        "event": "Goal",
                        "goal_t": float(goal_t),
                        "jump_t": float(jump_t),
                    }
                )

        # Fallback for files without phase-level goal metadata.
        elif bool(lead_to_goal.any()):
            is_goal_end = event_candidates["end_type_l"].isin(["goal", "own_goal"])
            is_shot = event_candidates["end_type_l"].eq("shot") | event_candidates["event_type_l"].eq("shot")
            fallback_rows = event_candidates[is_goal_end | (is_shot & lead_to_goal)].copy()

            if "frame_start" in fallback_rows.columns:
                fallback_rows["frame_start"] = pd.to_numeric(fallback_rows["frame_start"], errors="coerce")
            else:
                fallback_rows["frame_start"] = np.nan
            if "frame_end" in fallback_rows.columns:
                fallback_rows["frame_end"] = pd.to_numeric(fallback_rows["frame_end"], errors="coerce")
            else:
                fallback_rows["frame_end"] = np.nan

            fallback_rows["jump_frame"] = fallback_rows["frame_end"]
            missing_jump = fallback_rows["jump_frame"].isna()
            fallback_rows.loc[missing_jump, "jump_frame"] = fallback_rows.loc[missing_jump, "frame_start"]
            fallback_rows = fallback_rows[fallback_rows["jump_frame"].notna()].copy()
            fallback_rows["jump_frame"] = fallback_rows["jump_frame"].astype(int)
            fallback_rows = fallback_rows.sort_values("jump_frame")

            for _, ev in fallback_rows.iterrows():
                jump_frame = safe_int(ev.get("jump_frame", -1), -1)
                if jump_frame < 0:
                    continue

                idx = nearest_idx_for_frame(jump_frame)
                period = int(frame_periods[idx]) if len(frame_periods) > idx else safe_int(ev.get("period", 0), 0)
                if period not in (1, 2):
                    continue

                team_id = safe_int(ev.get("team_id", -1), -1)
                if team_id == home_team_id:
                    team_name = str(match_data["home_team"]["name"])
                elif team_id == away_team_id:
                    team_name = str(match_data["away_team"]["name"])
                else:
                    team_name = "Unknown"

                player_id = safe_int(ev.get("player_id", -1), -1)
                player_name = player_name_map.get(player_id, "")

                goal_t = float(frame_times[idx])
                jump_t = clamp_time_seconds(goal_t - 20.0)

                event_rows.append(
                    {
                        "time": f"{period}H {format_mmss(frame_half_clocks[idx])}",
                        "team": team_name,
                        "player": str(player_name),
                        "event": "Goal",
                        "goal_t": float(goal_t),
                        "jump_t": float(jump_t),
                    }
                )

    event_rows = sorted(event_rows, key=lambda r: safe_float(r.get("jump_t", np.nan), np.nan))
    for row in event_rows:
        goal_t = safe_float(row.get("goal_t", np.nan), np.nan)
        if not np.isfinite(goal_t):
            continue
        team_name = str(row.get("team", "")).strip()
        if team_name == home_team_name:
            goal_events_for_score.append((float(goal_t), home_team_id))
        elif team_name == away_team_name:
            goal_events_for_score.append((float(goal_t), away_team_id))

    goal_events_for_score = sorted(goal_events_for_score, key=lambda x: float(x[0]))

    goal_event_override_rows = GOAL_EVENT_OVERRIDES.get(match_id, None)
    if goal_event_override_rows:
        event_rows = []
        goal_events_for_score = []

        for item in goal_event_override_rows:
            team_side = str(item.get("team_side", "")).strip().lower()
            if team_side == "home":
                team_name = home_team_name
                team_id = home_team_id
            elif team_side == "away":
                team_name = away_team_name
                team_id = away_team_id
            else:
                continue

            game_clock_s = parse_minute_notation_to_game_clock_seconds(item.get("minute", ""))
            if not np.isfinite(game_clock_s):
                continue

            goal_t = timeline_from_game_clock(game_clock_s)
            idx = time_to_idx(goal_t)
            period = int(frame_periods[idx]) if len(frame_periods) > idx else 0
            if period not in (1, 2):
                continue

            event_rows.append(
                {
                    "time": f"{period}H {format_mmss(frame_half_clocks[idx])}",
                    "team": team_name,
                    "player": str(item.get("player", "Unknown")),
                    "event": "Goal",
                    "goal_t": float(goal_t),
                    "jump_t": float(clamp_time_seconds(goal_t - 20.0)),
                }
            )
            goal_events_for_score.append((float(goal_t), int(team_id)))

        event_rows = sorted(event_rows, key=lambda r: safe_float(r.get("goal_t", np.nan), np.nan))
        goal_events_for_score = sorted(goal_events_for_score, key=lambda x: float(x[0]))

    target_score_override = FINAL_SCORE_OVERRIDES.get(match_id, None)
    if target_score_override is not None:
        target_home_goals = max(0, safe_int(target_score_override[0], 0))
        target_away_goals = max(0, safe_int(target_score_override[1], 0))

        current_home_goals = sum(1 for _, tid in goal_events_for_score if tid == home_team_id)
        current_away_goals = sum(1 for _, tid in goal_events_for_score if tid == away_team_id)

        missing_home = max(0, target_home_goals - current_home_goals)
        missing_away = max(0, target_away_goals - current_away_goals)
        missing_total = int(missing_home + missing_away)

        if missing_total > 0:
            team_ids_to_add = [home_team_id] * missing_home + [away_team_id] * missing_away
            known_goal_times = [safe_float(gt, np.nan) for gt, _ in goal_events_for_score]
            known_goal_times = [gt for gt in known_goal_times if np.isfinite(gt)]
            last_goal_t = max(known_goal_times) if known_goal_times else (max_time_seconds * 0.85)

            synthetic_start_t = clamp_time_seconds(max(last_goal_t + frame_step_seconds, max_time_seconds * 0.90))
            synthetic_end_t = clamp_time_seconds(max(max_time_seconds - frame_step_seconds, synthetic_start_t))

            if missing_total == 1:
                synthetic_times = [float(synthetic_end_t)]
            else:
                synthetic_times = [float(x) for x in np.linspace(synthetic_start_t, synthetic_end_t, missing_total)]

            for goal_t, team_id in zip(synthetic_times, team_ids_to_add):
                idx = time_to_idx(goal_t)
                period = int(frame_periods[idx]) if len(frame_periods) > idx else 0
                if period not in (1, 2):
                    continue

                if team_id == home_team_id:
                    team_name = str(match_data["home_team"]["name"])
                else:
                    team_name = str(match_data["away_team"]["name"])

                event_rows.append(
                    {
                        "time": f"{period}H {format_mmss(frame_half_clocks[idx])}",
                        "team": team_name,
                        "player": "Unknown",
                        "event": "Goal",
                        "goal_t": float(goal_t),
                        "jump_t": float(clamp_time_seconds(goal_t - 20.0)),
                    }
                )
                goal_events_for_score.append((float(goal_t), int(team_id)))

            event_rows = sorted(event_rows, key=lambda r: safe_float(r.get("jump_t", np.nan), np.nan))
            goal_events_for_score = sorted(goal_events_for_score, key=lambda x: float(x[0]))

    event_table_override_rows = EVENT_TABLE_OVERRIDES.get(match_id, None)
    if event_table_override_rows:
        custom_event_rows: list[dict] = []
        for item in event_table_override_rows:
            team_side = str(item.get("team_side", "")).strip().lower()
            if team_side == "home":
                team_name = home_team_name
            elif team_side == "away":
                team_name = away_team_name
            else:
                team_name = "Unknown"

            game_clock_s = parse_mmss_to_seconds(item.get("time", ""))
            if not np.isfinite(game_clock_s):
                continue

            goal_t = timeline_from_game_clock(game_clock_s)
            idx = time_to_idx(goal_t)
            period = int(frame_periods[idx]) if len(frame_periods) > idx else 0
            if period not in (1, 2):
                continue

            custom_event_rows.append(
                {
                    "time": f"{period}H {format_mmss(frame_half_clocks[idx])}",
                    "team": team_name,
                    "player": str(item.get("player", "")),
                    "event": str(item.get("event", "Event")),
                    "goal_t": float(goal_t),
                    "jump_t": float(clamp_time_seconds(goal_t - 20.0)),
                }
            )

        if custom_event_rows:
            event_rows = sorted(custom_event_rows, key=lambda r: safe_float(r.get("goal_t", np.nan), np.nan))

    def get_metric_row(idx: int) -> dict:
        if idx in metrics_cache:
            return metrics_cache[idx]

        frame_number = frame_numbers[idx]
        if frame_number in precomputed_by_frame:
            row = dict(precomputed_by_frame[frame_number])
        else:
            fr = frame_map[frame_number]
            row = frame_metrics(fr, events_df, match_data)

        row["t_seconds"] = float(frame_times[idx]) if len(frame_times) > idx else safe_float(row.get("t_seconds", np.nan), np.nan)
        row["period"] = int(frame_periods[idx]) if len(frame_periods) > idx else safe_int(row.get("period", 0), 0)
        row["game_clock_s"] = float(frame_game_clocks[idx]) if len(frame_game_clocks) > idx else np.nan
        row["half_clock_s"] = float(frame_half_clocks[idx]) if len(frame_half_clocks) > idx else np.nan
        if not row.get("timestamp"):
            row["timestamp"] = frame_map[frame_number].get("timestamp", "")
        metrics_cache[idx] = row
        return row

    def warm_cache_range(start_idx: int, end_idx: int, max_new: int) -> int:
        if not frame_numbers or end_idx < start_idx or max_new <= 0:
            return 0
        start_idx = max(0, start_idx)
        end_idx = min(len(frame_numbers) - 1, end_idx)
        added = 0
        for i in range(start_idx, end_idx + 1):
            if i not in metrics_cache:
                get_metric_row(i)
                added += 1
                if added >= max_new:
                    break
        return added

    def count_cached_in_range(start_idx: int, end_idx: int) -> int:
        if not frame_numbers or end_idx < start_idx:
            return 0
        start_idx = max(0, start_idx)
        end_idx = min(len(frame_numbers) - 1, end_idx)
        return sum(1 for i in range(start_idx, end_idx + 1) if i in metrics_cache)

    def contiguous_cached_upto(start_idx: int, end_idx: int) -> int:
        if not frame_numbers or end_idx < start_idx:
            return start_idx - 1
        start_idx = max(0, start_idx)
        end_idx = min(len(frame_numbers) - 1, end_idx)
        upto = start_idx - 1
        for i in range(start_idx, end_idx + 1):
            if i in metrics_cache:
                upto = i
            else:
                break
        return upto

    def build_hist_from_cache(start_idx: int, end_idx: int) -> pd.DataFrame:
        if not frame_numbers or end_idx < start_idx:
            return pd.DataFrame(columns=["t_seconds", "flow_proxy", "team_side"])
        rows = [metrics_cache[i] for i in range(start_idx, end_idx + 1) if i in metrics_cache]
        return pd.DataFrame(rows)

    app = Dash(__name__, update_title=None)
    app.title = "Switzerland vs Germany Flow Dashboard"

    app.layout = html.Div(
        id="app-root",
        style={"fontFamily": "Verdana, sans-serif", "padding": "12px", "backgroundColor": "#0b1220", "color": "#e5e7eb", "minHeight": "100vh"},
        children=[
            html.Div(
                id="dashboard-header",
                style={"position": "relative", "display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "10px"},
                children=[
                    html.H2("Interactive Flow Dashboard", id="dashboard-title", style={"margin": "0", "color": "#f8fafc"}),
                    html.Button("Light Mode", id="btn-theme-toggle", n_clicks=0, style={"position": "absolute", "right": "0", "padding": "6px 12px", "borderRadius": "8px", "border": "1px solid #cbd5e1", "backgroundColor": "#f8fafc", "color": "#111827", "fontWeight": "bold"}),
                ],
            ),
            dcc.Interval(id="play-interval", interval=1000, n_intervals=0, disabled=True),
            dcc.Store(id="playback-time", data=0.0),
            dcc.Store(id="theme-mode", data="dark"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1.55fr 1fr", "gap": "10px", "marginTop": "10px"},
                children=[
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                        children=[
                            html.Div(
                                id="scoreboard-box",
                                style={
                                    "alignSelf": "center",
                                    "backgroundColor": "rgba(15,23,42,0.92)",
                                    "color": "#f8fafc",
                                    "border": "1px solid #334155",
                                    "borderRadius": "8px",
                                    "padding": "6px 12px",
                                    "minWidth": "220px",
                                    "textAlign": "center",
                                    "boxShadow": "0 2px 6px rgba(0,0,0,0.25)",
                                },
                                children=[
                                    html.Div(f"{home_abbr} 0 - 0 {away_abbr}", id="scoreboard-main", style={"fontWeight": "bold", "fontSize": "16px", "lineHeight": "1.2"}),
                                    html.Div("00:00", id="scoreboard-clock", style={"fontSize": "12px", "color": "#cbd5e1", "marginTop": "2px"}),
                                ],
                            ),
                            dcc.Graph(id="pitch-graph", config={"displayModeBar": False}, style={"height": "430px"}),
                        ],
                    ),
                    dcc.Graph(id="flow-graph", config={"displayModeBar": False}, style={"height": "430px"}),
                ],
            ),
            html.Div(
                id="controls-panel",
                style={"marginTop": "10px", "padding": "12px", "backgroundColor": "#0f172a", "border": "1px solid #0b1220", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.18)", "maxWidth": "980px", "marginLeft": "auto", "marginRight": "auto"},
                children=[
                    html.Div(
                        style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap", "marginBottom": "8px"},
                        children=[
                            html.Button("<<10", id="btn-back-10", n_clicks=0, style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #243041", "backgroundColor": "#1f2937", "color": "#f8fafc", "fontWeight": "bold"}),
                            html.Button("<1", id="btn-back", n_clicks=0, style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #243041", "backgroundColor": "#1f2937", "color": "#f8fafc", "fontWeight": "bold"}),
                            html.Button("Play", id="btn-play-pause", n_clicks=0, style={"padding": "7px 16px", "borderRadius": "999px", "border": "1px solid #2563eb", "backgroundColor": "#2563eb", "color": "white", "fontWeight": "bold"}),
                            html.Button("1>", id="btn-forward", n_clicks=0, style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #243041", "backgroundColor": "#1f2937", "color": "#f8fafc", "fontWeight": "bold"}),
                            html.Button("10>>", id="btn-forward-10", n_clicks=0, style={"padding": "6px 12px", "borderRadius": "6px", "border": "1px solid #243041", "backgroundColor": "#1f2937", "color": "#f8fafc", "fontWeight": "bold"}),
                            html.Div("Speed", style={"marginLeft": "8px", "fontWeight": "bold", "color": "#e2e8f0"}),
                            dcc.Dropdown(
                                id="speed-dropdown",
                                options=[
                                    {"label": "0.5x", "value": 0.5},
                                    {"label": "0.75x", "value": 0.75},
                                    {"label": "1x", "value": 1.0},
                                    {"label": "2x", "value": 2.0},
                                ],
                                value=1.0,
                                clearable=False,
                                style={"width": "120px", "color": "#111827"},
                            ),
                        ],
                    ),
                    html.Div("Match Time (minutes)", id="controls-time-label", style={"fontWeight": "bold", "color": "#cbd5e1", "marginBottom": "4px"}),
                    dcc.Slider(
                        id="time-slider",
                        min=0.0,
                        max=max_time_seconds,
                        value=0.0,
                        step=1.0,
                        marks=time_slider_marks_dark,
                    ),
                    html.Button("key-left", id="btn-key-left", n_clicks=0, style={"display": "none"}),
                    html.Button("key-right", id="btn-key-right", n_clicks=0, style={"display": "none"}),
                ],
            ),
            html.Div(id="metric-cards", style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(96px, 132px))", "justifyContent": "center", "gap": "8px", "marginTop": "8px"}),
            html.Div(
                id="options-panel",
                style={"marginTop": "8px"},
                children=[
                    html.H4("Top Options (Current Time)", style={"margin": "8px 0"}),
                    dash_table.DataTable(
                        id="options-table",
                        columns=[
                            {"name": "Action", "id": "action"},
                            {"name": "Target", "id": "target"},
                            {"name": "Utility", "id": "utility"},
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={"padding": "6px", "fontFamily": "Verdana, sans-serif", "fontSize": "13px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#eceff3"},
                    ),
                ],
            ),
            html.Div(
                id="events-panel",
                style={"marginTop": "10px", "backgroundColor": "#111827", "border": "1px solid #253045", "borderRadius": "8px", "padding": "8px", "color": "#e5e7eb", "maxWidth": "620px", "marginLeft": "auto", "marginRight": "auto"},
                children=[
                    html.H4("Events", style={"margin": "4px 0 8px 0"}),
                    dash_table.DataTable(
                        id="events-table",
                        columns=[
                            {"name": "Time", "id": "time"},
                            {"name": "Team", "id": "team"},
                            {"name": "Player", "id": "player"},
                            {"name": "Event", "id": "event"},
                        ],
                        data=event_rows,
                        style_table={"overflowY": "auto", "maxHeight": "240px", "overflowX": "auto"},
                        style_cell={"padding": "6px", "fontFamily": "Verdana, sans-serif", "fontSize": "12px", "textAlign": "left"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#eceff3"},
                    ),
                ],
            ),
        ],
    )

    def card(title: str, value: str, theme_mode: str = "light") -> html.Div:
        is_dark = get_theme_mode(theme_mode) == "dark"
        return html.Div(
            style={
                "backgroundColor": "#111827" if is_dark else "white",
                "border": "1px solid #334155" if is_dark else "1px solid #d6d9df",
                "borderRadius": "8px",
                "padding": "6px",
                "boxShadow": "0 1px 2px rgba(0,0,0,0.06)",
            },
            children=[
                html.Div(title, style={"fontSize": "11px", "color": "#94a3b8" if is_dark else "#555"}),
                html.Div(value, style={"fontSize": "16px", "fontWeight": "bold", "color": "#f8fafc" if is_dark else "#111827"}),
            ],
        )

    @app.callback(
        Output("theme-mode", "data"),
        Output("btn-theme-toggle", "children"),
        Input("btn-theme-toggle", "n_clicks"),
        State("theme-mode", "data"),
        prevent_initial_call=True,
    )
    def toggle_theme_mode(n_clicks, current_mode):
        del n_clicks
        mode = get_theme_mode(current_mode)
        new_mode = "dark" if mode == "light" else "light"
        btn_text = "Light Mode" if new_mode == "dark" else "Dark Mode"
        return new_mode, btn_text

    @app.callback(
        Output("playback-time", "data"),
        Output("time-slider", "value"),
        Output("play-interval", "disabled"),
        Output("play-interval", "interval"),
        Output("btn-play-pause", "children"),
        Input("btn-play-pause", "n_clicks"),
        Input("btn-back", "n_clicks"),
        Input("btn-forward", "n_clicks"),
        Input("btn-back-10", "n_clicks"),
        Input("btn-forward-10", "n_clicks"),
        Input("btn-key-left", "n_clicks"),
        Input("btn-key-right", "n_clicks"),
        Input("events-table", "active_cell"),
        Input("flow-graph", "clickData"),
        Input("play-interval", "n_intervals"),
        Input("speed-dropdown", "value"),
        Input("time-slider", "value"),
        State("events-table", "data"),
        State("playback-time", "data"),
        State("play-interval", "disabled"),
        prevent_initial_call=True,
    )
    def control_frame(toggle_clicks, back_clicks, forward_clicks, back10_clicks, forward10_clicks, key_left_clicks, key_right_clicks, events_active_cell, flow_click_data, n_intervals, speed, slider_value, events_table_data, playback_time, interval_disabled):
        del toggle_clicks, back_clicks, forward_clicks, back10_clicks, forward10_clicks, key_left_clicks, key_right_clicks, n_intervals

        trig = callback_context.triggered_id
        current_t = clamp_time_seconds(float(playback_time) if playback_time is not None else 0.0)

        speed = float(speed) if speed is not None else 1.0
        ms = int(max(20, round((frame_step_seconds * 1000.0) / speed)))

        if trig == "time-slider":
            current_t = clamp_time_seconds(float(slider_value) if slider_value is not None else 0.0)
        elif trig == "btn-play-pause":
            interval_disabled = not bool(interval_disabled)
        elif trig == "btn-back":
            current_t -= 1.0
        elif trig == "btn-forward":
            current_t += 1.0
        elif trig == "btn-back-10":
            current_t -= 10.0
        elif trig == "btn-forward-10":
            current_t += 10.0
        elif trig == "btn-key-left":
            current_t -= frame_step_seconds
        elif trig == "btn-key-right":
            current_t += frame_step_seconds
        elif trig == "events-table" and events_active_cell is not None and isinstance(events_table_data, list):
            row_idx = safe_int(events_active_cell.get("row", -1), -1)
            if 0 <= row_idx < len(events_table_data):
                jump_t = safe_float(events_table_data[row_idx].get("jump_t", np.nan), np.nan)
                if np.isfinite(jump_t):
                    current_t = float(jump_t)
                    interval_disabled = True
        elif trig == "flow-graph" and flow_click_data and flow_click_data.get("points"):
            try:
                clicked_t = float(flow_click_data["points"][0].get("x", 0.0))
                current_t = clicked_t
            except (TypeError, ValueError, KeyError, IndexError):
                pass
        elif trig == "play-interval" and not interval_disabled:
            current_t += frame_step_seconds

        current_t = clamp_time_seconds(current_t)
        slider_seconds = float(round(current_t))
        play_pause_text = "Play" if bool(interval_disabled) else "Pause"
        return current_t, slider_seconds, interval_disabled, ms, play_pause_text

    @app.callback(
        Output("pitch-graph", "figure"),
        Output("flow-graph", "figure"),
        Output("metric-cards", "children"),
        Output("options-table", "data"),
        Output("scoreboard-main", "children"),
        Output("scoreboard-clock", "children"),
        Output("options-panel", "style"),
        Output("app-root", "style"),
        Output("dashboard-title", "style"),
        Output("controls-panel", "style"),
        Output("controls-time-label", "style"),
        Output("events-panel", "style"),
        Output("options-table", "style_header"),
        Output("options-table", "style_cell"),
        Output("options-table", "style_data"),
        Output("events-table", "style_header"),
        Output("events-table", "style_cell"),
        Output("events-table", "style_data"),
        Output("btn-theme-toggle", "style"),
        Output("speed-dropdown", "style"),
        Output("time-slider", "marks"),
        Input("playback-time", "data"),
        Input("play-interval", "disabled"),
        Input("theme-mode", "data"),
    )
    def update_views(time_value: float, interval_disabled: bool, theme_mode_value: str):
        theme_mode = get_theme_mode(theme_mode_value)
        is_dark = theme_mode == "dark"
        is_paused = bool(interval_disabled)

        root_style = {
            "fontFamily": "Verdana, sans-serif",
            "padding": "12px",
            "backgroundColor": "#0b1220" if is_dark else "#f7f8fa",
            "color": "#e5e7eb" if is_dark else "#111827",
            "minHeight": "100vh",
        }
        title_style = {"margin": "0", "color": "#f8fafc" if is_dark else "#111827"}
        controls_style = {
            "marginTop": "10px",
            "padding": "12px",
            "backgroundColor": "#111827" if is_dark else "#ffffff",
            "border": "1px solid #253045" if is_dark else "1px solid #d6d9df",
            "borderRadius": "10px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.18)",
            "maxWidth": "980px",
            "marginLeft": "auto",
            "marginRight": "auto",
        }
        controls_time_label_style = {"fontWeight": "bold", "color": "#cbd5e1" if is_dark else "#334155", "marginBottom": "4px"}
        events_panel_style = {
            "marginTop": "10px",
            "backgroundColor": "#111827" if is_dark else "white",
            "border": "1px solid #253045" if is_dark else "1px solid #d6d9df",
            "borderRadius": "8px",
            "padding": "8px",
            "color": "#e5e7eb" if is_dark else "#111827",
            "maxWidth": "620px",
            "marginLeft": "auto",
            "marginRight": "auto",
        }
        options_panel_base = {
            "marginTop": "8px",
            "backgroundColor": "#111827" if is_dark else "white",
            "border": "1px solid #253045" if is_dark else "1px solid #d6d9df",
            "borderRadius": "8px",
            "padding": "8px",
            "color": "#e5e7eb" if is_dark else "#111827",
        }
        table_header_style = {
            "fontWeight": "bold",
            "backgroundColor": "#1f2937" if is_dark else "#eceff3",
            "color": "#f8fafc" if is_dark else "#111827",
        }
        table_cell_style = {
            "padding": "6px",
            "fontFamily": "Verdana, sans-serif",
            "fontSize": "12px",
            "textAlign": "left",
            "backgroundColor": "#0f172a" if is_dark else "white",
            "color": "#e5e7eb" if is_dark else "#111827",
            "border": "1px solid #1f2a44" if is_dark else "1px solid #e5e7eb",
        }
        table_data_style = {
            "backgroundColor": "#0f172a" if is_dark else "white",
            "color": "#e5e7eb" if is_dark else "#111827",
        }
        theme_btn_style = {
            "position": "absolute",
            "right": "0",
            "padding": "6px 12px",
            "borderRadius": "8px",
            "border": "1px solid #cbd5e1" if is_dark else "1px solid #475569",
            "backgroundColor": "#f8fafc" if is_dark else "#1f2937",
            "color": "#111827" if is_dark else "#f8fafc",
            "fontWeight": "bold",
        }
        speed_dropdown_style = {"width": "120px", "color": "#111827"}
        slider_marks = time_slider_marks_dark if is_dark else time_slider_marks_light

        if not frame_numbers:
            empty_fig = go.Figure()
            return (
                empty_fig,
                empty_fig,
                [card("Status", "No Data", theme_mode=theme_mode)],
                [],
                f"{home_abbr} 0 - 0 {away_abbr}",
                "00:00",
                {"display": "none"},
                root_style,
                title_style,
                controls_style,
                controls_time_label_style,
                events_panel_style,
                table_header_style,
                table_cell_style,
                table_data_style,
                table_header_style,
                table_cell_style,
                table_data_style,
                theme_btn_style,
                speed_dropdown_style,
                slider_marks,
            )

        requested_t = clamp_time_seconds(float(time_value) if time_value is not None else 0.0)
        idx = time_to_idx(requested_t)

        frame_number = frame_numbers[idx]
        frame_data = frame_map[frame_number]
        row = get_metric_row(idx)

        option_rows: list[dict] = []
        run_rows: list[dict] = []
        shot_info: dict | None = None

        cached_pass = parse_json_list(row.get("pass_options_json", "[]"))
        if cached_pass:
            sx = safe_float(row.get("carrier_x", np.nan), np.nan)
            sy = safe_float(row.get("carrier_y", np.nan), np.nan)
            ball = frame_data.get("ball_data", None)
            if (not np.isfinite(sx) or not np.isfinite(sy)) and ball is not None:
                sx = safe_float(ball.get("x", np.nan), np.nan)
                sy = safe_float(ball.get("y", np.nan), np.nan)

            for pr in cached_pass:
                rid = safe_int(pr.get("receiver_id", -1), -1)
                if rid < 0:
                    continue
                option_rows.append(
                    {
                        "receiver_id": rid,
                        "receiver_name": player_name_map.get(rid, str(rid)),
                        "utility": safe_float(pr.get("utility", np.nan), np.nan),
                        "target_x": safe_float(pr.get("target_x", pr.get("receiver_x", np.nan)), np.nan),
                        "target_y": safe_float(pr.get("target_y", pr.get("receiver_y", np.nan)), np.nan),
                        "start_x": sx,
                        "start_y": sy,
                        "is_best": False,
                    }
                )
            if option_rows:
                best_i = int(np.nanargmax([safe_float(o.get("utility", np.nan), -np.inf) for o in option_rows]))
                option_rows[best_i]["is_best"] = True
        elif is_paused:
            # Fallback is expensive, keep it only for paused inspection mode.
            options_df = top_options_for_frame(frame_data, events_df, player_name_map)
            option_rows = options_df.to_dict("records") if not options_df.empty else []

        cached_runs = parse_json_list(row.get("run_options_json", "[]"))
        if cached_runs:
            for rr in cached_runs:
                run_val = safe_float(rr.get("run_value", np.nan), np.nan)
                if not np.isfinite(run_val):
                    ctrl = safe_float(rr.get("control_capacity", np.nan), np.nan)
                    net = safe_float(rr.get("run_net_xthreat", rr.get("destination_xthreat", np.nan)), np.nan)
                    if np.isfinite(ctrl) and np.isfinite(net):
                        run_val = min(ctrl, net)
                run_rows.append(
                    {
                        "run_type": str(rr.get("run_type", "run")),
                        "target_x": safe_float(rr.get("target_x", np.nan), np.nan),
                        "target_y": safe_float(rr.get("target_y", np.nan), np.nan),
                        "run_value": run_val,
                    }
                )

        shot_cached = parse_json_obj(row.get("shot_option_json", ""))
        if shot_cached is not None:
            shot_info = {
                "start_x": safe_float(shot_cached.get("start_x", shot_cached.get("ball_x", np.nan)), np.nan),
                "start_y": safe_float(shot_cached.get("start_y", shot_cached.get("ball_y", np.nan)), np.nan),
                "target_x": safe_float(shot_cached.get("target_x", shot_cached.get("goal_x", np.nan)), np.nan),
                "target_y": safe_float(shot_cached.get("target_y", shot_cached.get("goal_y", np.nan)), np.nan),
                "xg": safe_float(shot_cached.get("xg", shot_cached.get("shot_probability", np.nan)), np.nan),
            }
        if (shot_info is None or not np.isfinite(safe_float(shot_info.get("xg", np.nan), np.nan))) and is_paused:
            shot_info = shot_option_for_frame(frame_data, match_data)

        pitch_fig = build_pitch_figure(
            frame_data,
            match_data,
            player_number_map,
            player_name_map,
            player_team_map,
            option_rows=option_rows,
            run_rows=run_rows,
            shot_info=shot_info,
            carrier_id=safe_int(row.get("carrier_id", -1), -1) if safe_int(row.get("carrier_id", -1), -1) >= 0 else None,
            best_pass_offensive_return=safe_float(row.get("best_pass_offensive_return", np.nan), np.nan),
            theme_mode=theme_mode,
        )

        history_window_seconds = 30.0
        history_start_t = max(0.0, float(row["t_seconds"]) - history_window_seconds)
        history_start_idx = time_to_idx(history_start_t)
        warm_cache_range(history_start_idx, idx, max_new=(40 if not is_paused else 90))
        flow_hist = build_hist_from_cache(history_start_idx, idx)
        if flow_hist.empty and idx in metrics_cache:
            flow_hist = pd.DataFrame([metrics_cache[idx]])
        elif not is_paused and len(flow_hist) > 520:
            stride = max(1, int(math.ceil(len(flow_hist) / 520.0)))
            flow_hist = flow_hist.iloc[::stride].copy()

        target_idx = min(len(frame_numbers) - 1, idx + 50)
        warm_cache_range(idx + 1, target_idx, max_new=(8 if not is_paused else 18))

        flow_fig = build_flow_figure(flow_hist, match_data, current_t=float(row["t_seconds"]), theme_mode=theme_mode)

        if not is_paused:
            # Keep playback lightweight: disable hover processing while playing.
            for fig_obj in (pitch_fig, flow_fig):
                fig_obj.update_layout(hovermode=False)
                for tr in fig_obj.data:
                    tr.hoverinfo = "skip"
                    tr.hovertemplate = None

        best_pass = safe_float(row.get("best_pass_utility", np.nan), np.nan)
        shot_xg = safe_float(row.get("shot_xg", np.nan), np.nan)
        frame_flow_capacity = safe_float(row.get("frame_flow_capacity", np.nan), np.nan)

        cards = [
            card("Best Action", title_case_label(row.get("best_action", "none")), theme_mode=theme_mode),
            card("Pass Utility", format_percent(best_pass), theme_mode=theme_mode),
            card("Shot XG", format_percent(shot_xg), theme_mode=theme_mode),
            card("Flow Capacity", format_percent(frame_flow_capacity), theme_mode=theme_mode),
            card("Pass Options", str(int(safe_int(row.get("n_options", 0), 0))), theme_mode=theme_mode),
            card("Team In Possession", title_case_label(row.get("team_side", "none")), theme_mode=theme_mode),
        ]

        option_actions: list[dict] = []
        for o in option_rows:
            util = safe_float(o.get("utility", np.nan), np.nan)
            if not np.isfinite(util):
                continue
            option_actions.append(
                {
                    "action": "Pass",
                    "target": str(o.get("receiver_name", "Unknown")),
                    "utility_num": util,
                }
            )

        best_run_action = None
        for r in run_rows:
            util = safe_float(r.get("run_value", np.nan), np.nan)
            if not np.isfinite(util):
                continue
            candidate = {
                "action": "Run",
                "target": str(r.get("run_type", "run")),
                "utility_num": util,
            }
            if best_run_action is None or float(candidate["utility_num"]) > float(best_run_action["utility_num"]):
                best_run_action = candidate
        if best_run_action is not None:
            option_actions.append(best_run_action)

        if shot_info is not None:
            shot_util = safe_float(shot_info.get("xg", np.nan), np.nan)
            if np.isfinite(shot_util):
                option_actions.append(
                    {
                        "action": "Shot",
                        "target": "Goal",
                        "utility_num": shot_util,
                    }
                )

        option_actions = sorted(option_actions, key=lambda r: float(r["utility_num"]), reverse=True)
        options_data = [
            {
                "action": title_case_label(r["action"]),
                "target": str(r["target"]) if str(r["target"]).strip() else "Unknown",
                "utility": format_percent(float(r["utility_num"])),
            }
            for r in option_actions
        ]

        panel_style = options_panel_base if bool(interval_disabled) else {"display": "none"}
        if not bool(interval_disabled):
            options_data = []

        current_timeline = float(row["t_seconds"]) if np.isfinite(safe_float(row.get("t_seconds", np.nan), np.nan)) else requested_t
        home_goals = sum(1 for gt, tid in goal_events_for_score if tid == home_team_id and gt <= current_timeline)
        away_goals = sum(1 for gt, tid in goal_events_for_score if tid == away_team_id and gt <= current_timeline)
        scoreboard_main = f"{home_abbr} {home_goals} - {away_goals} {away_abbr}"

        game_clock_s = float(frame_game_clocks[idx]) if len(frame_game_clocks) > idx else requested_t
        half_clock_s = float(frame_half_clocks[idx]) if len(frame_half_clocks) > idx else requested_t
        period_label = int(frame_periods[idx]) if len(frame_periods) > idx else 1
        del period_label, half_clock_s
        scoreboard_clock = format_mmss(game_clock_s)
        del game_clock_s
        return (
            pitch_fig,
            flow_fig,
            cards,
            options_data,
            scoreboard_main,
            scoreboard_clock,
            panel_style,
            root_style,
            title_style,
            controls_style,
            controls_time_label_style,
            events_panel_style,
            table_header_style,
            table_cell_style,
            table_data_style,
            table_header_style,
            table_cell_style,
            table_data_style,
            theme_btn_style,
            speed_dropdown_style,
            slider_marks,
        )

    return app


def main():
    print("[dashboard] Starting app initialization...")
    app = build_dashboard()
    print("[dashboard] Ready at http://127.0.0.1:8050")
    app.run(debug=False, host="127.0.0.1", port=8050, use_reloader=False)


if __name__ == "__main__":
    main()
