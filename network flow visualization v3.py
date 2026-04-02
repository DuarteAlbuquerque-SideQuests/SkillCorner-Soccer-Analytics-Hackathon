import math
import argparse
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import pandas as pd
import numpy as np
import json
from pathlib import Path
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import FancyArrowPatch

# Importing all data

DATA_DIR = Path(__file__).resolve().parent

dynamic_events_path = DATA_DIR / "match_data" / "dynamicevents.csv"
match_path = DATA_DIR / "match_data" / "match.json"
phases_path = DATA_DIR / "match_data" / "phasesdata.csv"
physical_path = DATA_DIR / "match_data" / "physicaldata.csv"
tracking_path = DATA_DIR / "match_data" / "trackingdata.jsonl"

# Run edge mode for SOURCE->RUN capacity in the frame flow graph.
# - "run_value": cap = max(run_net_xthreat, 0) (default)
# - "control_capacity": cap = control_capacity (legacy)
RUN_SOURCE_EDGE_MODE = "run_value"

# Run success-pressure calibration from nearby defender counts.
RUN_SMALL_PRESSURE_PER_DEFENDER = 0.03

# Run geometry: evaluate value at speed-based reachability, draw arrows at fixed length.
RUN_EVAL_TIME_SECONDS = 1.5
RUN_DISPLAY_DISTANCE_METERS = 7.0
RUN_EVAL_SPEED_FALLBACK_VMAX_FRACTION = 0.55

print("Loading CSV files...")

df_events = pd.read_csv(dynamic_events_path)
df_phases = pd.read_csv(phases_path)
df_physical = pd.read_csv(physical_path)

print("CSV files loaded.")

print("Loading match metadata...")

with open(match_path, "r", encoding="utf-8") as f:
    match_data = json.load(f)

print("Match metadata loaded.")

print("Loading tracking data (this may take a while)...")

tracking_data = []

with open(tracking_path, "r", encoding="utf-8") as f:
    for line in f:
        tracking_data.append(json.loads(line))

print("Tracking data loaded.")

print("\n=== DATA OVERVIEW ===")

print(f"Events shape: {df_events.shape}")
print(f"Phases shape: {df_phases.shape}")
print(f"Physical shape: {df_physical.shape}")
print(f"Tracking records: {len(tracking_data)}")

#print(tracking_data[150])

# -----------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pandas as pd

# ============================================================
# PLAYER / TEAM LOOKUP TABLES
# ============================================================

def build_player_lookup(match_data: dict) -> pd.DataFrame:
    """
    Build a player lookup table from match metadata.

    Returns a DataFrame with one row per player and the following fields:
    - player_id
    - short_name
    - team_id
    - team_name
    - jersey_number
    - position
    """
    rows = []

    home_team_id = match_data["home_team"]["id"]
    away_team_id = match_data["away_team"]["id"]
    home_team_name = match_data["home_team"]["name"]
    away_team_name = match_data["away_team"]["name"]

    for player in match_data["players"]:
        team_id = player["team_id"]

        if team_id == home_team_id:
            team_name = home_team_name
        elif team_id == away_team_id:
            team_name = away_team_name
        else:
            team_name = "Unknown"

        rows.append({
            "player_id": player["id"],
            "short_name": player["short_name"],
            "team_id": team_id,
            "team_name": team_name,
            "jersey_number": player["number"],
            "position": player["player_role"]["acronym"]
        })

    return pd.DataFrame(rows)


player_lookup = build_player_lookup(match_data)

# Optional dictionaries for fast lookup
player_name_map = dict(zip(player_lookup["player_id"], player_lookup["short_name"]))
player_team_map = dict(zip(player_lookup["player_id"], player_lookup["team_id"]))
player_number_map = dict(zip(player_lookup["player_id"], player_lookup["jersey_number"]))

home_team_id = match_data["home_team"]["id"]
away_team_id = match_data["away_team"]["id"]

home_team_name = match_data["home_team"]["name"]
away_team_name = match_data["away_team"]["name"]

home_color = match_data["home_team_kit"]["jersey_color"]
away_color = match_data["away_team_kit"]["jersey_color"]

pitch_length = match_data["pitch_length"]   # e.g. 105
pitch_width = match_data["pitch_width"]     # e.g. 68


# ============================================================
# NEW: FAST POSSESSION LOOKUP ARRAY & RESOLVER
# ============================================================
print("Building efficient possession lookup array from phases of play...")

# Get the maximum frame number to size our array
max_frame = int(df_phases["frame_end"].max())

# Initialize with -1 (meaning no team officially listed in possession for that frame)
frame_team_array = np.full(max_frame + 1, -1, dtype=np.int32)

# Vectorize the range filling
for _, row in df_phases.iterrows():
    f_start = int(row["frame_start"])
    f_end = int(row["frame_end"])
    team_id = int(row["team_in_possession_id"]) if pd.notna(row["team_in_possession_id"]) else -1
    
    # NumPy slice assignment is incredibly fast
    frame_team_array[f_start : f_end + 1] = team_id

print("Possession lookup array built successfully.")


def get_resolved_ball_carrier(frame_data: dict, frame_team_array: np.ndarray, player_team_map: dict) -> int | None:
    """
    Finds the player in possession for the current frame.
    1. Returns SkillCorner's AI detected carrier if available.
    2. If missing, finds the player from the team in possession who is closest to the ball.
    3. NEW: The closest player MUST be within 1 meter of the ball.
    """
    # 1. Try to use SkillCorner's native carrier first
    possession_info = frame_data.get("possession")
    if possession_info:
        ball_carrier_id = possession_info.get("player_id")
        if ball_carrier_id is not None:
            return int(ball_carrier_id)
            
    # 2. Fallback: Find closest player from the team possessing the ball
    ball = frame_data.get("ball_data")
    current_frame = frame_data.get("frame")
    
    if ball is not None and current_frame is not None:
        # Instant O(1) array lookup
        if current_frame < len(frame_team_array):
            team_poss_id = frame_team_array[current_frame]
        else:
            team_poss_id = -1
            
        if team_poss_id != -1:
            bx = float(ball["x"])
            by = float(ball["y"])
            
            min_dist = float('inf')
            closest_player = None
            
            # Loop through the players in the frame
            for p in frame_data.get("player_data", []):
                p_id = p.get("player_id")
                # Player must be detected and from the possessing team
                #if p.get("is_detected") and player_team_map.get(p_id) == team_poss_id:
                # Player is not detected here.
                if player_team_map.get(p_id) == team_poss_id:
                    dist = math.hypot(p["x"] - bx, p["y"] - by)
                    if dist < min_dist:
                        min_dist = dist
                        closest_player = p_id
                        
            # 3. SECONDARY CHECK: Only assign possession if within 2.0 meter
            if closest_player is not None and min_dist <= 2.0:
                return closest_player
            else:
                return None  # Ball is loose / mid-air
            
    return None

# ============================================================
# PITCH DRAWING
# ============================================================

def draw_pitch(ax, pitch_length=105, pitch_width=68):
    """
    Draw a simple 2D football pitch centered at (0, 0).

    SkillCorner tracking coordinates are typically centered at midfield:
    - x runs from approximately -pitch_length/2 to +pitch_length/2
    - y runs from approximately -pitch_width/2 to +pitch_width/2
    """
    half_length = pitch_length / 2
    half_width = pitch_width / 2

    # Outer boundaries
    ax.add_patch(
        Rectangle(
            (-half_length, -half_width),
            pitch_length,
            pitch_width,
            fill=False,
            edgecolor="black",
            linewidth=2
        )
    )

    # Halfway line
    ax.plot([0, 0], [-half_width, half_width], color="black", linewidth=1.5)

    # Center circle
    center_circle = Circle((0, 0), 9.15, fill=False, edgecolor="black", linewidth=1.5)
    ax.add_patch(center_circle)

    # Center spot
    ax.scatter(0, 0, color="black", s=15)

    # Penalty areas
    penalty_area_length = 16.5
    penalty_area_width = 40.32

    # Left penalty area
    ax.add_patch(
        Rectangle(
            (-half_length, -penalty_area_width / 2),
            penalty_area_length,
            penalty_area_width,
            fill=False,
            edgecolor="black",
            linewidth=1.5
        )
    )

    # Right penalty area
    ax.add_patch(
        Rectangle(
            (half_length - penalty_area_length, -penalty_area_width / 2),
            penalty_area_length,
            penalty_area_width,
            fill=False,
            edgecolor="black",
            linewidth=1.5
        )
    )

    # 6-yard boxes
    six_yard_length = 5.5
    six_yard_width = 18.32

    ax.add_patch(
        Rectangle(
            (-half_length, -six_yard_width / 2),
            six_yard_length,
            six_yard_width,
            fill=False,
            edgecolor="black",
            linewidth=1.5
        )
    )

    ax.add_patch(
        Rectangle(
            (half_length - six_yard_length, -six_yard_width / 2),
            six_yard_length,
            six_yard_width,
            fill=False,
            edgecolor="black",
            linewidth=1.5
        )
    )

    # Penalty spots
    ax.scatter(-half_length + 11, 0, color="black", s=15)
    ax.scatter(half_length - 11, 0, color="black", s=15)

    # Formatting
    ax.set_xlim(-half_length - 3, half_length + 3)
    ax.set_ylim(-half_width - 3, half_width + 3)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Tracking Frame")
    ax.grid(False)


# ============================================================
# FRAME PARSING
# ============================================================

def frame_to_player_dataframe(frame_data: dict, player_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one tracking frame into a player-level DataFrame.

    Expected frame_data structure:
    - frame_data["player_data"] is a list of dicts with:
      - x
      - y
      - player_id
      - is_detected
    """
    player_rows = frame_data.get("player_data", [])
    df_frame_players = pd.DataFrame(player_rows).copy()

    # Some early or edge frames can have empty player_data, which creates
    # an empty DataFrame with no columns and breaks merge on player_id.
    for col in ["player_id", "x", "y", "is_detected"]:
        if col not in df_frame_players.columns:
            df_frame_players[col] = np.nan

    df_frame_players = df_frame_players.merge(
        player_lookup,
        how="left",
        left_on="player_id",
        right_on="player_id"
    )

    return df_frame_players


# ============================================================
# FRAME PLOTTING
# ============================================================

def plot_tracking_frame(
    frame_data: dict,
    player_lookup: pd.DataFrame,
    match_data: dict,
    show_labels: bool = True,
    label_mode: str = "number",   # "number", "name", or "player_id"
    show_undetected: bool = True,
    highlight_possession: bool = True,
    figsize=(14, 9)
):
    """
    Plot a single tracking frame in 2D.

    Parameters
    ----------
    frame_data : dict
        One frame from tracking_data.
    player_lookup : pd.DataFrame
        Output of build_player_lookup(match_data).
    match_data : dict
        Match metadata dictionary.
    show_labels : bool
        Whether to label players.
    label_mode : str
        One of {"number", "name", "player_id"}.
    show_undetected : bool
        Whether to display players whose positions were not detected.
    highlight_possession : bool
        Whether to highlight the player in possession.
    figsize : tuple
        Figure size.
    """
    home_team_id = match_data["home_team"]["id"]
    away_team_id = match_data["away_team"]["id"]
    home_color = match_data["home_team_kit"]["jersey_color"]
    away_color = match_data["away_team_kit"]["jersey_color"]
    pitch_length = match_data["pitch_length"]
    pitch_width = match_data["pitch_width"]

    df_players = frame_to_player_dataframe(frame_data, player_lookup)

    if not show_undetected:
        df_players = df_players[df_players["is_detected"] == True].copy()

    fig, ax = plt.subplots(figsize=figsize)
    draw_pitch(ax, pitch_length=pitch_length, pitch_width=pitch_width)

    # Split by team
    df_home = df_players[df_players["team_id"] == home_team_id].copy()
    df_away = df_players[df_players["team_id"] == away_team_id].copy()
    df_unknown = df_players[df_players["team_id"].isna()].copy()

    # Plot home players
    if not df_home.empty:
        ax.scatter(
            df_home["x"],
            df_home["y"],
            s=220,
            c=home_color,
            edgecolors="black",
            linewidths=1.2,
            alpha=df_home["is_detected"].map(lambda v: 1.0 if v else 0.35),
            label=match_data["home_team"]["name"]
        )

    # Plot away players
    if not df_away.empty:
        ax.scatter(
            df_away["x"],
            df_away["y"],
            s=220,
            c=away_color,
            edgecolors="black",
            linewidths=1.2,
            alpha=df_away["is_detected"].map(lambda v: 1.0 if v else 0.35),
            label=match_data["away_team"]["name"]
        )

    # Plot unknown players if any
    if not df_unknown.empty:
        ax.scatter(
            df_unknown["x"],
            df_unknown["y"],
            s=220,
            c="gray",
            edgecolors="black",
            linewidths=1.2,
            alpha=0.7,
            label="Unknown team"
        )

    # Add labels
    if show_labels:
        for _, row in df_players.iterrows():
            if label_mode == "number":
                text_label = str(row["jersey_number"]) if pd.notna(row["jersey_number"]) else str(row["player_id"])
            elif label_mode == "name":
                text_label = row["short_name"] if pd.notna(row["short_name"]) else str(row["player_id"])
            else:
                text_label = str(row["player_id"])

            ax.text(
                row["x"],
                row["y"],
                text_label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if row["team_id"] in [home_team_id, away_team_id] else "black",
                weight="bold"
            )

    # Plot ball
    ball = frame_data.get("ball_data", None)
    if ball is not None:
        ax.scatter(
            ball["x"],
            ball["y"],
            s=120,
            c="orange",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
            label="Ball"
        )

    # Highlight player in possession
    if highlight_possession and frame_data.get("possession") is not None:
        possession_player_id = frame_data["possession"].get("player_id", None)

        if possession_player_id is not None:
            poss_row = df_players[df_players["player_id"] == possession_player_id]
            if not poss_row.empty:
                x_poss = poss_row.iloc[0]["x"]
                y_poss = poss_row.iloc[0]["y"]

                ax.scatter(
                    [x_poss],
                    [y_poss],
                    s=420,
                    facecolors="none",
                    edgecolors="gold",
                    linewidths=2.5,
                    zorder=6,
                    label="Player in possession"
                )

    frame_number = frame_data.get("frame", "Unknown")
    timestamp = frame_data.get("timestamp", "Unknown")
    period = frame_data.get("period", "Unknown")

    ax.set_title(f"Tracking Frame {frame_number} | Time {timestamp} | Period {period}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# EXAMPLE USAGE
# ============================================================

frame_example = tracking_data[150]

'''plot_tracking_frame(
    frame_data=frame_example,
    player_lookup=player_lookup,
    match_data=match_data,
    show_labels=True,
    label_mode="number",   # try "name" if you prefer
    show_undetected=True,
    highlight_possession=True
)'''


# --------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ============================================================
# PASSING OPTIONS HELPERS
# ============================================================

def get_active_passing_options(
    events_df: pd.DataFrame,
    frame_number: int,
    ball_carrier_id: int | None = None
) -> pd.DataFrame:
    """
    Return active passing options at a given frame.

    A passing option is considered active if:
    - event_type == "passing_option"
    - frame_start <= frame_number <= frame_end

    If ball_carrier_id is provided, only options associated with that
    current ball carrier are kept.
    """
    df_options = events_df[
        (events_df["event_type"] == "passing_option") &
        (events_df["frame_start"] <= frame_number) &
        (events_df["frame_end"] >= frame_number)
    ].copy()

    if ball_carrier_id is not None and "player_in_possession_id" in df_options.columns:
        df_options = df_options[df_options["player_in_possession_id"] == ball_carrier_id].copy()

    return df_options


def compute_option_value(option_row: pd.Series) -> float:
    """
    Compute the utility score for a passing option.

    Preferred formula:
        xpass_completion * player_targeted_xthreat

    Robust fallback:
        xpass_completion * xthreat

    This fallback is important because, in many passing_option rows,
    player_targeted_xthreat may be missing while xthreat is filled.
    """
    xpass = option_row.get("player_targeted_xpass_completion", np.nan)
    if pd.isna(xpass):
        xpass = option_row.get("xpass_completion", np.nan)

    xthreat = option_row.get("player_targeted_xthreat", np.nan)
    if pd.isna(xthreat):
        xthreat = option_row.get("xthreat", np.nan)

    if pd.isna(xpass) or pd.isna(xthreat):
        return np.nan

    return float(xpass) * float(xthreat)














import numpy as np
import requests
import math
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# 1. LOAD PRETRAINED MODEL & BUILD INTERPOLATOR
# ==========================================
# Fetch the Karun Singh 12x8 xT grid
url = "https://karun.in/blog/data/open_xt_12x8_v1.json"
xt_grid = np.array(requests.get(url).json())

# Define the centers of the grid cells for a standard 105x68 SPADL pitch
# 12 columns (x-axis) and 8 rows (y-axis)
x_coords = np.linspace(105/24, 105 - 105/24, 12)
y_coords = np.linspace(68/16, 68 - 68/16, 8)

# Create an ultra-fast scipy interpolator
# We transpose (xt_grid.T) to align with (x, y) input
xt_interpolator = RegularGridInterpolator(
    (x_coords, y_coords), 
    xt_grid.T, 
    bounds_error=False, 
    fill_value=None # Automatically extrapolates if a player steps slightly out of bounds
)


def get_xcost(x, y, team_id, period, match_data):
    """
    Returns the Xcost if the attacking team loses the ball at (x,y).
    Handles halftime side-switching and opponent directionality.
    """
    # 1. Find which direction THIS team is attacking
    sign = get_team_attack_sign(team_id, period, match_data)
    
    # 2. Convert SkillCorner coordinates to SPADL format [0, 105], [0, 68]
    spadl_x = x + 52.5
    spadl_y = y + 34.0
    
    # 3. Adjust for opponent's perspective
    if sign == 1:
        # We attack Right (+x), opponent attacks Left (-x). 
        # Karun Singh's grid assumes attacking Right, so we MIRROR the opponent's view.
        opp_x = 105.0 - spadl_x
        opp_y = 68.0 - spadl_y
    else:
        # We attack Left (-x), opponent attacks Right (+x).
        # Aligns perfectly with Karun Singh's grid on the X-axis.
        opp_x = spadl_x
        opp_y = 68.0 - spadl_y # Y is mirrored just to keep relative left/right symmetry

    opp_x = max(0.0, min(105.0, opp_x))
    opp_y = max(0.0, min(68.0, opp_y))
    
    return float(xt_interpolator((opp_x, opp_y)))

# ==========================================
# 3. GET EXPECTED GOALS (xG) FOR THE SINK NODE
# ==========================================
def get_xg(x, y):
    """
    Returns the spatial Expected Goals (xG) value for a shot taken from (x,y).
    Uses a standard distance and visible-angle logistic regression heuristic.
    This creates the edge directly from the player to the "Sink" (Goal).
    """
    # Assuming the attacking goal is at x = 52.5, y = 0
    goal_x = 52.5
    goal_y = 0.0
    
    # 1. Calculate Distance to goal
    distance = math.hypot(goal_x - x, goal_y - y)
    
    # 2. Calculate Visible Angle of the goalposts (Standard width is 7.32m)
    dx = goal_x - x
    # Prevent division by zero if player is literally standing on the goal line
    if dx <= 0: return 0.0 
    
    angle = math.atan2(7.32 * dx, dx**2 + y**2 - (7.32/2)**2)
    if angle < 0:
        angle += math.pi
        
    # 3. Spatial xG Logistic Regression Formula
    # These coefficients mimic standard open-play spatial xG models.
    logit = -1.0 - (0.15 * distance) + (1.5 * angle)
    
    # Convert log-odds to probability
    xg = 1 / (1 + math.exp(-logit))
    return xg























from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import math
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


# ============================================================
# ATTACKING-DIRECTION HELPERS
# ============================================================

def get_home_attacking_direction(match_data: dict, period: int) -> str:
    """
    Return the home team's attacking direction in the given period.

    Expected values:
    - "left_to_right"
    - "right_to_left"
    """
    return match_data["home_team_side"][period - 1]


def get_team_attack_sign(team_id: int, period: int, match_data: dict) -> int:
    """
    Return +1 if the given team is attacking toward +x in this period,
    and -1 if it is attacking toward -x.

    The external xG / xCost functions assume the attacking team always
    attacks toward +x, so we use this sign to normalize coordinates.
    """
    home_team_id = match_data["home_team"]["id"]
    away_team_id = match_data["away_team"]["id"]

    home_dir = get_home_attacking_direction(match_data, period)

    if team_id == home_team_id:
        return 1 if home_dir == "left_to_right" else -1
    elif team_id == away_team_id:
        return -1 if home_dir == "left_to_right" else 1
    else:
        return 1


def normalize_attacking_coordinates(x: float, y: float, team_id: int, period: int, match_data: dict):
    """
    Convert raw match coordinates into normalized attacking coordinates.

    Output convention:
    - the team in possession is always considered to be attacking toward +x

    We rotate the pitch by 180 degrees when the team is attacking toward -x.
    """
    sign = get_team_attack_sign(team_id, period, match_data)
    return sign * x, sign * y


def get_attacking_goal_coordinates(team_id: int, period: int, match_data: dict):
    """
    Return the actual goal coordinates in the original tracking coordinate system
    for the team that is currently attacking.
    """
    sign = get_team_attack_sign(team_id, period, match_data)
    goal_x = 52.5 * sign
    goal_y = 0.0
    return goal_x, goal_y


def get_position_xthreat(x: float, y: float, team_id: int, period: int, match_data: dict) -> float:
    """
    Return spatial xThreat at a raw tracking position for the attacking team.
    """
    norm_x, norm_y = normalize_attacking_coordinates(
        x=x,
        y=y,
        team_id=team_id,
        period=period,
        match_data=match_data
    )

    spadl_x = max(0.0, min(105.0, norm_x + 52.5))
    spadl_y = max(0.0, min(68.0, norm_y + 34.0))
    return float(xt_interpolator((spadl_x, spadl_y)))


def _clip_to_pitch(x: float, y: float, match_data: dict) -> tuple[float, float]:
    half_length = float(match_data["pitch_length"]) / 2.0
    half_width = float(match_data["pitch_width"]) / 2.0
    return (
        float(np.clip(x, -half_length, half_length)),
        float(np.clip(y, -half_width, half_width))
    )


def _player_motion_profile(player_row: pd.Series) -> dict:
    """
    Build a movement profile from one tracked player row with safe defaults.
    """
    vx = float(player_row["vx"]) if "vx" in player_row and pd.notna(player_row["vx"]) else 0.0
    vy = float(player_row["vy"]) if "vy" in player_row and pd.notna(player_row["vy"]) else 0.0

    vmax = float(player_row["vmax"]) if "vmax" in player_row and pd.notna(player_row["vmax"]) else 8.0
    accel = float(player_row["accel"]) if "accel" in player_row and pd.notna(player_row["accel"]) else 3.5
    reaction = float(player_row["reaction"]) if "reaction" in player_row and pd.notna(player_row["reaction"]) else 0.25

    return {
        "player_id": int(player_row["player_id"]),
        "x": float(player_row["x"]),
        "y": float(player_row["y"]),
        "vx": vx,
        "vy": vy,
        "vmax": max(vmax, 0.1),
        "accel": max(accel, 0.1),
        "reaction": max(reaction, 0.0)
    }


def _time_to_intercept(target_x: float, target_y: float, player: dict) -> float:
    """
    Notebook-equivalent arrival-time model using movement direction and acceleration.
    """
    px = float(player.get("x", 0.0))
    py = float(player.get("y", 0.0))
    vx = float(player.get("vx", 0.0))
    vy = float(player.get("vy", 0.0))
    vmax = max(float(player.get("vmax", 8.0)), 0.1)
    accel = max(float(player.get("accel", 3.5)), 0.1)
    reaction = max(float(player.get("reaction", 0.25)), 0.0)

    dx = target_x - px
    dy = target_y - py
    dist = math.hypot(dx, dy)

    if dist <= 1e-6:
        return reaction

    v_toward = (vx * dx + vy * dy) / dist
    v_toward = max(v_toward, 0.0)

    t_to_vmax = max(vmax - v_toward, 0.0) / accel
    d_accel = v_toward * t_to_vmax + 0.5 * accel * (t_to_vmax ** 2)

    if dist <= d_accel:
        disc = max(v_toward ** 2 + 2.0 * accel * dist, 1e-9)
        t_move = (-v_toward + np.sqrt(disc)) / accel
    else:
        t_move = t_to_vmax + (dist - d_accel) / vmax

    return reaction + max(t_move, 0.0)


def compute_zone_control_at_position(
    x: float,
    y: float,
    attackers: list[dict],
    defenders: list[dict],
    control_beta: float = 1.3
) -> float:
    """
    Movement-aware zone control in [0,1], matching SvG_v1 notebook logic.
    """
    if not attackers and not defenders:
        return 0.5
    if not attackers:
        return 0.0
    if not defenders:
        return 1.0

    t_att = min(_time_to_intercept(x, y, p) for p in attackers)
    t_def = min(_time_to_intercept(x, y, p) for p in defenders)

    delta_t = np.clip(t_def - t_att, -8.0, 8.0)
    return float(1.0 / (1.0 + np.exp(-control_beta * delta_t)))


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return 0.0, 0.0
    return dx / norm, dy / norm


def _exp_ramp_01(value: float, rate: float = 2.0) -> float:
    """
    Map a [0,1] input to [0,1] with exponential growth.
    Higher rate means faster penalty growth near 1.
    """
    v = float(np.clip(value, 0.0, 1.0))
    denom = math.exp(rate) - 1.0
    if denom <= 1e-12:
        return v
    return float((math.exp(rate * v) - 1.0) / denom)


def _resolve_carrier_team_id(
    carrier_row: pd.DataFrame,
    possession_info: dict,
    match_data: dict
) -> int | None:
    if "team_id" in carrier_row.columns and pd.notna(carrier_row.iloc[0]["team_id"]):
        return int(carrier_row.iloc[0]["team_id"])

    team_group = possession_info.get("group", None)
    if team_group == "home team":
        return int(match_data["home_team"]["id"])
    if team_group == "away team":
        return int(match_data["away_team"]["id"])

    return None


def _build_team_motion_profiles(df_frame_players: pd.DataFrame, team_id: int) -> tuple[list[dict], list[dict]]:
    detected_mask = (
        (df_frame_players["is_detected"] == True)
        if "is_detected" in df_frame_players.columns
        else pd.Series([True] * len(df_frame_players), index=df_frame_players.index)
    )

    attackers_df = df_frame_players[(df_frame_players["team_id"] == team_id) & detected_mask].copy()
    defenders_df = df_frame_players[(df_frame_players["team_id"] != team_id) & detected_mask].copy()

    attackers = [_player_motion_profile(row) for _, row in attackers_df.iterrows()]
    defenders = [_player_motion_profile(row) for _, row in defenders_df.iterrows()]
    return attackers, defenders


def _compute_defender_pressure(target_x: float, target_y: float, defenders: list[dict]) -> dict:
    """
    Quantify pressure around a target point using defender proximity and local crowding.
    """
    if not defenders:
        return {
            "pressure_index": 0.0,
            "nearest_distance": np.inf,
            "n_close": 0,
            "n_very_close": 0,
            "nearby_density": 0.0,
            "immediate_density": 0.0,
            "nearby_index": 0.0,
            "immediate_index": 0.0,
        }

    distances = np.array(
        [
            math.hypot(float(d.get("x", 0.0)) - target_x, float(d.get("y", 0.0)) - target_y)
            for d in defenders
        ],
        dtype=float
    )

    nearest_distance = float(np.min(distances))
    n_close = int(np.sum(distances <= 5.0))
    n_very_close = int(np.sum(distances <= 5.0))

    topk = np.sort(distances)[: min(3, len(distances))]
    proximity_term = float(np.mean(np.exp(-topk / 3.0)))
    nearby_density = float(np.sum(np.exp(-((distances / 4.5) ** 2))))
    immediate_density = float(np.sum(np.exp(-((distances / 2.0) ** 2))))
    nearby_index = float(np.clip(nearby_density / 3.0, 0.0, 1.0))
    immediate_index = float(np.clip(immediate_density / 2.0, 0.0, 1.0))
    crowding_term = nearby_index
    collision_term = float(np.clip((2.0 - nearest_distance) / 2.0, 0.0, 1.0))
    immediate_term = immediate_index

    pressure_index = float(
        np.clip(
            0.40 * proximity_term + 0.25 * crowding_term + 0.15 * collision_term + 0.20 * immediate_term,
            0.0,
            1.0,
        )
    )

    return {
        "pressure_index": pressure_index,
        "nearest_distance": nearest_distance,
        "n_close": n_close,
        "n_very_close": n_very_close,
        "nearby_density": nearby_density,
        "immediate_density": immediate_density,
        "nearby_index": nearby_index,
        "immediate_index": immediate_index,
    }


def _compute_lane_interception_metrics(
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    defenders: list[dict],
    lane_width: float = 2.25,
) -> dict:
    """
    Estimate interception pressure along the run lane (line segment start->target).
    """
    if not defenders:
        return {
            "lane_interception_index": 0.0,
            "n_lane": 0,
            "min_lane_distance": np.inf,
        }

    seg_dx = target_x - start_x
    seg_dy = target_y - start_y
    seg_len2 = (seg_dx * seg_dx) + (seg_dy * seg_dy)

    if seg_len2 <= 1e-9:
        dists = np.array(
            [
                math.hypot(float(d.get("x", 0.0)) - target_x, float(d.get("y", 0.0)) - target_y)
                for d in defenders
            ],
            dtype=float,
        )
    else:
        dvals = []
        for d in defenders:
            px = float(d.get("x", 0.0))
            py = float(d.get("y", 0.0))

            t = ((px - start_x) * seg_dx + (py - start_y) * seg_dy) / seg_len2
            t = float(np.clip(t, 0.0, 1.0))

            proj_x = start_x + t * seg_dx
            proj_y = start_y + t * seg_dy

            dvals.append(math.hypot(px - proj_x, py - proj_y))

        dists = np.array(dvals, dtype=float)

    min_lane_distance = float(np.min(dists))
    n_lane = int(np.sum(dists <= lane_width))

    topk = np.sort(dists)[: min(3, len(dists))]
    lane_proximity_term = float(np.mean(np.exp(-topk / max(lane_width, 0.5))))
    lane_density_term = float(min(n_lane / 3.0, 1.0))

    lane_interception_index = float(
        np.clip(0.60 * lane_proximity_term + 0.40 * lane_density_term, 0.0, 1.0)
    )

    return {
        "lane_interception_index": lane_interception_index,
        "n_lane": n_lane,
        "min_lane_distance": min_lane_distance,
    }


def _compute_directional_crowding_metrics(
    start_x: float,
    start_y: float,
    run_ux: float,
    run_uy: float,
    defenders: list[dict],
    max_ahead_distance: float = 12.0,
    half_width: float = 3.5,
) -> dict:
    """
    Measure defender crowding in the intended run direction (forward cone/lane).
    """
    if not defenders:
        return {
            "directional_crowding_index": 0.0,
            "n_directional": 0,
            "min_directional_distance": np.inf,
        }

    if math.hypot(run_ux, run_uy) <= 1e-9:
        return {
            "directional_crowding_index": 0.0,
            "n_directional": 0,
            "min_directional_distance": np.inf,
        }

    directional_weights = []
    n_directional = 0
    min_directional_distance = np.inf

    for d in defenders:
        dx = float(d.get("x", 0.0)) - start_x
        dy = float(d.get("y", 0.0)) - start_y

        # Projection along run direction and lateral distance from run axis.
        proj = dx * run_ux + dy * run_uy
        lat = abs((-run_uy * dx) + (run_ux * dy))

        if proj <= 0.0 or proj > max_ahead_distance:
            continue

        if lat <= half_width:
            n_directional += 1
            min_directional_distance = min(min_directional_distance, proj)

        if lat <= 2.0 * half_width:
            directional_weights.append(math.exp(-proj / 4.0) * math.exp(-lat / 2.5))

    if directional_weights:
        directional_crowding_index = float(np.clip(sum(directional_weights) / 2.0, 0.0, 1.0))
    else:
        directional_crowding_index = 0.0

    return {
        "directional_crowding_index": directional_crowding_index,
        "n_directional": n_directional,
        "min_directional_distance": min_directional_distance,
    }


def _compute_forward_pressure_weight(
    target_x: float,
    target_y: float,
    run_ux: float,
    run_uy: float,
    defenders: list[dict],
) -> dict:
    """
    Compute a forward-bias weight so defenders behind the run contribute less.
    """
    if not defenders or math.hypot(run_ux, run_uy) <= 1e-9:
        return {
            "forward_weight": 1.0,
            "ahead_share": 0.5,
        }

    ahead_mass = 0.0
    total_mass = 0.0

    for d in defenders:
        dx = float(d.get("x", 0.0)) - target_x
        dy = float(d.get("y", 0.0)) - target_y
        dist = math.hypot(dx, dy)
        mass = float(np.exp(-((dist / 5.0) ** 2)))
        if mass <= 1e-12:
            continue

        proj = dx * run_ux + dy * run_uy
        total_mass += mass
        if proj >= 0.0:
            ahead_mass += mass

    ahead_share = float(ahead_mass / total_mass) if total_mass > 1e-12 else 0.5
    forward_weight = 0.45 + 0.95 * _exp_ramp_01(ahead_share, rate=2.2)

    return {
        "forward_weight": float(forward_weight),
        "ahead_share": float(ahead_share),
    }


def _get_run_options_for_carrier(
    frame_data: dict,
    carrier_row: pd.DataFrame,
    df_frame_players: pd.DataFrame,
    match_data: dict,
    events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build directional run options for the ball carrier.

    Source -> run node capacity is min(current_zone_control, destination_zone_control).
    Run node -> goal capacity is pressure-adjusted destination run value.
    """
    possession_info = frame_data.get("possession", None) or {}

    team_id = _resolve_carrier_team_id(carrier_row, possession_info, match_data)
    if team_id is None:
        return pd.DataFrame()

    period = int(frame_data["period"])
    cx = float(carrier_row.iloc[0]["x"])
    cy = float(carrier_row.iloc[0]["y"])

    attackers, defenders = _build_team_motion_profiles(df_frame_players, team_id)
    if not attackers:
        return pd.DataFrame()

    current_control = compute_zone_control_at_position(cx, cy, attackers, defenders)
    current_xcost = get_xcost(cx, cy, team_id, period, match_data)

    # Crowd around the ball-carrier itself (decreasing with distance + count effects).
    ball_pressure_metrics = _compute_defender_pressure(cx, cy, defenders)
    near_ball_score = (
        0.65 * float(ball_pressure_metrics["pressure_index"])
        + 0.25 * float(ball_pressure_metrics["nearby_index"])
        + 0.45 * float(ball_pressure_metrics["immediate_index"])
    )
    near_ball_penalty = 1.10 * float(current_xcost) * _exp_ramp_01(near_ball_score, rate=2.4)

    goal_x, goal_y = get_attacking_goal_coordinates(team_id, period, match_data)
    goal_ux, goal_uy = _unit_vector(goal_x - cx, goal_y - cy)

    carrier_profile = _player_motion_profile(carrier_row.iloc[0])
    vel_ux, vel_uy = _unit_vector(carrier_profile["vx"], carrier_profile["vy"])
    if math.hypot(vel_ux, vel_uy) <= 1e-9:
        vel_ux, vel_uy = goal_ux, goal_uy

    if math.hypot(vel_ux, vel_uy) <= 1e-9:
        attack_sign = get_team_attack_sign(team_id, period, match_data)
        vel_ux, vel_uy = float(attack_sign), 0.0

    carrier_speed = math.hypot(float(carrier_profile["vx"]), float(carrier_profile["vy"]))
    carrier_vmax = max(float(carrier_profile.get("vmax", 8.0)), 0.1)
    run_eval_speed = max(carrier_speed, RUN_EVAL_SPEED_FALLBACK_VMAX_FRACTION * carrier_vmax)
    run_eval_distance = max(run_eval_speed * RUN_EVAL_TIME_SECONDS, 0.5)

    run_display_distance = RUN_DISPLAY_DISTANCE_METERS
    forward_offsets_deg = np.linspace(-95.0, 95.0, 8)
    backward_offsets_deg = np.linspace(140.0, 220.0, 4)

    run_defs = []

    # 8 forward-facing directions around attacking-goal heading.
    for idx, offset_deg in enumerate(forward_offsets_deg):
        angle = math.radians(float(offset_deg))
        dx = (goal_ux * math.cos(angle)) - (goal_uy * math.sin(angle))
        dy = (goal_ux * math.sin(angle)) + (goal_uy * math.cos(angle))
        eval_tx, eval_ty = _clip_to_pitch(cx + run_eval_distance * dx, cy + run_eval_distance * dy, match_data)
        vis_tx, vis_ty = _clip_to_pitch(cx + run_display_distance * dx, cy + run_display_distance * dy, match_data)
        run_defs.append(
            (
                f"forward_direction_{idx}",
                f"Forward {int(round(offset_deg))} deg",
                eval_tx,
                eval_ty,
                vis_tx,
                vis_ty,
            )
        )

    # 4 backward-facing directions around the opposite heading.
    for idx, offset_deg in enumerate(backward_offsets_deg):
        angle = math.radians(float(offset_deg))
        dx = (goal_ux * math.cos(angle)) - (goal_uy * math.sin(angle))
        dy = (goal_ux * math.sin(angle)) + (goal_uy * math.cos(angle))
        eval_tx, eval_ty = _clip_to_pitch(cx + run_eval_distance * dx, cy + run_eval_distance * dy, match_data)
        vis_tx, vis_ty = _clip_to_pitch(cx + run_display_distance * dx, cy + run_display_distance * dy, match_data)
        run_defs.append(
            (
                f"backward_direction_{idx}",
                f"Backward {int(round(offset_deg))} deg",
                eval_tx,
                eval_ty,
                vis_tx,
                vis_ty,
            )
        )

    # Force current direction option in (even if it overlaps another direction).
    current_eval_tx, current_eval_ty = _clip_to_pitch(
        cx + run_eval_distance * vel_ux,
        cy + run_eval_distance * vel_uy,
        match_data,
    )
    current_vis_tx, current_vis_ty = _clip_to_pitch(
        cx + run_display_distance * vel_ux,
        cy + run_display_distance * vel_uy,
        match_data,
    )
    run_defs.append(("current_direction", "Current direction", current_eval_tx, current_eval_ty, current_vis_tx, current_vis_ty))

    run_rows = []
    for run_type, run_label, eval_tx, eval_ty, vis_tx, vis_ty in run_defs:
        run_ux, run_uy = _unit_vector(eval_tx - cx, eval_ty - cy)

        defenders_shifted = []
        for defender in defenders:
            d = defender.copy()
            defender_vmax = max(float(d.get("vmax", 8.0)), 0.1)
            defender_speed = 0.70 * defender_vmax
            d["vx"] = run_ux * defender_speed
            d["vy"] = run_uy * defender_speed
            defenders_shifted.append(d)

        destination_control = compute_zone_control_at_position(
            eval_tx,
            eval_ty,
            attackers,
            defenders_shifted
        )

        control_capacity = min(current_control, destination_control)
        destination_xthreat = get_position_xthreat(
            eval_tx,
            eval_ty,
            team_id=team_id,
            period=period,
            match_data=match_data
        )

        destination_xcost = get_xcost(eval_tx, eval_ty, team_id, period, match_data)
        pressure_metrics = _compute_defender_pressure(eval_tx, eval_ty, defenders_shifted)
        forward_pressure_metrics = _compute_forward_pressure_weight(
            target_x=eval_tx,
            target_y=eval_ty,
            run_ux=run_ux,
            run_uy=run_uy,
            defenders=defenders_shifted,
        )
        forward_weight = float(forward_pressure_metrics["forward_weight"])
        forward_ahead_share = float(forward_pressure_metrics["ahead_share"])

        base_pressure_penalty = (
            2.50
            * float(pressure_metrics["pressure_index"])
            * float(destination_xcost)
            * forward_weight
        )

        # Extra penalty if run does not progress toward goal.
        progress_to_goal = (eval_tx - cx) * goal_ux + (eval_ty - cy) * goal_uy
        non_progress_penalty = 0.35 * float(destination_xcost) if progress_to_goal <= 0.0 else 0.0

        # Penalize defenders that can step into the run lane itself.
        lane_metrics = _compute_lane_interception_metrics(
            start_x=cx,
            start_y=cy,
            target_x=eval_tx,
            target_y=eval_ty,
            defenders=defenders_shifted,
            lane_width=2.25,
        )
        lane_penalty = 1.20 * float(lane_metrics["lane_interception_index"]) * float(destination_xcost)

        # Penalize defender density specifically in the intended run direction.
        directional_metrics = _compute_directional_crowding_metrics(
            start_x=cx,
            start_y=cy,
            run_ux=run_ux,
            run_uy=run_uy,
            defenders=defenders_shifted,
            max_ahead_distance=12.0,
            half_width=3.5,
        )
        directional_penalty = (
            1.20 * float(directional_metrics["directional_crowding_index"])
            + 0.75 * float(min(directional_metrics["n_directional"] / 3.0, 1.0))
        ) * float(destination_xcost)

        # Extra frontal-blocker penalty: strongly punish carries into packed lines,
        # especially when destination is close to goal/box.
        goal_distance_to_destination = float(math.hypot(goal_x - eval_tx, goal_y - eval_ty))
        box_proximity = float(np.clip((24.0 - goal_distance_to_destination) / 24.0, 0.0, 1.0))
        n_directional = int(directional_metrics["n_directional"])
        front_blocker_count = max(n_directional - 2, 0)
        front_blocker_count_score = _exp_ramp_01(min(front_blocker_count / 3.0, 1.0), rate=3.2)

        min_directional_distance = float(directional_metrics["min_directional_distance"])
        if np.isfinite(min_directional_distance):
            front_blocker_proximity = float(np.exp(-((min_directional_distance / 2.8) ** 2)))
        else:
            front_blocker_proximity = 0.0

        front_blocker_penalty = (
            float(destination_xcost)
            * (1.60 + 2.60 * box_proximity)
            * front_blocker_count_score
            * (0.60 + 0.80 * front_blocker_proximity)
        )

        # Stronger local congestion penalty near destination: many nearby defenders
        # should heavily discount the value of carrying into that zone.
        crowding_score = (
            0.55 * float(pressure_metrics["nearby_index"])
            + 0.95 * float(pressure_metrics["immediate_index"])
        )
        crowding_penalty = 1.30 * float(destination_xcost) * _exp_ramp_01(crowding_score, rate=2.8)

        # Smooth near-collision pressure: grows as nearest defender gets closer.
        nearest_def = float(pressure_metrics["nearest_distance"])
        collision_proximity = float(np.clip((5.0 - nearest_def) / 5.0, 0.0, 1.0))
        very_close_proximity = float(np.clip((2.5 - nearest_def) / 1.5, 0.0, 1.0))
        ultra_close_proximity = float(np.clip((1.8 - nearest_def) / 1.0, 0.0, 1.0))
        very_close_multiplier = (
            1.0
            + 4.20 * _exp_ramp_01(very_close_proximity, rate=8.0)
            + 2.00 * _exp_ramp_01(ultra_close_proximity, rate=10.0)
        )
        collision_boost_penalty = float(destination_xcost) * (
            1.20 * _exp_ramp_01(collision_proximity, rate=4.2)
            + 6.20 * _exp_ramp_01(very_close_proximity, rate=8.0)
            + 4.00 * _exp_ramp_01(ultra_close_proximity, rate=10.0)
        ) * very_close_multiplier

        # Multiply local congestion effects multiplicatively per additional
        # nearby defender so crowded pockets are penalized much more strongly.
        nearby_density_scaled = float(np.clip(pressure_metrics["nearby_density"] / 4.0, 0.0, 1.0))
        nearby_players = int(pressure_metrics["n_close"])
        extra_nearby_players = max(nearby_players - 1, 0)
        density_base_multiplier = 1.0 + 0.55 * _exp_ramp_01(nearby_density_scaled, rate=2.3)
        count_multiplier = 1.80 ** extra_nearby_players
        close_cluster_scale = float(np.clip((nearby_players - 2) / 4.0, 0.0, 1.0))
        cluster_spike_multiplier = 1.0 + 1.40 * _exp_ramp_01(close_cluster_scale, rate=3.2)
        front_blocker_multiplier = (
            1.0
            + 2.40
            * front_blocker_count_score
            * (0.50 + 0.50 * front_blocker_proximity)
            * (0.50 + 0.70 * box_proximity)
        )
        nearby_multiplier = float(
            min(
                density_base_multiplier
                * count_multiplier
                * cluster_spike_multiplier
                * front_blocker_multiplier,
                32.0,
            )
        )
        local_congestion_penalty = nearby_multiplier * (crowding_penalty + collision_boost_penalty)

        pressure_penalty = (
            base_pressure_penalty
            + non_progress_penalty
            + lane_penalty
            + directional_penalty
            + front_blocker_penalty
            + local_congestion_penalty
            + near_ball_penalty
        )
        defender_count_score = (
            float(pressure_metrics["n_close"])
            + 1.50 * float(pressure_metrics["n_very_close"])
            + 0.50 * float(lane_metrics["n_lane"])
            + 0.50 * float(directional_metrics["n_directional"])
        )
        scaled_pressure_penalty = RUN_SMALL_PRESSURE_PER_DEFENDER * defender_count_score

        prob_success = max(float(destination_control) - float(scaled_pressure_penalty), 0.0)
        prob_success = min(prob_success, 1.0)
        run_xthreat = float(prob_success) * float(destination_xthreat)
        run_net_xthreat = float(run_xthreat) - ((1.0 - float(prob_success)) * float(destination_xcost))

        if not np.isfinite(destination_xthreat):
            continue

        run_rows.append(
            {
                "run_type": run_type,
                "run_label": run_label,
                "start_x": cx,
                "start_y": cy,
                "target_x": float(vis_tx),
                "target_y": float(vis_ty),
                "eval_target_x": float(eval_tx),
                "eval_target_y": float(eval_ty),
                "current_control": float(current_control),
                "destination_control": float(destination_control),
                "control_capacity": float(max(control_capacity, 0.0)),
                "destination_xthreat": float(destination_xthreat),
                "destination_xcost": float(destination_xcost),
                "prob_success": float(prob_success),
                "run_xthreat": float(run_xthreat),
                "defender_pressure": float(pressure_metrics["pressure_index"]),
                "nearest_defender_distance": float(pressure_metrics["nearest_distance"]),
                "n_close_defenders": int(pressure_metrics["n_close"]),
                "n_very_close_defenders": int(pressure_metrics["n_very_close"]),
                "progress_to_goal": float(progress_to_goal),
                "non_progress_penalty": float(non_progress_penalty),
                "lane_interception_index": float(lane_metrics["lane_interception_index"]),
                "lane_defenders": int(lane_metrics["n_lane"]),
                "min_lane_distance": float(lane_metrics["min_lane_distance"]),
                "lane_penalty": float(lane_penalty),
                "directional_crowding_index": float(directional_metrics["directional_crowding_index"]),
                "directional_defenders": int(directional_metrics["n_directional"]),
                "min_directional_distance": float(directional_metrics["min_directional_distance"]),
                "directional_penalty": float(directional_penalty),
                "goal_distance_to_destination": float(goal_distance_to_destination),
                "box_proximity": float(box_proximity),
                "front_blocker_count": int(front_blocker_count),
                "front_blocker_count_score": float(front_blocker_count_score),
                "front_blocker_proximity": float(front_blocker_proximity),
                "front_blocker_penalty": float(front_blocker_penalty),
                "front_blocker_multiplier": float(front_blocker_multiplier),
                "forward_pressure_weight": float(forward_weight),
                "forward_ahead_share": float(forward_ahead_share),
                "nearby_multiplier": float(nearby_multiplier),
                "nearby_players": int(nearby_players),
                "extra_nearby_players": int(extra_nearby_players),
                "cluster_spike_multiplier": float(cluster_spike_multiplier),
                "local_congestion_penalty": float(local_congestion_penalty),
                "collision_proximity": float(collision_proximity),
                "very_close_proximity": float(very_close_proximity),
                "ultra_close_proximity": float(ultra_close_proximity),
                "very_close_multiplier": float(very_close_multiplier),
                "crowding_penalty": float(crowding_penalty),
                "collision_boost_penalty": float(collision_boost_penalty),
                "near_ball_pressure_index": float(ball_pressure_metrics["pressure_index"]),
                "near_ball_penalty": float(near_ball_penalty),
                "pressure_penalty": float(pressure_penalty),
                "defender_count_score": float(defender_count_score),
                "scaled_pressure_penalty": float(scaled_pressure_penalty),
                "run_net_xthreat": float(run_net_xthreat)
            }
        )

    run_df = pd.DataFrame(run_rows) if run_rows else pd.DataFrame()

    if not run_df.empty:
        run_df = run_df.drop_duplicates(subset=["run_type"]).copy()

    return run_df


# ============================================================
# PASSING / SHOT VALUE HELPERS
# ============================================================

def get_xpass_value(option_row: pd.Series) -> float:
    """
    Get pass completion probability for one passing option row.
    """
    xpass = option_row.get("player_targeted_xpass_completion", np.nan)
    if pd.isna(xpass):
        xpass = option_row.get("xpass_completion", np.nan)

    return float(xpass) if pd.notna(xpass) else np.nan


def get_target_xthreat_value(option_row: pd.Series) -> float:
    """
    Get target xThreat for one passing option row.
    """
    xt = option_row.get("player_targeted_xthreat", np.nan)
    if pd.isna(xt):
        xt = option_row.get("xthreat", np.nan)

    return float(xt) if pd.notna(xt) else np.nan


def compute_pass_metrics_for_frame_option(
    option_row: pd.Series,
    df_frame_players: pd.DataFrame,
    frame_data: dict,
    match_data: dict
):
    """
    Compute all pass-related metrics for a passing option.

    Returns:
    - xpass
    - xthreat
    - xcost
    - offensive_return = xpass * xthreat
    - utility = (xpass * xthreat) - ((1 - xpass) * xcost)
    """
    receiver_id = option_row["player_id"]

    receiver_row = df_frame_players[df_frame_players["player_id"] == receiver_id]
    if receiver_row.empty:
        return None

    rx = float(receiver_row.iloc[0]["x"])
    ry = float(receiver_row.iloc[0]["y"])

    team_id = int(option_row["team_id"]) if "team_id" in option_row and pd.notna(option_row["team_id"]) else None
    if team_id is None:
        return None

    period = int(frame_data["period"])

    norm_x, norm_y = normalize_attacking_coordinates(
        x=rx,
        y=ry,
        team_id=team_id,
        period=period,
        match_data=match_data
    )

    xpass = get_xpass_value(option_row)
    xthreat = get_target_xthreat_value(option_row)

    if pd.isna(xpass) or pd.isna(xthreat):
        return None

    xcost = get_xcost(rx, ry, team_id, period, match_data)

    offensive_return = float(xpass) * float(xthreat)
    utility = offensive_return - ((1.0 - float(xpass)) * float(xcost))

    passing_option_event_id = option_row.get("event_id", np.nan)
    passing_option_event_id = str(passing_option_event_id) if pd.notna(passing_option_event_id) else ""

    return {
        "passing_option_event_id": passing_option_event_id,
        "receiver_id": int(receiver_id),
        "receiver_x": rx,
        "receiver_y": ry,
        "xpass": float(xpass),
        "xthreat": float(xthreat),
        "xcost": float(xcost),
        "offensive_return": float(offensive_return),
        "utility": float(utility)
    }


def compute_shot_metric(
    frame_data: dict,
    match_data: dict,
    ball_carrier_id: int | None = None,
    df_frame_players: pd.DataFrame | None = None,
):
    """
    Compute xG for a direct shot from the current ball location.

    The external get_xg(x, y) function assumes the attacking team always
    attacks toward +x, so we normalize coordinates first.
    """
    ball = frame_data.get("ball_data", None)
    if ball is None:
        return None

    bx = ball.get("x", None)
    by = ball.get("y", None)
    period_raw = frame_data.get("period", None)

    if bx is None or by is None or period_raw is None:
        return None

    possession = frame_data.get("possession", None) or {}
    period = int(period_raw)
    team_id = None

    candidate_carrier_ids = []
    if ball_carrier_id is not None:
        candidate_carrier_ids.append(int(ball_carrier_id))

    possession_player_id = possession.get("player_id", None)
    if possession_player_id is not None:
        candidate_carrier_ids.append(int(possession_player_id))

    for candidate_id in candidate_carrier_ids:
        if (
            df_frame_players is not None
            and "player_id" in df_frame_players.columns
            and "team_id" in df_frame_players.columns
        ):
            carrier_row = df_frame_players[
                (df_frame_players["player_id"] == candidate_id)
                & (df_frame_players["team_id"].notna())
            ]
            if not carrier_row.empty:
                team_id = int(carrier_row.iloc[0]["team_id"])
                break

        mapped_team_id = player_team_map.get(candidate_id)
        if mapped_team_id is not None and pd.notna(mapped_team_id):
            team_id = int(mapped_team_id)
            break

    if team_id is None:
        team_group = possession.get("group", None)
        if team_group == "home team":
            team_id = int(match_data["home_team"]["id"])
        elif team_group == "away team":
            team_id = int(match_data["away_team"]["id"])
        else:
            return None

    bx = float(bx)
    by = float(by)

    norm_x, norm_y = normalize_attacking_coordinates(
        x=bx,
        y=by,
        team_id=team_id,
        period=period,
        match_data=match_data
    )

    xg = get_xg(norm_x, norm_y)

    goal_x, goal_y = get_attacking_goal_coordinates(
        team_id=team_id,
        period=period,
        match_data=match_data
    )

    return {
        "team_id": team_id,
        "ball_x": bx,
        "ball_y": by,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "xg": float(xg)
    }


def get_active_passing_options(
    events_df: pd.DataFrame,
    frame_number: int,
    ball_carrier_id: int | None = None
) -> pd.DataFrame:
    """
    Return active passing options at a given frame.
    """
    df_options = events_df[
        (events_df["event_type"] == "passing_option") &
        (events_df["frame_start"] <= frame_number) &
        (events_df["frame_end"] >= frame_number)
    ].copy()

    if ball_carrier_id is not None and "player_in_possession_id" in df_options.columns:
        df_options = df_options[df_options["player_in_possession_id"] == ball_carrier_id].copy()

    return df_options



# ============================================================
# VISUALIZATION OF PASS OPTIONS + SHOT OPTION
# ============================================================

def draw_decision_arrows(
    ax,
    frame_data: dict,
    df_frame_players: pd.DataFrame,
    events_df: pd.DataFrame,
    match_data: dict,
    show_all_passes: bool = True,
    show_best_pass: bool = True,
    show_run_options: bool = True,
    show_best_run: bool = True,
    show_shot_option: bool = True,
    shot_display_threshold: float = 0.01,
    pass_color: str = "gray",
    best_color: str = "limegreen",
    run_color: str = "deepskyblue",
    best_run_color: str = "royalblue",
    shot_default_color: str = "gray",
    all_alpha: float = 0.35,
    best_alpha: float = 0.60,
    linewidth_all: float = 1.6,
    linewidth_best: float = 3.0,
    mutation_scale: int = 14
):
    # NEW HYBRID LOOKUP
    ball_carrier_id = get_resolved_ball_carrier(frame_data, frame_team_array, player_team_map)
    
    if ball_carrier_id is None:
        return {
            "best_pass_offensive_return": np.nan,
            "best_pass_utility": np.nan,
            "best_run_value": np.nan,
            "shot_xg": np.nan
        }

    carrier_row = df_frame_players[df_frame_players["player_id"] == ball_carrier_id]
    if carrier_row.empty:
        return {
            "best_pass_offensive_return": np.nan,
            "best_pass_utility": np.nan,
            "best_run_value": np.nan,
            "shot_xg": np.nan
        }

    carrier_x = float(carrier_row.iloc[0]["x"])
    carrier_y = float(carrier_row.iloc[0]["y"])

    # Active passing options
    df_options = get_active_passing_options(
        events_df=events_df,
        frame_number=frame_data["frame"],
        ball_carrier_id=ball_carrier_id
    )

    visible_player_ids = set(df_frame_players["player_id"].dropna().tolist())
    df_options = df_options[df_options["player_id"].isin(visible_player_ids)].copy()

    pass_metrics = []

    for _, option in df_options.iterrows():
        metrics = compute_pass_metrics_for_frame_option(
            option_row=option,
            df_frame_players=df_frame_players,
            frame_data=frame_data,
            match_data=match_data
        )
        if metrics is not None:
            pass_metrics.append(metrics)

    best_pass_offensive_return = np.nan
    best_pass_utility = np.nan
    best_run_value = np.nan

    if pass_metrics:
        pass_df = pd.DataFrame(pass_metrics)

        # Remove duplicated receivers if needed
        pass_df = (
            pass_df.sort_values("utility", ascending=False)
            .drop_duplicates(subset=["receiver_id"])
            .copy()
        )

        best_pass_offensive_return = pass_df["offensive_return"].max()
        best_pass_utility = pass_df["utility"].max()

        # Draw all passes
        if show_all_passes:
            for _, row in pass_df.iterrows():
                rx = row["receiver_x"]
                ry = row["receiver_y"]

                arrow = FancyArrowPatch(
                    (carrier_x, carrier_y),
                    (rx, ry),
                    arrowstyle="->",
                    mutation_scale=mutation_scale,
                    linewidth=linewidth_all,
                    color=pass_color,
                    alpha=all_alpha,
                    zorder=4
                )
                ax.add_patch(arrow)

                mid_x = (carrier_x + rx) / 2
                mid_y = (carrier_y + ry) / 2

                ax.text(
                    mid_x,
                    mid_y,
                    f"{100 * row['utility']:.1f}%",
                    fontsize=8,
                    color=pass_color,
                    ha="center",
                    va="bottom",
                    zorder=5,
                    bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=0.2)
                )

        # Draw best pass
        if show_best_pass:
            best_row = pass_df.loc[pass_df["utility"].idxmax()]
            rx = best_row["receiver_x"]
            ry = best_row["receiver_y"]

            arrow = FancyArrowPatch(
                (carrier_x, carrier_y),
                (rx, ry),
                arrowstyle="->",
                mutation_scale=mutation_scale + 2,
                linewidth=linewidth_best,
                color=best_color,
                alpha=best_alpha,
                zorder=6
            )
            ax.add_patch(arrow)

            mid_x = (carrier_x + rx) / 2
            mid_y = (carrier_y + ry) / 2

            ax.text(
                mid_x,
                mid_y,
                f"{100 * best_row['utility']:.1f}%",
                fontsize=9,
                color=best_color,
                weight="bold",
                ha="center",
                va="bottom",
                zorder=7,
                bbox=dict(facecolor="white", alpha=0.45, edgecolor="none", pad=0.25)
            )

    shot_metrics = compute_shot_metric(
        frame_data=frame_data,
        match_data=match_data,
        ball_carrier_id=ball_carrier_id,
        df_frame_players=df_frame_players,
    )
    shot_xg = float(shot_metrics["xg"]) if shot_metrics is not None else np.nan

    # Draw run options
    if show_run_options:
        run_df = _get_run_options_for_carrier(
            frame_data=frame_data,
            carrier_row=carrier_row,
            df_frame_players=df_frame_players,
            match_data=match_data,
            events_df=events_df,
        )

        if not run_df.empty:
            run_df = run_df.copy()
            if "run_net_xthreat" in run_df.columns:
                run_df["run_value"] = run_df["run_net_xthreat"]
            else:
                run_df["run_value"] = run_df["destination_xthreat"]

            run_df = (
                run_df.sort_values("run_value", ascending=False)
                .drop_duplicates(subset=["run_type"])
                .copy()
            )

            best_run_value = float(run_df["run_value"].max())

            if show_best_run:
                best_run_row = run_df.loc[run_df["run_value"].idxmax()]
                tx = float(best_run_row["target_x"])
                ty = float(best_run_row["target_y"])

                run_beats_best_pass_and_shot = bool(
                    pd.notna(best_pass_utility)
                    and pd.notna(best_run_value)
                    and pd.notna(shot_xg)
                    and (best_run_value > best_pass_utility)
                    and (best_run_value > shot_xg)
                )

                if run_beats_best_pass_and_shot:
                    run_highlight_color = best_run_color
                    run_highlight_alpha = best_alpha
                    run_highlight_lw = linewidth_best
                    run_highlight_ms = mutation_scale + 2
                    run_label_weight = "bold"
                else:
                    run_highlight_color = "lightskyblue"
                    run_highlight_alpha = all_alpha
                    run_highlight_lw = linewidth_all
                    run_highlight_ms = mutation_scale
                    run_label_weight = "normal"

                run_ux, run_uy = _unit_vector(tx - carrier_x, ty - carrier_y)
                tx_long, ty_long = _clip_to_pitch(
                    tx + 1.5 * run_ux,
                    ty + 1.5 * run_uy,
                    match_data
                )

                arrow = FancyArrowPatch(
                    (carrier_x, carrier_y),
                    (tx_long, ty_long),
                    arrowstyle="->",
                    mutation_scale=run_highlight_ms,
                    linewidth=run_highlight_lw,
                    linestyle="--",
                    color=run_highlight_color,
                    alpha=run_highlight_alpha,
                    zorder=6
                )
                ax.add_patch(arrow)

                mid_x = (carrier_x + tx_long) / 2
                mid_y = (carrier_y + ty_long) / 2

                ax.text(
                    mid_x,
                    mid_y,
                    f"{100 * best_run_row['run_value']:.1f}%",
                    fontsize=9,
                    color=run_highlight_color,
                    weight=run_label_weight,
                    ha="center",
                    va="bottom",
                    zorder=7,
                    bbox=dict(facecolor="white", alpha=0.45, edgecolor="none", pad=0.25)
                )

    # Draw shot option
    if show_shot_option and shot_metrics is not None:
        if shot_xg > shot_display_threshold:
            goal_x = shot_metrics["goal_x"]
            goal_y = shot_metrics["goal_y"]

            # Green if shot is better than best offensive pass return
            shot_color = shot_default_color
            shot_alpha = all_alpha
            shot_lw = linewidth_all

            if pd.notna(best_pass_offensive_return) and shot_xg > best_pass_offensive_return:
                shot_color = best_color
                shot_alpha = best_alpha
                shot_lw = linewidth_best

            arrow = FancyArrowPatch(
                (shot_metrics["ball_x"], shot_metrics["ball_y"]),
                (goal_x, goal_y),
                arrowstyle="->",
                mutation_scale=mutation_scale + 2,
                linewidth=shot_lw,
                color=shot_color,
                alpha=shot_alpha,
                zorder=6
            )
            ax.add_patch(arrow)

            mid_x = (shot_metrics["ball_x"] + goal_x) / 2
            mid_y = (shot_metrics["ball_y"] + goal_y) / 2

            ax.text(
                mid_x,
                mid_y,
                f"{100 * shot_xg:.1f}%",
                fontsize=9,
                color=shot_color,
                weight="bold" if shot_color == best_color else "normal",
                ha="center",
                va="bottom",
                zorder=7,
                bbox=dict(facecolor="white", alpha=0.45, edgecolor="none", pad=0.25)
            )

    return {
        "best_pass_offensive_return": best_pass_offensive_return,
        "best_pass_utility": best_pass_utility,
        "best_run_value": best_run_value,
        "shot_xg": shot_xg
    }



# ============================================================
# FORD-FULKERSON
# ============================================================

def bfs_find_augmenting_path(residual_graph: dict, source: str, sink: str):
    visited = set([source])
    queue = deque([source])
    parent = {source: None}

    while queue:
        u = queue.popleft()
        for v, capacity in residual_graph[u].items():
            if v not in visited and capacity > 1e-12:
                visited.add(v)
                parent[v] = u
                if v == sink:
                    return parent
                queue.append(v)

    return None


def ford_fulkerson_max_flow(capacity_graph: dict, source: str, sink: str):
    residual_graph = {}

    for u in capacity_graph:
        residual_graph.setdefault(u, {})
        for v, cap in capacity_graph[u].items():
            residual_graph.setdefault(v, {})
            residual_graph[u][v] = float(cap)
            residual_graph[v].setdefault(u, 0.0)

    max_flow = 0.0

    while True:
        parent = bfs_find_augmenting_path(residual_graph, source, sink)
        if parent is None:
            break

        path_flow = math.inf
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual_graph[u][v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            v = u

        max_flow += path_flow

    flow_dict = {}
    for u in capacity_graph:
        flow_dict[u] = {}
        for v in capacity_graph[u]:
            flow_dict[u][v] = capacity_graph[u][v] - residual_graph[u][v]

    return max_flow, flow_dict, residual_graph


# ============================================================
# FLOW GRAPH WITH PASSES + SHOT
# ============================================================

def build_frame_flow_graph(
    frame_data: dict,
    df_frame_players: pd.DataFrame,
    events_df: pd.DataFrame,
    match_data: dict
):
    # NEW HYBRID LOOKUP
    ball_carrier_id = get_resolved_ball_carrier(frame_data, frame_team_array, player_team_map)
    
    if ball_carrier_id is None:
        return {}, pd.DataFrame(), None

    carrier_row = df_frame_players[df_frame_players["player_id"] == ball_carrier_id]
    if carrier_row.empty:
        return {}, pd.DataFrame(), None

    df_options = get_active_passing_options(
        events_df=events_df,
        frame_number=frame_data["frame"],
        ball_carrier_id=ball_carrier_id
    )

    visible_player_ids = set(df_frame_players["player_id"].dropna().tolist())
    df_options = df_options[df_options["player_id"].isin(visible_player_ids)].copy()

    pass_rows = []
    for _, option in df_options.iterrows():
        metrics = compute_pass_metrics_for_frame_option(
            option_row=option,
            df_frame_players=df_frame_players,
            frame_data=frame_data,
            match_data=match_data
        )
        if metrics is not None:
            pass_rows.append(metrics)

    pass_df = pd.DataFrame(pass_rows) if pass_rows else pd.DataFrame()

    source_node = f"SOURCE_{ball_carrier_id}"
    sink_node = "GOAL"
    capacity_graph = {source_node: {}, sink_node: {}}

    run_df = _get_run_options_for_carrier(
        frame_data=frame_data,
        carrier_row=carrier_row,
        df_frame_players=df_frame_players,
        match_data=match_data,
        events_df=events_df,
    )

    if not run_df.empty:
        run_df_for_flow = run_df.copy()
        if "run_net_xthreat" in run_df_for_flow.columns:
            run_df_for_flow["run_value"] = run_df_for_flow["run_net_xthreat"]
        else:
            run_df_for_flow["run_value"] = run_df_for_flow["destination_xthreat"]

        # Keep only the top run options so added angular granularity
        # does not disproportionately inflate total flow.
        run_df_for_flow = (
            run_df_for_flow[run_df_for_flow["run_value"] > 0.0]
            .sort_values("run_value", ascending=False)
            .head(5)
            .copy()
        )

        for _, run_row in run_df_for_flow.iterrows():
            run_node = f"RUN_{run_row['run_type']}"

            run_value_capacity = max(float(run_row.get("run_value", 0.0)), 0.0)
            if RUN_SOURCE_EDGE_MODE == "control_capacity":
                source_capacity = max(float(run_row["control_capacity"]), 0.0)
            else:
                source_capacity = run_value_capacity
            sink_capacity = max(float(run_row.get("run_net_xthreat", run_row["destination_xthreat"])), 0.0)

            capacity_graph.setdefault(source_node, {})
            capacity_graph.setdefault(run_node, {})
            capacity_graph.setdefault(sink_node, {})

            capacity_graph[source_node][run_node] = source_capacity
            capacity_graph[run_node][sink_node] = sink_capacity

    if not pass_df.empty:
        pass_df = (
            pass_df.sort_values("utility", ascending=False)
            .drop_duplicates(subset=["receiver_id"])
            .copy()
        )

        for _, row in pass_df.iterrows():
            receiver_node = f"PLAYER_{int(row['receiver_id'])}"

            pass_capacity = max(float(row["utility"]), 0.0)

            capacity_graph.setdefault(source_node, {})
            capacity_graph.setdefault(receiver_node, {})
            capacity_graph.setdefault(sink_node, {})

            capacity_graph[source_node][receiver_node] = pass_capacity
            capacity_graph[receiver_node][sink_node] = 1.0

    shot_metrics = compute_shot_metric(
        frame_data=frame_data,
        match_data=match_data,
        ball_carrier_id=ball_carrier_id,
        df_frame_players=df_frame_players,
    )
    if shot_metrics is not None and shot_metrics["xg"] > 0:
        capacity_graph[source_node][sink_node] = float(shot_metrics["xg"])
    else:
        shot_metrics = None

    return capacity_graph, pass_df, shot_metrics


def compute_frame_network_flow(
    frame_data: dict,
    df_frame_players: pd.DataFrame,
    events_df: pd.DataFrame,
    match_data: dict
):
    capacity_graph, pass_df, shot_metrics = build_frame_flow_graph(
        frame_data=frame_data,
        df_frame_players=df_frame_players,
        events_df=events_df,
        match_data=match_data
    )

    if not capacity_graph or "GOAL" not in capacity_graph:
        return 0.0, capacity_graph, {}, pass_df, shot_metrics

    source_node = next(iter([k for k in capacity_graph.keys() if k.startswith("SOURCE_")]))
    sink_node = "GOAL"

    flow_value, flow_dict, _ = ford_fulkerson_max_flow(
        capacity_graph=capacity_graph,
        source=source_node,
        sink=sink_node
    )

    return flow_value, capacity_graph, flow_dict, pass_df, shot_metrics







# ============================================================
# ANIMATION WITH PASSES + SHOT + FLOW
# ============================================================

def animate_tracking_sequence_with_flow(
    tracking_data: list,
    events_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    player_lookup: pd.DataFrame,
    match_data: dict,
    output_path: str = "tracking_sequence_with_flow.gif",
    show_labels: bool = True,
    label_mode: str = "number",
    show_undetected: bool = True,
    highlight_possession: bool = True,
    show_decision_arrows: bool = True,
    frame_interval_seconds: float = 0.1,
    figsize: tuple = (14, 9),
    dpi: int = 120,
    flow_box_loc: tuple = (0.02, 0.95),
    flow_series_output_path: str | None = None
):
    if end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame.")

    frame_map = {frame["frame"]: frame for frame in tracking_data}
    selected_frame_numbers = [
        frame_number
        for frame_number in range(start_frame, end_frame + 1)
        if frame_number in frame_map
    ]

    if not selected_frame_numbers:
        raise ValueError("No frames found in the requested interval.")

    home_team_id = match_data["home_team"]["id"]
    away_team_id = match_data["away_team"]["id"]
    home_color = match_data["home_team_kit"]["jersey_color"]
    away_color = match_data["away_team_kit"]["jersey_color"]
    pitch_length = match_data["pitch_length"]
    pitch_width = match_data["pitch_width"]

    # Precompute flow values
    flow_records = []

    for frame_number in selected_frame_numbers:
        frame_data = frame_map[frame_number]
        df_players = frame_to_player_dataframe(frame_data, player_lookup)

        if not show_undetected:
            df_players = df_players[df_players["is_detected"] == True].copy()

        flow_value, _, _, pass_df, shot_metrics = compute_frame_network_flow(
            frame_data=frame_data,
            df_frame_players=df_players,
            events_df=events_df,
            match_data=match_data
        )

        flow_records.append({
            "frame": frame_number,
            "timestamp": frame_data.get("timestamp", None),
            "time_seconds_from_start": (frame_number - selected_frame_numbers[0]) * frame_interval_seconds,
            "network_flow": flow_value,
            "n_pass_options": len(pass_df),
            "shot_xg": shot_metrics["xg"] if shot_metrics is not None else np.nan
        })

    flow_df = pd.DataFrame(flow_records)

    fig, ax = plt.subplots(figsize=figsize)
    # Create a wider figure with two side-by-side axes: ax (pitch) and ax_flow (graph)
    fig, (ax, ax_flow) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})

    def update(frame_number):
        """
        Animation update function for each frame.
        Clears the axis, redraws the pitch, plots player & ball positions,
        computes decision arrows, and tracks the continuous network flow.
        """
        # 1. CLEAR BOTH PANELS
        ax.clear()
        ax_flow.clear()
        
        # 2. DRAW PITCH & PLAYERS (Left Panel)
        draw_pitch(ax, pitch_length=pitch_length, pitch_width=pitch_width)

        frame_data = frame_map.get(frame_number)
        if frame_data is None:
            return

        # Parse tracking frame into player DataFrame
        df_players = frame_to_player_dataframe(frame_data, player_lookup)

        # Filter out undetected players if requested
        if not show_undetected:
            df_players = df_players[df_players["is_detected"] == True].copy()

        # Split players by team
        df_home = df_players[df_players["team_id"] == home_team_id]
        df_away = df_players[df_players["team_id"] == away_team_id]
        df_unknown = df_players[df_players["team_id"].isna()]

        # Plot home players (ADDED LABEL)
        if not df_home.empty:
            ax.scatter(df_home["x"], df_home["y"], s=220, c=home_color, edgecolors="black", 
                       linewidths=1.2, alpha=df_home["is_detected"].map(lambda v: 1.0 if v else 0.35),
                       label=match_data["home_team"]["name"])

        # Plot away players (ADDED LABEL)
        if not df_away.empty:
            ax.scatter(df_away["x"], df_away["y"], s=220, c=away_color, edgecolors="black", 
                       linewidths=1.2, alpha=df_away["is_detected"].map(lambda v: 1.0 if v else 0.35),
                       label=match_data["away_team"]["name"])

        # Plot unknown players
        if not df_unknown.empty:
            ax.scatter(df_unknown["x"], df_unknown["y"], s=220, c="gray", edgecolors="black", 
                       linewidths=1.2, alpha=0.7)

        # Add jersey numbers or player ID labels
        for _, row in df_players.iterrows():
            text_label = str(row["jersey_number"]) if pd.notna(row["jersey_number"]) else str(row["player_id"])
            ax.text(row["x"], row["y"], text_label, ha="center", va="center", fontsize=8,
                    color="white" if row["team_id"] in [home_team_id, away_team_id] else "black", weight="bold")

        # Resolve Ball Carrier using the robust hybrid method
        ball_carrier_id = get_resolved_ball_carrier(frame_data, frame_team_array, player_team_map)

        # Draw decision arrows (Passes only, we will draw the custom shot arrow next)
        if show_decision_arrows:
            draw_decision_arrows(
                ax=ax, frame_data=frame_data, df_frame_players=df_players, events_df=events_df,
                match_data=match_data, show_all_passes=True, show_best_pass=True, 
                show_shot_option=False # Disable default so we can use our Gold Arrow
            )

        # Draw Custom Gold Shot Arrow
        if show_decision_arrows and ball_carrier_id is not None:
            carrier_row = df_players[df_players["player_id"] == ball_carrier_id]
            if not carrier_row.empty:
                ball_carrier_x = carrier_row.iloc[0]["x"]
                ball_carrier_y = carrier_row.iloc[0]["y"]
                shot_metrics = compute_shot_metric(
                    frame_data=frame_data,
                    match_data=match_data,
                    ball_carrier_id=ball_carrier_id,
                    df_frame_players=df_players,
                )
                
                # Threshold set to 0.02 (2% xG)
                if shot_metrics and shot_metrics["xg"] > 0.02:
                    goal_x = shot_metrics["goal_x"]
                    goal_y = shot_metrics["goal_y"]
                    shot_arrow = FancyArrowPatch((ball_carrier_x, ball_carrier_y), (goal_x, goal_y),
                                                 arrowstyle='->,head_width=5,head_length=8', color='gold', 
                                                 linewidth=3, zorder=10, alpha=0.9, 
                                                 label=f'Potential Shot (xG: {shot_metrics["xg"]*100:.1f}%)')
                    ax.add_patch(shot_arrow)

        # Plot ball (ADDED LABEL)
        ball = frame_data.get("ball_data", None)
        if ball is not None:
            ax.scatter(ball["x"], ball["y"], s=120, c="orange", edgecolors="black", linewidths=1.2, zorder=8, label="Ball")

        # Highlight player in possession (ADDED LABEL)
        if highlight_possession and ball_carrier_id is not None:
            poss_row = df_players[df_players["player_id"] == ball_carrier_id]
            if not poss_row.empty:
                ax.scatter([poss_row.iloc[0]["x"]], [poss_row.iloc[0]["y"]], s=420, facecolors="none",
                           edgecolors="gold", linewidths=2.5, zorder=9, label="Player in possession")

        # Update left panel title
        timestamp = frame_data.get("timestamp", "Unknown")
        period = frame_data.get("period", "Unknown")
        ax.set_title(f"Tracking Frame {frame_number} | Time {timestamp} | Period {period}")
        
        # Prevent duplicate legend items and draw the legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        # ========================================================
        # 3. DRAW DYNAMIC FLOW GRAPH (Right Panel)
        # ========================================================
        # Retrieve historical flow data up to current frame
        current_idx = flow_df.index[flow_df['frame'] == frame_number].tolist()[0]
        history = flow_df.iloc[:current_idx + 1].copy()
        
        # Smooth out gaps using np.nan
        history["continuous_flow"] = history["network_flow"].replace(0.0, np.nan).ffill().fillna(0.0)

        times = history["time_seconds_from_start"].values
        flows = history["continuous_flow"].values
        
        # Use the fast lookup array to resolve historical teams
        historical_teams = []
        for f in history["frame"]:
            if f < len(frame_team_array):
                historical_teams.append(frame_team_array[f])
            else:
                historical_teams.append(-1)
        historical_teams = np.array(historical_teams)

        # Split flow by team
        swiss_flow = np.where(historical_teams == home_team_id, flows, np.nan)
        german_flow = np.where(historical_teams == away_team_id, -flows, np.nan)

        swiss_valid = np.isfinite(swiss_flow)
        german_valid = np.isfinite(german_flow)

        ax_flow.fill_between(
            times,
            0.0,
            swiss_flow,
            where=swiss_valid,
            color='red',
            alpha=0.22,
            interpolate=True,
        )
        ax_flow.fill_between(
            times,
            0.0,
            german_flow,
            where=german_valid,
            color='black',
            alpha=0.22,
            interpolate=True,
        )
        
        ax_flow.plot(times, swiss_flow, color='red', linewidth=3, label=match_data["home_team"]["name"])
        ax_flow.plot(times, german_flow, color='black', linewidth=3, label=match_data["away_team"]["name"])

        # Center Line and Limits
        max_limit = flow_df["network_flow"].max() * 1.1
        if pd.isna(max_limit) or max_limit == 0: max_limit = 1.0 # Safety fallback
        
        ax_flow.axhline(0, color='gray', linewidth=1)
        ax_flow.set_xlim(flow_df["time_seconds_from_start"].min(), flow_df["time_seconds_from_start"].max())
        ax_flow.set_ylim(-max_limit, max_limit) # Bottom to Top
        
        # Format Y-axis to absolute values (No negative signs for Germany)
        ticks = ax_flow.get_yticks()
        ax_flow.set_yticks(ticks)  # This line prevents the UserWarning
        ax_flow.set_yticklabels([f"{abs(t):.2f}" for t in ticks])
        
        ax_flow.set_title("Live Maximum Flow Capacity")
        ax_flow.set_xlabel("Sequence Time (s)")
        ax_flow.set_ylabel("Network Flow (Swiss ↑ | Germany ↓)")
        ax_flow.grid(True, linestyle='--', alpha=0.6)
        
        # Legend handling for the right panel
        handles_flow, labels_flow = ax_flow.get_legend_handles_labels()
        if handles_flow:
            by_label_flow = dict(zip(labels_flow, handles_flow))
            ax_flow.legend(by_label_flow.values(), by_label_flow.keys(), loc="upper left")

    # ========================================================
    # OUTSIDE THE UPDATE LOOP: SAVE ANIMATION AND PLOT
    # ========================================================
    interval_ms = int(frame_interval_seconds * 1000)

    anim = FuncAnimation(
        fig,
        update,
        frames=selected_frame_numbers,
        interval=interval_ms,
        repeat=False
    )

    if output_path.lower().endswith(".gif"):
        writer = PillowWriter(fps=round(1 / frame_interval_seconds))
        anim.save(output_path, writer=writer, dpi=dpi)
    elif output_path.lower().endswith(".mp4"):
        writer = FFMpegWriter(fps=round(1 / frame_interval_seconds))
        anim.save(output_path, writer=writer, dpi=dpi)
    else:
        plt.close(fig)
        raise ValueError("output_path must end with '.gif' or '.mp4'.")

    plt.close(fig)
    print(f"Animation saved to: {output_path}")

    return flow_df



def precompute_full_game_network_flow(
    tracking_data: list,
    events_df: pd.DataFrame,
    player_lookup: pd.DataFrame,
    match_data: dict,
    output_path: str | Path,
    show_undetected: bool = True,
    frame_interval_seconds: float = 0.1,
    progress_every: int = 1000,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> pd.DataFrame:
    def to_json_safe(value):
        if isinstance(value, dict):
            return {str(k): to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, np.bool_):
            return bool(value)
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def df_to_json_records(df: pd.DataFrame, keep_cols: list[str] | None = None) -> str:
        if df is None or df.empty:
            return "[]"
        if keep_cols is not None:
            cols = [c for c in keep_cols if c in df.columns]
            safe_df = df[cols].copy()
        else:
            safe_df = df.copy()
        return json.dumps(to_json_safe(safe_df.to_dict("records")), ensure_ascii=True, separators=(",", ":"))

    def obj_to_json(obj) -> str:
        return json.dumps(to_json_safe(obj), ensure_ascii=True, separators=(",", ":"))

    frame_map = {
        int(fr["frame"]): fr
        for fr in tracking_data
        if fr.get("frame") is not None
    }
    frame_numbers = sorted(frame_map.keys())

    if not frame_numbers:
        raise ValueError("No tracking frames were found for precomputation.")

    if start_frame is not None:
        frame_numbers = [f for f in frame_numbers if f >= int(start_frame)]
    if end_frame is not None:
        frame_numbers = [f for f in frame_numbers if f <= int(end_frame)]

    if not frame_numbers:
        raise ValueError("No tracking frames left after applying precompute frame range.")

    start_frame = frame_numbers[0]
    total_frames = len(frame_numbers)
    records = []

    giveaway_end_types = {
        "possession_loss",
        "clearance",
        "direct_disruption",
        "indirect_disruption",
        "foul_committed",
    }
    possession_terminal_end_types = set(giveaway_end_types) | {"pass", "shot"}

    possession_events = events_df[
        events_df["event_type"].astype(str).str.lower() == "player_possession"
    ].copy()

    frame_to_possession_event = {}
    if not possession_events.empty:
        possession_events = possession_events[
            possession_events["frame_start"].notna() & possession_events["frame_end"].notna()
        ].copy()
        possession_events["frame_start"] = possession_events["frame_start"].astype(int)
        possession_events["frame_end"] = possession_events["frame_end"].astype(int)
        possession_events = possession_events.sort_values(["frame_start", "frame_end"]).copy()

        next_possession_id = 1
        for _, ev in possession_events.iterrows():
            ev_id = ev.get("event_id", np.nan)
            ev_id = str(ev_id) if pd.notna(ev_id) else ""

            ev_start = int(ev["frame_start"])
            ev_end = int(ev["frame_end"])
            ev_end_type = str(ev.get("end_type", "")).strip().lower()
            ev_team_id = ev.get("team_id", np.nan)
            ev_player_id = ev.get("player_id", np.nan)
            ev_target_option = ev.get("targeted_passing_option_event_id", np.nan)
            ev_target_option = str(ev_target_option) if pd.notna(ev_target_option) else ""

            event_payload = {
                "event_id": ev_id,
                "frame_start": ev_start,
                "frame_end": ev_end,
                "end_type": ev_end_type,
                "team_id": ev_team_id,
                "player_id": ev_player_id,
                "targeted_passing_option_event_id": ev_target_option,
                "possession_id": next_possession_id,
            }

            for ff in range(ev_start, ev_end + 1):
                prev = frame_to_possession_event.get(ff)
                if prev is None or ev_start >= int(prev["frame_start"]):
                    frame_to_possession_event[ff] = event_payload

            if ev_end_type in possession_terminal_end_types:
                next_possession_id += 1

    print(f"[precompute] Computing max-flow cache for {total_frames} frames...")

    for i, frame_number in enumerate(frame_numbers, start=1):
        frame_data = frame_map[frame_number]
        df_players = frame_to_player_dataframe(frame_data, player_lookup)

        if not show_undetected:
            df_players = df_players[df_players["is_detected"] == True].copy()

        flow_value, capacity_graph, flow_dict, pass_df, shot_metrics = compute_frame_network_flow(
            frame_data=frame_data,
            df_frame_players=df_players,
            events_df=events_df,
            match_data=match_data,
        )

        carrier_id = get_resolved_ball_carrier(frame_data, frame_team_array, player_team_map)
        carrier_x = np.nan
        carrier_y = np.nan
        carrier_row = pd.DataFrame()
        if carrier_id is not None and "player_id" in df_players.columns:
            carrier_row = df_players[df_players["player_id"] == carrier_id]
            if not carrier_row.empty:
                carrier_x = float(carrier_row.iloc[0]["x"])
                carrier_y = float(carrier_row.iloc[0]["y"])

        pass_df = pass_df.copy() if not pass_df.empty else pd.DataFrame()
        if not pass_df.empty and "xpass" in pass_df.columns:
            pass_df["pass_probability"] = pass_df["xpass"]
        if not pass_df.empty and "utility" in pass_df.columns:
            pass_df["display_probability"] = pass_df["utility"]

        best_pass_utility = float(pass_df["utility"].max()) if (not pass_df.empty and "utility" in pass_df.columns) else np.nan
        best_pass_offensive_return = float(pass_df["offensive_return"].max()) if (not pass_df.empty and "offensive_return" in pass_df.columns) else np.nan
        n_options = int(len(pass_df))
        pass_capacity_sum = float(np.clip(pass_df["utility"], 0.0, None).sum()) if (not pass_df.empty and "utility" in pass_df.columns) else 0.0

        run_df_for_vis = pd.DataFrame()
        run_df_for_flow = pd.DataFrame()
        if carrier_id is not None and not carrier_row.empty:
            run_df_raw = _get_run_options_for_carrier(
                frame_data=frame_data,
                carrier_row=carrier_row,
                df_frame_players=df_players,
                match_data=match_data,
                events_df=events_df,
            )
            if not run_df_raw.empty:
                run_df_for_vis = run_df_raw.copy()
                if "run_net_xthreat" in run_df_for_vis.columns:
                    run_df_for_vis["run_value"] = run_df_for_vis["run_net_xthreat"]
                else:
                    run_df_for_vis["run_value"] = run_df_for_vis["destination_xthreat"]

                run_df_for_vis = (
                    run_df_for_vis.sort_values("run_value", ascending=False)
                    .drop_duplicates(subset=["run_type"])
                    .copy()
                )
                if "prob_success" in run_df_for_vis.columns:
                    run_df_for_vis["run_probability"] = run_df_for_vis["prob_success"]
                else:
                    run_df_for_vis["run_probability"] = run_df_for_vis["run_value"]

                run_df_for_flow = (
                    run_df_for_vis[run_df_for_vis["run_value"] > 0.0]
                    .sort_values("run_value", ascending=False)
                    .head(5)
                    .copy()
                )

        best_run_value = float(run_df_for_vis["run_value"].max()) if (not run_df_for_vis.empty and "run_value" in run_df_for_vis.columns) else np.nan

        shot_xg = float(shot_metrics["xg"]) if shot_metrics is not None else np.nan
        shot_capacity = float(max(shot_xg, 0.0)) if pd.notna(shot_xg) else 0.0
        shot_option_payload = None
        if shot_metrics is not None:
            shot_option_payload = dict(shot_metrics)
            shot_option_payload["shot_probability"] = float(shot_metrics.get("xg", np.nan))

        active_possession = frame_to_possession_event.get(int(frame_number))
        possession_id = np.nan
        action_taken = ""
        action_value = np.nan
        action_taken_option_type = ""
        action_taken_option_id = ""
        action_taken_matched_in_options = False

        if active_possession is not None:
            possession_id = int(active_possession["possession_id"])

            if int(frame_number) == int(active_possession["frame_end"]):
                end_type = str(active_possession.get("end_type", "")).strip().lower()

                if end_type == "pass":
                    action_taken = "pass"
                    targeted_option_id = str(active_possession.get("targeted_passing_option_event_id", "")).strip()
                    if targeted_option_id:
                        action_taken_option_type = "pass_option"
                        action_taken_option_id = targeted_option_id

                        if not pass_df.empty and "passing_option_event_id" in pass_df.columns:
                            match_mask = pass_df["passing_option_event_id"].astype(str) == targeted_option_id
                            if bool(match_mask.any()):
                                matched_row = pass_df.loc[match_mask].iloc[0]
                                matched_utility = matched_row.get("utility", np.nan)
                                if pd.notna(matched_utility):
                                    action_value = float(matched_utility)
                                    action_taken_matched_in_options = True

                elif end_type == "shot":
                    action_taken = "shot"
                    action_taken_option_type = "shot_option"
                    action_taken_option_id = "direct_shot"
                    if pd.notna(shot_xg):
                        action_value = float(shot_xg)
                        action_taken_matched_in_options = True

                elif end_type in giveaway_end_types:
                    action_taken = "giveaway"

                    action_x = carrier_x
                    action_y = carrier_y
                    if (not np.isfinite(action_x) or not np.isfinite(action_y)) and frame_data.get("ball_data") is not None:
                        ball = frame_data.get("ball_data") or {}
                        action_x = float(ball.get("x", np.nan))
                        action_y = float(ball.get("y", np.nan))

                    action_team_id = active_possession.get("team_id", np.nan)
                    action_period = int(frame_data.get("period", 1))
                    if pd.notna(action_team_id) and np.isfinite(action_x) and np.isfinite(action_y):
                        loss_cost = get_xcost(
                            float(action_x),
                            float(action_y),
                            int(action_team_id),
                            action_period,
                            match_data,
                        )
                        if np.isfinite(loss_cost):
                            action_value = -float(loss_cost)

        poss_team_id = frame_team_array[frame_number] if frame_number < len(frame_team_array) else -1
        if poss_team_id == home_team_id:
            team_side = "home"
            signed_flow = float(flow_value)
        elif poss_team_id == away_team_id:
            team_side = "away"
            signed_flow = -float(flow_value)
        else:
            team_side = "none"
            signed_flow = 0.0

        best_action = "none"
        pass_val = best_pass_utility if pd.notna(best_pass_utility) else -np.inf
        shot_val = shot_xg if pd.notna(shot_xg) else -np.inf
        if np.isfinite(pass_val) and pass_val >= shot_val:
            best_action = "pass"
        elif np.isfinite(shot_val):
            best_action = "shot"

        records.append(
            {
                "frame": int(frame_number),
                "timestamp": frame_data.get("timestamp", ""),
                "period": frame_data.get("period", ""),
                "t_seconds": float((frame_number - start_frame) * frame_interval_seconds),
                "possession_id": possession_id,
                "team_side": team_side,
                "carrier_id": int(carrier_id) if carrier_id is not None else np.nan,
                "carrier_x": carrier_x,
                "carrier_y": carrier_y,
                "action_taken": action_taken,
                "action_value": action_value,
                "action_taken_option_type": action_taken_option_type,
                "action_taken_option_id": action_taken_option_id,
                "action_taken_matched_in_options": bool(action_taken_matched_in_options),
                "n_options": n_options,
                "best_pass_utility": best_pass_utility,
                "best_pass_offensive_return": best_pass_offensive_return,
                "best_run_value": best_run_value,
                "pass_capacity_sum": pass_capacity_sum,
                "shot_xg": shot_xg,
                "shot_capacity": shot_capacity,
                "best_action": best_action,
                "frame_flow_capacity": float(flow_value),
                "network_flow": float(flow_value),
                "flow_proxy": signed_flow,
                "pass_options_json": df_to_json_records(
                    pass_df,
                    keep_cols=[
                        "passing_option_event_id",
                        "receiver_id",
                        "receiver_x",
                        "receiver_y",
                        "xpass",
                        "pass_probability",
                        "xthreat",
                        "xcost",
                        "offensive_return",
                        "utility",
                        "display_probability",
                    ],
                ),
                "run_options_json": df_to_json_records(
                    run_df_for_vis,
                    keep_cols=[
                        "run_type",
                        "target_x",
                        "target_y",
                        "eval_target_x",
                        "eval_target_y",
                        "destination_xthreat",
                        "control_capacity",
                        "prob_success",
                        "run_xthreat",
                        "run_net_xthreat",
                        "run_value",
                        "run_probability",
                    ],
                ),
                "run_options_flow_top5_json": df_to_json_records(
                    run_df_for_flow,
                    keep_cols=[
                        "run_type",
                        "target_x",
                        "target_y",
                        "eval_target_x",
                        "eval_target_y",
                        "destination_xthreat",
                        "control_capacity",
                        "prob_success",
                        "run_xthreat",
                        "run_net_xthreat",
                        "run_value",
                        "run_probability",
                    ],
                ),
                "shot_option_json": obj_to_json(shot_option_payload),
                "capacity_graph_json": obj_to_json(capacity_graph),
                "flow_dict_json": obj_to_json(flow_dict),
            }
        )

        if progress_every > 0 and (i % progress_every == 0 or i == total_frames):
            print(f"[precompute] Processed {i}/{total_frames} frames...")

    flow_df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flow_df.to_csv(output_path, index=False)

    metadata = {
        "output_csv": str(output_path),
        "cache_version": 3,
        "n_frames": int(len(flow_df)),
        "frame_start": int(flow_df["frame"].min()),
        "frame_end": int(flow_df["frame"].max()),
        "frame_interval_seconds": float(frame_interval_seconds),
        "columns": list(flow_df.columns),
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[precompute] Cache saved to: {output_path}")
    print(f"[precompute] Metadata saved to: {meta_path}")
    return flow_df


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Network flow visualization and full-game max-flow precompute utility."
    )
    parser.add_argument(
        "--mode",
        choices=["animate", "precompute"],
        default="animate",
        help="Run animation rendering or precompute full-game max-flow cache.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output file path (.gif/.mp4 for animate, .csv for precompute).",
    )
    parser.add_argument("--start-frame", type=int, default=9560)
    parser.add_argument("--end-frame", type=int, default=9850)
    parser.add_argument("--frame-interval-seconds", type=float, default=0.1)
    parser.add_argument(
        "--hide-undetected",
        action="store_true",
        help="If set, drop undetected players from frame-level calculations.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Progress print interval in frames for precompute mode.",
    )
    parser.add_argument(
        "--precompute-start-frame",
        type=int,
        default=None,
        help="Optional inclusive start frame for precompute mode.",
    )
    parser.add_argument(
        "--precompute-end-frame",
        type=int,
        default=None,
        help="Optional inclusive end frame for precompute mode.",
    )
    return parser.parse_args()


def main():
    args = parse_cli_args()
    output_dir = DATA_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    show_undetected = not bool(args.hide_undetected)

    if args.mode == "precompute":
        output_path = (
            Path(args.output_path)
            if args.output_path
            else output_dir / "full_game_max_flow_cache.csv"
        )
        precompute_full_game_network_flow(
            tracking_data=tracking_data,
            events_df=df_events,
            player_lookup=player_lookup,
            match_data=match_data,
            output_path=output_path,
            show_undetected=show_undetected,
            frame_interval_seconds=args.frame_interval_seconds,
            progress_every=args.progress_every,
            start_frame=args.precompute_start_frame,
            end_frame=args.precompute_end_frame,
        )
        return

    output_path = (
        str(Path(args.output_path))
        if args.output_path
        else str(output_dir / "play_with_flow_and_shot.gif")
    )

    animate_tracking_sequence_with_flow(
        tracking_data=tracking_data,
        events_df=df_events,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        player_lookup=player_lookup,
        match_data=match_data,
        output_path=output_path,
        show_labels=True,
        label_mode="number",
        show_undetected=show_undetected,
        highlight_possession=True,
        show_decision_arrows=True,
        frame_interval_seconds=args.frame_interval_seconds,
    )
    print(f"Process complete. Check {output_path} for the final output.")


if __name__ == "__main__":
    main()