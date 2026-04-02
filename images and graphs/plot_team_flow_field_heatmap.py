import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_match_context(match_path: Path) -> dict:
    with open(match_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {
        "pitch_length": float(m.get("pitch_length", 105.0)),
        "pitch_width": float(m.get("pitch_width", 68.0)),
        "home_name": str(m.get("home_team", {}).get("name", "Home")),
        "away_name": str(m.get("away_team", {}).get("name", "Away")),
    }


def compute_team_grid(df: pd.DataFrame, value_col: str, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _, _ = np.histogram2d(df["carrier_y"], df["carrier_x"], bins=[y_edges, x_edges])
    sums, _, _ = np.histogram2d(df["carrier_y"], df["carrier_x"], bins=[y_edges, x_edges], weights=df[value_col])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    return mean, counts


def build_grid_summary(team: str, mean: np.ndarray, counts: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> pd.DataFrame:
    rows = []
    for yi in range(mean.shape[0]):
        for xi in range(mean.shape[1]):
            x0 = float(x_edges[xi])
            x1 = float(x_edges[xi + 1])
            y0 = float(y_edges[yi])
            y1 = float(y_edges[yi + 1])
            rows.append(
                {
                    "team_side": team,
                    "x_bin": int(xi),
                    "y_bin": int(yi),
                    "x_left": x0,
                    "x_right": x1,
                    "y_bottom": y0,
                    "y_top": y1,
                    "x_center": 0.5 * (x0 + x1),
                    "y_center": 0.5 * (y0 + y1),
                    "count": int(counts[yi, xi]),
                    "mean_flow": float(mean[yi, xi]) if np.isfinite(mean[yi, xi]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def draw_pitch_overlay(ax, half_length: float, half_width: float):
    ax.plot(
        [-half_length, half_length, half_length, -half_length, -half_length],
        [-half_width, -half_width, half_width, half_width, -half_width],
        color="black",
        linewidth=1.0,
    )
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)


def main():
    parser = argparse.ArgumentParser(description="Plot team flow amount by field location using square bins.")
    parser.add_argument("--cache", default="outputs/full_game_max_flow_cache.csv", help="Input full-game cache CSV")
    parser.add_argument("--match", default="match.json", help="Match metadata JSON")
    parser.add_argument("--metric", choices=["frame_flow_capacity", "abs_flow_proxy"], default="frame_flow_capacity", help="Flow metric to aggregate")
    parser.add_argument("--x-bins", type=int, default=8, help="Number of field bins along x")
    parser.add_argument("--y-bins", type=int, default=6, help="Number of field bins along y")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum rows in a bin to display a value")
    parser.add_argument("--output", default="outputs/team_flow_field_heatmap.png", help="Output PNG path")
    parser.add_argument("--output-grid", default="outputs/team_flow_field_heatmap_grid.csv", help="Output CSV with bin summaries")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    match_path = Path(args.match)
    output_path = Path(args.output)
    output_grid = Path(args.output_grid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_grid.parent.mkdir(parents=True, exist_ok=True)

    ctx = load_match_context(match_path)
    half_length = 0.5 * ctx["pitch_length"]
    half_width = 0.5 * ctx["pitch_width"]

    usecols = ["team_side", "carrier_x", "carrier_y", "frame_flow_capacity", "flow_proxy"]
    df = pd.read_csv(cache_path, usecols=usecols)

    df["team_side"] = df["team_side"].fillna("").astype(str).str.strip().str.lower()
    df["carrier_x"] = pd.to_numeric(df["carrier_x"], errors="coerce")
    df["carrier_y"] = pd.to_numeric(df["carrier_y"], errors="coerce")
    df["frame_flow_capacity"] = pd.to_numeric(df["frame_flow_capacity"], errors="coerce")
    df["flow_proxy"] = pd.to_numeric(df["flow_proxy"], errors="coerce")
    df["abs_flow_proxy"] = df["flow_proxy"].abs()

    value_col = str(args.metric)
    df = df[
        df["team_side"].isin(["home", "away"])
        & np.isfinite(df["carrier_x"])
        & np.isfinite(df["carrier_y"])
        & np.isfinite(df[value_col])
    ].copy()

    if df.empty:
        raise ValueError("No finite rows with team_side, carrier location, and selected flow metric.")

    x_edges = np.linspace(-half_length, half_length, max(int(args.x_bins), 1) + 1)
    y_edges = np.linspace(-half_width, half_width, max(int(args.y_bins), 1) + 1)

    home_df = df[df["team_side"] == "home"].copy()
    away_df = df[df["team_side"] == "away"].copy()

    home_mean, home_counts = compute_team_grid(home_df, value_col=value_col, x_edges=x_edges, y_edges=y_edges)
    away_mean, away_counts = compute_team_grid(away_df, value_col=value_col, x_edges=x_edges, y_edges=y_edges)

    home_mean = home_mean.astype(float)
    away_mean = away_mean.astype(float)
    home_mean[home_counts < int(args.min_count)] = np.nan
    away_mean[away_counts < int(args.min_count)] = np.nan

    all_vals = np.concatenate([home_mean[np.isfinite(home_mean)], away_mean[np.isfinite(away_mean)]])
    if all_vals.size == 0:
        raise ValueError("No bins meet min-count threshold.")

    vmin = float(np.nanpercentile(all_vals, 5))
    vmax = float(np.nanpercentile(all_vals, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-9:
        vmin = float(np.nanmin(all_vals))
        vmax = float(np.nanmax(all_vals))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140, sharex=True, sharey=True)
    fig.subplots_adjust(top=0.76, wspace=0.08)

    im0 = axes[0].pcolormesh(x_edges, y_edges, home_mean, cmap="YlGnBu", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{ctx['home_name']} Avg Flow by Location")
    draw_pitch_overlay(axes[0], half_length, half_width)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].pcolormesh(x_edges, y_edges, away_mean, cmap="YlGnBu", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{ctx['away_name']} Avg Flow by Location")
    draw_pitch_overlay(axes[1], half_length, half_width)
    axes[1].set_xlabel("x (m)")
    axes[1].set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(
        im0,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.90,
        pad=0.12,
    )
    metric_label = "Average flow amount (frame flow capacity)" if value_col == "frame_flow_capacity" else "Average flow amount (abs flow proxy)"
    cbar.set_label(metric_label)

    fig.suptitle("Team Flow Heatmaps by Field Location", fontsize=14, y=0.95)
    fig.text(0.5, 0.90, "Direction of play is to the right (+x)", ha="center", va="center", fontsize=10)
    fig.savefig(output_path)
    plt.close(fig)

    grid_summary = pd.concat(
        [
            build_grid_summary("home", home_mean, home_counts, x_edges, y_edges),
            build_grid_summary("away", away_mean, away_counts, x_edges, y_edges),
        ],
        ignore_index=True,
    )
    grid_summary.to_csv(output_grid, index=False)

    print(f"Saved heatmap: {output_path}")
    print(f"Saved grid summary: {output_grid}")
    print(f"Rows analyzed: {len(df)}")
    print(f"Metric used: {value_col}")


if __name__ == "__main__":
    main()
