import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    as_text = s.astype(str).str.strip().str.lower()
    return as_text.isin({"1", "true", "t", "yes", "y"})


def load_match_context(match_path: Path) -> dict:
    with open(match_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {
        "pitch_length": float(m.get("pitch_length", 105.0)),
        "home_name": str(m.get("home_team", {}).get("name", "Home")),
        "away_name": str(m.get("away_team", {}).get("name", "Away")),
    }


def assign_third(x: float, half_length: float) -> str:
    if x < (-half_length / 3.0):
        return "Defensive"
    if x <= (half_length / 3.0):
        return "Middle"
    return "Offensive"


def main():
    parser = argparse.ArgumentParser(description="Plot thirds-based optimal-choice rate bar chart for each team.")
    parser.add_argument("--rows", default="outputs/player_action_optimality_rows.csv", help="Input action rows CSV")
    parser.add_argument("--match", default="match.json", help="Match metadata JSON")
    parser.add_argument("--output", default="outputs/team_optimal_choice_rate_thirds_bars.png", help="Output PNG path")
    parser.add_argument("--output-summary", default="outputs/team_optimal_choice_rate_thirds_summary.csv", help="Output CSV summary path")
    args = parser.parse_args()

    rows_path = Path(args.rows)
    match_path = Path(args.match)
    output_path = Path(args.output)
    output_summary = Path(args.output_summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    ctx = load_match_context(match_path)
    half_length = 0.5 * ctx["pitch_length"]

    df = pd.read_csv(rows_path)
    needed = ["team_side", "carrier_x", "took_optimal_action"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input rows missing columns: {missing}")

    df["team_side"] = df["team_side"].fillna("").astype(str).str.strip().str.lower()
    df["carrier_x"] = pd.to_numeric(df["carrier_x"], errors="coerce")
    df["optimal_flag"] = parse_bool_series(df["took_optimal_action"]).astype(float)

    df = df[df["team_side"].isin(["home", "away"]) & np.isfinite(df["carrier_x"])].copy()
    if df.empty:
        raise ValueError("No valid action rows for team_side and carrier_x.")

    df["third"] = df["carrier_x"].apply(lambda x: assign_third(float(x), half_length=half_length))
    third_order = ["Defensive", "Middle", "Offensive"]

    summary = (
        df.groupby(["team_side", "third"], dropna=False)
        .agg(n_actions=("optimal_flag", "size"), optimal_choice_rate_pct=("optimal_flag", lambda s: 100.0 * float(np.mean(s))))
        .reset_index()
    )

    # Ensure all thirds are present for each team.
    expanded_rows = []
    for side in ["home", "away"]:
        for third in third_order:
            chunk = summary[(summary["team_side"] == side) & (summary["third"] == third)]
            if chunk.empty:
                expanded_rows.append({"team_side": side, "third": third, "n_actions": 0, "optimal_choice_rate_pct": np.nan})
            else:
                expanded_rows.append(chunk.iloc[0].to_dict())
    summary = pd.DataFrame(expanded_rows)
    summary.to_csv(output_summary, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    x = np.arange(len(third_order), dtype=float)
    width = 0.34

    home_vals = [
        float(summary[(summary["team_side"] == "home") & (summary["third"] == t)]["optimal_choice_rate_pct"].iloc[0])
        if len(summary[(summary["team_side"] == "home") & (summary["third"] == t)]) > 0
        else np.nan
        for t in third_order
    ]
    away_vals = [
        float(summary[(summary["team_side"] == "away") & (summary["third"] == t)]["optimal_choice_rate_pct"].iloc[0])
        if len(summary[(summary["team_side"] == "away") & (summary["third"] == t)]) > 0
        else np.nan
        for t in third_order
    ]

    bars_home = ax.bar(x - width / 2.0, home_vals, width=width, color="#dc2626", alpha=0.88, label=ctx["home_name"])
    bars_away = ax.bar(x + width / 2.0, away_vals, width=width, color="#1d4ed8", alpha=0.88, label=ctx["away_name"])

    for bars in (bars_home, bars_away):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title("Optimal Choices by Field Third", fontsize=16, fontweight="bold")
    ax.set_ylabel("Optimal choices (%)")
    ax.set_xlabel("Field third")
    ax.set_xticks(x)
    ax.set_xticklabels(third_order)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved bar chart: {output_path}")
    print(f"Saved summary: {output_summary}")
    print(f"Rows analyzed: {len(df)}")


if __name__ == "__main__":
    main()
