from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ALERT_LINE = 8.0
SAFE_LINE = 6.5


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / ".csv"
    clean_path = project_root / "data" / "clean" / "_clean_source.csv"
    summary_path = project_root / "outputs" / "tables" / "_clean_source_summary.csv"
    figure_path = project_root / "outputs" / "figures" / "_clean_source_plot.png"

    for path in (clean_path.parent, summary_path.parent, figure_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    input_columns = list(df.columns)
    used_columns = ["timestamp", "water_level"]
    missing_required = [col for col in used_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    ignored_reference_columns = [col for col in input_columns if col not in used_columns]

    clean = df.loc[:, used_columns].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean["water_level"] = pd.to_numeric(clean["water_level"], errors="coerce")

    invalid_timestamp_count = int(clean["timestamp"].isna().sum())
    if invalid_timestamp_count:
        raise ValueError(f"Found {invalid_timestamp_count} invalid timestamp rows.")

    clean = clean.sort_values("timestamp").reset_index(drop=True)
    duplicated_timestamps = int(clean["timestamp"].duplicated(keep="first").sum())
    clean = clean.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    row_count = int(len(clean))
    start_time = clean["timestamp"].min()
    end_time = clean["timestamp"].max()

    if row_count:
        expected_minute_count = int((end_time - start_time).total_seconds() // 60) + 1
    else:
        expected_minute_count = 0
    missing_minutes = int(expected_minute_count - row_count)

    water_level_nan_count = int(clean["water_level"].isna().sum())
    water_level_zero_or_negative_count = int((clean["water_level"] <= 0).sum())

    summary = pd.DataFrame(
        [
            {
                "row_count": row_count,
                "start_time": start_time,
                "end_time": end_time,
                "expected_minute_count": expected_minute_count,
                "missing_minutes": missing_minutes,
                "duplicated_timestamps": duplicated_timestamps,
                "water_level_min": clean["water_level"].min(),
                "water_level_max": clean["water_level"].max(),
                "water_level_nan_count": water_level_nan_count,
                "water_level_zero_or_negative_count": water_level_zero_or_negative_count,
                "input_columns": "|".join(input_columns),
                "used_columns": "|".join(used_columns),
                "ignored_reference_columns": "|".join(ignored_reference_columns),
            }
        ]
    )

    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(clean["timestamp"], clean["water_level"], linewidth=1.2, color="#1f77b4")
    ax.axhline(ALERT_LINE, color="#d62728", linestyle="--", linewidth=1.0, label="alert_line=8.0")
    ax.axhline(SAFE_LINE, color="#2ca02c", linestyle="--", linewidth=1.0, label="safe_line=6.5")
    ax.set_title("Clean Source Water Level")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("water_level")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Saved clean source: {clean_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {figure_path}")
    print(f"1-minute continuous: {missing_minutes == 0}")


if __name__ == "__main__":
    main()
