from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager


ALERT_LINE = 8.0
SAFE_RETURN_LINE = 6.5

RISE_START_THRESHOLD = 0.08
RISE_STRENGTH_THRESHOLD = 0.04
FALL_START_THRESHOLD = 0.08
FALL_STRENGTH_THRESHOLD = 0.04

S1_GATE_MIN_LEVEL = 6.7
S1_GATE_MIN_DELTA_10M = 0.15
S1_GATE_MIN_RISE_STRENGTH = 0.10

MEANINGFUL_RISE_THRESHOLD = 0.30

PEAK_DROP_THRESHOLD_SMALL = 0.12
PEAK_DROP_THRESHOLD_FLOOD = 0.22

PEAK_CONFIRM_MINUTES_SMALL = 3
PEAK_CONFIRM_MINUTES_FLOOD = 3

HIGH_WATER_EXIT_LEVEL = 8.60

STABLE_BAND_THRESHOLD = 0.05

SHORT_WINDOW_MIN = 5
DELTA_10M_SHIFT = 10

STATE_IDS = {
    "S0_STABLE": 0,
    "S1_RISING": 1,
    "S2_FLOOD": 2,
    "S3_RECEDING": 3,
}


def setup_chinese_font() -> None:
    candidate_fonts = [
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Microsoft JhengHei",
        "PingFang TC",
        "SimHei",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available_fonts:
            mpl.rcParams["font.family"] = font_name
            break
    mpl.rcParams["axes.unicode_minus"] = False


def safe_float(value) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def format_timestamp(value) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def prepare_causal_inputs(source: pd.DataFrame) -> pd.DataFrame:
    df = source.loc[:, ["timestamp", "water_level"]].copy()
    df["delta_10m"] = df["water_level"] - df["water_level"].shift(DELTA_10M_SHIFT)
    df["rise_strength"] = df["water_level"] - df["water_level"].shift(SHORT_WINDOW_MIN)
    df["fall_strength"] = df["water_level"].shift(SHORT_WINDOW_MIN) - df["water_level"]

    delta_10m = df["delta_10m"].fillna(0.0)
    rise_strength = df["rise_strength"].fillna(0.0)
    fall_strength = df["fall_strength"].fillna(0.0)

    df["is_local_rising"] = (delta_10m > RISE_START_THRESHOLD) & (rise_strength > RISE_STRENGTH_THRESHOLD)
    df["is_local_falling"] = (delta_10m < -FALL_START_THRESHOLD) & (fall_strength > FALL_STRENGTH_THRESHOLD)
    df["near_safe_zone"] = df["water_level"] <= SAFE_RETURN_LINE
    df["is_trend_flat"] = (
        (delta_10m.abs() <= STABLE_BAND_THRESHOLD)
        & (rise_strength.abs() <= STABLE_BAND_THRESHOLD)
        & (fall_strength.abs() <= STABLE_BAND_THRESHOLD)
    )
    return df


def official_s1_gate(water_level: float, delta_10m: float, rise_strength: float, is_local_rising: bool) -> bool:
    return (
        is_local_rising
        and water_level >= S1_GATE_MIN_LEVEL
        and (
            delta_10m >= S1_GATE_MIN_DELTA_10M
            or rise_strength >= S1_GATE_MIN_RISE_STRENGTH
        )
    )


def run_fsm_v3c_official(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    transitions = []

    current_state = "S0_STABLE"
    current_event_id = 0
    next_event_id = 1

    event_active = False
    event_start_level = None
    event_peak_level = None
    event_peak_idx = None
    has_meaningful_rise = False
    has_entered_flood = False

    for i, row in enumerate(df.itertuples(index=False)):
        timestamp = row.timestamp
        water_level = float(row.water_level)
        delta_10m = safe_float(row.delta_10m)
        rise_strength = safe_float(row.rise_strength)
        fall_strength = safe_float(row.fall_strength)
        is_local_rising = bool(row.is_local_rising)
        is_local_falling = bool(row.is_local_falling)
        near_safe_zone = bool(row.near_safe_zone)
        is_trend_flat = bool(row.is_trend_flat)
        s1_gate_condition = official_s1_gate(water_level, delta_10m, rise_strength, is_local_rising)

        if event_active and current_state in {"S1_RISING", "S2_FLOOD"}:
            if event_peak_level is None or water_level > event_peak_level:
                event_peak_level = water_level
                event_peak_idx = i

        if event_active and event_start_level is not None:
            if (water_level - event_start_level) >= MEANINGFUL_RISE_THRESHOLD:
                has_meaningful_rise = True

        peak_drop = 0.0 if event_peak_level is None else event_peak_level - water_level
        minutes_since_peak = 0 if event_peak_idx is None else i - event_peak_idx

        previous_state = current_state
        transition_reason = "no_transition"

        if current_state == "S0_STABLE":
            if s1_gate_condition:
                current_state = "S1_RISING"
                current_event_id = next_event_id
                next_event_id += 1
                event_active = True
                event_start_level = water_level
                event_peak_level = water_level
                event_peak_idx = i
                has_meaningful_rise = False
                has_entered_flood = False
                peak_drop = 0.0
                minutes_since_peak = 0
                transition_reason = "S0_to_S1_v3c_official_gate"
            elif water_level >= ALERT_LINE:
                current_state = "S2_FLOOD"
                current_event_id = next_event_id
                next_event_id += 1
                event_active = True
                event_start_level = water_level
                event_peak_level = water_level
                event_peak_idx = i
                has_meaningful_rise = False
                has_entered_flood = True
                peak_drop = 0.0
                minutes_since_peak = 0
                transition_reason = "S0_to_S2_initial_above_alert"

        elif current_state == "S1_RISING":
            if water_level >= ALERT_LINE:
                current_state = "S2_FLOOD"
                has_entered_flood = True
                transition_reason = "S1_to_S2_above_alert_line"
            elif (
                event_active
                and has_meaningful_rise
                and peak_drop >= PEAK_DROP_THRESHOLD_SMALL
                and minutes_since_peak >= PEAK_CONFIRM_MINUTES_SMALL
                and is_local_falling
            ):
                current_state = "S3_RECEDING"
                transition_reason = "S1_to_S3_peak_drop_confirmed"
            elif (not has_meaningful_rise) and near_safe_zone and (not is_local_rising) and is_trend_flat:
                current_state = "S0_STABLE"
                current_event_id = 0
                event_active = False
                event_start_level = None
                event_peak_level = None
                event_peak_idx = None
                has_meaningful_rise = False
                has_entered_flood = False
                peak_drop = 0.0
                minutes_since_peak = 0
                transition_reason = "S1_to_S0_no_meaningful_rise_safe_flat"

        elif current_state == "S2_FLOOD":
            if (
                has_entered_flood
                and peak_drop >= PEAK_DROP_THRESHOLD_FLOOD
                and minutes_since_peak >= PEAK_CONFIRM_MINUTES_FLOOD
                and is_local_falling
                and water_level <= HIGH_WATER_EXIT_LEVEL
            ):
                current_state = "S3_RECEDING"
                transition_reason = "S2_to_S3_peak_based_receding"

        elif current_state == "S3_RECEDING":
            if is_local_rising and water_level >= ALERT_LINE:
                current_state = "S2_FLOOD"
                has_entered_flood = True
                if event_peak_level is None or water_level > event_peak_level:
                    event_peak_level = water_level
                    event_peak_idx = i
                    peak_drop = 0.0
                    minutes_since_peak = 0
                transition_reason = "S3_to_S2_local_rise_above_alert"
            elif is_local_rising and water_level < ALERT_LINE and water_level >= 7.5:
                current_state = "S1_RISING"
                if event_peak_level is None or water_level > event_peak_level:
                    event_peak_level = water_level
                    event_peak_idx = i
                    peak_drop = 0.0
                    minutes_since_peak = 0
                transition_reason = "S3_to_S1_limited_local_re_rise_below_alert"
            elif near_safe_zone and is_trend_flat:
                current_state = "S0_STABLE"
                current_event_id = 0
                event_active = False
                event_start_level = None
                event_peak_level = None
                event_peak_idx = None
                has_meaningful_rise = False
                has_entered_flood = False
                peak_drop = 0.0
                minutes_since_peak = 0
                transition_reason = "S3_to_S0_safe_return_flat"

        else:
            raise ValueError(f"Unknown FSM state: {current_state}")

        if previous_state != current_state:
            transitions.append(
                {
                    "timestamp": timestamp,
                    "water_level": water_level,
                    "from_state": previous_state,
                    "to_state": current_state,
                    "reason": transition_reason,
                    "delta_10m": delta_10m,
                    "rise_strength": rise_strength,
                    "fall_strength": fall_strength,
                    "peak_drop": peak_drop,
                    "minutes_since_peak": minutes_since_peak,
                }
            )

        rows.append(
            {
                "timestamp": timestamp,
                "water_level": water_level,
                "delta_10m": delta_10m,
                "rise_strength": rise_strength,
                "fall_strength": fall_strength,
                "fsm_state_name": current_state,
                "fsm_state_id": STATE_IDS[current_state],
                "fsm_event_id": current_event_id,
                "transition_reason": transition_reason,
                "event_peak_level_running": pd.NA if event_peak_level is None else event_peak_level,
                "has_meaningful_rise": has_meaningful_rise,
                "has_entered_flood": has_entered_flood,
                "peak_drop": peak_drop,
                "minutes_since_peak": minutes_since_peak,
                "is_local_rising": is_local_rising,
                "is_local_falling": is_local_falling,
                "near_safe_zone": near_safe_zone,
                "is_trend_flat": is_trend_flat,
                "s1_gate_condition": s1_gate_condition,
            }
        )

    transitions_df = pd.DataFrame(
        transitions,
        columns=[
            "timestamp",
            "water_level",
            "from_state",
            "to_state",
            "reason",
            "delta_10m",
            "rise_strength",
            "fall_strength",
            "peak_drop",
            "minutes_since_peak",
        ],
    )
    return pd.DataFrame(rows), transitions_df


def make_state_counts(labeled: pd.DataFrame) -> pd.DataFrame:
    state_counts = (
        labeled["fsm_state_name"]
        .value_counts()
        .rename_axis("state")
        .reset_index(name="count")
        .sort_values("state")
        .reset_index(drop=True)
    )
    state_counts["ratio"] = state_counts["count"] / len(labeled)
    return state_counts


def make_event_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    event_rows = []
    event_ids = [event_id for event_id in sorted(labeled["fsm_event_id"].unique()) if int(event_id) != 0]

    for event_id in event_ids:
        event = labeled[labeled["fsm_event_id"] == event_id].copy()
        state_counts = event["fsm_state_name"].value_counts()
        peak_water_level = float(event["water_level"].max())
        has_s2 = bool((event["fsm_state_name"] == "S2_FLOOD").any())
        has_s1 = bool((event["fsm_state_name"] == "S1_RISING").any())
        tier = 2 if has_s2 else 1 if has_s1 or peak_water_level >= SAFE_RETURN_LINE else 0

        first_s1 = event.loc[event["fsm_state_name"] == "S1_RISING", "timestamp"].min()
        first_s2 = event.loc[event["fsm_state_name"] == "S2_FLOOD", "timestamp"].min()
        first_s3 = event.loc[event["fsm_state_name"] == "S3_RECEDING", "timestamp"].min()
        if pd.notna(first_s1) and pd.notna(first_s2):
            lead_s1_to_s2_min = (first_s2 - first_s1).total_seconds() / 60.0
        else:
            lead_s1_to_s2_min = pd.NA

        states = event["fsm_state_name"].tolist()
        number_of_s2_s3_switches = sum(
            1
            for previous, current in zip(states[:-1], states[1:])
            if {previous, current} == {"S2_FLOOD", "S3_RECEDING"}
        )

        event_rows.append(
            {
                "fsm_event_id": event_id,
                "start_timestamp": event["timestamp"].min(),
                "end_timestamp": event["timestamp"].max(),
                "rows": int(len(event)),
                "peak_water_level": peak_water_level,
                "tier": tier,
                "first_s1": first_s1,
                "first_s2": first_s2,
                "first_s3": first_s3,
                "lead_s1_to_s2_min": lead_s1_to_s2_min,
                "n_s0": int(state_counts.get("S0_STABLE", 0)),
                "n_s1": int(state_counts.get("S1_RISING", 0)),
                "n_s2": int(state_counts.get("S2_FLOOD", 0)),
                "n_s3": int(state_counts.get("S3_RECEDING", 0)),
                "number_of_s2_s3_switches": int(number_of_s2_s3_switches),
            }
        )

    return pd.DataFrame(
        event_rows,
        columns=[
            "fsm_event_id",
            "start_timestamp",
            "end_timestamp",
            "rows",
            "peak_water_level",
            "tier",
            "first_s1",
            "first_s2",
            "first_s3",
            "lead_s1_to_s2_min",
            "n_s0",
            "n_s1",
            "n_s2",
            "n_s3",
            "number_of_s2_s3_switches",
        ],
    )


def pick_main_flood_event(event_summary: pd.DataFrame) -> pd.Series | None:
    if event_summary.empty:
        return None
    tier2 = event_summary[event_summary["tier"] == 2]
    candidates = tier2 if not tier2.empty else event_summary
    return candidates.sort_values(["peak_water_level", "rows"], ascending=[False, False]).iloc[0]


def save_check_plot(labeled: pd.DataFrame, figure_path: Path) -> None:
    setup_chinese_font()
    colors = {
        "S0_STABLE": "gray",
        "S1_RISING": "orange",
        "S2_FLOOD": "red",
        "S3_RECEDING": "blue",
    }
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(labeled["timestamp"], labeled["water_level"], label="water_level", linewidth=1.8, color="tab:blue")
    for state_name, color in colors.items():
        part = labeled[labeled["fsm_state_name"] == state_name]
        ax.scatter(part["timestamp"], part["water_level"], label=state_name, s=16, color=color, alpha=0.8)
    ax.axhline(ALERT_LINE, linestyle="--", linewidth=1.5, color="black", label="alert_line = 8.0")
    ax.axhline(SAFE_RETURN_LINE, linestyle=":", linewidth=1.5, color="green", label="safe_line = 6.5")
    ax.set_title("4-State Semantic Labeling", fontsize=20)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("water_level (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    summary_path: Path,
    input_path: Path,
    output_path: Path,
    labeled: pd.DataFrame,
    state_counts: pd.DataFrame,
    transitions: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> None:
    main_event = pick_main_flood_event(event_summary)
    transition_counts = transitions.groupby(["from_state", "to_state"]).size().reset_index(name="count")

    lines = [
        "FSM v3c Official Summary",
        "",
        f"input file path: {input_path}",
        f"output file path: {output_path}",
        "FSM version: v3c official working label",
        f"row count: {len(labeled)}",
        f"time range: {format_timestamp(labeled['timestamp'].min())} to {format_timestamp(labeled['timestamp'].max())}",
        "",
        "FSM parameters:",
        f"- ALERT_LINE = {ALERT_LINE}",
        f"- SAFE_RETURN_LINE = {SAFE_RETURN_LINE}",
        f"- RISE_START_THRESHOLD = {RISE_START_THRESHOLD}",
        f"- RISE_STRENGTH_THRESHOLD = {RISE_STRENGTH_THRESHOLD}",
        f"- FALL_START_THRESHOLD = {FALL_START_THRESHOLD}",
        f"- FALL_STRENGTH_THRESHOLD = {FALL_STRENGTH_THRESHOLD}",
        f"- S1_GATE_MIN_LEVEL = {S1_GATE_MIN_LEVEL}",
        f"- S1_GATE_MIN_DELTA_10M = {S1_GATE_MIN_DELTA_10M}",
        f"- S1_GATE_MIN_RISE_STRENGTH = {S1_GATE_MIN_RISE_STRENGTH}",
        f"- MEANINGFUL_RISE_THRESHOLD = {MEANINGFUL_RISE_THRESHOLD}",
        f"- PEAK_DROP_THRESHOLD_SMALL = {PEAK_DROP_THRESHOLD_SMALL}",
        f"- PEAK_DROP_THRESHOLD_FLOOD = {PEAK_DROP_THRESHOLD_FLOOD}",
        f"- PEAK_CONFIRM_MINUTES_SMALL = {PEAK_CONFIRM_MINUTES_SMALL}",
        f"- PEAK_CONFIRM_MINUTES_FLOOD = {PEAK_CONFIRM_MINUTES_FLOOD}",
        f"- HIGH_WATER_EXIT_LEVEL = {HIGH_WATER_EXIT_LEVEL}",
        f"- STABLE_BAND_THRESHOLD = {STABLE_BAND_THRESHOLD}",
        f"- SHORT_WINDOW_MIN = {SHORT_WINDOW_MIN}",
        f"- DELTA_10M_SHIFT = {DELTA_10M_SHIFT}",
        "",
        "S1 gate official rule:",
        "is_local_rising AND water_level >= S1_GATE_MIN_LEVEL AND "
        "(delta_10m >= S1_GATE_MIN_DELTA_10M OR rise_strength >= S1_GATE_MIN_RISE_STRENGTH)",
        "",
        "state counts:",
        state_counts.to_string(index=False),
        "",
        "transition counts:",
        transition_counts.to_string(index=False) if len(transition_counts) else "No transitions.",
        "",
        "event summary:",
        event_summary.to_string(index=False) if len(event_summary) else "No non-S0 events.",
        "",
        f"main flood event first_s1: {'' if main_event is None else main_event['first_s1']}",
        f"main flood event first_s2: {'' if main_event is None else main_event['first_s2']}",
        f"main flood event first_s3: {'' if main_event is None else main_event['first_s3']}",
        f"lead_s1_to_s2_min: {'' if main_event is None else main_event['lead_s1_to_s2_min']}",
        f"number_of_s2_s3_switches: {'' if main_event is None else main_event['number_of_s2_s3_switches']}",
        "",
        "S1_RISING is the low-cost early-warning transmission trigger.",
        "S2_FLOOD is the high-priority flood transmission state.",
        "S3_RECEDING is the peak-based receding monitoring state and does not mean safe.",
        "FSM uses only causal information derived from timestamp and water_level.",
        "Trace/debug columns in this file must not be used as model inputs.",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "clean" / "_clean_source.csv"
    labeled_path = project_root / "data" / "labeled" / "_fsm_v3c_official.csv"
    state_counts_path = project_root / "outputs" / "tables" / "_fsm_v3c_official_state_counts.csv"
    transitions_path = project_root / "outputs" / "tables" / "_fsm_v3c_official_transitions.csv"
    event_summary_path = project_root / "outputs" / "tables" / "_fsm_v3c_official_event_summary.csv"
    figure_path = project_root / "outputs" / "figures" / "_fsm_ground_truth.png"
    summary_path = project_root / "outputs" / "tables" / "_fsm_v3c_official_summary.txt"

    for path in (
        labeled_path.parent,
        state_counts_path.parent,
        transitions_path.parent,
        event_summary_path.parent,
        figure_path.parent,
        summary_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    source = pd.read_csv(input_path, usecols=["timestamp", "water_level"])
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source["water_level"] = pd.to_numeric(source["water_level"], errors="coerce")
    if source["timestamp"].isna().any():
        raise ValueError("Input contains invalid timestamp values.")
    if source["water_level"].isna().any():
        raise ValueError("Input contains invalid water_level values.")
    source = source.sort_values("timestamp").reset_index(drop=True)

    causal_inputs = prepare_causal_inputs(source)
    labeled, transitions = run_fsm_v3c_official(causal_inputs)
    state_counts = make_state_counts(labeled)
    event_summary = make_event_summary(labeled)

    labeled.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    state_counts.to_csv(state_counts_path, index=False, encoding="utf-8-sig")
    transitions.to_csv(transitions_path, index=False, encoding="utf-8-sig")
    event_summary.to_csv(event_summary_path, index=False, encoding="utf-8-sig")
    save_check_plot(labeled, figure_path)
    write_summary(summary_path, input_path, labeled_path, labeled, state_counts, transitions, event_summary)

    print(f"Saved labeled data: {labeled_path}")
    print(f"Saved state counts: {state_counts_path}")
    print(f"Saved transitions: {transitions_path}")
    print(f"Saved event summary: {event_summary_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved text summary: {summary_path}")
    print("State counts:")
    print(state_counts.to_string(index=False))
    print("Event summary:")
    print(event_summary.to_string(index=False) if len(event_summary) else "No non-S0 events.")


if __name__ == "__main__":
    main()
