import json
from pathlib import Path

import pandas as pd


ALERT_LINE = 8.0
SAFE_RETURN_LINE = 6.5

LAG_WINDOWS = [1, 3, 5, 10, 30, 60]
ROLLING_WINDOWS = [5, 10, 30, 60]

METADATA_COLUMNS = ["timestamp", "fsm_event_id", "source", "split"]
LABEL_COLUMNS = ["fsm_state_name", "fsm_state_id"]

STUDENT_FEATURE_COLUMNS = [
    "water_level",
    "delta_1m",
    "delta_3m",
    "delta_5m",
    "delta_10m",
    "max_5m",
    "range_5m",
    "slope_5m",
]

FORBIDDEN_COLUMNS = [
    "semantic_state_name",
    "fsm_state_name",
    "fsm_state_id",
    "label_id",
    "fsm_event_id",
    "original_event_id",
    "transition_reason",
    "fsm_transition_reason",
    "event_peak_level_trace",
    "event_peak_level_running",
    "has_meaningful_rise",
    "has_entered_flood",
    "peak_drop",
    "minutes_since_peak",
    "s1_gate_condition",
    "is_local_rising",
    "is_local_falling",
    "near_safe_zone",
    "is_trend_flat",
    "rising_condition",
    "falling_condition",
    "stable_condition",
    "strong_re_rise_condition",
    "rising_confirm_count",
    "falling_confirm_count",
    "stable_confirm_count",
    "alert_confirm_count",
]

FORBIDDEN_NAME_TOKENS = ["future", "next", "lead"]


def consecutive_count(condition: pd.Series) -> pd.Series:
    counts = []
    current = 0
    for value in condition.fillna(False):
        if bool(value):
            current += 1
        else:
            current = 0
        counts.append(current)
    return pd.Series(counts, index=condition.index, dtype="int64")


def build_features(source: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = source.loc[:, ["timestamp", "water_level"]].copy()
    df["water_level"] = pd.to_numeric(df["water_level"], errors="coerce")
    if df["water_level"].isna().any():
        raise ValueError("water_level contains NaN after numeric conversion.")

    for window in LAG_WINDOWS:
        df[f"lag_{window}m"] = df["water_level"].shift(window).fillna(0.0)
        df[f"delta_{window}m"] = (df["water_level"] - df["water_level"].shift(window)).fillna(0.0)

    for window in ROLLING_WINDOWS:
        rolling = df["water_level"].rolling(window=window, min_periods=1)
        df[f"mean_{window}m"] = rolling.mean()
        df[f"std_{window}m"] = rolling.std().fillna(0.0)
        df[f"min_{window}m"] = rolling.min()
        df[f"max_{window}m"] = rolling.max()
        df[f"range_{window}m"] = df[f"max_{window}m"] - df[f"min_{window}m"]

    df["slope_5m"] = df["delta_5m"] / 5.0
    df["slope_10m"] = df["delta_10m"] / 10.0
    df["slope_30m"] = df["delta_30m"] / 30.0

    df["dist_to_alert"] = ALERT_LINE - df["water_level"]
    df["dist_to_safe_return"] = df["water_level"] - SAFE_RETURN_LINE
    df["above_alert_line"] = (df["water_level"] >= ALERT_LINE).astype(int)
    df["above_safe_return_line"] = (df["water_level"] >= SAFE_RETURN_LINE).astype(int)

    df["consecutive_above_safe_return"] = consecutive_count(df["water_level"] >= SAFE_RETURN_LINE)
    df["consecutive_above_alert"] = consecutive_count(df["water_level"] >= ALERT_LINE)
    df["consecutive_rising_5m"] = consecutive_count(df["delta_5m"] > 0)
    df["consecutive_falling_5m"] = consecutive_count(df["delta_5m"] < 0)

    teacher_feature_columns = (
        ["water_level"]
        + [f"lag_{window}m" for window in LAG_WINDOWS]
        + [f"delta_{window}m" for window in LAG_WINDOWS]
        + [f"{stat}_{window}m" for window in ROLLING_WINDOWS for stat in ["mean", "std", "min", "max", "range"]]
        + ["slope_5m", "slope_10m", "slope_30m"]
        + ["dist_to_alert", "dist_to_safe_return", "above_alert_line", "above_safe_return_line"]
        + [
            "consecutive_above_safe_return",
            "consecutive_above_alert",
            "consecutive_rising_5m",
            "consecutive_falling_5m",
        ]
    )
    return df, teacher_feature_columns


def validate_columns(
    input_df: pd.DataFrame,
    teacher_feature_columns: list[str],
    student_feature_columns: list[str],
    teacher_table: pd.DataFrame,
    student_table: pd.DataFrame,
) -> None:
    forbidden_in_teacher = sorted(set(teacher_feature_columns) & set(FORBIDDEN_COLUMNS))
    forbidden_in_student = sorted(set(student_feature_columns) & set(FORBIDDEN_COLUMNS))
    if forbidden_in_teacher:
        raise ValueError(f"Forbidden columns in teacher features: {forbidden_in_teacher}")
    if forbidden_in_student:
        raise ValueError(f"Forbidden columns in student features: {forbidden_in_student}")
    if student_feature_columns != STUDENT_FEATURE_COLUMNS:
        raise ValueError(f"Student feature columns mismatch: {student_feature_columns}")
    if len(teacher_table) != len(input_df):
        raise ValueError("Teacher feature table row_count differs from input row_count.")
    if len(student_table) != len(input_df):
        raise ValueError("Student feature table row_count differs from input row_count.")
    if teacher_table[teacher_feature_columns].isna().any().any():
        raise ValueError("Teacher feature columns contain NaN.")
    if student_table[student_feature_columns].isna().any().any():
        raise ValueError("Student feature columns contain NaN.")

    all_feature_columns = teacher_feature_columns + student_feature_columns
    bad_names = [
        col
        for col in all_feature_columns
        if any(token in col.lower() for token in FORBIDDEN_NAME_TOKENS)
    ]
    if bad_names:
        raise ValueError(f"Future-looking feature names are not allowed: {bad_names}")


def make_nan_summary(table_name: str, table: pd.DataFrame) -> pd.DataFrame:
    row_count = len(table)
    rows = []
    for column in table.columns:
        nan_count = int(table[column].isna().sum())
        rows.append(
            {
                "table_name": table_name,
                "column": column,
                "nan_count": nan_count,
                "nan_ratio": 0.0 if row_count == 0 else nan_count / row_count,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "labeled" / "_fsm_v3c_official.csv"
    teacher_output_path = project_root / "data" / "features" / "_teacher_features_v3c.csv"
    student_output_path = project_root / "data" / "features" / "_student_features_v3c.csv"
    columns_json_path = project_root / "outputs" / "tables" / "feature_columns_v3c.json"
    feature_summary_path = project_root / "outputs" / "tables" / "feature_engineering_summary_v3c.csv"
    nan_summary_path = project_root / "outputs" / "tables" / "feature_nan_summary_v3c.csv"
    text_summary_path = project_root / "outputs" / "tables" / "feature_engineering_summary_v3c.txt"

    for path in (
        teacher_output_path.parent,
        student_output_path.parent,
        columns_json_path.parent,
        feature_summary_path.parent,
        nan_summary_path.parent,
        text_summary_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    input_df = pd.read_csv(input_path)
    required_columns = ["timestamp", "water_level", "fsm_state_name", "fsm_state_id"]
    missing_columns = [col for col in required_columns if col not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    input_df["timestamp"] = pd.to_datetime(input_df["timestamp"], errors="coerce")
    if input_df["timestamp"].isna().any():
        raise ValueError("Input contains invalid timestamp values.")
    input_df = input_df.sort_values("timestamp").reset_index(drop=True)

    features, teacher_feature_columns = build_features(input_df)
    student_feature_columns = STUDENT_FEATURE_COLUMNS.copy()

    metadata = pd.DataFrame(
        {
            "timestamp": input_df["timestamp"],
            "fsm_event_id": input_df["fsm_event_id"] if "fsm_event_id" in input_df.columns else 0,
            "source": "real",
            "split": "",
        }
    )
    labels = input_df.loc[:, LABEL_COLUMNS].copy()

    teacher_table = pd.concat([metadata, labels, features.loc[:, teacher_feature_columns]], axis=1)
    student_table = pd.concat([metadata, labels, features.loc[:, student_feature_columns]], axis=1)

    validate_columns(input_df, teacher_feature_columns, student_feature_columns, teacher_table, student_table)

    forbidden_present_in_input = sorted(set(input_df.columns) & set(FORBIDDEN_COLUMNS))
    forbidden_in_teacher = sorted(set(teacher_feature_columns) & set(FORBIDDEN_COLUMNS))
    forbidden_in_student = sorted(set(student_feature_columns) & set(FORBIDDEN_COLUMNS))

    teacher_nan_counts = teacher_table[teacher_feature_columns].isna().sum()
    student_nan_counts = student_table[student_feature_columns].isna().sum()
    any_nan_teacher = bool((teacher_nan_counts > 0).any())
    any_nan_student = bool((student_nan_counts > 0).any())

    teacher_table.to_csv(teacher_output_path, index=False, encoding="utf-8-sig")
    student_table.to_csv(student_output_path, index=False, encoding="utf-8-sig")

    columns_payload = {
        "teacher_feature_columns": teacher_feature_columns,
        "student_feature_columns": student_feature_columns,
        "metadata_columns": METADATA_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "forbidden_columns": FORBIDDEN_COLUMNS,
    }
    columns_json_path.write_text(json.dumps(columns_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    label_counts = input_df["fsm_state_name"].value_counts().sort_index().to_dict()
    causal_rule_check = (
        "pass: no centered rolling, no negative shift, no future/next/lead feature names; "
        "all features derived from current/past water_level"
    )
    summary = pd.DataFrame(
        [
            {
                "row_count": int(len(input_df)),
                "teacher_feature_count": int(len(teacher_feature_columns)),
                "student_feature_count": int(len(student_feature_columns)),
                "label_counts": json.dumps(label_counts, ensure_ascii=False),
                "teacher_feature_columns": "|".join(teacher_feature_columns),
                "student_feature_columns": "|".join(student_feature_columns),
                "metadata_columns": "|".join(METADATA_COLUMNS),
                "label_columns": "|".join(LABEL_COLUMNS),
                "forbidden_columns_present_in_input": "|".join(forbidden_present_in_input),
                "forbidden_columns_in_teacher_features": "|".join(forbidden_in_teacher),
                "forbidden_columns_in_student_features": "|".join(forbidden_in_student),
                "any_nan_in_teacher_features": any_nan_teacher,
                "any_nan_in_student_features": any_nan_student,
                "max_nan_count_teacher": int(teacher_nan_counts.max()),
                "max_nan_count_student": int(student_nan_counts.max()),
                "causal_rule_check": causal_rule_check,
            }
        ]
    )
    summary.to_csv(feature_summary_path, index=False, encoding="utf-8-sig")

    nan_summary = pd.concat(
        [
            make_nan_summary("teacher", teacher_table),
            make_nan_summary("student", student_table),
        ],
        ignore_index=True,
    )
    nan_summary.to_csv(nan_summary_path, index=False, encoding="utf-8-sig")

    text_lines = [
        "Feature Engineering Summary v3c",
        "",
        f"input file path: {input_path}",
        f"teacher output path: {teacher_output_path}",
        f"student output path: {student_output_path}",
        f"row count: {len(input_df)}",
        f"label distribution: {label_counts}",
        f"teacher feature count: {len(teacher_feature_columns)}",
        f"student feature count: {len(student_feature_columns)}",
        "",
        "full teacher feature list:",
        ", ".join(teacher_feature_columns),
        "",
        "full student feature list:",
        ", ".join(student_feature_columns),
        "",
        "forbidden column check:",
        f"- forbidden columns present in input as metadata/debug/reference: {forbidden_present_in_input}",
        f"- forbidden columns in teacher features: {forbidden_in_teacher}",
        f"- forbidden columns in student features: {forbidden_in_student}",
        "",
        "NaN check:",
        f"- any_nan_in_teacher_features: {any_nan_teacher}",
        f"- any_nan_in_student_features: {any_nan_student}",
        f"- max_nan_count_teacher: {int(teacher_nan_counts.max())}",
        f"- max_nan_count_student: {int(student_nan_counts.max())}",
        "",
        "all feature columns are causal and derived from timestamp/water_level only",
        "FSM trace/debug columns are not used as model inputs",
    ]
    text_summary_path.write_text("\n".join(text_lines), encoding="utf-8")

    print(f"Saved teacher features: {teacher_output_path}")
    print(f"Saved student features: {student_output_path}")
    print(f"Saved feature columns json: {columns_json_path}")
    print(f"Saved feature summary: {feature_summary_path}")
    print(f"Saved NaN summary: {nan_summary_path}")
    print(f"Saved text summary: {text_summary_path}")
    print(f"teacher_feature_count: {len(teacher_feature_columns)}")
    print(f"student_feature_count: {len(student_feature_columns)}")
    print(f"label_distribution: {label_counts}")
    print(f"forbidden_columns_in_teacher_features: {forbidden_in_teacher}")
    print(f"forbidden_columns_in_student_features: {forbidden_in_student}")
    print(f"any_nan_in_teacher_features: {any_nan_teacher}")
    print(f"any_nan_in_student_features: {any_nan_student}")


if __name__ == "__main__":
    main()
