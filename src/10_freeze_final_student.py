import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


ALERT_LINE = 8.0
SAFE_RETURN_LINE = 6.5
STUDENT_NAME = "student_17f_dt_d3_catboost_confidence_weighted"
CONTROL_NAME = "student_17f_dt_d4_catboost_confidence_weighted"
FEATURE_SET_NAME = "student_17f_domain_receding"
TEACHER_NAME = "CatBoost raw_s2_weight_2p0"
STATE_ID_TO_NAME = {0: "S0_STABLE", 1: "S1_RISING", 2: "S2_FLOOD", 3: "S3_RECEDING"}
STATE_ORDER = ["S0_STABLE", "S1_RISING", "S2_FLOOD", "S3_RECEDING"]
STATE_COLORS = {"S0_STABLE": "gray", "S1_RISING": "orange", "S2_FLOOD": "red", "S3_RECEDING": "blue"}
STUDENT_FEATURES = [
    "water_level",
    "dist_to_alert",
    "dist_to_safe_return",
    "above_alert_line",
    "delta_1m",
    "delta_3m",
    "delta_5m",
    "delta_10m",
    "slope_5m",
    "slope_10m",
    "max_5m",
    "range_10m",
    "recent_max_30m",
    "drawdown_from_recent_max_30m",
    "delta_30m",
    "slope_30m",
    "range_30m",
]
FORMULAS = {
    "water_level": "current water_level",
    "dist_to_alert": "ALERT_LINE - water_level",
    "dist_to_safe_return": "water_level - SAFE_RETURN_LINE",
    "above_alert_line": "int(water_level >= ALERT_LINE)",
    "delta_1m": "water_level - water_level.shift(1)",
    "delta_3m": "water_level - water_level.shift(3)",
    "delta_5m": "water_level - water_level.shift(5)",
    "delta_10m": "water_level - water_level.shift(10)",
    "slope_5m": "delta_5m / 5",
    "slope_10m": "delta_10m / 10",
    "max_5m": "rolling max over current and previous 4 rows",
    "range_10m": "rolling max previous 9/current - rolling min previous 9/current",
    "recent_max_30m": "rolling max over current and previous 29 rows",
    "drawdown_from_recent_max_30m": "recent_max_30m - water_level",
    "delta_30m": "water_level - water_level.shift(30)",
    "slope_30m": "delta_30m / 30",
    "range_30m": "rolling max previous 29/current - rolling min previous 29/current",
}


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def validate_features():
    forbidden_exact = {
        "true_state_id",
        "true_state_name",
        "catboost_hard_id",
        "catboost_hard_name",
        "catboost_soft_S0",
        "catboost_soft_S1",
        "catboost_soft_S2",
        "catboost_soft_S3",
        "catboost_confidence",
        "source",
        "split",
        "event_id",
        "fsm_event_id",
        "synthetic_id",
        "template_event_id",
        "scenario_type",
    }
    forbidden_substrings = ["label", "future", "trace", "pred", "teacher"]
    bad_exact = sorted(set(STUDENT_FEATURES) & forbidden_exact)
    bad_sub = sorted({f for f in STUDENT_FEATURES for s in forbidden_substrings if s in f.lower()})
    if bad_exact or bad_sub:
        raise ValueError(f"Frozen student features failed leakage check: exact={bad_exact}, substrings={bad_sub}")


def load_source_model_path(complexity):
    direct = Path(r"C:_stage1\outputs\student_receding_dt_v3c\models") / f"{STUDENT_NAME}.joblib"
    if direct.exists():
        return direct
    row = complexity[complexity["experiment_id"] == STUDENT_NAME]
    if row.empty:
        raise FileNotFoundError(f"Cannot find complexity row for {STUDENT_NAME}")
    path = Path(row.iloc[0]["model_path"])
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def feature_spec():
    roles = {
        "water_level": "current absolute level",
        "dist_to_alert": "distance to fixed alert threshold",
        "dist_to_safe_return": "distance from fixed safe-return threshold",
        "above_alert_line": "fixed alert-line gate",
        "delta_1m": "1-minute direction",
        "delta_3m": "short direction",
        "delta_5m": "S1/S3 short trend",
        "delta_10m": "medium short trend",
        "slope_5m": "5-minute slope",
        "slope_10m": "10-minute slope",
        "max_5m": "recent local max",
        "range_10m": "short volatility",
        "recent_max_30m": "receding-memory recent peak",
        "drawdown_from_recent_max_30m": "receding-memory drawdown",
        "delta_30m": "longer trend",
        "slope_30m": "longer slope",
        "range_30m": "longer volatility",
    }
    buffers = {
        "water_level": 0,
        "dist_to_alert": 0,
        "dist_to_safe_return": 0,
        "above_alert_line": 0,
        "delta_1m": 1,
        "delta_3m": 3,
        "delta_5m": 5,
        "delta_10m": 10,
        "slope_5m": 5,
        "slope_10m": 10,
        "max_5m": 5,
        "range_10m": 10,
        "recent_max_30m": 30,
        "drawdown_from_recent_max_30m": 30,
        "delta_30m": 30,
        "slope_30m": 30,
        "range_30m": 30,
    }
    rows = []
    for idx, feature in enumerate(STUDENT_FEATURES, 1):
        rows.append(
            {
                "feature_order": idx,
                "feature_name": feature,
                "feature_role": roles[feature],
                "formula": FORMULAS[feature],
                "required_buffer_minutes": buffers[feature],
                "causal": True,
                "edge_computable": True,
                "uses_fixed_threshold": feature in {"dist_to_alert", "dist_to_safe_return", "above_alert_line"},
                "notes": "Uses current/past water level only; fixed thresholds are public system constants.",
            }
        )
    return pd.DataFrame(rows)


def generate_ifelse_code(model):
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value

    def recurse(node, indent):
        space = " " * indent
        if children_left[node] == children_right[node]:
            state_id = int(model.classes_[value[node][0].argmax()])
            state_name = STATE_ID_TO_NAME[state_id]
            return [f"{space}return {state_id}, \"{state_name}\""]
        feat_name = STUDENT_FEATURES[feature[node]]
        th = threshold[node]
        lines = [f"{space}if features[\"{feat_name}\"] <= {th:.10f}:"]
        lines.extend(recurse(children_left[node], indent + 4))
        lines.append(f"{space}else:")
        lines.extend(recurse(children_right[node], indent + 4))
        return lines

    header = [
        "# Auto-generated from frozen DecisionTree student.",
        "# Do not edit thresholds manually; regenerate from the frozen model if needed.",
        "STATE_ID_TO_NAME = {0: \"S0_STABLE\", 1: \"S1_RISING\", 2: \"S2_FLOOD\", 3: \"S3_RECEDING\"}",
        f"REQUIRED_FEATURES = {repr(STUDENT_FEATURES)}",
        "",
        "def predict_state_from_features(features: dict) -> tuple[int, str]:",
        "    for name in REQUIRED_FEATURES:",
        "        if name not in features:",
        "            raise KeyError(name)",
    ]
    return "\n".join(header + recurse(0, 4)) + "\n"


def feature_reference_code():
    return f'''# Edge feature computation reference for frozen d3 student.
ALERT_LINE = {ALERT_LINE}
SAFE_RETURN_LINE = {SAFE_RETURN_LINE}

def _lag(buffer, minutes):
    if len(buffer) > minutes:
        return float(buffer[-1 - minutes])
    return None

def _delta(buffer, minutes):
    lag = _lag(buffer, minutes)
    if lag is None:
        return 0.0
    return float(buffer[-1]) - lag

def compute_student_17f_features_from_water_level_buffer(water_levels: list[float]) -> dict:
    if not water_levels:
        raise ValueError("water_levels buffer must not be empty")
    buf = [float(x) for x in water_levels[-30:]]
    current = buf[-1]
    delta_1m = _delta(buf, 1)
    delta_3m = _delta(buf, 3)
    delta_5m = _delta(buf, 5)
    delta_10m = _delta(buf, 10)
    delta_30m = _delta(buf, 30)
    last5 = buf[-5:]
    last10 = buf[-10:]
    last30 = buf[-30:]
    recent_max_30m = max(last30)
    return {{
        "water_level": current,
        "dist_to_alert": ALERT_LINE - current,
        "dist_to_safe_return": current - SAFE_RETURN_LINE,
        "above_alert_line": int(current >= ALERT_LINE),
        "delta_1m": delta_1m,
        "delta_3m": delta_3m,
        "delta_5m": delta_5m,
        "delta_10m": delta_10m,
        "slope_5m": delta_5m / 5.0,
        "slope_10m": delta_10m / 10.0,
        "max_5m": max(last5),
        "range_10m": max(last10) - min(last10),
        "recent_max_30m": recent_max_30m,
        "drawdown_from_recent_max_30m": recent_max_30m - current,
        "delta_30m": delta_30m,
        "slope_30m": delta_30m / 30.0,
        "range_30m": max(last30) - min(last30),
    }}
'''


def streaming_demo_code():
    return '''# Minimal streaming demo for frozen d3 student.
from student_d3_feature_computation_reference_v3c import compute_student_17f_features_from_water_level_buffer
from pathlib import Path
import importlib.util

RULE_PATH = Path(__file__).resolve().parents[1] / "rules" / "frozen_student_d3_tree_rules_ifelse_python_v3c.py"
spec = importlib.util.spec_from_file_location("student_rules", RULE_PATH)
student_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(student_rules)

def run_demo():
    water_levels = [6.72, 6.75, 6.82, 6.93, 7.05, 7.22, 7.48, 7.8, 8.12, 8.34, 8.30, 8.18, 7.96]
    buffer = []
    for idx, level in enumerate(water_levels):
        buffer.append(level)
        features = compute_student_17f_features_from_water_level_buffer(buffer)
        state_id, state_name = student_rules.predict_state_from_features(features)
        print(f"t={idx:03d}, water_level={level:.2f}, state_id={state_id}, state_name={state_name}")

if __name__ == "__main__":
    run_demo()
'''


def plot_timeline(df, path, cols, titles):
    fig, axes = plt.subplots(len(cols), 1, figsize=(18, 3.2 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col, title in zip(axes, cols, titles):
        ax.plot(df["timestamp"], df["water_level"], color="tab:blue", linewidth=1.1)
        for state, color in STATE_COLORS.items():
            part = df[df[col] == state]
            ax.scatter(part["timestamp"], part["water_level"], s=10, color=color, alpha=0.85, label=state)
        ax.axhline(ALERT_LINE, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(SAFE_RETURN_LINE, color="green", linestyle=":", linewidth=0.9)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=5, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    out_root = root / "outputs/student_frozen_final_d3_v3c"
    dirs = {name: out_root / name for name in ["models", "tables", "figures", "metadata", "rules", "edge_export"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    paths = {
        "rules": root / "PROJECT_RULES.md",
        "metrics": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_metrics_v3c.csv",
        "complexity": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_complexity_v3c.csv",
        "core_pred": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_predictions_core_v3c.csv",
        "full_pred": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_predictions_full_cycle_v3c.csv",
        "audit_metrics": root / "outputs/tables/student_final_candidate_audit_v3c/student_final_candidate_metrics_v3c.csv",
        "audit_decision": root / "outputs/tables/student_final_candidate_audit_v3c/student_final_candidate_decision_v3c.csv",
        "teacher_metrics": root / "outputs/teacher_frozen_catboost_v3c/tables/frozen_catboost_teacher_metrics_v3c.csv",
    }
    for path in paths.values():
        require_file(path)
    validate_features()

    metrics = pd.read_csv(paths["metrics"])
    complexity = pd.read_csv(paths["complexity"])
    core_pred = pd.read_csv(paths["core_pred"])
    full_pred = pd.read_csv(paths["full_pred"])
    for df in [core_pred, full_pred]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    source_model_path = load_source_model_path(complexity)
    model = joblib.load(source_model_path)
    if not isinstance(model, DecisionTreeClassifier):
        raise ValueError(f"Frozen model must be DecisionTreeClassifier, got {type(model)}")
    if int(model.n_features_in_) != len(STUDENT_FEATURES):
        raise ValueError(f"Feature order/count mismatch: model n_features_in_={model.n_features_in_}, features={len(STUDENT_FEATURES)}")
    if int(model.get_depth()) != 3:
        raise ValueError(f"Expected model depth 3, got {model.get_depth()}")
    frozen_model_path = dirs["models"] / f"{STUDENT_NAME}_frozen.joblib"
    shutil.copy2(source_model_path, frozen_model_path)
    frozen_size_kb = frozen_model_path.stat().st_size / 1024.0

    cx = complexity[complexity["experiment_id"] == STUDENT_NAME].iloc[0]
    d4_cx = complexity[complexity["experiment_id"] == CONTROL_NAME].iloc[0]
    if int(cx["actual_depth"]) != 3 or int(cx["node_count"]) != 15:
        raise ValueError("Unexpected d3 complexity values.")
    if len(STUDENT_FEATURES) != 17:
        raise ValueError("Feature count must be 17.")

    d3_metrics = metrics[metrics["experiment_id"] == STUDENT_NAME].copy()
    d3_rows = []
    for _, row in d3_metrics.iterrows():
        d3_rows.append(
            {
                "student_name": STUDENT_NAME,
                "evaluation_scope": row["evaluation_scope"],
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
                "s0_recall": row["s0_recall"],
                "s1_recall": row["s1_recall"],
                "s2_recall": row["s2_recall"],
                "s3_recall": row["s3_recall"],
                "s1_precision": row["s1_precision"],
                "s2_precision": row["s2_precision"],
                "s3_precision": row["s3_precision"],
                "transition_count": row["transition_count"],
                "true_S1_pred_S3_count": row["true_S1_pred_S3_count"],
                "true_S2_pred_S3_count": row["true_S2_pred_S3_count"],
                "true_S3_pred_S1_or_S2_ratio": row["true_S3_pred_S1_or_S2_ratio"],
                "post_flood_false_trigger_rows": row["post_flood_false_trigger_rows"],
                "student_vs_catboost_agreement": row["student_vs_catboost_agreement"],
                "actual_depth": cx["actual_depth"],
                "node_count": cx["node_count"],
                "model_file_size_kb": frozen_size_kb,
                "interpretation": "preliminary final d3 student",
            }
        )
    d3_metrics_out = pd.DataFrame(d3_rows)
    d3_metrics_out.to_csv(dirs["tables"] / "frozen_student_d3_metrics_v3c.csv", index=False, encoding="utf-8-sig")

    teacher_metrics = pd.read_csv(paths["teacher_metrics"])
    teacher_core = teacher_metrics[teacher_metrics["dataset_name"] == "core_event3_test"].iloc[0]
    teacher_full = teacher_metrics[teacher_metrics["dataset_name"] == "full_cycle_event3_test"].iloc[0]
    d3_core = d3_metrics[d3_metrics["evaluation_scope"] == "core_event3_test"].iloc[0]
    d3_full = d3_metrics[d3_metrics["evaluation_scope"] == "full_cycle_event3_test"].iloc[0]
    d4_core = metrics[(metrics["experiment_id"] == CONTROL_NAME) & (metrics["evaluation_scope"] == "core_event3_test")].iloc[0]
    d4_full = metrics[(metrics["experiment_id"] == CONTROL_NAME) & (metrics["evaluation_scope"] == "full_cycle_event3_test")].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "model_role": "teacher",
                "model_name": TEACHER_NAME,
                "macro_f1": teacher_core["macro_f1"],
                "s1_recall": teacher_core["s1_recall"],
                "s2_recall": teacher_core["s2_recall"],
                "s3_recall": teacher_core["s3_recall"],
                "transition_count": teacher_core["transition_count"],
                "true_S3_pred_S1_or_S2_ratio": "",
                "full_cycle_s0_recall": teacher_full["s0_recall"],
                "post_flood_false_trigger_rows": teacher_full["post_flood_false_trigger_rows"],
                "actual_depth": "",
                "node_count": "",
                "interpretation": "high-capacity teacher reference",
            },
            {
                "model_role": "preliminary_final_student",
                "model_name": STUDENT_NAME,
                "macro_f1": d3_core["macro_f1"],
                "s1_recall": d3_core["s1_recall"],
                "s2_recall": d3_core["s2_recall"],
                "s3_recall": d3_core["s3_recall"],
                "transition_count": d3_core["transition_count"],
                "true_S3_pred_S1_or_S2_ratio": d3_core["true_S3_pred_S1_or_S2_ratio"],
                "full_cycle_s0_recall": d3_full["s0_recall"],
                "post_flood_false_trigger_rows": d3_full["post_flood_false_trigger_rows"],
                "actual_depth": cx["actual_depth"],
                "node_count": cx["node_count"],
                "interpretation": "balanced/control-stable edge student",
            },
            {
                "model_role": "aggressive_trigger_control",
                "model_name": CONTROL_NAME,
                "macro_f1": d4_core["macro_f1"],
                "s1_recall": d4_core["s1_recall"],
                "s2_recall": d4_core["s2_recall"],
                "s3_recall": d4_core["s3_recall"],
                "transition_count": d4_core["transition_count"],
                "true_S3_pred_S1_or_S2_ratio": d4_core["true_S3_pred_S1_or_S2_ratio"],
                "full_cycle_s0_recall": d4_full["s0_recall"],
                "post_flood_false_trigger_rows": d4_full["post_flood_false_trigger_rows"],
                "actual_depth": d4_cx["actual_depth"],
                "node_count": d4_cx["node_count"],
                "interpretation": "higher S2/macro control, weaker S3 stability",
            },
        ]
    )
    comparison.to_csv(dirs["tables"] / "frozen_student_d3_vs_teacher_vs_d4_v3c.csv", index=False, encoding="utf-8-sig")
    spec = feature_spec()
    spec.to_csv(dirs["tables"] / "frozen_student_d3_feature_spec_v3c.csv", index=False, encoding="utf-8-sig")

    tree_text = export_text(model, feature_names=STUDENT_FEATURES, decimals=6)
    (dirs["rules"] / "frozen_student_d3_tree_rules_text_v3c.txt").write_text(tree_text, encoding="utf-8")
    (dirs["rules"] / "frozen_student_d3_tree_rules_ifelse_python_v3c.py").write_text(generate_ifelse_code(model), encoding="utf-8")
    (dirs["edge_export"] / "student_d3_feature_computation_reference_v3c.py").write_text(feature_reference_code(), encoding="utf-8")
    (dirs["edge_export"] / "student_d3_streaming_inference_demo_v3c.py").write_text(streaming_demo_code(), encoding="utf-8")

    metadata = {
        "task_name": "Step 9E - Freeze d3 preliminary final student package and export edge rules",
        "student_name": STUDENT_NAME,
        "student_status": "preliminary_final_candidate",
        "student_role": "edge_4state_semantic_control_student",
        "teacher_name": TEACHER_NAME,
        "feature_set_name": FEATURE_SET_NAME,
        "student_features": STUDENT_FEATURES,
        "feature_count": len(STUDENT_FEATURES),
        "feature_recompute_policy": "current/past water_level only, fixed threshold derived domain features, warmup fillna zero",
        "alert_line": ALERT_LINE,
        "safe_return_line": SAFE_RETURN_LINE,
        "model_type": "DecisionTreeClassifier",
        "actual_depth": int(cx["actual_depth"]),
        "node_count": int(cx["node_count"]),
        "leaf_count": int(cx["leaf_count"]),
        "model_file_size_kb": frozen_size_kb,
        "training_variant": "catboost_confidence_weighted",
        "training_data_source": "CatBoost train probabilities for B2_semdiv_balanced train only",
        "evaluation_scope": "Event3 core and full-cycle evaluation only",
        "event3_usage": "evaluation_only_no_training_no_tuning",
        "leakage_check_status": "passed",
        "aggressive_control_student": CONTROL_NAME,
        "known_limitations": [
            "only one Event3 holdout",
            "S2 recall is sacrificed compared with d4 and teacher",
            "student remains below CatBoost teacher",
            "tree-based KD is confidence-weighted hard-label approximation, not standard soft-label KL KD",
            "no cross-event generalization claim",
        ],
        "created_by_script": Path(__file__).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (dirs["metadata"] / "frozen_student_d3_metadata_v3c.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    d3_col = f"{STUDENT_NAME}_pred_name"
    d4_col = f"{CONTROL_NAME}_pred_name"
    plot_timeline(core_pred, dirs["figures"] / "frozen_student_d3_core_timeline.png", ["true_state_name", "catboost_hard_name", d3_col], ["ground truth", "CatBoost teacher", "d3 preliminary final student"])
    plot_timeline(full_pred, dirs["figures"] / "frozen_student_d3_full_cycle_timeline.png", ["true_state_name", "catboost_hard_name", d3_col], ["ground truth", "CatBoost teacher", "d3 preliminary final student"])
    first_s2 = core_pred.loc[core_pred["true_state_name"] == "S2_FLOOD", "timestamp"].min()
    first_s3 = core_pred.loc[core_pred["true_state_name"] == "S3_RECEDING", "timestamp"].min()
    zoom = core_pred[(core_pred["timestamp"] >= first_s2 - pd.Timedelta(minutes=30)) & (core_pred["timestamp"] <= first_s3 + pd.Timedelta(minutes=240))].copy()
    plot_timeline(zoom, dirs["figures"] / "frozen_student_d3_vs_d4_s2_s3_zoom.png", ["true_state_name", "catboost_hard_name", d3_col, d4_col], ["ground truth", "CatBoost teacher", "d3", "d4"])

    fig, ax = plt.subplots(figsize=(11, 5))
    comparison.set_index("model_role")[["macro_f1", "s1_recall", "s2_recall", "s3_recall", "transition_count", "node_count"]].replace("", 0).astype(float).plot(kind="bar", ax=ax)
    ax.set_title("Frozen d3 Student Tradeoff")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "frozen_student_d3_model_tradeoff.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(model, feature_names=STUDENT_FEATURES, class_names=STATE_ORDER, filled=True, rounded=True, fontsize=8, ax=ax)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "frozen_student_d3_tree_plot.png", dpi=180)
    plt.close(fig)

    summary = f"""task name: Step 9E - Freeze d3 preliminary final student package and export edge rules
no training performed.
no Event3 tuning performed.
preliminary final student = {STUDENT_NAME}
teacher = {TEACHER_NAME}
d4 = {CONTROL_NAME} as aggressive trigger control baseline

why d3 selected:
d3 is selected for edge 4-state semantic control because it preserves S3 stability better than d4, has lower true-S3 false-trigger ratio, and is smaller.

why d4 not selected as main student:
d4 has higher macro F1 and S2 recall, but lower S3 recall and higher true-S3 S1/S2 false-trigger ratio. It is kept as aggressive trigger control.

feature set = {FEATURE_SET_NAME}
added receding-memory features: recent_max_30m, drawdown_from_recent_max_30m, delta_30m, slope_30m, range_30m
model complexity: depth={int(cx['actual_depth'])}, nodes={int(cx['node_count'])}, leaves={int(cx['leaf_count'])}, file_size_kb={frozen_size_kb:.3f}

core metrics:
macro_f1={d3_core['macro_f1']}, S1 recall={d3_core['s1_recall']}, S2 recall={d3_core['s2_recall']}, S3 recall={d3_core['s3_recall']}, transition_count={d3_core['transition_count']}, true_S3_pred_S1_or_S2_ratio={d3_core['true_S3_pred_S1_or_S2_ratio']}

full-cycle metrics:
macro_f1={d3_full['macro_f1']}, S0 recall={d3_full['s0_recall']}, post_flood_false_trigger_rows={d3_full['post_flood_false_trigger_rows']}

S2/S3 tradeoff:
d3 sacrifices S2 recall compared with d4 and teacher, but keeps receding stability much better than d4.

leakage checks passed.
Event3 usage = evaluation only.
edge deployment requirement: maintain a 30-min water-level buffer.
KD method note: confidence-weighted teacher-guided tree distillation, not standard KL KD.

limitations:
- only one Event3 holdout
- S2 recall is sacrificed compared with d4 and teacher
- student remains below CatBoost teacher
- no cross-event generalization claim

next steps:
- standard soft-label MLP KD control
- optional S3-aware weighting ablation
- optional hysteresis ablation
- Raspberry Pi / MCU inference simulation
"""
    (dirs["metadata"] / "frozen_student_d3_summary_v3c.txt").write_text(summary, encoding="utf-8")
    report = f"""## Frozen Preliminary Final Student

We froze a 17-feature depth-3 DecisionTree student (`{STUDENT_NAME}`) distilled from the CatBoost teacher (`{TEACHER_NAME}`). The teacher uses 44 causal water-level features, while the student uses edge-computable current/past water-level features, including 30-minute receding-memory features. These receding-memory features substantially improve S3 receding stability compared with the earlier 12-feature student.

The d3 student is selected as the preliminary final student because it better preserves 4-state semantic control: S1 early warning remains strong, S0 full-cycle recovery remains safe, and S3 receding stability is much better than the aggressive d4 control. The d4 student is retained as an aggressive trigger baseline because it has higher S2 recall and macro F1, but it sacrifices S3 stability.

This is a preliminary final candidate, not a claim of general flood-event generalization. The evaluation still relies on one Event3 holdout, and the student remains below the CatBoost teacher. The distillation method is confidence-weighted hard-label tree compression, not standard soft-label KL distillation.
"""
    (dirs["metadata"] / "frozen_student_d3_report_paragraph_v3c.md").write_text(report, encoding="utf-8")

    print("freeze_success: True")
    print(f"frozen_model_path: {frozen_model_path}")
    print(f"feature_count: {len(STUDENT_FEATURES)}")
    print(f"actual_depth: {int(cx['actual_depth'])}")
    print(f"node_count: {int(cx['node_count'])}")
    print(f"model_file_size_kb: {frozen_size_kb:.3f}")
    print(f"core_macro_f1: {d3_core['macro_f1']}")
    print(f"full_cycle_s0_recall: {d3_full['s0_recall']}")


if __name__ == "__main__":
    main()
