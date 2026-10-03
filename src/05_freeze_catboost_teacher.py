import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

try:
    import joblib
except ImportError as exc:
    raise ImportError("joblib is required to load/copy the CatBoost teacher model.") from exc


TEACHER_NAME = "catboost_raw_s2_weight_2p0"
SYN_SOURCE = "synthetic_B2_semdiv_balanced"
ALERT_LINE = 8.0
STATE_ID_TO_NAME = {0: "S0_STABLE", 1: "S1_RISING", 2: "S2_FLOOD", 3: "S3_RECEDING"}
STATE_ORDER = ["S0_STABLE", "S1_RISING", "S2_FLOOD", "S3_RECEDING"]
STATE_COLORS = {"S0_STABLE": "gray", "S1_RISING": "orange", "S2_FLOOD": "red", "S3_RECEDING": "blue"}
BLOCKED_FEATURES = {
    "fsm_state_name",
    "fsm_state_id",
    "true_state_name",
    "true_state_id",
    "semantic_state_name",
    "label_id",
    "event_id",
    "fsm_event_id",
    "source",
    "split",
    "synthetic_id",
    "template_event_id",
    "scenario_type",
    "target_peak",
    "actual_peak",
    "rise_time_warp",
    "recession_time_warp",
    "peak_plateau_width",
    "scale_factor",
    "quality_flag",
    "quality_notes",
    "correction_flags",
    "semantic_reject_reasons",
    "diversity_status",
    "accepted_for_train",
    "max_similarity_to_selected",
    "ranking_score",
    "selection_reason",
}
BLOCKED_SUBSTRINGS = ["label", "future", "trace", "pred", "ground_truth", "gt"]


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")


def normalize_ground_truth(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    if {"true_state_name", "true_state_id"}.issubset(out.columns):
        return out
    if {"fsm_state_name", "fsm_state_id"}.issubset(out.columns):
        return out.rename(columns={"fsm_state_name": "true_state_name", "fsm_state_id": "true_state_id"})
    raise ValueError(f"{name} missing ground truth columns: expected true_state_* or fsm_state_*")


def load_features(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    features = payload["teacher_feature_columns"]
    exact = sorted(set(features) & (BLOCKED_FEATURES | set(payload.get("forbidden_columns", []))))
    substr = sorted(feature for feature in features if any(token in feature for token in BLOCKED_SUBSTRINGS))
    if exact:
        raise ValueError(f"Teacher features contain blocked columns: {exact}")
    if substr:
        raise ValueError(f"Teacher features contain blocked substrings: {substr}")
    if len(features) != 44:
        raise ValueError(f"Expected 44 teacher features, got {len(features)}")
    return features


def validate_data(train: pd.DataFrame, core: pd.DataFrame, full: pd.DataFrame, features: list[str]) -> None:
    for name, df in [("train", train), ("core", core), ("full", full)]:
        required = ["timestamp", "water_level", "source", "split", "true_state_name", "true_state_id"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        feature_missing = [col for col in features if col not in df.columns]
        if feature_missing:
            raise ValueError(f"{name} missing teacher features: {feature_missing}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError(f"{name} has invalid timestamp values.")
    if set(train["source"].dropna().unique()) != {"real", SYN_SOURCE}:
        raise ValueError(f"Train source must be real/{SYN_SOURCE}, got {sorted(train['source'].unique())}")
    if not (core["source"] == "real").all():
        raise ValueError("Core Event3 test source must be 100% real.")
    if not (full["source"] == "real").all():
        raise ValueError("Full-cycle Event3 test source must be 100% real.")
    train_ts = set(train["timestamp"])
    if set(core["timestamp"]) & train_ts:
        raise ValueError("Core Event3 test timestamps overlap train.")
    if set(full["timestamp"]) & train_ts:
        raise ValueError("Full-cycle Event3 test timestamps overlap train.")
    if "fsm_event_id" not in train.columns:
        raise ValueError("Train missing fsm_event_id; cannot verify Event3 exclusion.")
    real_train = train[train["source"] == "real"]
    if (real_train["fsm_event_id"].astype(int) == 3).any():
        raise ValueError("Event3 appears in real training rows.")


def validate_model(model) -> None:
    if not hasattr(model, "classes_"):
        raise ValueError("CatBoost model missing classes_.")
    classes = [int(x) for x in list(model.classes_)]
    if classes != [0, 1, 2, 3]:
        raise ValueError(f"CatBoost classes_ must be [0,1,2,3], got {classes}")


def entropy(proba: np.ndarray) -> np.ndarray:
    return -np.sum(proba * np.log(proba + 1e-12), axis=1)


def probability_table(df: pd.DataFrame, model, features: list[str]) -> pd.DataFrame:
    proba = model.predict_proba(df[features])
    pred_id = proba.argmax(axis=1).astype(int)
    sorted_proba = np.sort(proba, axis=1)
    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "water_level": df["water_level"],
            "source": df["source"],
            "split": df["split"],
            "true_state_id": df["true_state_id"].astype(int),
            "true_state_name": df["true_state_name"],
            "catboost_hard_id": pred_id,
            "catboost_hard_name": pd.Series(pred_id).map(STATE_ID_TO_NAME).values,
            "catboost_soft_S0": proba[:, 0],
            "catboost_soft_S1": proba[:, 1],
            "catboost_soft_S2": proba[:, 2],
            "catboost_soft_S3": proba[:, 3],
            "catboost_confidence": proba.max(axis=1),
            "catboost_entropy": entropy(proba),
            "catboost_margin_top1_top2": sorted_proba[:, -1] - sorted_proba[:, -2],
        }
    )
    out["catboost_is_correct"] = out["catboost_hard_id"] == out["true_state_id"]
    out["is_high_confidence"] = out["catboost_confidence"] >= 0.80
    out["is_medium_confidence"] = (out["catboost_confidence"] >= 0.60) & (out["catboost_confidence"] < 0.80)
    out["is_low_confidence"] = out["catboost_confidence"] < 0.60
    return out


def state_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().to_dict()
    return {state: int(counts.get(state, 0)) for state in STATE_ORDER}


def transition_count(series: pd.Series) -> int:
    return int((series != series.shift(1)).sum() - 1)


def evaluate(dataset_name: str, prob: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y_true = prob["true_state_name"]
    y_pred = prob["catboost_hard_name"]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=STATE_ORDER, zero_division=0)
    post_trigger = (y_true == "S0_STABLE") & (y_pred.isin(["S1_RISING", "S2_FLOOD"]))
    row = {
        "dataset_name": dataset_name,
        "rows": int(len(prob)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=STATE_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=STATE_ORDER, average="weighted", zero_division=0),
        "s0_precision": precision[0],
        "s0_recall": recall[0],
        "s0_f1": f1[0],
        "s1_precision": precision[1],
        "s1_recall": recall[1],
        "s1_f1": f1[1],
        "s2_precision": precision[2],
        "s2_recall": recall[2],
        "s2_f1": f1[2],
        "s3_precision": precision[3],
        "s3_recall": recall[3],
        "s3_f1": f1[3],
        "pred_state_counts": json.dumps(state_counts(y_pred), ensure_ascii=False),
        "true_state_counts": json.dumps(state_counts(y_true), ensure_ascii=False),
        "transition_count": transition_count(y_pred) if "test" in dataset_name else "",
        "true_S1_pred_S3_count": int(((y_true == "S1_RISING") & (y_pred == "S3_RECEDING")).sum()),
        "true_S2_pred_S3_count": int(((y_true == "S2_FLOOD") & (y_pred == "S3_RECEDING")).sum()),
        "true_S3_pred_S2_count": int(((y_true == "S3_RECEDING") & (y_pred == "S2_FLOOD")).sum()),
        "false_S1_in_true_S0": int(((y_true == "S0_STABLE") & (y_pred == "S1_RISING")).sum()),
        "false_S2_in_true_S0": int(((y_true == "S0_STABLE") & (y_pred == "S2_FLOOD")).sum()),
        "false_S3_in_true_S0": int(((y_true == "S0_STABLE") & (y_pred == "S3_RECEDING")).sum()),
        "post_flood_false_trigger_rows": int(post_trigger.sum()),
        "s2_false_positive_below_alert_rows": int(((prob["water_level"] < ALERT_LINE) & (y_pred == "S2_FLOOD")).sum()),
        "mean_confidence": float(prob["catboost_confidence"].mean()),
        "median_confidence": float(prob["catboost_confidence"].median()),
        "mean_entropy": float(prob["catboost_entropy"].mean()),
        "median_entropy": float(prob["catboost_entropy"].median()),
        "high_confidence_rows": int(prob["is_high_confidence"].sum()),
        "medium_confidence_rows": int(prob["is_medium_confidence"].sum()),
        "low_confidence_rows": int(prob["is_low_confidence"].sum()),
        "notes": "Train probabilities are KD-usable only for train dataset; Event3 probabilities are evaluation-only.",
    }
    matrix = confusion_matrix(y_true, y_pred, labels=STATE_ORDER)
    rows = []
    for i, true_state in enumerate(STATE_ORDER):
        for j, pred_state in enumerate(STATE_ORDER):
            rows.append({"dataset_name": dataset_name, "true_state": true_state, "pred_state": pred_state, "count": int(matrix[i, j])})
    return row, pd.DataFrame(rows)


def plot_state_axis(ax, df: pd.DataFrame, col: str, title: str) -> None:
    ax.plot(df["timestamp"], df["water_level"], color="tab:blue", linewidth=1.1)
    for state, color in STATE_COLORS.items():
        part = df[df[col] == state]
        ax.scatter(part["timestamp"], part["water_level"], color=color, s=12, alpha=0.82, label=state)
    ax.axhline(ALERT_LINE, color="black", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_ylabel("water_level")


def save_timeline(prob: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    plot_state_axis(axes[0], prob, "true_state_name", f"{title} - ground truth")
    plot_state_axis(axes[1], prob, "catboost_hard_name", f"{title} - CatBoost prediction")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confidence_plot(prob: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    axes[0].plot(prob["timestamp"], prob["water_level"], color="tab:blue")
    axes[0].axhline(ALERT_LINE, color="black", linestyle="--")
    axes[0].set_title("water_level")
    axes[1].plot(prob["timestamp"], prob["catboost_confidence"], color="tab:green")
    axes[1].set_title("catboost_confidence")
    axes[2].plot(prob["timestamp"], prob["catboost_entropy"], color="tab:red")
    axes[2].set_title("catboost_entropy")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("timestamp")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_probability_curves(prob: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(18, 7))
    for col, color in [
        ("catboost_soft_S0", "gray"),
        ("catboost_soft_S1", "orange"),
        ("catboost_soft_S2", "red"),
        ("catboost_soft_S3", "blue"),
    ]:
        ax1.plot(prob["timestamp"], prob[col], label=col, color=color)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("probability")
    ax2 = ax1.twinx()
    ax2.plot(prob["timestamp"], prob["water_level"], color="black", alpha=0.35, label="water_level")
    ax2.axhline(ALERT_LINE, color="black", linestyle="--", alpha=0.7)
    ax2.set_ylabel("water_level")
    ax1.set_title("Frozen CatBoost probability curves - Core Event3")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source_model = root / "outputs/catboost_teacher_control_v3c/models/catboost_raw_s2_weight_2p0.joblib"
    train_path = root / "data/synthetic/B2_semdiv_balanced_event_shape/experiment_B_train_B2_semdiv_balanced_teacher_v3c.csv"
    core_path = root / "data/splits/experiment_B_test_real_teacher_v3c.csv"
    full_path = root / "data/splits/event3_full_cycle_test_teacher_features_v3c.csv"
    feature_path = root / "outputs/tables/feature_columns_v3c.json"
    feature_audit = root / "outputs/tables/feature_leakage_audit_v3c/teacher_feature_leakage_audit_summary_v3c.txt"
    leakage_audit = root / "outputs/tables/event3_full_cycle_leakage_audit_v3c/event3_full_cycle_leakage_audit_summary_v3c.txt"
    secondary_summary = root / "outputs/tables/secondary_rise_audit_v3c/secondary_rise_audit_summary_v3c.txt"
    secondary_fig = root / "outputs/figures/secondary_rise_audit_v3c/secondary_rise_segment_overview.png"
    out_root = root / "outputs/teacher_frozen_catboost_v3c"
    model_dir = out_root / "models"
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    meta_dir = out_root / "metadata"
    for directory in [model_dir, table_dir, fig_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    require_files([source_model, train_path, core_path, full_path, feature_path, feature_audit, leakage_audit, secondary_summary, secondary_fig])

    features = load_features(feature_path)
    train = normalize_ground_truth(pd.read_csv(train_path), "train")
    core = normalize_ground_truth(pd.read_csv(core_path), "core")
    full = normalize_ground_truth(pd.read_csv(full_path), "full")
    validate_data(train, core, full, features)

    model = joblib.load(source_model)
    validate_model(model)
    frozen_model = model_dir / "catboost_raw_s2_weight_2p0_frozen_teacher.joblib"
    shutil.copy2(source_model, frozen_model)

    datasets = [
        ("train_B2_semdiv_balanced", train, table_dir / "catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv"),
        ("core_event3_test", core, table_dir / "catboost_teacher_probabilities_core_event3_test_v3c.csv"),
        ("full_cycle_event3_test", full, table_dir / "catboost_teacher_probabilities_full_cycle_event3_test_v3c.csv"),
    ]
    metrics_rows = []
    matrices = []
    probs = {}
    for name, df, out_path in datasets:
        prob = probability_table(df, model, features)
        prob.to_csv(out_path, index=False, encoding="utf-8-sig")
        row, matrix = evaluate(name, prob)
        metrics_rows.append(row)
        matrices.append(matrix)
        probs[name] = prob
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(table_dir / "frozen_catboost_teacher_metrics_v3c.csv", index=False, encoding="utf-8-sig")
    pd.concat(matrices, ignore_index=True).to_csv(table_dir / "frozen_catboost_teacher_confusion_matrices_v3c.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"feature_index": i, "feature_name": name, "category": "causal_water_level_feature"} for i, name in enumerate(features)]
    ).to_csv(table_dir / "frozen_catboost_teacher_feature_columns_v3c.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "teacher_name": TEACHER_NAME,
        "teacher_type": "CatBoostClassifier",
        "source_model_path": str(source_model),
        "frozen_model_path": str(frozen_model),
        "state_mapping": STATE_ID_TO_NAME,
        "teacher_feature_columns": features,
        "teacher_feature_count": len(features),
        "train_dataset_path": str(train_path),
        "core_test_path": str(core_path),
        "full_cycle_test_path": str(full_path),
        "final_teacher_status": "recommended_after_secondary_rise_audit",
        "xgboost_status": "control_baseline",
        "guard_status": "ablation_only_not_final",
        "secondary_rise_audit_status": "CatBoost better preserves secondary-rise S1/S2 semantics.",
        "leakage_audit_status": "passed",
        "feature_leakage_audit_status": "passed",
        "known_limitations": [
            "one Event3 holdout only",
            "S2/S3 boundary not fully solved",
            "CatBoost improves secondary-rise semantics but still requires future validation",
            "Event3 probabilities are evaluation-only",
        ],
        "created_by_script": "src/09_03_freeze_catboost_teacher_v3c.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (meta_dir / "frozen_catboost_teacher_metadata_v3c.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    save_timeline(probs["core_event3_test"], fig_dir / "frozen_catboost_core_event3_timeline.png", "Frozen CatBoost core Event3")
    save_timeline(probs["full_cycle_event3_test"], fig_dir / "frozen_catboost_full_cycle_event3_timeline.png", "Frozen CatBoost full-cycle Event3")
    save_confidence_plot(probs["core_event3_test"], fig_dir / "frozen_catboost_confidence_core_event3.png")
    save_probability_curves(probs["core_event3_test"], fig_dir / "frozen_catboost_probability_curves_core_event3.png")
    shutil.copy2(secondary_fig, fig_dir / "frozen_catboost_secondary_rise_overview.png")

    core_m = metrics[metrics["dataset_name"] == "core_event3_test"].iloc[0]
    full_m = metrics[metrics["dataset_name"] == "full_cycle_event3_test"].iloc[0]
    summary = f"""task name: Step 9A-extra-2 - Freeze CatBoost teacher package and export CatBoost probabilities
no training performed.
no student training performed.
final teacher candidate = catboost_raw_s2_weight_2p0.
XGBoost kept as control baseline.
synthetic version = B2_semdiv_balanced.
teacher feature count = {len(features)}.
only causal water-level features are used.
Event3 is not used in train / synthetic.
core/full-cycle Event3 probabilities are evaluation-only.
leakage checks passed.
feature leakage audit passed.
guard candidates are ablation only, not final.
secondary-rise audit conclusion: CatBoost better preserves secondary-rise S1/S2 semantics; recommended after freeze.

CatBoost core test metrics:
macro_f1={core_m['macro_f1']:.6f}
S1 recall={core_m['s1_recall']:.6f}
S2 recall={core_m['s2_recall']:.6f}
S3 recall={core_m['s3_recall']:.6f}
transition_count={core_m['transition_count']}

CatBoost full-cycle test metrics:
macro_f1={full_m['macro_f1']:.6f}
S0 recall={full_m['s0_recall']:.6f}
post_flood_false_trigger_rows={int(full_m['post_flood_false_trigger_rows'])}

confidence statistics:
core mean confidence={core_m['mean_confidence']:.6f}; low confidence rows={int(core_m['low_confidence_rows'])}
full-cycle mean confidence={full_m['mean_confidence']:.6f}; low confidence rows={int(full_m['low_confidence_rows'])}

KD usage:
catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv can be used for student training / KD.
catboost_teacher_probabilities_core_event3_test_v3c.csv is evaluation-only and must not be used for student training.
catboost_teacher_probabilities_full_cycle_event3_test_v3c.csv is evaluation-only and must not be used for student training.
Event3 remains unseen real-only evaluation.

limitations:
one Event3 holdout only.
S2/S3 boundary not fully solved.
CatBoost improves secondary-rise semantics but still requires future validation.
"""
    (meta_dir / "frozen_catboost_teacher_summary_v3c.txt").write_text(summary, encoding="utf-8")

    print(f"frozen_catboost_model_path: {frozen_model}")
    print(f"teacher_feature_count: {len(features)}")
    print("leakage_checks_passed: True")
    print(metrics[["dataset_name", "rows", "macro_f1", "s0_recall", "s1_recall", "s2_recall", "s3_recall", "transition_count", "mean_confidence", "low_confidence_rows", "post_flood_false_trigger_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
