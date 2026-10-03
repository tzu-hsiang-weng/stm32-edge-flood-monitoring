import json
import runpy
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


ALERT_LINE = 8.0
SAFE_RETURN_LINE = 6.5
STATE_ID_TO_NAME = {0: "S0_STABLE", 1: "S1_RISING", 2: "S2_FLOOD", 3: "S3_RECEDING"}
STATE_ORDER = ["S0_STABLE", "S1_RISING", "S2_FLOOD", "S3_RECEDING"]
STATE_COLORS = {"S0_STABLE": "gray", "S1_RISING": "orange", "S2_FLOOD": "red", "S3_RECEDING": "blue"}
FEATURE_SET_NAME = "student_17f_domain_receding"
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
VARIANTS = [
    "hard_label_baseline",
    "catboost_teacher_hard",
    "catboost_confidence_weighted",
    "catboost_mixed_label",
    "catboost_mixed_label_conf_weighted",
]
DEPTHS = [3, 4, 5, 6]
TEACHER_REF = {
    "core_macro_f1": 0.662399,
    "core_s1_recall": 0.954545,
    "core_s2_recall": 0.617647,
    "core_s3_recall": 0.997674,
    "core_transition_count": 13,
    "full_macro_f1": 0.900123,
    "full_s0_recall": 1.0,
    "full_post_flood_false_trigger_rows": 0,
}


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def json_text(value):
    return json.dumps(value, ensure_ascii=False)


def add_receding_features(df):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    original_index = out.index
    group_cols = [col for col in ["source", "split"] if col in out.columns]
    out = out.sort_values(group_cols + ["timestamp"] if group_cols else ["timestamp"]).copy()

    def transform_group(g):
        g = g.sort_values("timestamp").copy()
        wl = g["water_level"].astype(float)
        g["dist_to_alert"] = ALERT_LINE - wl
        g["dist_to_safe_return"] = wl - SAFE_RETURN_LINE
        g["above_alert_line"] = (wl >= ALERT_LINE).astype(int)
        for window in [1, 3, 5, 10, 30]:
            g[f"delta_{window}m"] = wl - wl.shift(window)
        g["slope_5m"] = g["delta_5m"] / 5.0
        g["slope_10m"] = g["delta_10m"] / 10.0
        g["slope_30m"] = g["delta_30m"] / 30.0
        g["max_5m"] = wl.rolling(window=5, min_periods=1).max()
        roll10 = wl.rolling(window=10, min_periods=1)
        g["range_10m"] = roll10.max() - roll10.min()
        roll30 = wl.rolling(window=30, min_periods=1)
        g["recent_max_30m"] = roll30.max()
        g["range_30m"] = roll30.max() - roll30.min()
        g["drawdown_from_recent_max_30m"] = g["recent_max_30m"] - wl
        return g

    if group_cols:
        out = pd.concat([transform_group(group) for _, group in out.groupby(group_cols, sort=False)], axis=0).sort_index()
    else:
        out = transform_group(out)
    out[STUDENT_FEATURES] = out[STUDENT_FEATURES].fillna(0.0)
    out = out.loc[original_index].copy()
    missing = [feature for feature in STUDENT_FEATURES if feature not in out.columns]
    if missing:
        raise ValueError(f"Student feature recompute failed: {missing}")
    forbidden_substrings = ["label", "future", "trace", "pred", "teacher", "ground_truth", "gt"]
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
    bad_exact = sorted(set(STUDENT_FEATURES) & forbidden_exact)
    bad_substrings = sorted(
        {feature for feature in STUDENT_FEATURES for marker in forbidden_substrings if marker.lower() in feature.lower()}
    )
    if bad_exact or bad_substrings:
        raise ValueError(f"Student features contain leakage-risk columns: exact={bad_exact}, substrings={bad_substrings}")
    return out


def add_s3_false_trigger_metrics(row, df, pred_col):
    true_s3 = df["true_state_name"] == "S3_RECEDING"
    false_s1s2 = true_s3 & df[pred_col].isin(["S1_RISING", "S2_FLOOD"])
    row["true_S3_pred_S1_or_S2"] = int(false_s1s2.sum())
    row["true_S3_pred_S1_or_S2_ratio"] = float(false_s1s2.sum() / true_s3.sum()) if true_s3.any() else np.nan
    return row


def score_experiment(core_row, full_row):
    score = float(core_row["macro_f1"])
    if core_row["s1_recall"] >= 0.90:
        score += 0.20
    if full_row["s0_recall"] >= 0.98:
        score += 0.10
    if full_row["post_flood_false_trigger_rows"] == 0:
        score += 0.10
    if core_row["s3_recall"] >= 0.60:
        score += 0.10
    if core_row["actual_depth"] <= 5:
        score += 0.05
    if core_row["node_count"] <= 31:
        score += 0.05
    if core_row["s1_recall"] < 0.80:
        score -= 0.30
    if full_row["s0_recall"] < 0.95:
        score -= 0.30
    if full_row["post_flood_false_trigger_rows"] > 0:
        score -= 0.20
    if core_row["s3_recall"] < 0.50:
        score -= 0.25
    if core_row["transition_count"] > 50:
        score -= 0.15
    if core_row["true_S3_pred_S1_or_S2_ratio"] > 0.40:
        score -= 0.10
    if core_row["s2_false_positive_below_alert_rows"] > 0:
        score -= 0.10
    return score


def save_state_timeline(df, path, pred_col, title):
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    for ax, col, subplot_title in [
        (axes[0], "true_state_name", "ground truth"),
        (axes[1], "catboost_hard_name", "CatBoost teacher"),
        (axes[2], pred_col, title),
    ]:
        ax.plot(df["timestamp"], df["water_level"], color="tab:blue", linewidth=1.1)
        for state, color in STATE_COLORS.items():
            part = df[df[col] == state]
            ax.scatter(part["timestamp"], part["water_level"], s=10, color=color, alpha=0.82, label=state)
        ax.axhline(ALERT_LINE, color="black", linestyle="--", linewidth=0.9)
        ax.axhline(SAFE_RETURN_LINE, color="green", linestyle=":", linewidth=0.9)
        ax.set_title(subplot_title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=5, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_figures(metrics, old_metrics, core_pred, full_pred, best_exp, comparison, fig_dir):
    core = metrics[metrics["evaluation_scope"] == "core_event3_test"].copy()
    full = metrics[metrics["evaluation_scope"] == "full_cycle_event3_test"].copy()
    core["label"] = core["experiment_id"] + "\n" + core["training_variant"].str.replace("catboost_", "cb_", regex=False)
    fig, ax = plt.subplots(figsize=(18, 7))
    core.set_index("label")[["s1_recall", "s2_recall", "s3_recall", "macro_f1"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("17f Receding-memory DT Students - Core Recall / Macro F1")
    fig.tight_layout()
    fig.savefig(fig_dir / "student_receding_dt_core_recall_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(18, 7))
    core.set_index("label")[["s3_recall", "transition_count", "true_S3_pred_S1_or_S2_ratio"]].plot(kind="bar", ax=ax)
    ax.set_title("17f Receding-memory DT Students - S3 / Transition")
    fig.tight_layout()
    fig.savefig(fig_dir / "student_receding_dt_s3_transition_comparison.png", dpi=180)
    plt.close(fig)

    plot_items = comparison[
        comparison["comparison_item"].isin(["best_core_s3_recall", "lowest_transition_count", "lowest_true_S3_pred_S1_or_S2_ratio"])
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_items))
    ax.bar(x - 0.18, plot_items["old_12f_best_value"].astype(float), width=0.36, label="old 12f")
    ax.bar(x + 0.18, plot_items["new_17f_best_value"].astype(float), width=0.36, label="new 17f")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_items["comparison_item"], rotation=20, ha="right")
    ax.legend()
    ax.set_title("12f vs 17f S3 Comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "student_12f_vs_17f_s3_comparison.png", dpi=180)
    plt.close(fig)

    pred_col = f"{best_exp}_pred_name"
    save_state_timeline(core_pred, fig_dir / "best_receding_student_core_timeline.png", pred_col, f"best balanced 17f student: {best_exp}")
    save_state_timeline(full_pred, fig_dir / "best_receding_student_full_cycle_timeline.png", pred_col, f"best balanced 17f student: {best_exp}")
    first_s2 = core_pred.loc[core_pred["true_state_name"] == "S2_FLOOD", "timestamp"].min()
    first_s3 = core_pred.loc[core_pred["true_state_name"] == "S3_RECEDING", "timestamp"].min()
    if pd.notna(first_s2) and pd.notna(first_s3):
        zoom = core_pred[
            (core_pred["timestamp"] >= first_s2 - pd.Timedelta(minutes=30))
            & (core_pred["timestamp"] <= first_s3 + pd.Timedelta(minutes=240))
        ].copy()
    else:
        zoom = core_pred.copy()
    save_state_timeline(zoom, fig_dir / "best_receding_student_s2_s3_zoom.png", pred_col, f"best balanced 17f student: {best_exp}")


def main():
    root = Path(__file__).resolve().parents[1]
    helper = runpy.run_path(str(root / "src/09_07_train_required_dt_students_v3c.py"))
    validate_columns = helper["validate_columns"]
    make_target_and_weight = helper["make_target_and_weight"]
    evaluate = helper["evaluate"]

    train_path = root / "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv"
    core_path = root / "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_core_event3_test_v3c.csv"
    full_path = root / "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_full_cycle_event3_test_v3c.csv"
    old_metrics_path = root / "outputs/student_required_dt_v3c/tables/student_required_dt_metrics_v3c.csv"
    old_complexity_path = root / "outputs/student_required_dt_v3c/tables/student_required_dt_complexity_v3c.csv"
    old_core_pred_path = root / "outputs/student_required_dt_v3c/tables/student_required_dt_predictions_core_v3c.csv"
    audit_path = root / "outputs/tables/student_s3_failure_audit_v3c/student_s3_failure_audit_summary_v3c.txt"
    out_root = root / "outputs/student_receding_dt_v3c"
    model_dir = out_root / "models"
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    meta_dir = out_root / "metadata"
    for path in [train_path, core_path, full_path, old_metrics_path, old_complexity_path, old_core_pred_path, audit_path, root / "PROJECT_RULES.md"]:
        require_file(path)
    for directory in [model_dir, table_dir, fig_dir, meta_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    core = pd.read_csv(core_path)
    full = pd.read_csv(full_path)
    required_train_cols = [
        "timestamp",
        "water_level",
        "source",
        "split",
        "true_state_id",
        "true_state_name",
        "catboost_hard_id",
        "catboost_hard_name",
        "catboost_soft_S0",
        "catboost_soft_S1",
        "catboost_soft_S2",
        "catboost_soft_S3",
        "catboost_confidence",
    ]
    required_eval_cols = ["timestamp", "water_level", "true_state_id", "true_state_name", "catboost_hard_id", "catboost_hard_name", "catboost_confidence"]
    validate_columns(train, required_train_cols, "train")
    validate_columns(core, required_eval_cols, "core")
    validate_columns(full, required_eval_cols, "full_cycle")
    for name, df in [("train", train), ("core", core), ("full_cycle", full)]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["water_level"] = df["water_level"].astype(float)
        df["true_state_id"] = df["true_state_id"].astype(int)
        df["catboost_hard_id"] = df["catboost_hard_id"].astype(int)
        if "source" in df.columns and name != "train" and set(df["source"].unique()) != {"real"}:
            raise ValueError(f"{name} source must be all real: {df['source'].value_counts().to_dict()}")
    if not set(train["source"].unique()).issubset({"real", "synthetic_B2_semdiv_balanced"}):
        raise ValueError(f"Unexpected train source: {train['source'].value_counts().to_dict()}")
    if set(train["timestamp"]) & set(core["timestamp"]) or set(train["timestamp"]) & set(full["timestamp"]):
        raise ValueError("Train timestamps overlap Event3 evaluation timestamps.")

    train = add_receding_features(train)
    core = add_receding_features(core)
    full = add_receding_features(full)
    X_train = train[STUDENT_FEATURES].astype(float)
    X_core = core[STUDENT_FEATURES].astype(float)
    X_full = full[STUDENT_FEATURES].astype(float)
    core_pred = core[["timestamp", "water_level", "true_state_id", "true_state_name", "catboost_hard_id", "catboost_hard_name"]].copy()
    full_pred = full[["timestamp", "water_level", "true_state_id", "true_state_name", "catboost_hard_id", "catboost_hard_name"]].copy()

    metrics_rows, matrix_rows, complexity_rows, experiments = [], [], [], []
    for depth in DEPTHS:
        for variant in VARIANTS:
            experiment_id = f"student_17f_dt_d{depth}_{variant}"
            target, sample_weight = make_target_and_weight(train, variant)
            model = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight=None)
            model.fit(X_train, target, sample_weight=sample_weight)
            model_path = model_dir / f"{experiment_id}.joblib"
            joblib.dump(model, model_path)
            model_size_kb = model_path.stat().st_size / 1024.0
            core_ids = model.predict(X_core).astype(int)
            full_ids = model.predict(X_full).astype(int)
            core_pred[f"{experiment_id}_pred_id"] = core_ids
            core_pred[f"{experiment_id}_pred_name"] = pd.Series(core_ids).map(STATE_ID_TO_NAME).values
            full_pred[f"{experiment_id}_pred_id"] = full_ids
            full_pred[f"{experiment_id}_pred_name"] = pd.Series(full_ids).map(STATE_ID_TO_NAME).values

            core_eval = core_pred[["timestamp", "water_level", "true_state_name", "catboost_hard_name", f"{experiment_id}_pred_name"]].rename(
                columns={f"{experiment_id}_pred_name": "student_pred_name"}
            )
            full_eval = full_pred[["timestamp", "water_level", "true_state_name", "catboost_hard_name", f"{experiment_id}_pred_name"]].rename(
                columns={f"{experiment_id}_pred_name": "student_pred_name"}
            )
            core_row, core_matrix = evaluate(core_eval, "student_pred_name", experiment_id, FEATURE_SET_NAME, variant, depth, model, "core_event3_test")
            full_row, full_matrix = evaluate(full_eval, "student_pred_name", experiment_id, FEATURE_SET_NAME, variant, depth, model, "full_cycle_event3_test")
            core_row = add_s3_false_trigger_metrics(core_row, core_eval, "student_pred_name")
            full_row = add_s3_false_trigger_metrics(full_row, full_eval, "student_pred_name")
            score = score_experiment(core_row, full_row)
            for row in [core_row, full_row]:
                row["feature_count"] = len(STUDENT_FEATURES)
                row["model_file_size_kb"] = model_size_kb
                row["deployment_selection_score"] = score
            metrics_rows.extend([core_row, full_row])
            matrix_rows.extend(core_matrix + full_matrix)
            complexity_rows.append(
                {
                    "experiment_id": experiment_id,
                    "feature_set_name": FEATURE_SET_NAME,
                    "training_variant": variant,
                    "model_type": "DecisionTreeClassifier",
                    "max_depth_setting": depth,
                    "actual_depth": int(model.get_depth()),
                    "node_count": int(model.tree_.node_count),
                    "leaf_count": int(model.get_n_leaves()),
                    "estimated_rule_count": int(model.get_n_leaves()),
                    "model_file_size_kb": model_size_kb,
                    "used_features": json_text(STUDENT_FEATURES),
                    "model_path": str(model_path),
                }
            )
            experiments.append(experiment_id)

    if len(experiments) != 20:
        raise ValueError(f"Expected 20 required experiments, got {len(experiments)}.")
    metrics = pd.DataFrame(metrics_rows)
    confusion = pd.DataFrame(matrix_rows)
    complexity = pd.DataFrame(complexity_rows)
    metrics.to_csv(table_dir / "student_receding_dt_metrics_v3c.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(table_dir / "student_receding_dt_confusion_matrices_v3c.csv", index=False, encoding="utf-8-sig")
    core_pred.to_csv(table_dir / "student_receding_dt_predictions_core_v3c.csv", index=False, encoding="utf-8-sig")
    full_pred.to_csv(table_dir / "student_receding_dt_predictions_full_cycle_v3c.csv", index=False, encoding="utf-8-sig")
    complexity.to_csv(table_dir / "student_receding_dt_complexity_v3c.csv", index=False, encoding="utf-8-sig")

    core_m = metrics[metrics["evaluation_scope"] == "core_event3_test"].copy()
    full_m = metrics[metrics["evaluation_scope"] == "full_cycle_event3_test"][
        ["experiment_id", "s0_recall", "post_flood_false_trigger_rows", "transition_count"]
    ].rename(columns={"s0_recall": "full_cycle_s0_recall", "post_flood_false_trigger_rows": "full_cycle_post_flood_false_trigger_rows", "transition_count": "full_cycle_transition_count"})
    merged = core_m.merge(full_m, on="experiment_id", how="left")

    def sel_row(criterion, selected_id, reason):
        row = merged[merged["experiment_id"] == selected_id].iloc[0]
        return {
            "criterion": criterion,
            "selected_experiment_id": selected_id,
            "reason": reason,
            "core_macro_f1": row["macro_f1"],
            "core_s1_recall": row["s1_recall"],
            "core_s2_recall": row["s2_recall"],
            "core_s3_recall": row["s3_recall"],
            "transition_count": row["transition_count"],
            "true_S3_pred_S1_or_S2_ratio": row["true_S3_pred_S1_or_S2_ratio"],
            "full_cycle_s0_recall": row["full_cycle_s0_recall"],
            "post_flood_false_trigger_rows": row["full_cycle_post_flood_false_trigger_rows"],
            "actual_depth": row["actual_depth"],
            "node_count": row["node_count"],
            "training_variant": row["training_variant"],
        }

    best_macro = merged.sort_values(["macro_f1", "s1_recall", "s3_recall"], ascending=False).iloc[0]["experiment_id"]
    best_s1 = merged[(merged["s1_recall"] >= 0.90) & (merged["full_cycle_post_flood_false_trigger_rows"] == 0)]
    best_s1 = (best_s1 if not best_s1.empty else merged).sort_values(["s1_recall", "macro_f1"], ascending=False).iloc[0]["experiment_id"]
    best_s3 = merged.sort_values(["s3_recall", "transition_count"], ascending=[False, True]).iloc[0]["experiment_id"]
    best_s0 = merged[(merged["full_cycle_s0_recall"] >= 0.98) & (merged["full_cycle_post_flood_false_trigger_rows"] == 0)]
    best_s0 = (best_s0 if not best_s0.empty else merged).sort_values(["full_cycle_s0_recall", "full_cycle_post_flood_false_trigger_rows", "macro_f1"], ascending=[False, True, False]).iloc[0]["experiment_id"]
    best_score = merged.sort_values(["deployment_selection_score", "macro_f1"], ascending=False).iloc[0]["experiment_id"]
    low = merged[(merged["actual_depth"] <= 4) & (merged["node_count"] <= 31)]
    best_low = (low if not low.empty else merged).sort_values(["macro_f1", "node_count"], ascending=[False, True]).iloc[0]["experiment_id"]
    balanced = merged[(merged["s1_recall"] >= 0.80) & (merged["s2_recall"] >= 0.50) & (merged["s3_recall"] >= 0.50) & (merged["full_cycle_post_flood_false_trigger_rows"] == 0)]
    best_bal = (balanced if not balanced.empty else merged).sort_values(["s3_recall", "macro_f1", "transition_count"], ascending=[False, False, True]).iloc[0]["experiment_id"]
    selection = pd.DataFrame(
        [
            sel_row("best_by_core_macro_f1", best_macro, "Highest core macro F1 with recall tie-breakers."),
            sel_row("best_by_s1_trigger_safety", best_s1, "Prioritizes S1 recall and zero post-flood trigger."),
            sel_row("best_by_s3_receding_stability", best_s3, "Prioritizes S3 recall and lower transition count."),
            sel_row("best_by_full_cycle_s0_safety", best_s0, "Prioritizes full-cycle S0 safety."),
            sel_row("best_by_deployment_score", best_score, "Highest deployment selection score."),
            sel_row("best_low_complexity_student", best_low, "Best low-complexity student."),
            sel_row("best_balanced_4state_student", best_bal, "Best available S1/S2/S3/S0 balance."),
        ]
    )
    selection.to_csv(table_dir / "student_receding_dt_selection_summary_v3c.csv", index=False, encoding="utf-8-sig")

    old_metrics = pd.read_csv(old_metrics_path)
    old_complexity = pd.read_csv(old_complexity_path)
    old_core_pred = pd.read_csv(old_core_pred_path)
    old_core = old_metrics[old_metrics["evaluation_scope"] == "core_event3_test"].merge(old_complexity[["experiment_id", "actual_depth", "node_count"]], on="experiment_id", how="left", suffixes=("", "_cx"))
    old_ratio_rows = []
    for exp in old_core["experiment_id"].unique():
        pred_col = f"{exp}_pred_name"
        if pred_col not in old_core_pred.columns:
            continue
        true_s3 = old_core_pred["true_state_name"] == "S3_RECEDING"
        false_s1s2 = true_s3 & old_core_pred[pred_col].isin(["S1_RISING", "S2_FLOOD"])
        old_ratio_rows.append(
            {
                "experiment_id": exp,
                "true_S3_pred_S1_or_S2_ratio": float(false_s1s2.sum() / true_s3.sum()) if true_s3.any() else np.nan,
            }
        )
    old_core = old_core.merge(pd.DataFrame(old_ratio_rows), on="experiment_id", how="left")
    old_full = old_metrics[old_metrics["evaluation_scope"] == "full_cycle_event3_test"]

    def old_best_value(metric_name, mode="max"):
        if metric_name == "full_s0":
            df, col = old_full, "s0_recall"
        elif metric_name == "post_trigger":
            df, col, mode = old_full, "post_flood_false_trigger_rows", "min"
        else:
            df, col = old_core, metric_name
        idx = df[col].astype(float).idxmax() if mode == "max" else df[col].astype(float).idxmin()
        return float(df.loc[idx, col]), str(df.loc[idx, "experiment_id"])

    def new_best_value(metric_name, mode="max"):
        if metric_name == "full_s0":
            df, col = metrics[metrics["evaluation_scope"] == "full_cycle_event3_test"], "s0_recall"
        elif metric_name == "post_trigger":
            df, col, mode = metrics[metrics["evaluation_scope"] == "full_cycle_event3_test"], "post_flood_false_trigger_rows", "min"
        else:
            df, col = core_m, metric_name
        idx = df[col].astype(float).idxmax() if mode == "max" else df[col].astype(float).idxmin()
        return float(df.loc[idx, col]), str(df.loc[idx, "experiment_id"])

    comparison_specs = [
        ("best_core_macro_f1", "macro_f1", "max"),
        ("best_core_s1_recall", "s1_recall", "max"),
        ("best_core_s2_recall", "s2_recall", "max"),
        ("best_core_s3_recall", "s3_recall", "max"),
        ("lowest_transition_count", "transition_count", "min"),
        ("lowest_true_S3_pred_S1_or_S2_ratio", "true_S3_pred_S1_or_S2_ratio", "min"),
        ("best_full_cycle_s0_recall", "full_s0", "max"),
        ("lowest_post_flood_false_trigger_rows", "post_trigger", "min"),
        ("best_deployment_score", "deployment_selection_score", "max"),
    ]
    comp_rows = []
    for item, metric_name, mode in comparison_specs:
        old_value, old_exp = old_best_value(metric_name, mode)
        new_value, new_exp = new_best_value(metric_name, mode)
        delta = new_value - old_value
        interpretation = "improved" if (delta > 0 and mode == "max") or (delta < 0 and mode == "min") else "not_improved"
        comp_rows.append(
            {
                "comparison_item": item,
                "old_12f_best_value": old_value,
                "old_12f_experiment": old_exp,
                "new_17f_best_value": new_value,
                "new_17f_experiment": new_exp,
                "delta": delta,
                "interpretation": interpretation,
            }
        )
    comparison = pd.DataFrame(comp_rows)
    comparison.to_csv(table_dir / "student_12f_vs_17f_comparison_v3c.csv", index=False, encoding="utf-8-sig")

    best_balanced = selection[selection["criterion"] == "best_balanced_4state_student"]["selected_experiment_id"].iloc[0]
    save_figures(metrics, old_metrics, core_pred, full_pred, best_balanced, comparison, fig_dir)

    hard_best_macro = core_m[core_m["training_variant"] == "hard_label_baseline"]["macro_f1"].max()
    kd_best_macro = core_m[core_m["training_variant"] != "hard_label_baseline"]["macro_f1"].max()
    kd_improves = bool(kd_best_macro > hard_best_macro)
    s3_improved = comparison[comparison["comparison_item"] == "best_core_s3_recall"]["interpretation"].iloc[0] == "improved"
    transition_improved = comparison[comparison["comparison_item"] == "lowest_transition_count"]["interpretation"].iloc[0] == "improved"
    s1_safe = bool((core_m["s1_recall"] >= 0.90).any())
    s0_safe = bool(((metrics[metrics["evaluation_scope"] == "full_cycle_event3_test"]["s0_recall"] >= 0.98) & (metrics[metrics["evaluation_scope"] == "full_cycle_event3_test"]["post_flood_false_trigger_rows"] == 0)).any())
    can_final = bool(
        (
            (merged["s1_recall"] >= 0.90)
            & (merged["s3_recall"] >= 0.60)
            & (merged["transition_count"] <= 50)
            & (merged["full_cycle_s0_recall"] >= 0.98)
            & (merged["full_cycle_post_flood_false_trigger_rows"] == 0)
        ).any()
    )

    metadata = {
        "task_name": "Step 9C-fix - Add receding-memory causal features and rerun required DecisionTree students",
        "primary_teacher": "CatBoost raw_s2_weight_2p0",
        "feature_set_name": FEATURE_SET_NAME,
        "student_features": STUDENT_FEATURES,
        "required_experiment_count": len(experiments),
        "training_data_path": str(train_path),
        "core_eval_path": str(core_path),
        "full_cycle_eval_path": str(full_path),
        "leakage_check_status": "passed",
        "event3_usage": "evaluation_only_no_training_validation_tuning",
        "warmup_fillna_zero": True,
        "no_s3_class_weight": True,
        "no_hysteresis_postprocessing": True,
        "created_by_script": Path(__file__).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (meta_dir / "student_receding_dt_metadata_v3c.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def sel_text(name):
        row = selection[selection["criterion"] == name].iloc[0]
        return (
            f"{row['selected_experiment_id']} | macro_f1={row['core_macro_f1']:.6f}, S1={row['core_s1_recall']:.6f}, "
            f"S2={row['core_s2_recall']:.6f}, S3={row['core_s3_recall']:.6f}, transition={int(row['transition_count'])}, "
            f"S3_false_ratio={row['true_S3_pred_S1_or_S2_ratio']:.6f}, full_S0={row['full_cycle_s0_recall']:.6f}, "
            f"post_trigger={int(row['post_flood_false_trigger_rows'])}, depth={int(row['actual_depth'])}, nodes={int(row['node_count'])}"
        )

    summary = f"""task name: Step 9C-fix - Add receding-memory causal features and rerun required DecisionTree students
no teacher training performed.
no teacher re-inference performed.
no Event3 training, validation, early stopping, hyperparameter tuning, or threshold tuning.
no S3 class weight added.
no hysteresis/post-processing applied.
primary teacher = CatBoost raw_s2_weight_2p0
feature set = student_17f_domain_receding
added receding-memory features = recent_max_30m, drawdown_from_recent_max_30m, delta_30m, slope_30m, range_30m
required experiments = {len(experiments)}
feature recompute method: causal shift/rolling within source/split groups sorted by timestamp.
warmup_fillna_zero=True
leakage checks result: passed

teacher reference metrics:
Core macro_f1={TEACHER_REF['core_macro_f1']}, S1 recall={TEACHER_REF['core_s1_recall']}, S2 recall={TEACHER_REF['core_s2_recall']}, S3 recall={TEACHER_REF['core_s3_recall']}, transition_count={TEACHER_REF['core_transition_count']}
Full-cycle macro_f1={TEACHER_REF['full_macro_f1']}, S0 recall={TEACHER_REF['full_s0_recall']}, post_flood_false_trigger_rows={TEACHER_REF['full_post_flood_false_trigger_rows']}

best_by_core_macro_f1:
{sel_text('best_by_core_macro_f1')}

best_by_s1_trigger_safety:
{sel_text('best_by_s1_trigger_safety')}

best_by_s3_receding_stability:
{sel_text('best_by_s3_receding_stability')}

best_by_full_cycle_s0_safety:
{sel_text('best_by_full_cycle_s0_safety')}

best_by_deployment_score:
{sel_text('best_by_deployment_score')}

best_balanced_4state_student:
{sel_text('best_balanced_4state_student')}

12f vs 17f comparison:
{comparison.to_string(index=False)}

receding-memory features improve S3: {s3_improved}
transition count improves: {transition_improved}
S1 safety preserved by at least one 17f student: {s1_safe}
S0 recovery preserved by at least one 17f student: {s0_safe}
KD variants improve hard-label by core macro F1: {kd_improves}
current 17f student can be final candidate under preliminary criteria: {can_final}

recommendation:
If 17f improves S3 but transition remains high, proceed to S3 weighting / hysteresis ablation.
If 17f still cannot preserve S3, run standard KD control or consider binary trigger fallback while reporting 4-state limitation.
"""
    (meta_dir / "student_receding_dt_summary_v3c.txt").write_text(summary, encoding="utf-8")

    print("leakage_checks_passed: True")
    print(f"required_experiments: {len(experiments)}")
    for criterion in selection["criterion"]:
        print(f"{criterion}: {selection[selection['criterion'] == criterion]['selected_experiment_id'].iloc[0]}")
    print(f"receding_features_improve_s3: {s3_improved}")
    print(f"transition_count_improves: {transition_improved}")
    print(f"s1_safety_preserved: {s1_safe}")
    print(f"s0_recovery_preserved: {s0_safe}")
    print(f"kd_improves_hard_label_macro_f1: {kd_improves}")
    print(f"can_be_final_candidate_preliminary: {can_final}")


if __name__ == "__main__":
    main()
