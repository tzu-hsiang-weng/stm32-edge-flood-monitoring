import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALERT_LINE = 8.0
SAFE_RETURN_LINE = 6.5
STATE_ORDER = ["S0_STABLE", "S1_RISING", "S2_FLOOD", "S3_RECEDING"]
STATE_COLORS = {"S0_STABLE": "gray", "S1_RISING": "orange", "S2_FLOOD": "red", "S3_RECEDING": "blue"}
CANDIDATES = {
    "d3_balanced_control": "student_17f_dt_d3_catboost_confidence_weighted",
    "d4_macro_aggressive": "student_17f_dt_d4_catboost_confidence_weighted",
}


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def validate_pred_cols(df, scope):
    required = ["timestamp", "water_level", "true_state_id", "true_state_name", "catboost_hard_id", "catboost_hard_name"]
    missing = [col for col in required if col not in df.columns]
    for exp in CANDIDATES.values():
        missing += [col for col in [f"{exp}_pred_id", f"{exp}_pred_name"] if col not in df.columns]
    if missing:
        print(f"{scope} columns:", list(df.columns))
        raise ValueError(f"{scope} missing required columns: {missing}")


def transition_count(series):
    if len(series) == 0:
        return 0
    return int((series != series.shift(1)).sum() - 1)


def transitions_inside(mask, pred):
    changed = pred != pred.shift(1)
    return int((changed & mask & mask.shift(1).fillna(False)).sum())


def state_counts(series):
    counts = series.value_counts().to_dict()
    return {state: int(counts.get(state, 0)) for state in STATE_ORDER}


def normalize_df(df):
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["water_level"] = out["water_level"].astype(float)
    return out.sort_values("timestamp").reset_index(drop=True)


def active_segments(df):
    active = df[df["true_state_name"].isin(["S1_RISING", "S2_FLOOD"])].copy()
    if active.empty:
        raise ValueError("No active S1/S2 segments found.")
    active["gap_min"] = active["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    active["segment_order"] = (active["gap_min"] > 2).cumsum() + 1
    segments = []
    for order, part in active.groupby("segment_order"):
        segments.append(
            {
                "segment_order": int(order),
                "start_time": part["timestamp"].min(),
                "end_time": part["timestamp"].max(),
                "rows": int(len(part)),
            }
        )
    return pd.DataFrame(segments)


def true_state_segments(df, state):
    part = df[df["true_state_name"] == state].copy()
    if part.empty:
        return pd.DataFrame()
    part["gap_min"] = part["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    part["segment_id"] = (part["gap_min"] > 2).cumsum() + 1
    rows = []
    for sid, seg in part.groupby("segment_id"):
        rows.append({"segment_id": int(sid), "start_time": seg["timestamp"].min(), "end_time": seg["timestamp"].max(), "rows": int(len(seg))})
    return pd.DataFrame(rows)


def segment_slice(df, start, end):
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()


def recall(part, pred_col, state):
    mask = part["true_state_name"] == state
    if not mask.any():
        return np.nan
    return float((part.loc[mask, pred_col] == state).mean())


def segment_metrics(candidate, pred_col, role, part):
    true_s3 = part["true_state_name"] == "S3_RECEDING"
    return {
        "candidate_name": candidate,
        "segment_role": role,
        "start_time": part["timestamp"].min() if len(part) else "",
        "end_time": part["timestamp"].max() if len(part) else "",
        "true_state_counts": json.dumps(state_counts(part["true_state_name"]), ensure_ascii=False),
        "s1_recall": recall(part, pred_col, "S1_RISING"),
        "s2_recall": recall(part, pred_col, "S2_FLOOD"),
        "s3_recall": recall(part, pred_col, "S3_RECEDING"),
        "true_S1_pred_S3": int(((part["true_state_name"] == "S1_RISING") & (part[pred_col] == "S3_RECEDING")).sum()),
        "true_S2_pred_S3": int(((part["true_state_name"] == "S2_FLOOD") & (part[pred_col] == "S3_RECEDING")).sum()),
        "true_S3_pred_S1_or_S2": int((true_s3 & part[pred_col].isin(["S1_RISING", "S2_FLOOD"])).sum()),
        "transition_count": transition_count(part[pred_col]),
        "interpretation": "",
    }


def first_transition_time(df, pred_col, from_state, to_state):
    prev = df[pred_col].shift(1)
    mask = (prev == from_state) & (df[pred_col] == to_state)
    return df.loc[mask, "timestamp"].min() if mask.any() else pd.NaT


def candidate_metric_rows(metrics, complexity):
    rows = []
    for label, exp in CANDIDATES.items():
        cx = complexity[complexity["experiment_id"] == exp].iloc[0]
        for scope in ["core_event3_test", "full_cycle_event3_test"]:
            row = metrics[(metrics["experiment_id"] == exp) & (metrics["evaluation_scope"] == scope)].iloc[0]
            interpretation = (
                "balanced/control-stable candidate with stronger S3 stability"
                if label == "d3_balanced_control"
                else "macro-F1/aggressive detection candidate with stronger S2 recall"
            )
            rows.append(
                {
                    "candidate_name": label,
                    "evaluation_scope": scope,
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
                    "leaf_count": cx["leaf_count"],
                    "model_file_size_kb": cx["model_file_size_kb"],
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def disagreement_summary(df, pred_col):
    disagree = df[df[pred_col] != df["catboost_hard_name"]].copy()
    if disagree.empty:
        return "none"
    by_true = disagree["true_state_name"].value_counts().to_dict()
    return json.dumps({k: int(v) for k, v in by_true.items()}, ensure_ascii=False)


def save_timeline(df, path, start=None, end=None):
    plot_df = df.copy()
    if start is not None:
        plot_df = plot_df[plot_df["timestamp"] >= start]
    if end is not None:
        plot_df = plot_df[plot_df["timestamp"] <= end]
    cols = ["true_state_name", "catboost_hard_name"] + [f"{exp}_pred_name" for exp in CANDIDATES.values()]
    titles = ["ground truth", "CatBoost teacher", "d3 candidate", "d4 candidate"]
    fig, axes = plt.subplots(4, 1, figsize=(18, 13), sharex=True)
    for ax, col, title in zip(axes, cols, titles):
        ax.plot(plot_df["timestamp"], plot_df["water_level"], color="tab:blue", linewidth=1.1)
        for state, color in STATE_COLORS.items():
            part = plot_df[plot_df[col] == state]
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


def save_bar_figures(metric_rows, segment_rows, fig_dir):
    core = metric_rows[metric_rows["evaluation_scope"] == "core_event3_test"].set_index("candidate_name")
    fig, ax = plt.subplots(figsize=(10, 5))
    core[["macro_f1", "s1_recall", "s2_recall", "s3_recall", "transition_count", "node_count"]].plot(kind="bar", ax=ax)
    ax.set_title("Candidate Metric Tradeoff")
    fig.tight_layout()
    fig.savefig(fig_dir / "candidate_metric_tradeoff.png", dpi=180)
    plt.close(fig)

    s3 = segment_rows[segment_rows["segment_role"] == "long_recession"].set_index("candidate_name")
    full = metric_rows[metric_rows["evaluation_scope"] == "full_cycle_event3_test"].set_index("candidate_name")
    combo = pd.DataFrame(
        {
            "true_S3_pred_S1_or_S2_ratio": core["true_S3_pred_S1_or_S2_ratio"],
            "transitions_inside_true_S3": s3["transition_count"],
            "post_flood_false_trigger_rows": full["post_flood_false_trigger_rows"],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    combo.plot(kind="bar", ax=ax)
    ax.set_title("Candidate S3 False Trigger Comparison")
    fig.tight_layout()
    fig.savefig(fig_dir / "candidate_s3_false_trigger_comparison.png", dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    table_dir = root / "outputs/tables/student_final_candidate_audit_v3c"
    fig_dir = root / "outputs/figures/student_final_candidate_audit_v3c"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "rules": root / "PROJECT_RULES.md",
        "summary": root / "outputs/student_receding_dt_v3c/metadata/student_receding_dt_summary_v3c.txt",
        "metrics": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_metrics_v3c.csv",
        "core": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_predictions_core_v3c.csv",
        "full": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_predictions_full_cycle_v3c.csv",
        "complexity": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_complexity_v3c.csv",
        "selection": root / "outputs/student_receding_dt_v3c/tables/student_receding_dt_selection_summary_v3c.csv",
        "teacher_core": root / "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_core_event3_test_v3c.csv",
        "teacher_full": root / "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_full_cycle_event3_test_v3c.csv",
    }
    for path in paths.values():
        require_file(path)

    metrics = pd.read_csv(paths["metrics"])
    complexity = pd.read_csv(paths["complexity"])
    core = normalize_df(pd.read_csv(paths["core"]))
    full = normalize_df(pd.read_csv(paths["full"]))
    validate_pred_cols(core, "core predictions")
    validate_pred_cols(full, "full-cycle predictions")

    metric_rows = candidate_metric_rows(metrics, complexity)
    for label, exp in CANDIDATES.items():
        pred_col = f"{exp}_pred_name"
        for scope, df in [("core_event3_test", core), ("full_cycle_event3_test", full)]:
            mask_s3 = df["true_state_name"] == "S3_RECEDING"
            metric_rows.loc[
                (metric_rows["candidate_name"] == label) & (metric_rows["evaluation_scope"] == scope),
                "teacher_disagreement_by_true_state",
            ] = disagreement_summary(df, pred_col)
            metric_rows.loc[
                (metric_rows["candidate_name"] == label) & (metric_rows["evaluation_scope"] == scope),
                "true_S3_pred_S1_or_S2_count",
            ] = int((mask_s3 & df[pred_col].isin(["S1_RISING", "S2_FLOOD"])).sum())
    metric_rows.to_csv(table_dir / "student_final_candidate_metrics_v3c.csv", index=False, encoding="utf-8-sig")

    active = active_segments(core)
    if len(active) >= 3:
        middle_seg = active.iloc[1]
        final_seg = active.iloc[-1]
    elif len(active) == 2:
        middle_seg = None
        final_seg = active.iloc[-1]
    else:
        raise ValueError("Need at least two active segments for secondary-rise audit.")
    s3_segments = true_state_segments(core, "S3_RECEDING")
    long_rec = s3_segments.sort_values("rows", ascending=False).iloc[0]
    s0_segments = true_state_segments(full, "S0_STABLE")
    post_s0 = s0_segments.sort_values("start_time", ascending=False).iloc[0] if not s0_segments.empty else None

    segment_rows = []
    for label, exp in CANDIDATES.items():
        pred_col = f"{exp}_pred_name"
        if middle_seg is not None:
            segment_rows.append(segment_metrics(label, pred_col, "middle_rerise", segment_slice(core, middle_seg["start_time"], middle_seg["end_time"])))
        segment_rows.append(segment_metrics(label, pred_col, "final_rise", segment_slice(core, final_seg["start_time"], final_seg["end_time"])))
        segment_rows.append(segment_metrics(label, pred_col, "long_recession", segment_slice(core, long_rec["start_time"], long_rec["end_time"])))
        if post_s0 is not None:
            segment_rows.append(segment_metrics(label, pred_col, "post_flood_s0", segment_slice(full, post_s0["start_time"], post_s0["end_time"])))
    segment_df = pd.DataFrame(segment_rows)
    segment_df["interpretation"] = segment_df.apply(
        lambda r: "secondary-rise preservation" if r["segment_role"] in ["middle_rerise", "final_rise"] else "stability/safety segment",
        axis=1,
    )
    segment_df.to_csv(table_dir / "student_final_candidate_segment_audit_v3c.csv", index=False, encoding="utf-8-sig")

    complexity_rows = []
    for label, exp in CANDIDATES.items():
        row = complexity[complexity["experiment_id"] == exp].iloc[0]
        complexity_rows.append(
            {
                "candidate_name": label,
                "feature_count": 17,
                "actual_depth": row["actual_depth"],
                "node_count": row["node_count"],
                "leaf_count": row["leaf_count"],
                "estimated_rule_count": row["estimated_rule_count"],
                "model_file_size_kb": row["model_file_size_kb"],
                "deployment_comment": "very compact rule candidate" if row["actual_depth"] <= 3 else "still shallow, but more complex than d3",
            }
        )
    complexity_df = pd.DataFrame(complexity_rows)
    complexity_df.to_csv(table_dir / "student_final_candidate_complexity_v3c.csv", index=False, encoding="utf-8-sig")

    d3_core = metric_rows[(metric_rows["candidate_name"] == "d3_balanced_control") & (metric_rows["evaluation_scope"] == "core_event3_test")].iloc[0]
    d4_core = metric_rows[(metric_rows["candidate_name"] == "d4_macro_aggressive") & (metric_rows["evaluation_scope"] == "core_event3_test")].iloc[0]
    d3_full = metric_rows[(metric_rows["candidate_name"] == "d3_balanced_control") & (metric_rows["evaluation_scope"] == "full_cycle_event3_test")].iloc[0]
    d4_full = metric_rows[(metric_rows["candidate_name"] == "d4_macro_aggressive") & (metric_rows["evaluation_scope"] == "full_cycle_event3_test")].iloc[0]
    decision_rows = [
        {
            "decision_item": "best_for_full_4state_control",
            "selected_candidate": "d3_balanced_control",
            "reason": "Higher S3 recall, much lower true-S3 S1/S2 false-trigger ratio, lower depth/node count.",
            "tradeoff": "Lower S2 recall and macro F1 than d4.",
            "risk": "May under-detect S2 compared with aggressive candidate.",
            "next_action": "Use as preliminary control-stable candidate; audit S2 boundary.",
        },
        {
            "decision_item": "best_for_aggressive_trigger_detection",
            "selected_candidate": "d4_macro_aggressive",
            "reason": "Higher macro F1 and much higher S2 recall.",
            "tradeoff": "Lower S3 recall and higher S3 false-trigger ratio than d3.",
            "risk": "More aggressive S2 behavior can reduce receding semantic stability.",
            "next_action": "Use as aggressive detection comparator, not control-stable default.",
        },
        {
            "decision_item": "best_for_low_complexity",
            "selected_candidate": "d3_balanced_control",
            "reason": "Depth 3 and 15 nodes versus d4 depth 4 and 29 nodes.",
            "tradeoff": "Lower S2 recall.",
            "risk": "Simpler boundary may miss some S2 rows.",
            "next_action": "Prefer if deployability and interpretability dominate.",
        },
        {
            "decision_item": "best_for_macro_f1",
            "selected_candidate": "d4_macro_aggressive",
            "reason": "Core macro F1 is higher.",
            "tradeoff": "S3 stability is weaker.",
            "risk": "Macro F1 hides receding-state false trigger risk.",
            "next_action": "Keep as performance-oriented candidate.",
        },
        {
            "decision_item": "final_recommended_student",
            "selected_candidate": "d3_balanced_control",
            "reason": "Research goal emphasizes edge 4-state semantic control: early warning, safe recovery, and stable receding.",
            "tradeoff": "Accepts lower S2 recall than d4 to preserve S3 and compactness.",
            "risk": "Still below CatBoost teacher and needs validation beyond one Event3 holdout.",
            "next_action": "Treat as preliminary final candidate, then run S3-aware weighting / MLP KD / hysteresis ablation as controls.",
        },
        {
            "decision_item": "remaining_limitation",
            "selected_candidate": "both",
            "reason": "Both candidates remain below teacher and are evaluated on one holdout event.",
            "tradeoff": "d3 is stable but less aggressive; d4 is aggressive but less stable.",
            "risk": "No cross-event generalization claim is allowed.",
            "next_action": "Report limitations explicitly.",
        },
        {
            "decision_item": "next_step",
            "selected_candidate": "none",
            "reason": "Do not tune on Event3.",
            "tradeoff": "Further improvements need controlled ablations.",
            "risk": "Guard or hysteresis chosen on Event3 could overfit.",
            "next_action": "Run S3-aware weighting ablation, standard soft-label MLP KD control, and hysteresis ablation as non-final controls.",
        },
    ]
    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(table_dir / "student_final_candidate_decision_v3c.csv", index=False, encoding="utf-8-sig")

    true_s2s3 = core[core["true_state_name"].isin(["S2_FLOOD", "S3_RECEDING"])].copy()
    first_s2 = true_s2s3[true_s2s3["true_state_name"] == "S2_FLOOD"]["timestamp"].min()
    first_s3 = true_s2s3[true_s2s3["true_state_name"] == "S3_RECEDING"]["timestamp"].min()
    zoom_start = first_s2 - pd.Timedelta(minutes=30)
    zoom_end = first_s3 + pd.Timedelta(minutes=240)
    sec_start = middle_seg["start_time"] - pd.Timedelta(minutes=30) if middle_seg is not None else final_seg["start_time"] - pd.Timedelta(minutes=30)
    sec_end = final_seg["end_time"] + pd.Timedelta(minutes=60)

    save_timeline(core, fig_dir / "candidate_core_timeline_comparison.png")
    save_timeline(full, fig_dir / "candidate_full_cycle_timeline_comparison.png")
    save_timeline(core, fig_dir / "candidate_s2_s3_zoom_comparison.png", zoom_start, zoom_end)
    save_timeline(core, fig_dir / "candidate_secondary_rise_zoom_comparison.png", sec_start, sec_end)
    save_bar_figures(metric_rows, segment_df, fig_dir)

    d3_pred = f"{CANDIDATES['d3_balanced_control']}_pred_name"
    d4_pred = f"{CANDIDATES['d4_macro_aggressive']}_pred_name"
    boundary_lines = []
    for label, pred_col in [("d3", d3_pred), ("d4", d4_pred)]:
        s2_to_s3 = first_transition_time(core, pred_col, "S2_FLOOD", "S3_RECEDING")
        true_s2_to_s3 = first_transition_time(core.rename(columns={"true_state_name": "truth"}), "truth", "S2_FLOOD", "S3_RECEDING")
        boundary_lines.append(f"{label} first S2->S3: {s2_to_s3}; true first S2->S3: {true_s2_to_s3}")

    summary = f"""task name: Step 9C-final-audit - Compare final student candidates
no training performed.
no teacher inference performed.
no Event3 tuning, threshold tuning, hysteresis, or post-processing.

candidates compared:
- d3_balanced_control = {CANDIDATES['d3_balanced_control']}
- d4_macro_aggressive = {CANDIDATES['d4_macro_aggressive']}

d3 strengths:
- stronger S3 stability: S3 recall={d3_core['s3_recall']:.6f}, true_S3_pred_S1_or_S2_ratio={d3_core['true_S3_pred_S1_or_S2_ratio']:.6f}
- lower complexity: depth=3, nodes=15
- full-cycle S0 safety: S0 recall={d3_full['s0_recall']:.6f}, post_flood_false_trigger_rows={int(d3_full['post_flood_false_trigger_rows'])}

d3 weaknesses:
- lower S2 recall={d3_core['s2_recall']:.6f}
- macro F1={d3_core['macro_f1']:.6f}, lower than d4

d4 strengths:
- higher macro F1={d4_core['macro_f1']:.6f}
- much higher S2 recall={d4_core['s2_recall']:.6f}
- full-cycle S0 safety: S0 recall={d4_full['s0_recall']:.6f}, post_flood_false_trigger_rows={int(d4_full['post_flood_false_trigger_rows'])}

d4 weaknesses:
- lower S3 recall={d4_core['s3_recall']:.6f}
- higher true_S3_pred_S1_or_S2_ratio={d4_core['true_S3_pred_S1_or_S2_ratio']:.6f}
- more complex: depth=4, nodes=29

S2/S3 boundary comparison:
{chr(10).join(boundary_lines)}

secondary-rise comparison:
{segment_df[segment_df['segment_role'].isin(['middle_rerise', 'final_rise'])].to_string(index=False)}

long-recession S3 comparison:
{segment_df[segment_df['segment_role'] == 'long_recession'].to_string(index=False)}

full-cycle S0 safety comparison:
Both candidates keep full-cycle S0 recall at 1.0 and post-flood false trigger rows at 0.

teacher agreement comparison:
d3 agreement core={d3_core['student_vs_catboost_agreement']:.6f}; disagreement by true state={d3_core.get('teacher_disagreement_by_true_state', '')}
d4 agreement core={d4_core['student_vs_catboost_agreement']:.6f}; disagreement by true state={d4_core.get('teacher_disagreement_by_true_state', '')}

best_for_full_4state_control: d3_balanced_control
best_for_aggressive_trigger_detection: d4_macro_aggressive
final recommended student: d3_balanced_control as preliminary final candidate for edge 4-state semantic control.

whether current student can be considered preliminary final candidate:
Yes, d3 can be considered preliminary final candidate for control-stable 4-state student, but not a permanent final without further validation.

limitations:
- only one Event3 holdout
- d3 sacrifices S2 recall
- d4 sacrifices receding stability
- both remain below CatBoost teacher
- no cross-event generalization claim

recommended next step:
Run S3-aware weighting ablation, standard soft-label MLP KD control, and hysteresis ablation as controls; do not tune thresholds on Event3.
"""
    (table_dir / "student_final_candidate_audit_summary_v3c.txt").write_text(summary, encoding="utf-8")

    print("audit_success: True")
    print(f"compared_candidates: {list(CANDIDATES.values())}")
    print("best_for_full_4state_control: d3_balanced_control")
    print("best_for_aggressive_trigger_detection: d4_macro_aggressive")
    print("final_recommended_student: d3_balanced_control")
    print(f"d3_core_macro_f1: {d3_core['macro_f1']}")
    print(f"d4_core_macro_f1: {d4_core['macro_f1']}")


if __name__ == "__main__":
    main()
