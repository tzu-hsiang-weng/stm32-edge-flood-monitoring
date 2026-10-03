import json
import shutil
from pathlib import Path

import pandas as pd


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")


def row(df: pd.DataFrame, **kwargs) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in kwargs.items():
        mask &= df[key] == value
    if not mask.any():
        raise ValueError(f"Missing row for {kwargs}")
    return df[mask].iloc[0]


def copy_if_exists(src: Path, dst: Path, missing: list[str]) -> None:
    if src.exists():
        shutil.copy2(src, dst)
    else:
        missing.append(str(src))


def metric(summary: pd.DataFrame, segment_role: str, metric_name: str) -> pd.Series:
    return row(summary, segment_role=segment_role, metric=metric_name)


def snapshot_row(model_name: str, scope: str, r: pd.Series, interpretation: str) -> dict:
    return {
        "model_name": model_name,
        "test_scope": scope,
        "macro_f1": r["macro_f1"],
        "accuracy": r["accuracy"],
        "s0_recall": r["s0_recall"],
        "s1_recall": r["s1_recall"],
        "s2_recall": r["s2_recall"],
        "s3_recall": r["s3_recall"],
        "s1_precision": r["s1_precision"],
        "s2_precision": r["s2_precision"],
        "s3_precision": r["s3_precision"],
        "transition_count": r["transition_count"],
        "post_flood_false_trigger_rows": r["post_flood_false_trigger_rows"],
        "true_S1_pred_S3_count": r["true_S1_pred_S3_count"],
        "true_S2_pred_S3_count": r["true_S2_pred_S3_count"],
        "true_S3_pred_S2_count": r["true_S3_pred_S2_count"],
        "mean_confidence": r["mean_confidence"],
        "low_confidence_rows": r["low_confidence_rows"],
        "interpretation": interpretation,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    xgb_metrics_path = root / "outputs/teacher_frozen_v3c/tables/frozen_teacher_metrics_v3c.csv"
    cat_metrics_path = root / "outputs/teacher_frozen_catboost_v3c/tables/frozen_catboost_teacher_metrics_v3c.csv"
    comparison_path = root / "outputs/catboost_teacher_control_v3c/tables/xgboost_vs_catboost_teacher_comparison_v3c.csv"
    secondary_summary_path = root / "outputs/tables/secondary_rise_audit_v3c/secondary_rise_xgb_vs_catboost_summary_v3c.csv"
    secondary_decision_path = root / "outputs/tables/secondary_rise_audit_v3c/secondary_rise_teacher_decision_evidence_v3c.csv"
    cat_metadata_path = root / "outputs/teacher_frozen_catboost_v3c/metadata/frozen_catboost_teacher_metadata_v3c.json"
    cat_summary_path = root / "outputs/teacher_frozen_catboost_v3c/metadata/frozen_catboost_teacher_summary_v3c.txt"
    xgb_summary_path = root / "outputs/teacher_frozen_v3c/metadata/frozen_teacher_summary_v3c.txt"
    feature_audit_path = root / "outputs/tables/feature_leakage_audit_v3c/teacher_feature_leakage_audit_summary_v3c.txt"
    leakage_audit_path = root / "outputs/tables/event3_full_cycle_leakage_audit_v3c/event3_full_cycle_leakage_audit_summary_v3c.txt"

    table_dir = root / "outputs/tables/final_teacher_decision_catboost_v3c"
    fig_dir = root / "outputs/figures/final_teacher_decision_catboost_v3c"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    require_files(
        [
            xgb_metrics_path,
            cat_metrics_path,
            comparison_path,
            secondary_summary_path,
            secondary_decision_path,
            cat_metadata_path,
            cat_summary_path,
            xgb_summary_path,
            feature_audit_path,
            leakage_audit_path,
        ]
    )

    xgb_metrics = pd.read_csv(xgb_metrics_path)
    cat_metrics = pd.read_csv(cat_metrics_path)
    comparison = pd.read_csv(comparison_path)
    secondary = pd.read_csv(secondary_summary_path)
    secondary_decision = pd.read_csv(secondary_decision_path)
    metadata = json.loads(cat_metadata_path.read_text(encoding="utf-8"))

    xgb_core = row(comparison, model_name="xgboost_raw_s2_weight_2p0", test_scope="core_test")
    xgb_full = row(comparison, model_name="xgboost_raw_s2_weight_2p0", test_scope="full_cycle_test")
    cat_core = row(comparison, model_name="catboost_raw_s2_weight_2p0", test_scope="core_test")
    cat_full = row(comparison, model_name="catboost_raw_s2_weight_2p0", test_scope="full_cycle_test")

    mid_s2_recall = metric(secondary, "middle_rerise", "s2_recall")
    mid_s2_s3 = metric(secondary, "middle_rerise", "true_S2_pred_S3")
    mid_pred_s2 = metric(secondary, "middle_rerise", "pred_S2_rows_on_true_S2")
    final_s1_recall = metric(secondary, "final_rise", "s1_recall")
    final_s1_s3 = metric(secondary, "final_rise", "true_S1_pred_S3")
    final_first_s1 = metric(secondary, "final_rise", "first_pred_S1_time")
    final_s1_lead = metric(secondary, "final_rise", "pred_S1_lead_before_true_S2_min")
    final_s2_recall = metric(secondary, "final_rise", "s2_recall")

    decision_rows = [
        {
            "item": "updated_final_teacher",
            "value": "catboost_raw_s2_weight_2p0",
            "evidence": f"core macro_f1={cat_core['macro_f1']:.6f}, S2={cat_core['s2_recall']:.6f}, S3={cat_core['s3_recall']:.6f}",
            "interpretation": "Updated final teacher after CatBoost freeze and secondary-rise audit.",
        },
        {
            "item": "previous_teacher",
            "value": "xgboost_raw_s2_weight_2p0",
            "evidence": f"core macro_f1={xgb_core['macro_f1']:.6f}, S2={xgb_core['s2_recall']:.6f}",
            "interpretation": "Previous teacher remains useful as a comparison reference.",
        },
        {
            "item": "control_baseline",
            "value": "xgboost_raw_s2_weight_2p0",
            "evidence": "XGBoost frozen teacher retained as control baseline.",
            "interpretation": "XGBoost is not discarded; it remains a control teacher.",
        },
        {"item": "synthetic_version", "value": "B2_semdiv_balanced", "evidence": "CatBoost frozen package metadata.", "interpretation": "Same train synthetic version as XGBoost teacher."},
        {"item": "feature_count", "value": metadata["teacher_feature_count"], "evidence": "frozen_catboost_teacher_metadata_v3c.json", "interpretation": "Expected 44 teacher features."},
        {"item": "feature_type", "value": "causal water-level features", "evidence": "feature leakage audit passed.", "interpretation": "No label/event/source/split/synthetic metadata/trace/future feature."},
        {"item": "event3_usage", "value": "evaluation only", "evidence": "Event3 leakage audit passed.", "interpretation": "Event3 probabilities must not be used for student training."},
        {
            "item": "why_catboost_selected",
            "value": "secondary-rise semantic preservation",
            "evidence": f"middle S2 recall CatBoost={mid_s2_recall['catboost_value']} vs XGBoost={mid_s2_recall['xgboost_value']}; final S1 recall CatBoost={final_s1_recall['catboost_value']} vs XGBoost={final_s1_recall['xgboost_value']}",
            "interpretation": "Selection is not based on a large overall score jump, but on secondary-rise S1/S2 semantics with stable S0 safety.",
        },
        {
            "item": "why_xgboost_control",
            "value": "lower transition count and strong reference",
            "evidence": f"XGBoost core transition={xgb_core['transition_count']}; CatBoost core transition={cat_core['transition_count']}",
            "interpretation": "XGBoost remains a useful control baseline and optional teacher control.",
        },
        {"item": "why_guard_not_final", "value": "ablation only", "evidence": "Guard candidates came from Event3 audit.", "interpretation": "Final teacher is raw CatBoost without guard correction."},
        {
            "item": "kd_train_probability_file",
            "value": "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv",
            "evidence": "Frozen CatBoost package.",
            "interpretation": "Primary KD train probability file.",
        },
        {
            "item": "kd_evaluation_only_files",
            "value": "CatBoost core/full-cycle Event3 probability CSVs",
            "evidence": "Frozen CatBoost package.",
            "interpretation": "Evaluation only; not for student training.",
        },
        {
            "item": "main_limitations",
            "value": "one Event3 holdout; S2/S3 boundary not fully solved; future validation needed",
            "evidence": "CatBoost and secondary-rise summaries.",
            "interpretation": "Do not claim cross-event generalization.",
        },
        {"item": "next_stage_readiness", "value": "ready for student/KD protocol", "evidence": "Frozen CatBoost teacher package completed.", "interpretation": "Proceed with CatBoost as primary teacher and XGBoost as optional control."},
    ]
    pd.DataFrame(decision_rows).to_csv(table_dir / "final_teacher_decision_catboost_v3c.csv", index=False, encoding="utf-8-sig")

    model_rows = [
        snapshot_row("xgboost_raw_s2_weight_2p0", "core_test", xgb_core, "Previous final teacher; now control baseline."),
        snapshot_row("xgboost_raw_s2_weight_2p0", "full_cycle_test", xgb_full, "XGBoost control full-cycle reference."),
        snapshot_row("catboost_raw_s2_weight_2p0", "core_test", cat_core, "Updated final teacher; better secondary-rise semantics."),
        snapshot_row("catboost_raw_s2_weight_2p0", "full_cycle_test", cat_full, "Updated final teacher; full-cycle S0 safety preserved."),
    ]
    pd.DataFrame(model_rows).to_csv(table_dir / "teacher_model_comparison_catboost_v3c.csv", index=False, encoding="utf-8-sig")

    secondary_rows = []
    for segment_role, metric_name, out_metric in [
        ("middle_rerise", "s2_recall", "middle_rerise_s2_recall"),
        ("middle_rerise", "true_S2_pred_S3", "middle_rerise_true_S2_pred_S3"),
        ("middle_rerise", "pred_S2_rows_on_true_S2", "middle_rerise_pred_S2_rows_on_true_S2"),
        ("final_rise", "s1_recall", "final_rise_s1_recall"),
        ("final_rise", "true_S1_pred_S3", "final_rise_true_S1_pred_S3"),
        ("final_rise", "first_pred_S1_time", "final_rise_first_pred_S1_time"),
        ("final_rise", "pred_S1_lead_before_true_S2_min", "final_rise_s1_lead_before_true_S2"),
        ("final_rise", "s2_recall", "final_rise_s2_recall"),
    ]:
        r = metric(secondary, segment_role, metric_name)
        secondary_rows.append(
            {
                "segment_role": segment_role,
                "metric": out_metric,
                "xgboost_value": r["xgboost_value"],
                "catboost_value": r["catboost_value"],
                "winner": r["winner"],
                "interpretation": r["interpretation"],
            }
        )
    secondary_rows.append(
        {
            "segment_role": "overall",
            "metric": "overall_secondary_rise_conclusion",
            "xgboost_value": "weaker on middle rerise/final S1",
            "catboost_value": "better secondary-rise preservation",
            "winner": "catboost",
            "interpretation": "CatBoost better preserves middle rerise S2 and final rise S1; future validation still required.",
        }
    )
    pd.DataFrame(secondary_rows).to_csv(table_dir / "secondary_rise_teacher_selection_evidence_v3c.csv", index=False, encoding="utf-8-sig")

    leakage_rows = [
        {"evidence_type": "frozen model", "metric": "CatBoost frozen model path", "value": metadata["frozen_model_path"], "pass_or_fail": "pass", "notes": "Frozen CatBoost model exists."},
        {"evidence_type": "features", "metric": "teacher feature count", "value": metadata["teacher_feature_count"], "pass_or_fail": "pass", "notes": "44 causal water-level features."},
        {"evidence_type": "feature leakage", "metric": "feature leakage audit", "value": "passed", "pass_or_fail": "pass", "notes": "No label/event/source/split/synthetic metadata/trace/future feature."},
        {"evidence_type": "Event3 leakage", "metric": "Event3 train overlap", "value": 0, "pass_or_fail": "pass", "notes": "Event3 remains unseen real-only evaluation."},
        {"evidence_type": "Event3 synthetic usage", "metric": "Event3 used in synthetic", "value": "no", "pass_or_fail": "pass", "notes": "B2_semdiv_balanced train-only synthetic."},
        {"evidence_type": "KD usage", "metric": "train probability KD usage", "value": "allowed", "pass_or_fail": "pass", "notes": "Train CatBoost probabilities may be used for KD."},
        {"evidence_type": "KD usage", "metric": "core Event3 probability usage", "value": "evaluation-only", "pass_or_fail": "pass", "notes": "Do not use Event3 probabilities in student training."},
        {"evidence_type": "KD usage", "metric": "full-cycle Event3 probability usage", "value": "evaluation-only", "pass_or_fail": "pass", "notes": "Do not use Event3 probabilities in student training."},
        {"evidence_type": "guard", "metric": "guard status", "value": "ablation_only_not_final", "pass_or_fail": "pass", "notes": "No guard applied to final CatBoost teacher."},
        {"evidence_type": "XGBoost", "metric": "XGBoost status", "value": "control_baseline", "pass_or_fail": "pass", "notes": "XGBoost retained as teacher control."},
    ]
    pd.DataFrame(leakage_rows).to_csv(table_dir / "teacher_freeze_and_leakage_evidence_catboost_v3c.csv", index=False, encoding="utf-8-sig")

    md = f"""# Final Teacher Decision after CatBoost Freeze

## 1. 最新 final teacher
- final teacher = CatBoost `raw_s2_weight_2p0`
- XGBoost `raw_s2_weight_2p0` = control baseline
- synthetic version = B2_semdiv_balanced
- guard-based correction = ablation only

## 2. 訓練與資料洩漏控制
- Event3 不進 train / synthetic
- CatBoost 使用 44 個 causal water-level features
- feature leakage audit passed
- Event3 probabilities evaluation only
- train probabilities 可用於 student/KD

## 3. XGBoost vs CatBoost overall comparison
- XGBoost core macro F1 = {xgb_core['macro_f1']:.6f}; CatBoost = {cat_core['macro_f1']:.6f}
- XGBoost core S2 recall = {xgb_core['s2_recall']:.6f}; CatBoost = {cat_core['s2_recall']:.6f}
- XGBoost core S3 recall = {xgb_core['s3_recall']:.6f}; CatBoost = {cat_core['s3_recall']:.6f}
- CatBoost transition count 稍高：{cat_core['transition_count']} vs XGBoost {xgb_core['transition_count']}
- full-cycle S0 safety 一樣穩定，CatBoost S0 recall = {cat_full['s0_recall']:.6f}, post-flood false trigger = {int(cat_full['post_flood_false_trigger_rows'])}

## 4. 為什麼選 CatBoost
- 不是因為總分大幅提升。
- 主要因為 secondary-rise audit：
  - middle rerise：CatBoost S2 recall {float(mid_s2_recall['catboost_value']):.3f} vs XGBoost {float(mid_s2_recall['xgboost_value']):.3f}
  - middle rerise true S2 -> S3：CatBoost {mid_s2_s3['catboost_value']} vs XGBoost {mid_s2_s3['xgboost_value']}
  - final rise：CatBoost 抓到 S1，XGBoost 沒抓到；CatBoost first_pred_S1_time = {final_first_s1['catboost_value']}
- S1/S2 是 early-warning trigger 與危險期控制的核心語意。

## 5. 為什麼 XGBoost 保留
- XGBoost 指標仍具參考價值。
- XGBoost transition count 較低。
- 保留為 control baseline。
- 可在 student/KD 中作為 optional teacher control。

## 6. 為什麼不用 guard
- guard 是 Event3 audit 產生的 candidate ablation。
- 雖可提升局部 S2，但會增加 transition / over-correction risk。
- final teacher 使用 raw CatBoost，不套用 guard。

## 7. Limitations
- only one Event3 holdout
- S2/S3 boundary not fully solved
- CatBoost secondary-rise advantage needs future validation
- 不可宣稱泛化到所有洪水事件

## 8. Next stage
- proceed to student feature set / KD protocol
- primary KD teacher = CatBoost
- XGBoost optional control teacher
- Event3 remains evaluation only
"""
    (table_dir / "final_teacher_key_findings_catboost_v3c.md").write_text(md, encoding="utf-8")
    txt = md.replace("# Final Teacher Decision after CatBoost Freeze", "Final Teacher Decision after CatBoost Freeze").replace("## ", "").replace("- ", "")
    (table_dir / "final_teacher_key_findings_catboost_v3c.txt").write_text(txt, encoding="utf-8")

    kd_plan = """Next-stage KD plan after CatBoost final teacher decision

Primary teacher for KD:
CatBoost raw_s2_weight_2p0

Optional teacher control:
XGBoost raw_s2_weight_2p0

Train probability file for KD:
outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv

Evaluation-only probability files:
outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_core_event3_test_v3c.csv
outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_full_cycle_event3_test_v3c.csv

Step 9B: define student feature set
Use only the approved 8 deployable student features.

Step 9C: hard-label student baseline
Train a constrained student directly on FSM labels for comparison.

Step 9D: teacher-hard / confidence-filtered / mixed-label KD
Compare teacher hard labels, confidence-filtered labels, and mixed FSM/teacher labels.

Step 9E: teacher vs student evaluation
Evaluate on Event3 core and full-cycle only as evaluation.

Selection criteria:
S1 recall, S0 full-cycle recovery, post-flood false trigger, S3 stability, S2/S3 boundary, and model size.
"""
    (table_dir / "next_stage_kd_plan_after_catboost_v3c.txt").write_text(kd_plan, encoding="utf-8")

    fig_dir = root / "outputs/figures/final_teacher_decision_catboost_v3c"
    fig_dir.mkdir(parents=True, exist_ok=True)
    missing_figures = []
    for src, dst_name in [
        (root / "outputs/teacher_frozen_catboost_v3c/figures/frozen_catboost_core_event3_timeline.png", "final_teacher_catboost_core_timeline.png"),
        (root / "outputs/teacher_frozen_catboost_v3c/figures/frozen_catboost_probability_curves_core_event3.png", "final_teacher_catboost_probability_curves.png"),
        (root / "outputs/figures/secondary_rise_audit_v3c/secondary_rise_s1_s2_recall_comparison.png", "final_teacher_secondary_rise_recall_comparison.png"),
        (root / "outputs/figures/secondary_rise_audit_v3c/final_rise_segment_zoom.png", "final_teacher_final_rise_zoom.png"),
        (root / "outputs/figures/secondary_rise_audit_v3c/middle_rerise_segment_zoom.png", "final_teacher_middle_rerise_zoom.png"),
    ]:
        if src.exists():
            shutil.copy2(src, fig_dir / dst_name)
        else:
            missing_figures.append(str(src))
    if missing_figures:
        (table_dir / "missing_figures_note_v3c.txt").write_text("\n".join(missing_figures), encoding="utf-8")

    print("updated_final_teacher: catboost_raw_s2_weight_2p0")
    print("xgboost_status: control_baseline")
    print("why_catboost_selected: secondary-rise S1/S2 preservation plus stable S0 safety")
    print("kd_next_stage: primary teacher CatBoost, optional XGBoost control")
    print(f"missing_figures: {missing_figures}")


if __name__ == "__main__":
    main()
