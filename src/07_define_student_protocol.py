import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ALERT_LINE = 8.0
SAFE_RETURN_LINE = 6.5

STRICT_8F = [
    "water_level",
    "delta_1m",
    "delta_3m",
    "delta_5m",
    "delta_10m",
    "max_5m",
    "range_5m",
    "slope_5m",
]

DOMAIN_12F = [
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
]

TINY_6F = [
    "water_level",
    "dist_to_alert",
    "delta_5m",
    "delta_10m",
    "slope_5m",
    "above_alert_line",
]

FORBIDDEN_FEATURES = [
    "fsm_state_name",
    "fsm_state_id",
    "true_state_name",
    "true_state_id",
    "semantic_state_name",
    "label_id",
    "label_flood_in_20m",
    "event_id",
    "fsm_event_id",
    "source",
    "split",
    "synthetic_id",
    "template_event_id",
    "scenario_type",
    "target_peak",
    "actual_peak",
    "quality_flag",
    "quality_notes",
    "ranking_score",
    "selection_reason",
    "teacher_prediction_columns",
    "future_columns",
    "trace_columns",
    "ground_truth_derived_columns",
]

DOMAIN_FEATURES = {
    "dist_to_alert": f"{ALERT_LINE} - water_level",
    "dist_to_safe_return": f"water_level - {SAFE_RETURN_LINE}",
    "above_alert_line": f"int(water_level >= {ALERT_LINE})",
    "above_safe_return_line": f"int(water_level >= {SAFE_RETURN_LINE})",
}

CAUSAL_WATER_FEATURES = sorted(
    set(STRICT_8F + ["lag_1m", "lag_3m", "lag_5m", "lag_10m", "mean_5m", "std_5m", "range_10m", "slope_10m"])
)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def json_text(value):
    return json.dumps(value, ensure_ascii=False)


def feature_role(feature):
    roles = {
        "water_level": "current level",
        "dist_to_alert": "distance to fixed alert threshold",
        "dist_to_safe_return": "distance from fixed safe-return threshold",
        "above_alert_line": "fixed alert-line gate",
        "above_safe_return_line": "fixed safe-return gate",
        "delta_1m": "1-minute direction",
        "delta_3m": "short rising/falling direction",
        "delta_5m": "S1/S3 trend evidence",
        "delta_10m": "medium short-term trend evidence",
        "slope_5m": "5-minute trend rate",
        "slope_10m": "10-minute trend rate",
        "max_5m": "short local high-water context",
        "range_5m": "short local volatility",
        "range_10m": "10-minute local volatility",
    }
    return roles.get(feature, "causal water-level feature")


def feature_buffer(feature):
    if feature == "water_level":
        return 0
    if feature in DOMAIN_FEATURES:
        return 0
    for minutes in [1, 3, 5, 10, 30, 60]:
        if f"{minutes}m" in feature:
            return minutes
    return 0


def feature_category(feature):
    if feature in DOMAIN_FEATURES:
        return "domain_informed_causal"
    return "causal_water_level"


def recompute_reason(feature):
    if feature in DOMAIN_FEATURES:
        return f"Can be recomputed from current water_level and fixed threshold: {DOMAIN_FEATURES[feature]}."
    if feature == "water_level":
        return "Already present in probability file."
    return "Can be recomputed causally from current/past water_level before student training."


def build_feature_policy(train_columns):
    rows = []
    for feature in FORBIDDEN_FEATURES:
        rows.append(
            {
                "feature_name": feature,
                "category": "forbidden",
                "allowed_status": "forbidden",
                "reason": "Forbidden because it is a label, event, source/split, metadata, teacher prediction, trace, future, or ground-truth-derived column.",
                "leakage_risk": "high",
                "edge_computable": False,
                "requires_feature_recompute": False,
            }
        )
    for feature in CAUSAL_WATER_FEATURES:
        rows.append(
            {
                "feature_name": feature,
                "category": "causal_water_level",
                "allowed_status": "allowed",
                "reason": "Allowed if recomputed only from current/past water_level.",
                "leakage_risk": "low if computed causally",
                "edge_computable": True,
                "requires_feature_recompute": feature not in train_columns,
            }
        )
    for feature, formula in DOMAIN_FEATURES.items():
        rows.append(
            {
                "feature_name": feature,
                "category": "domain_informed_causal",
                "allowed_status": "conditional_allowed",
                "reason": f"Allowed because it uses current water_level and fixed public system threshold only: {formula}. It does not use labels, Event3, future data, or metadata.",
                "leakage_risk": "low; threshold-derived, not label-derived",
                "edge_computable": True,
                "requires_feature_recompute": feature not in train_columns,
            }
        )
    return pd.DataFrame(rows)


def build_feature_sets(train_columns):
    specs = [
        (
            "student_8f_strict_project_rules",
            STRICT_8F,
            "Strict baseline preserving the early PROJECT_RULES.md deployable feature list.",
        ),
        (
            "student_12f_domain_edge",
            DOMAIN_12F,
            "Primary practical student set with conditional allowed domain-informed causal threshold features.",
        ),
        (
            "student_6f_tiny_domain",
            TINY_6F,
            "Extreme edge ablation retaining current level, fixed alert threshold context, and short trends.",
        ),
    ]
    rows = []
    for feature_set, features, reason_prefix in specs:
        for order, feature in enumerate(features, start=1):
            rows.append(
                {
                    "feature_set_name": feature_set,
                    "feature_name": feature,
                    "feature_order": order,
                    "feature_role": feature_role(feature),
                    "causal": True,
                    "edge_friendly": True,
                    "domain_informed": feature in DOMAIN_FEATURES,
                    "required_buffer_minutes": feature_buffer(feature),
                    "requires_feature_recompute": feature not in train_columns,
                    "reason": f"{reason_prefix} {recompute_reason(feature)}",
                }
            )
    return pd.DataFrame(rows)


def build_kd_variants():
    rows = [
        (
            "hard_label_baseline",
            "none",
            "true_state_id",
            "base_sample_weight: real=1.0, synthetic_B2_semdiv_balanced=0.7",
            False,
            False,
            True,
            False,
            True,
            "No-KD baseline.",
            "Event3 excluded from training.",
        ),
        (
            "catboost_teacher_hard",
            "CatBoost raw_s2_weight_2p0",
            "catboost_hard_id",
            "base_sample_weight",
            True,
            False,
            False,
            False,
            True,
            "Teacher-hard imitation.",
            "Use train probability file only.",
        ),
        (
            "catboost_confidence_weighted",
            "CatBoost raw_s2_weight_2p0",
            "catboost_hard_id",
            "base_sample_weight * catboost_confidence",
            True,
            True,
            False,
            False,
            True,
            "Downweight low-confidence teacher decisions.",
            "Confidence is from frozen train probabilities only.",
        ),
        (
            "catboost_mixed_label",
            "CatBoost raw_s2_weight_2p0",
            "catboost_hard_id if catboost_confidence >= 0.80 else true_state_id",
            "base_sample_weight",
            True,
            True,
            True,
            False,
            True,
            "Use teacher when confident, official label otherwise.",
            "No Event3 probability is used for training.",
        ),
        (
            "catboost_mixed_label_conf_weighted",
            "CatBoost raw_s2_weight_2p0",
            "catboost_hard_id if catboost_confidence >= 0.80 else true_state_id",
            "base_sample_weight * max(catboost_confidence, 0.5)",
            True,
            True,
            True,
            False,
            True,
            "Mixed target plus confidence weighting.",
            "No full KL claim; pragmatic teacher-guided compression only.",
        ),
        (
            "xgboost_teacher_hard_control",
            "XGBoost raw_s2_weight_2p0 control",
            "xgboost teacher hard id",
            "base_sample_weight",
            True,
            False,
            False,
            False,
            False,
            "Optional one-run control teacher.",
            "Do not expand to complete XGBoost KD grid.",
        ),
    ]
    columns = [
        "variant_name",
        "teacher_source",
        "target_definition",
        "sample_weight_definition",
        "uses_teacher_hard_label",
        "uses_teacher_confidence",
        "uses_true_label",
        "uses_soft_label",
        "first_round_required",
        "reason",
        "leakage_risk_note",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_experiment_matrix():
    rows = []
    eid = 1
    target_map = {
        "hard_label_baseline": "true_state_id",
        "catboost_teacher_hard": "catboost_hard_id",
        "catboost_confidence_weighted": "catboost_hard_id",
        "catboost_mixed_label": "catboost_hard_id if catboost_confidence >= 0.80 else true_state_id",
        "catboost_mixed_label_conf_weighted": "catboost_hard_id if catboost_confidence >= 0.80 else true_state_id",
        "xgboost_teacher_hard_control": "xgboost teacher hard id",
    }
    weight_map = {
        "hard_label_baseline": "base_sample_weight",
        "catboost_teacher_hard": "base_sample_weight",
        "catboost_confidence_weighted": "base_sample_weight * catboost_confidence",
        "catboost_mixed_label": "base_sample_weight",
        "catboost_mixed_label_conf_weighted": "base_sample_weight * max(catboost_confidence, 0.5)",
        "xgboost_teacher_hard_control": "base_sample_weight",
    }

    required_variants = [
        "hard_label_baseline",
        "catboost_teacher_hard",
        "catboost_confidence_weighted",
        "catboost_mixed_label",
        "catboost_mixed_label_conf_weighted",
    ]
    for depth in [3, 4, 5, 6]:
        for variant in required_variants:
            rows.append(
                {
                    "experiment_id": f"stu9c_fix_{eid:03d}",
                    "run_priority": "required",
                    "feature_set_name": "student_12f_domain_edge",
                    "model_family": "DecisionTree",
                    "model_name": f"DecisionTree_depth_{depth}",
                    "model_params": json_text({"max_depth": depth, "random_state": 42}),
                    "training_variant": variant,
                    "teacher_source": "CatBoost raw_s2_weight_2p0" if variant != "hard_label_baseline" else "none",
                    "target_definition": target_map[variant],
                    "sample_weight_definition": weight_map[variant],
                    "expected_output_model_name": f"student_12f_dt_d{depth}_{variant}",
                    "notes": "Required first-round primary domain-edge DecisionTree experiment.",
                }
            )
            eid += 1

    for feature_set, priority_label in [
        ("student_8f_strict_project_rules", "recommended_strict"),
        ("student_6f_tiny_domain", "recommended_tiny"),
    ]:
        for depth in [4, 5]:
            for variant in ["hard_label_baseline", "catboost_teacher_hard", "catboost_mixed_label"]:
                rows.append(
                    {
                        "experiment_id": f"stu9c_fix_{eid:03d}",
                        "run_priority": priority_label,
                        "feature_set_name": feature_set,
                        "model_family": "DecisionTree",
                        "model_name": f"DecisionTree_depth_{depth}",
                        "model_params": json_text({"max_depth": depth, "random_state": 42}),
                        "training_variant": variant,
                        "teacher_source": "CatBoost raw_s2_weight_2p0" if variant != "hard_label_baseline" else "none",
                        "target_definition": target_map[variant],
                        "sample_weight_definition": weight_map[variant],
                        "expected_output_model_name": f"{feature_set}_dt_d{depth}_{variant}",
                        "notes": "Recommended baseline/ablation after required primary experiments.",
                    }
                )
                eid += 1

    rows.append(
        {
            "experiment_id": f"stu9c_fix_{eid:03d}",
            "run_priority": "optional",
            "feature_set_name": "student_12f_domain_edge",
            "model_family": "DecisionTree",
            "model_name": "DecisionTree_depth_4",
            "model_params": json_text({"max_depth": 4, "random_state": 42}),
            "training_variant": "xgboost_teacher_hard_control",
            "teacher_source": "XGBoost raw_s2_weight_2p0 control",
            "target_definition": target_map["xgboost_teacher_hard_control"],
            "sample_weight_definition": weight_map["xgboost_teacher_hard_control"],
            "expected_output_model_name": "student_12f_dt_d4_xgboost_teacher_hard_control",
            "notes": "Optional teacher control.",
        }
    )
    eid += 1

    for n_estimators, depth in [(10, 2), (20, 2), (20, 3)]:
        for variant in ["hard_label_baseline", "catboost_teacher_hard", "catboost_mixed_label"]:
            rows.append(
                {
                    "experiment_id": f"stu9c_fix_{eid:03d}",
                    "run_priority": "optional",
                    "feature_set_name": "student_12f_domain_edge",
                    "model_family": "TinyXGBoost",
                    "model_name": f"TinyXGB_{n_estimators}trees_depth{depth}",
                    "model_params": json_text(
                        {
                            "n_estimators": n_estimators,
                            "max_depth": depth,
                            "learning_rate": 0.1,
                            "subsample": 1.0,
                            "colsample_bytree": 1.0,
                            "objective": "multi:softprob",
                            "num_class": 4,
                        }
                    ),
                    "training_variant": variant,
                    "teacher_source": "CatBoost raw_s2_weight_2p0" if variant != "hard_label_baseline" else "none",
                    "target_definition": target_map[variant],
                    "sample_weight_definition": weight_map[variant],
                    "expected_output_model_name": f"student_12f_tinyxgb_{n_estimators}t_d{depth}_{variant}",
                    "notes": "Optional Tiny XGBoost control.",
                }
            )
            eid += 1
    return pd.DataFrame(rows)


def build_selection_criteria():
    rows = [
        ("Feature policy", "primary_feature_set", "student_12f_domain_edge", "Using Event3 or label-derived features", "Primary practical feature set allows fixed-threshold causal domain features."),
        ("Feature policy", "strict_feature_set", "student_8f_strict_project_rules", "Dropping strict baseline entirely", "Strict baseline keeps comparability to early PROJECT_RULES.md."),
        ("Feature policy", "tiny_feature_set", "student_6f_tiny_domain", "Claiming tiny is final without evidence", "Tiny feature set is an ablation, not assumed best."),
        ("S1 early warning", "core_s1_recall", ">= 0.90 preferred", "< 0.80 usually reject", "S1 is the trigger state."),
        ("S0 safety", "full_cycle_s0_recall", ">= 0.98", "< 0.98 reject unless justified", "Full-cycle recovery is required for deployment."),
        ("S0 safety", "post_flood_false_trigger_rows", "0 preferred", "> 0 requires review", "Avoid post-flood false triggers."),
        ("S3 stability", "transition_count", "not much higher than teacher", "transition explosion", "Fragmented outputs are hard to deploy."),
        ("S2 boundary", "core_s2_recall", "as close to teacher as possible", "improves only by sacrificing S1/S0", "S2/S3 boundary remains a known limitation."),
        ("KD value", "improvement_over_hard_label_baseline", "clear system metric improvement", "no benefit over same-capacity baseline", "Only claim teacher guidance is useful if it beats hard-label baseline."),
    ]
    return pd.DataFrame(
        rows,
        columns=["criterion_group", "criterion_name", "preferred_threshold", "reject_condition", "reason"],
    )


def save_plots(feature_sets, matrix, fig_dir):
    counts = pd.Series(
        {
            "teacher": 44,
            "student_12f_domain_edge": 12,
            "student_8f_strict_project_rules": 8,
            "student_6f_tiny_domain": 6,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax, color=["tab:blue", "tab:orange", "tab:green", "tab:purple"])
    ax.set_ylabel("feature count")
    ax.set_title("Updated Student Feature Set Overview")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "student_feature_set_overview_fix.png", dpi=180)
    plt.close(fig)

    priority_counts = (
        matrix["run_priority"].value_counts().reindex(["required", "recommended_strict", "recommended_tiny", "optional"]).fillna(0).astype(int)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    priority_counts.plot(kind="bar", ax=ax, color=["tab:red", "tab:gray", "tab:green", "tab:purple"])
    ax.set_ylabel("experiment count")
    ax.set_title("Updated Student Experiment Matrix Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "student_experiment_matrix_count_fix.png", dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    table_dir = root / "outputs/tables/student_kd_protocol_v3c_fix"
    fig_dir = root / "outputs/figures/student_kd_protocol_v3c_fix"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        root / "PROJECT_RULES.md",
        root / "outputs/tables/final_teacher_decision_catboost_v3c/final_teacher_key_findings_catboost_v3c.md",
        root / "outputs/teacher_frozen_catboost_v3c/metadata/frozen_catboost_teacher_metadata_v3c.json",
        root / "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv",
    ]
    for path in required_files:
        require_file(path)

    train_path = root / "outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv"
    train_columns = set(pd.read_csv(train_path, nrows=1).columns)
    if "water_level" not in train_columns:
        raise ValueError("Train probability file must contain water_level to allow causal feature recomputation.")

    with (root / "outputs/teacher_frozen_catboost_v3c/metadata/frozen_catboost_teacher_metadata_v3c.json").open(
        "r", encoding="utf-8"
    ) as f:
        meta = json.load(f)
    if meta.get("teacher_feature_count") != 44:
        raise ValueError("Frozen CatBoost teacher metadata must report 44 teacher features.")

    feature_policy = build_feature_policy(train_columns)
    feature_sets = build_feature_sets(train_columns)
    kd_variants = build_kd_variants()
    matrix = build_experiment_matrix()
    selection = build_selection_criteria()

    feature_policy.to_csv(table_dir / "student_feature_policy_audit_v3c.csv", index=False, encoding="utf-8-sig")
    feature_sets.to_csv(table_dir / "student_feature_sets_v3c_fix.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(table_dir / "student_experiment_matrix_v3c_fix.csv", index=False, encoding="utf-8-sig")
    kd_variants.to_csv(table_dir / "student_kd_variants_v3c_fix.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(table_dir / "student_selection_criteria_v3c_fix.csv", index=False, encoding="utf-8-sig")
    save_plots(feature_sets, matrix, fig_dir)

    counts = matrix["run_priority"].value_counts().to_dict()
    required_count = int(counts.get("required", 0))
    recommended_strict_count = int(counts.get("recommended_strict", 0))
    recommended_tiny_count = int(counts.get("recommended_tiny", 0))
    optional_count = int(counts.get("optional", 0))
    domain_missing = feature_sets[
        (feature_sets["domain_informed"]) & (feature_sets["requires_feature_recompute"])
    ]["feature_name"].drop_duplicates().tolist()
    missing_all = feature_sets[feature_sets["requires_feature_recompute"]]["feature_name"].drop_duplicates().tolist()

    md = f"""# Student / KD Protocol - v3c-fix

## 1. 為什麼修正

- 舊 `PROJECT_RULES.md` 的 student feature list 對目前 KD 階段過度保守。
- Step 9B 原本排除了 alert-line / safe-return-line derived features。
- 本修正版保留防止 data leakage 的核心精神，但允許由 current `water_level` 與固定公開門檻計算的 domain-informed causal features。
- 這不是調 threshold；`ALERT_LINE = {ALERT_LINE}` 與 `SAFE_RETURN_LINE = {SAFE_RETURN_LINE}` 是既有系統門檻。

## 2. Feature policy

- Forbidden features: labels, event id, source/split, synthetic metadata, teacher prediction columns, future columns, trace columns, ground-truth-derived columns。
- Allowed causal water-level features: `water_level`, lag/delta/rolling/slope/range/min/max/std/mean，只能由 current/past `water_level` 計算。
- Conditional allowed domain-informed causal features: `dist_to_alert`, `dist_to_safe_return`, `above_alert_line`, `above_safe_return_line`。

Domain-informed causal features are allowed because they are computed from current water_level and fixed thresholds, not from labels or future data.

## 3. Updated student feature sets

- `student_8f_strict_project_rules`: `{", ".join(STRICT_8F)}`
- `student_12f_domain_edge`: `{", ".join(DOMAIN_12F)}`
- `student_6f_tiny_domain`: `{", ".join(TINY_6F)}`

目前 CatBoost train probability file 中只有 `water_level` 已存在；缺少的 student features 需要在 Step 9C 由 `water_level` causal recompute。需要重算的 feature 包含：`{", ".join(missing_all)}`。

## 4. 為什麼 student_12f_domain_edge 是 primary

- 更符合 deployment：所有 feature 都可在 edge 端用 current/past water-level 與固定門檻計算。
- `dist_to_alert / above_alert_line` 對 S2 danger boundary 有幫助。
- `dist_to_safe_return` 對 S0/S3 recovery boundary 有幫助。
- delta / slope / max / range 保留 S1 rising 與 S3 falling 的方向與局部波動資訊。

## 5. Experiment matrix

- required = `{required_count}`：student_12f_domain_edge × DecisionTree depth 3/4/5/6 × 5 training variants。
- recommended strict baseline = `{recommended_strict_count}`：student_8f_strict_project_rules × DecisionTree depth 4/5 × 3 variants。
- recommended tiny = `{recommended_tiny_count}`：student_6f_tiny_domain × DecisionTree depth 4/5 × 3 variants。
- optional = `{optional_count}`：XGBoost teacher control 1 run + Tiny XGBoost 9 runs。

## 6. Leakage rule

- Event3 evaluation only。
- 不使用 label/event/future/metadata/trace/teacher prediction columns 作 student input。
- domain features 只能由 current `water_level` + fixed thresholds 計算。
- Core/full-cycle Event3 probabilities 不可用於 training、validation、early stopping、hyperparameter tuning 或 threshold tuning。

## 7. Next step

- Step 9C: train required 20 DecisionTree experiments on `student_12f_domain_edge`。
- 若 feature 在 train probability file 缺失，Step 9C 需要 causal recompute，不可修改原始 input CSV。
- recommended strict/tiny baselines 可在 required 結果後執行。
"""
    (table_dir / "student_kd_protocol_summary_v3c_fix.md").write_text(md, encoding="utf-8")
    (table_dir / "student_kd_protocol_summary_v3c_fix.txt").write_text(
        md.replace("# ", "").replace("## ", "").replace("`", ""), encoding="utf-8"
    )

    outline = f"""Step 9C 任務輪廓：

1. 讀取 CatBoost train probabilities：
   outputs/teacher_frozen_catboost_v3c/tables/catboost_teacher_probabilities_train_B2_semdiv_balanced_v3c.csv
2. Primary feature set = student_12f_domain_edge。
3. 若 feature 缺失，從 water_level causal recompute：
   - dist_to_alert = {ALERT_LINE} - water_level
   - dist_to_safe_return = water_level - {SAFE_RETURN_LINE}
   - above_alert_line = int(water_level >= {ALERT_LINE})
   - deltas/slopes/ranges/max only from current/past water_level
4. 訓練 required 20 DecisionTree experiments。
5. Event3 core/full-cycle 只做 evaluation，不可 training / validation / tuning。
6. 輸出 student metrics、complexity、teacher agreement、timelines。
7. 禁止修改 input CSV、生成 synthetic、調 threshold、套用 Event3 tuning。
"""
    (table_dir / "student_kd_next_step_prompt_outline_v3c_fix.txt").write_text(outline, encoding="utf-8")

    print("strict_baseline: student_8f_strict_project_rules")
    print("primary_feature_set: student_12f_domain_edge")
    print("tiny_feature_set: student_6f_tiny_domain")
    print(f"required_experiment_count: {required_count}")
    print(f"recommended_strict_experiment_count: {recommended_strict_count}")
    print(f"recommended_tiny_experiment_count: {recommended_tiny_count}")
    print(f"optional_experiment_count: {optional_count}")
    print(f"domain_informed_conditional_allowed: {sorted(DOMAIN_FEATURES)}")
    print(f"domain_features_require_recompute: {sorted(domain_missing)}")
    print("next_step: Step 9C train required 20 DecisionTree experiments on student_12f_domain_edge")


if __name__ == "__main__":
    main()
