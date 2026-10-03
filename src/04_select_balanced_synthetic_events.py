import json
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import shorten

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALERT_LINE = 8.0
SAFE_RETURN_LINE = 6.5
STATE_ORDER = ["S0_STABLE", "S1_RISING", "S2_FLOOD", "S3_RECEDING"]
STATE_COLORS = {"S0_STABLE": "gray", "S1_RISING": "orange", "S2_FLOOD": "red", "S3_RECEDING": "blue"}
SYN_SOURCE = "synthetic_B2_semdiv_balanced"
SYN_SPLIT = "train_synthetic_B2_semdiv_balanced"
STUDENT_FEATURES_REQUIRED = ["water_level", "delta_1m", "delta_3m", "delta_5m", "delta_10m", "max_5m", "range_5m", "slope_5m"]
SCENARIO_MIN = {"near_alert": 2, "boundary_crossing": 2, "light_flood": 2, "moderate_flood_stress": 0}
SCENARIO_TARGET = {"near_alert": 2, "boundary_crossing": 3, "light_flood": 2, "moderate_flood_stress": 1}
SCENARIO_MAX = {"near_alert": 3, "boundary_crossing": 4, "light_flood": 3, "moderate_flood_stress": 1}
SCENARIO_RANK = {"boundary_crossing": 0, "light_flood": 1, "near_alert": 2, "moderate_flood_stress": 3}
SYN_META = [
    "synthetic_id",
    "template_event_id",
    "scenario_type",
    "target_peak",
    "actual_peak",
    "rise_time_warp",
    "recession_time_warp",
    "peak_plateau_width",
    "scale_factor",
    "correction_flags",
    "semantic_reject_reasons",
    "quality_flag",
    "diversity_status",
    "max_similarity_to_selected",
    "ranking_score",
    "selection_reason",
]


def state_counts(df):
    counts = df["fsm_state_name"].value_counts().to_dict()
    return {s: int(counts.get(s, 0)) for s in STATE_ORDER}


def consecutive_count(condition):
    out, cur = [], 0
    for val in condition.fillna(False):
        cur = cur + 1 if bool(val) else 0
        out.append(cur)
    return pd.Series(out, index=condition.index, dtype="int64")


def clean_empty(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def flag_count(value):
    text = clean_empty(value)
    if not text:
        return 0
    return len([x for x in text.replace(";", "; ").split("; ") if x.strip()])


def shape_vector(df, length=100):
    y = df.sort_values("timestamp")["water_level"].to_numpy(float)
    spread = float(y.max() - y.min()) if len(y) else 0.0
    norm = np.zeros_like(y) if spread <= 1e-9 else (y - y.min()) / spread
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(norm)), norm)


def shape_corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pair_allowed(candidate, cand_vec, selected_rows, selected_vectors, relaxed_same_scenario=False):
    max_sim = 0.0
    for row, vec in zip(selected_rows, selected_vectors):
        sim = shape_corr(cand_vec, vec)
        max_sim = max(max_sim, sim)
        same_template = int(candidate["template_event_id"]) == int(row["template_event_id"])
        same_scenario = candidate["scenario_type"] == row["scenario_type"]
        target_gap = abs(float(candidate["target_peak"]) - float(row["target_peak"]))
        same_scenario_threshold = 0.985 if relaxed_same_scenario else 0.970
        if sim > 0.995:
            return False, max_sim
        if same_template and same_scenario and sim > same_scenario_threshold:
            return False, max_sim
        if same_template and (not same_scenario) and sim > 0.985 and target_gap < 0.3:
            return False, max_sim
        if (not same_template) and sim > 0.990:
            return False, max_sim
    return True, max_sim


def ranking_score(row):
    score = 0
    score += 3 if row["quality_flag"] == "ok" else 1
    score += {"boundary_crossing": 3, "light_flood": 3, "near_alert": 2, "moderate_flood_stress": 1}.get(row["scenario_type"], 0)
    if 0.05 <= float(row["s1_ratio"]) <= 0.35:
        score += 1
    if float(row["target_peak"]) >= ALERT_LINE and 0.05 <= float(row["s2_ratio"]) <= 0.35:
        score += 1
    if 0.20 <= float(row["s3_ratio"]) <= 0.60:
        score += 1
    if float(row["s3_ratio"]) > 0.65:
        score -= 3
    if float(row["s1_ratio"]) > 0.50:
        score -= 2
    lead = row.get("lead_s1_to_s2_min", np.nan)
    if pd.notna(lead) and str(lead) != "" and float(row["target_peak"]) >= ALERT_LINE and float(lead) < 3:
        score -= 2
    if float(row["target_peak"]) >= ALERT_LINE and int(row["n_s2"]) < 5:
        score -= 2
    score -= flag_count(row.get("warning_reasons", ""))
    return float(score)


def select_balanced(eligible, all_labeled, train_rows):
    seq_map = {int(sid): g.copy() for sid, g in all_labeled.groupby("synthetic_id")}
    vec_map = {sid: shape_vector(seq) for sid, seq in seq_map.items()}
    pool = eligible.copy()
    pool["ranking_score"] = pool.apply(ranking_score, axis=1)
    pool["scenario_rank"] = pool["scenario_type"].map(SCENARIO_RANK)
    pool["quality_rank"] = pool["quality_flag"].map({"ok": 0, "warning": 1}).fillna(2)
    pool = pool.sort_values(
        ["scenario_rank", "quality_rank", "ranking_score", "template_event_id", "synthetic_id"],
        ascending=[True, True, False, True, True],
    )

    selected = []
    selected_vectors = []
    selected_rows = []
    selected_ids = set()
    scenario_counts = Counter()
    template_counts = Counter()
    rows_total = 0
    max_rows = int(1.5 * train_rows)

    def try_add(row, reason, relaxed=False):
        nonlocal rows_total
        sid = int(row["synthetic_id"])
        if sid in selected_ids:
            return False
        if scenario_counts[row["scenario_type"]] >= SCENARIO_MAX[row["scenario_type"]]:
            return False
        if rows_total + int(row["rows"]) > max_rows:
            return False
        ok, max_sim = pair_allowed(row, vec_map[sid], selected_rows, selected_vectors, relaxed_same_scenario=relaxed)
        if not ok:
            return False
        selected_ids.add(sid)
        rec = row.to_dict()
        rec["selection_reason"] = reason
        rec["max_similarity_to_selected"] = max_sim
        selected.append(rec)
        selected_rows.append(rec)
        selected_vectors.append(vec_map[sid])
        scenario_counts[row["scenario_type"]] += 1
        template_counts[int(row["template_event_id"])] += 1
        rows_total += int(row["rows"])
        return True

    # First pass: meet scenario minima using strict diversity.
    for scenario, minimum in SCENARIO_MIN.items():
        for _, row in pool[pool["scenario_type"] == scenario].sort_values(
            ["quality_rank", "ranking_score", "template_event_id", "synthetic_id"],
            ascending=[True, False, True, True],
        ).iterrows():
            if scenario_counts[scenario] >= minimum:
                break
            try_add(row, f"scenario_min_{scenario}", relaxed=False)

    # Second pass: fill target counts; allow relaxed same-scenario threshold only if needed.
    for relaxed in [False, True]:
        for scenario, target in SCENARIO_TARGET.items():
            for _, row in pool[pool["scenario_type"] == scenario].sort_values(
                ["quality_rank", "ranking_score", "template_event_id", "synthetic_id"],
                ascending=[True, False, True, True],
            ).iterrows():
                if scenario_counts[scenario] >= target:
                    break
                try_add(row, f"scenario_target_{scenario}{'_relaxed_corr' if relaxed else ''}", relaxed=relaxed)

    # Third pass: reach at least 8 if possible while respecting max scenario and ratio.
    for relaxed in [False, True]:
        for _, row in pool.iterrows():
            if len(selected) >= 12:
                break
            if len(selected) >= 8 and template_counts[1] >= 3 and template_counts[2] >= 3:
                break
            try_add(row, f"fill_balance{'_relaxed_corr' if relaxed else ''}", relaxed=relaxed)

    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        raise ValueError("No B2_semdiv_balanced candidates selected.")
    return selected_df, selected_ids


def semantic_safety_check(df, selected_summary):
    for sid, seq in df.groupby("synthetic_id"):
        seq = seq.sort_values("timestamp").reset_index(drop=True)
        row = selected_summary[selected_summary["synthetic_id"] == sid].iloc[0]
        counts = state_counts(seq)
        rows = len(seq)
        peak_idx = int(seq["water_level"].idxmax())
        peak_water = float(seq["water_level"].max())
        if ((seq["fsm_state_name"] == "S2_FLOOD") & (seq["water_level"] < ALERT_LINE)).any():
            raise ValueError(f"S2 below alert in final accepted synthetic_id={sid}")
        if float(row["target_peak"]) < ALERT_LINE and counts["S2_FLOOD"] > 0:
            raise ValueError(f"Below-alert target has S2 in synthetic_id={sid}")
        if float(row["target_peak"]) >= ALERT_LINE and counts["S2_FLOOD"] == 0:
            raise ValueError(f"Above-alert target has no S2 in synthetic_id={sid}")
        post = seq.iloc[peak_idx + 1 :].copy()
        if len(post):
            long_s1 = ((peak_water - post["water_level"] >= 0.25) & (post["fsm_state_name"] == "S1_RISING")).sum()
            if long_s1 / len(post) > 0.40:
                raise ValueError(f"Post-peak long S1 in synthetic_id={sid}")
        pre = seq.iloc[:peak_idx].copy()
        if len(pre) and (pre["fsm_state_name"] == "S3_RECEDING").sum() / len(pre) > 0.10:
            raise ValueError(f"S3 before peak in synthetic_id={sid}")
        if counts["S1_RISING"] / rows > 0.55:
            raise ValueError(f"S1 ratio too high in synthetic_id={sid}")
        if counts["S3_RECEDING"] / rows > 0.75:
            raise ValueError(f"S3 ratio too high in synthetic_id={sid}")
        if counts["S2_FLOOD"] / rows > 0.45:
            raise ValueError(f"S2 ratio too high in synthetic_id={sid}")


def build_features_by_sequence(labeled, teacher_features, student_features):
    parts = []
    for _, group in labeled.groupby("synthetic_id", sort=False):
        df = group.copy().reset_index(drop=True)
        for w in [1, 3, 5, 10, 30, 60]:
            df[f"lag_{w}m"] = df["water_level"].shift(w).fillna(0.0)
            df[f"delta_{w}m"] = (df["water_level"] - df["water_level"].shift(w)).fillna(0.0)
        for w in [5, 10, 30, 60]:
            roll = df["water_level"].rolling(window=w, min_periods=1)
            df[f"mean_{w}m"] = roll.mean()
            df[f"std_{w}m"] = roll.std().fillna(0.0)
            df[f"min_{w}m"] = roll.min()
            df[f"max_{w}m"] = roll.max()
            df[f"range_{w}m"] = df[f"max_{w}m"] - df[f"min_{w}m"]
        df["slope_5m"] = df["delta_5m"] / 5
        df["slope_10m"] = df["delta_10m"] / 10
        df["slope_30m"] = df["delta_30m"] / 30
        df["dist_to_alert"] = ALERT_LINE - df["water_level"]
        df["dist_to_safe_return"] = df["water_level"] - SAFE_RETURN_LINE
        df["above_alert_line"] = (df["water_level"] >= ALERT_LINE).astype(int)
        df["above_safe_return_line"] = (df["water_level"] >= SAFE_RETURN_LINE).astype(int)
        df["consecutive_above_safe_return"] = consecutive_count(df["water_level"] >= SAFE_RETURN_LINE)
        df["consecutive_above_alert"] = consecutive_count(df["water_level"] >= ALERT_LINE)
        df["consecutive_rising_5m"] = consecutive_count(df["delta_5m"] > 0)
        df["consecutive_falling_5m"] = consecutive_count(df["delta_5m"] < 0)
        parts.append(df)
    allf = pd.concat(parts, ignore_index=True)
    meta = ["timestamp", "fsm_event_id", "source", "split", "fsm_state_name", "fsm_state_id"] + SYN_META
    return allf.loc[:, meta + teacher_features].copy(), allf.loc[:, meta + student_features].copy()


def pairwise_similarity(selected_labeled):
    ids = sorted(selected_labeled["synthetic_id"].unique())
    vectors = [shape_vector(selected_labeled[selected_labeled["synthetic_id"] == sid]) for sid in ids]
    mat = np.eye(len(ids))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            mat[i, j] = mat[j, i] = shape_corr(vectors[i], vectors[j])
    vals = mat[np.triu_indices_from(mat, k=1)] if len(ids) > 1 else np.array([0.0])
    return ids, mat, float(vals.max()) if len(vals) else 0.0, float(vals.mean()) if len(vals) else 0.0


def parse_counts(value):
    return json.loads(value) if isinstance(value, str) and value else {}


def add_state_plot(ax, seq, title):
    local = seq.sort_values("timestamp").reset_index(drop=True)
    ax.plot(local.index, local["water_level"], color="tab:blue", linewidth=1.35)
    for state, color in STATE_COLORS.items():
        part = local[local["fsm_state_name"] == state]
        ax.scatter(part.index, part["water_level"], s=14, color=color, alpha=0.85, label=state)
    if len(local):
        peak_idx = int(local["water_level"].idxmax())
        ax.scatter([peak_idx], [local.loc[peak_idx, "water_level"]], marker="x", color="black", s=45, zorder=4)
    ax.axhline(ALERT_LINE, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(SAFE_RETURN_LINE, color="green", linestyle=":", linewidth=1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title(shorten(str(title), width=150, placeholder="..."), fontsize=9)


def write_plots(fig_dir, selected_labeled, selected_summary, sim_ids, sim_mat, comparison_counts, candidate_summary):
    fig_dir.mkdir(parents=True, exist_ok=True)
    groups = list(selected_labeled.groupby("synthetic_id"))
    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 2.9 * len(groups)))
    axes = np.atleast_1d(axes)
    for ax, (sid, seq) in zip(axes, groups):
        row = selected_summary[selected_summary["synthetic_id"] == sid].iloc[0]
        add_state_plot(
            ax,
            seq,
            f"id={sid} template={row.template_event_id} {row.scenario_type} target={row.target_peak} actual={row.actual_peak:.2f} quality={row.quality_flag} score={row.ranking_score}",
        )
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(fig_dir / "B2_semdiv_balanced_accepted_synthetic_panels.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, scenario in zip(axes, ["near_alert", "boundary_crossing", "light_flood", "moderate_flood_stress"]):
        for _, seq in selected_labeled[selected_labeled["scenario_type"] == scenario].groupby("synthetic_id"):
            ax.plot(np.arange(len(seq)), seq["water_level"], alpha=0.8)
        ax.axhline(ALERT_LINE, color="black", linestyle="--")
        ax.axhline(SAFE_RETURN_LINE, color="green", linestyle=":")
        ax.set_title(scenario)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "B2_semdiv_balanced_overlay_by_scenario.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_mat, vmin=0.9, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_xticks(range(len(sim_ids)))
    ax.set_yticks(range(len(sim_ids)))
    ax.set_xticklabels(sim_ids, rotation=45)
    ax.set_yticklabels(sim_ids)
    ax.set_title("B2_semdiv_balanced Shape Similarity")
    fig.tight_layout()
    fig.savefig(fig_dir / "B2_semdiv_balanced_shape_similarity_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(candidate_summary["target_peak"], candidate_summary["actual_peak"], color="lightgray", label="eligible")
    ax.scatter(selected_summary["target_peak"], selected_summary["actual_peak"], color="tab:green", label="selected")
    lo, hi = candidate_summary["target_peak"].min() - 0.05, candidate_summary["target_peak"].max() + 0.05
    ax.plot([lo, hi], [lo, hi], "k--")
    ax.set_xlabel("target_peak")
    ax.set_ylabel("actual_peak")
    ax.legend()
    ax.set_title("B2_semdiv_balanced Target vs Actual Peak")
    fig.tight_layout()
    fig.savefig(fig_dir / "B2_semdiv_balanced_target_vs_actual_peak.png", dpi=160)
    plt.close(fig)

    versions = list(comparison_counts.keys())
    x = np.arange(len(versions))
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, state in enumerate(STATE_ORDER):
        ax.bar(x + (i - 1.5) * width, [comparison_counts[v].get(state, 0) for v in versions], width, label=state)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, rotation=22, ha="right")
    ax.legend()
    ax.set_title("B2_semdiv_balanced vs Previous State Distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "B2_semdiv_balanced_vs_previous_state_distribution.png", dpi=160)
    plt.close(fig)

    for sid, seq in groups:
        row = selected_summary[selected_summary["synthetic_id"] == sid].iloc[0]
        fig, ax = plt.subplots(figsize=(13, 5))
        add_state_plot(
            ax,
            seq,
            f"id={sid} template={row.template_event_id} scenario={row.scenario_type} target={row.target_peak} actual={row.actual_peak:.2f} quality={row.quality_flag} score={row.ranking_score}",
        )
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(fig_dir / f"sequence_{sid}_balanced_accepted.png", dpi=170)
        plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    all_candidates_path = root / "data/synthetic/B2_semdiv_event_shape/_B2_semdiv_all_candidates_labeled_v3c.csv"
    candidate_summary_path = root / "outputs/tables/B2_semdiv_event_shape/B2_semdiv_candidate_sequence_summary_v3c.csv"
    train_teacher_path = root / "data/splits/experiment_B_train_real_teacher_v3c.csv"
    train_student_path = root / "data/splits/experiment_B_train_real_student_v3c.csv"
    test_teacher_path = root / "data/splits/experiment_B_test_real_teacher_v3c.csv"
    feature_columns_path = root / "outputs/tables/feature_columns_v3c.json"
    b1_lite_summary_path = root / "outputs/tables/B1_lite_rule_prior/B1_lite_augmented_train_summary_v3c.csv"
    b1_full_summary_path = root / "outputs/tables/B1_rule_prior/B1_augmented_train_summary_v3c.csv"
    b2_lite_summary_path = root / "outputs/tables/B2_lite_event_shape/B2_lite_augmented_train_summary_v3c.csv"
    b2_semdiv_summary_path = root / "outputs/tables/B2_semdiv_event_shape/B2_semdiv_augmented_train_summary_v3c.csv"
    out_dir = root / "data/synthetic/B2_semdiv_balanced_event_shape"
    tab_dir = root / "outputs/tables/B2_semdiv_balanced_event_shape"
    fig_dir = root / "outputs/figures/B2_semdiv_balanced_event_shape"
    for path in [out_dir, tab_dir, fig_dir]:
        path.mkdir(parents=True, exist_ok=True)

    all_labeled = pd.read_csv(all_candidates_path)
    candidate_summary = pd.read_csv(candidate_summary_path)
    train_teacher = pd.read_csv(train_teacher_path)
    train_student = pd.read_csv(train_student_path)
    test_teacher = pd.read_csv(test_teacher_path)
    for df in [all_labeled, train_teacher, train_student, test_teacher]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    with feature_columns_path.open("r", encoding="utf-8") as f:
        fc = json.load(f)
    teacher_features = fc["teacher_feature_columns"]
    student_features = fc["student_feature_columns"]
    forbidden = set(fc["forbidden_columns"])
    if set(teacher_features) & forbidden or set(student_features) & forbidden:
        raise ValueError("Forbidden columns found in feature columns.")
    if student_features != STUDENT_FEATURES_REQUIRED:
        raise ValueError("Student features do not match required 8-feature list.")
    if set(SYN_META) & set(teacher_features + student_features):
        raise ValueError("Synthetic metadata found in model features.")

    required_label_cols = {
        "timestamp",
        "water_level",
        "fsm_state_name",
        "fsm_state_id",
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
    }
    required_summary_cols = {
        "synthetic_id",
        "template_event_id",
        "scenario_type",
        "target_peak",
        "actual_peak",
        "quality_flag",
        "semantic_reject_reasons",
        "correction_flags",
        "diversity_status",
        "accepted_for_train",
        "s0_ratio",
        "s1_ratio",
        "s2_ratio",
        "s3_ratio",
        "lead_s1_to_s2_min",
        "n_s0",
        "n_s1",
        "n_s2",
        "n_s3",
        "rows",
    }
    if required_label_cols - set(all_labeled.columns):
        raise ValueError(f"All-candidate labels missing columns: {sorted(required_label_cols - set(all_labeled.columns))}")
    if required_summary_cols - set(candidate_summary.columns):
        raise ValueError(f"Candidate summary missing columns: {sorted(required_summary_cols - set(candidate_summary.columns))}")

    if set(all_labeled["timestamp"]) & set(test_teacher["timestamp"]):
        raise ValueError("All-candidate synthetic timestamps overlap test_real timestamps.")
    if (all_labeled["template_event_id"].astype(int) == 3).any():
        raise ValueError("Event 3 appears in synthetic template_event_id.")
    if (all_labeled["source"] != "synthetic_B2_semdiv").any():
        raise ValueError("Input all candidates must have source synthetic_B2_semdiv.")
    if not train_teacher[["timestamp", "fsm_event_id", "fsm_state_id"]].equals(train_student[["timestamp", "fsm_event_id", "fsm_state_id"]]):
        raise ValueError("Train teacher/student alignment mismatch.")

    source_by_id = all_labeled.groupby("synthetic_id")["source"].first().to_dict()
    eligible = candidate_summary.copy()
    eligible["semantic_reject_reasons"] = eligible["semantic_reject_reasons"].apply(clean_empty)
    eligible["source"] = eligible["synthetic_id"].map(source_by_id)
    eligible = eligible[
        eligible["quality_flag"].isin(["ok", "warning"])
        & (eligible["semantic_reject_reasons"] == "")
        & (eligible["source"] == "synthetic_B2_semdiv")
    ].copy()
    eligible_count = len(eligible)

    selected_summary, selected_ids = select_balanced(eligible, all_labeled, len(train_teacher))
    selected_labeled = all_labeled[all_labeled["synthetic_id"].isin(selected_ids)].copy()
    selected_labeled["source"] = SYN_SOURCE
    selected_labeled["split"] = SYN_SPLIT
    selected_labeled = selected_labeled.merge(
        selected_summary[["synthetic_id", "ranking_score", "selection_reason", "max_similarity_to_selected"]],
        on="synthetic_id",
        how="left",
        suffixes=("", "_selected"),
    )
    if "ranking_score_selected" in selected_labeled.columns:
        selected_labeled["ranking_score"] = selected_labeled["ranking_score_selected"]
        selected_labeled = selected_labeled.drop(columns=["ranking_score_selected"])
    if "max_similarity_to_selected_selected" in selected_labeled.columns:
        selected_labeled["max_similarity_to_selected"] = selected_labeled["max_similarity_to_selected_selected"]
        selected_labeled = selected_labeled.drop(columns=["max_similarity_to_selected_selected"])
    selected_labeled["diversity_status"] = "accepted"

    semantic_safety_check(selected_labeled, selected_summary)
    real_ts = set(train_teacher["timestamp"]) | set(test_teacher["timestamp"])
    if set(selected_labeled["timestamp"]) & real_ts:
        raise ValueError("Selected synthetic timestamps overlap real timestamps.")
    if set(selected_labeled["source"].unique()) != {SYN_SOURCE}:
        raise ValueError("Final selected source check failed.")
    if set(selected_labeled["split"].unique()) != {SYN_SPLIT}:
        raise ValueError("Final selected split check failed.")

    synth_teacher, synth_student = build_features_by_sequence(selected_labeled, teacher_features, student_features)
    if synth_teacher[teacher_features].isna().any().any() or synth_student[student_features].isna().any().any():
        raise ValueError("Synthetic feature tables contain NaN.")
    real_teacher_aug = train_teacher.copy()
    real_student_aug = train_student.copy()
    for col in SYN_META:
        real_teacher_aug[col] = pd.NA
        real_student_aug[col] = pd.NA
    augmented_teacher = pd.concat([real_teacher_aug, synth_teacher.loc[:, real_teacher_aug.columns]], ignore_index=True)
    augmented_student = pd.concat([real_student_aug, synth_student.loc[:, real_student_aug.columns]], ignore_index=True)
    if not set(augmented_teacher["source"].unique()).issubset({"real", SYN_SOURCE}):
        raise ValueError("Augmented teacher has invalid source.")
    if not set(augmented_student["source"].unique()).issubset({"real", SYN_SOURCE}):
        raise ValueError("Augmented student has invalid source.")

    sim_ids, sim_mat, max_corr, mean_corr = pairwise_similarity(selected_labeled)
    final_counts = state_counts(selected_labeled)
    aug_counts = state_counts(augmented_teacher)
    real_counts = state_counts(train_teacher)
    scenario_dist = selected_summary["scenario_type"].value_counts().to_dict()
    template_dist = selected_summary["template_event_id"].astype(int).value_counts().to_dict()
    synthetic_ratio = len(selected_labeled) / len(train_teacher)
    scenario_balance_pass = (
        2 <= scenario_dist.get("near_alert", 0) <= 3
        and 2 <= scenario_dist.get("boundary_crossing", 0) <= 4
        and 2 <= scenario_dist.get("light_flood", 0) <= 3
        and scenario_dist.get("moderate_flood_stress", 0) <= 1
    )
    final_count = len(selected_summary)
    template_balance_pass = (
        template_dist.get(1, 0) >= 3
        and template_dist.get(2, 0) >= 3
        and max(template_dist.values()) <= final_count * 0.65
    )
    ratio_pass = synthetic_ratio <= 1.5
    semantic_safety_pass = True

    selected_ids_out = selected_summary.copy()
    selected_ids_out["accepted_for_train"] = True
    selected_ids_out = selected_ids_out[
        [
            "synthetic_id",
            "template_event_id",
            "scenario_type",
            "target_peak",
            "actual_peak",
            "quality_flag",
            "ranking_score",
            "selection_reason",
            "rows",
            "n_s0",
            "n_s1",
            "n_s2",
            "n_s3",
            "s1_ratio",
            "s2_ratio",
            "s3_ratio",
            "max_similarity_to_selected",
            "accepted_for_train",
        ]
    ].sort_values("synthetic_id")

    selected_labeled.to_csv(out_dir / "_B2_semdiv_balanced_synthetic_labeled_v3c.csv", index=False, encoding="utf-8-sig")
    synth_teacher.to_csv(out_dir / "_B2_semdiv_balanced_synthetic_teacher_features_v3c.csv", index=False, encoding="utf-8-sig")
    synth_student.to_csv(out_dir / "_B2_semdiv_balanced_synthetic_student_features_v3c.csv", index=False, encoding="utf-8-sig")
    augmented_teacher.to_csv(out_dir / "experiment_B_train_B2_semdiv_balanced_teacher_v3c.csv", index=False, encoding="utf-8-sig")
    augmented_student.to_csv(out_dir / "experiment_B_train_B2_semdiv_balanced_student_v3c.csv", index=False, encoding="utf-8-sig")
    selected_ids_out.to_csv(out_dir / "B2_semdiv_balanced_selected_ids_v3c.csv", index=False, encoding="utf-8-sig")

    summary_rows = [
        ("eligible_count", eligible_count, "quality ok/warning and no semantic reject"),
        ("final_accepted_count", final_count, ""),
        ("near_alert_count", scenario_dist.get("near_alert", 0), ""),
        ("boundary_crossing_count", scenario_dist.get("boundary_crossing", 0), ""),
        ("light_flood_count", scenario_dist.get("light_flood", 0), ""),
        ("moderate_flood_stress_count", scenario_dist.get("moderate_flood_stress", 0), ""),
        ("template_1_count", template_dist.get(1, 0), ""),
        ("template_2_count", template_dist.get(2, 0), ""),
        ("synthetic_rows", len(selected_labeled), ""),
        ("synthetic_ratio", synthetic_ratio, ""),
        ("n_s0", final_counts["S0_STABLE"], ""),
        ("n_s1", final_counts["S1_RISING"], ""),
        ("n_s2", final_counts["S2_FLOOD"], ""),
        ("n_s3", final_counts["S3_RECEDING"], ""),
        ("augmented_n_s0", aug_counts["S0_STABLE"], ""),
        ("augmented_n_s1", aug_counts["S1_RISING"], ""),
        ("augmented_n_s2", aug_counts["S2_FLOOD"], ""),
        ("augmented_n_s3", aug_counts["S3_RECEDING"], ""),
        ("s2_before", real_counts["S2_FLOOD"], ""),
        ("s2_after", aug_counts["S2_FLOOD"], ""),
        ("max_pairwise_corr", max_corr, ""),
        ("mean_pairwise_corr", mean_corr, ""),
        ("scenario_balance_pass", scenario_balance_pass, ""),
        ("template_balance_pass", template_balance_pass, ""),
        ("ratio_pass", ratio_pass, ""),
        ("semantic_safety_pass", semantic_safety_pass, ""),
    ]
    selection_summary = pd.DataFrame(summary_rows, columns=["metric", "value", "notes"])
    selection_summary.to_csv(tab_dir / "B2_semdiv_balanced_selection_summary_v3c.csv", index=False, encoding="utf-8-sig")

    def version_row(version, seq_count, syn_rows, ratio, syn_counts, aug_c, s2_before, s2_after, notes):
        return {
            "version": version,
            "sequence_count": seq_count,
            "synthetic_rows": syn_rows,
            "synthetic_ratio": ratio,
            "synthetic_n_s0": syn_counts.get("S0_STABLE", 0),
            "synthetic_n_s1": syn_counts.get("S1_RISING", 0),
            "synthetic_n_s2": syn_counts.get("S2_FLOOD", 0),
            "synthetic_n_s3": syn_counts.get("S3_RECEDING", 0),
            "augmented_n_s0": aug_c.get("S0_STABLE", ""),
            "augmented_n_s1": aug_c.get("S1_RISING", ""),
            "augmented_n_s2": aug_c.get("S2_FLOOD", ""),
            "augmented_n_s3": aug_c.get("S3_RECEDING", ""),
            "s2_count_before": s2_before,
            "s2_count_after": s2_after,
            "notes": notes,
        }

    b1_lite = pd.read_csv(b1_lite_summary_path).iloc[0]
    b1_full = pd.read_csv(b1_full_summary_path).iloc[0]
    b2_lite = pd.read_csv(b2_lite_summary_path).iloc[0]
    b2_semdiv = pd.read_csv(b2_semdiv_summary_path).iloc[0]
    b1_lite_syn, b1_lite_aug = parse_counts(b1_lite["synthetic_B1_state_counts"]), parse_counts(b1_lite["augmented_train_state_counts"])
    b1_full_syn, b1_full_aug = parse_counts(b1_full["synthetic_B1_state_counts"]), parse_counts(b1_full["augmented_train_state_counts"])
    b2_lite_syn, b2_lite_aug = parse_counts(b2_lite["B2_lite_accepted_synthetic_state_counts"]), parse_counts(b2_lite["augmented_train_state_counts"])
    b2_semdiv_syn, b2_semdiv_aug = parse_counts(b2_semdiv["final_accepted_synthetic_state_counts"]), parse_counts(b2_semdiv["augmented_train_state_counts"])
    comparison = pd.DataFrame(
        [
            version_row("B1_lite", 6, int(b1_lite["synthetic_teacher_rows"]), float(b1_lite["synthetic_ratio"]), b1_lite_syn, b1_lite_aug, int(b1_lite["s2_count_before"]), int(b1_lite["s2_count_after"]), "B1-lite anchored scaling"),
            version_row("B1_full", 12, int(b1_full["synthetic_teacher_rows"]), float(b1_full["synthetic_ratio"]), b1_full_syn, b1_full_aug, int(b1_full["s2_count_before"]), int(b1_full["s2_count_after"]), "B1-full anchored scaling"),
            version_row("B2_lite", int(b2_lite["accepted_sequence_count"]), int(b2_lite["accepted_synthetic_rows"]), float(b2_lite["synthetic_ratio_after_ratio_control"]), b2_lite_syn, b2_lite_aug, int(b2_lite["s2_count_before"]), int(b2_lite["s2_count_after"]), "B2-lite quality-controlled event-shape"),
            version_row("B2_semdiv", int(b2_semdiv["accepted_count"]), int(b2_semdiv["final_accepted_rows"]), float(b2_semdiv["synthetic_ratio_after_filtering"]), b2_semdiv_syn, b2_semdiv_aug, int(b2_semdiv["s2_count_before"]), int(b2_semdiv["s2_count_after"]), "B2_semdiv strict diversity selection"),
            version_row("B2_semdiv_balanced", final_count, len(selected_labeled), synthetic_ratio, final_counts, aug_counts, real_counts["S2_FLOOD"], aug_counts["S2_FLOOD"], "B2_semdiv reselected balanced subset"),
        ]
    )
    comparison.to_csv(tab_dir / "B2_semdiv_balanced_vs_previous_comparison_v3c.csv", index=False, encoding="utf-8-sig")

    limitations = []
    if not scenario_balance_pass:
        limitations.append("scenario balance target not fully met")
    if not template_balance_pass:
        limitations.append("template balance target not fully met")
    if final_count < 8 or final_count > 12:
        limitations.append("final accepted count outside 8-12 target")
    if synthetic_ratio > 1.5:
        limitations.append("synthetic ratio exceeds 1.5")

    summary_text = [
        "Task name: Step 6A-4 - Build B2_semdiv_balanced final synthetic candidate",
        "No Event 3 rows are used.",
        "Selection pool: B2_semdiv all candidates with quality_flag ok/warning and no semantic_reject_reasons.",
        "Scenario balance target: near_alert 2-3, boundary_crossing 2-4, light_flood 2-3, moderate_flood_stress <=1.",
        "Template balance target: template 1 and 2 at least 3 each, no template above 65%.",
        "Diversity thresholds: same template/scenario 0.970, relaxed to 0.985 if needed; same template different scenario 0.985 unless target gap >=0.3; global 0.995.",
        f"eligible count: {eligible_count}",
        f"final accepted count: {final_count}",
        f"selected synthetic ids: {sorted(selected_ids)}",
        f"scenario distribution: {scenario_dist}",
        f"template distribution: {template_dist}",
        f"state distribution: {final_counts}",
        f"synthetic ratio: {synthetic_ratio}",
        f"pairwise similarity summary: max={max_corr}, mean={mean_corr}",
        f"semantic safety check result: {semantic_safety_pass}",
        f"limitations: {limitations if limitations else 'none'}",
        "No model training performed.",
    ]
    (tab_dir / "B2_semdiv_balanced_quality_summary_v3c.txt").write_text("\n".join(summary_text) + "\n", encoding="utf-8")

    comparison_counts = {
        "B1_lite": b1_lite_syn,
        "B1_full": b1_full_syn,
        "B2_lite": b2_lite_syn,
        "B2_semdiv": b2_semdiv_syn,
        "B2_semdiv_balanced": final_counts,
    }
    write_plots(fig_dir, selected_labeled, selected_summary, sim_ids, sim_mat, comparison_counts, eligible)

    print(f"eligible_count: {eligible_count}")
    print(f"final_accepted_count: {final_count}")
    print(f"selected_synthetic_ids: {sorted(selected_ids)}")
    print(f"scenario_distribution: {scenario_dist}")
    print(f"template_distribution: {template_dist}")
    print(f"final_synthetic_state_counts: {final_counts}")
    print(f"augmented_train_state_counts: {aug_counts}")
    print(f"final_synthetic_ratio: {synthetic_ratio:.4f}")
    print(f"max_pairwise_corr: {max_corr:.6f}")
    print(f"mean_pairwise_corr: {mean_corr:.6f}")
    print(f"scenario_balance_pass: {scenario_balance_pass}")
    print(f"template_balance_pass: {template_balance_pass}")
    print(f"ratio_pass: {ratio_pass}")
    print(f"semantic_safety_pass: {semantic_safety_pass}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
