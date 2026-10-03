# Lightweight Semantic Water-Level Inference on STM32 for Edge Flood Monitoring

Research repository for the paper **“Lightweight Semantic Water-Level Inference on STM32 for Edge Flood Monitoring”**, accepted for **Oral Presentation at the 2026 IEEE 15th Global Conference on Consumer Electronics (IEEE GCCE 2026)**.

This work studies how a compact, interpretable water-level semantic model can be trained under limited flood-event data and deployed on a resource-constrained MCU.

> **Status:** First-author paper · Accepted for Oral Presentation · IEEE GCCE 2026

## Overview

The system converts a minute-level water-level stream into four operational semantic states:

- **S0 — Stable**
- **S1 — Rising**
- **S2 — Flood**
- **S3 — Receding**

The research workflow emphasizes three issues: event-aware evaluation, compact teacher-guided learning, and actual MCU deployment.

![Four-state semantic labeling](artifacts/figures/four_state_semantic_labels.svg)

## Research pipeline

```text
Water-level stream
        ↓
Cleaning and quality checks
        ↓
Four-state causal semantic labeling
        ↓
Event-aware split
        ↓
Train-side event augmentation
        ↓
CatBoost teacher
        ↓
Teacher-guided compact student
        ↓
Depth-3 Decision Tree
        ↓
C99 export → STM32L476RG
```

### 1. Event-aware evaluation

The cleaned series contains **2,878 minute-level samples**. Instead of randomly splitting adjacent time windows, the experiment keeps flood events as units. The complete Event 3 flood cycle is reserved as **unseen real-only evaluation data** and is not used for training, augmentation, or parameter tuning.

### 2. Train-side augmentation

Because the training side contains only smaller real events and very limited flood-state samples, event-level synthetic sequences are used only on the training side. The final evaluation remains entirely real.

### 3. Teacher-guided compact student

A CatBoost model is used as the high-capacity teacher. The selected student is a **depth-3 Decision Tree** with **15 nodes / 8 leaves**.

The student is trained using teacher hard predictions with teacher confidence as sample weights. In this repository this is described as **teacher-guided confidence-weighted supervision**; it is not standard KL-divergence soft-label knowledge distillation.

## Key results

| Model / deployment item | Result |
|---|---:|
| CatBoost teacher — full-cycle Event 3 Macro-F1 | **0.9001** |
| Depth-3 student — full-cycle Event 3 Macro-F1 | **0.8588** |
| Student S1 recall | **0.9364** |
| Student S2 recall | **0.5441** |
| Student S3 recall | **0.8930** |
| Student depth / nodes / leaves | **3 / 15 / 8** |
| Frozen student model size | **3.149 KB** |
| STM32L476RG average inference latency | **≈ 79 μs** |
| PC–MCU implementation agreement | **100%** |

The **100% PC–MCU agreement** measures implementation equivalence during stream replay. It should not be interpreted as 100% classification accuracy.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/                     # curated experiment-stage Python scripts
├── data/
│   ├── processed/           # representative processed excerpt
│   └── summaries/           # event / state / training summaries
├── artifacts/
│   ├── figures/             # water-level and semantic-state figures
│   ├── teacher/             # frozen teacher metrics and selection evidence
│   └── student/             # final student metrics, rules, features, model
├── docs/
│   └── experiment_protocol.md
└── deployment/
    └── README.md            # STM32 deployment notes
```

## Representative artifacts

- [`artifacts/teacher/metrics.csv`](artifacts/teacher/metrics.csv) — teacher evaluation metrics.
- [`artifacts/teacher/leakage_audit.csv`](artifacts/teacher/leakage_audit.csv) — leakage-control evidence.
- [`artifacts/student/final_metrics.csv`](artifacts/student/final_metrics.csv) — frozen depth-3 student metrics.
- [`artifacts/student/feature_spec.csv`](artifacts/student/feature_spec.csv) — edge-computable feature specification.
- [`artifacts/student/tree_rules.txt`](artifacts/student/tree_rules.txt) — interpretable decision rules.
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md) — split, leakage, and teacher/student protocol.

## Limitations

This study has an important limitation: evaluation currently relies on **one held-out complete flood event**. The depth-3 student also trades some flood-state sensitivity for better receding-state stability; in particular, S2 recall remains limited. The results therefore support the feasibility of compact deployment in this case study, but do not establish generalization across all flood events or sensing domains.

## Relationship to the undergraduate capstone

This research extends the AI water-level semantic inference component of the broader undergraduate capstone on VoI-driven adaptive LoRa/Wi-Fi flood monitoring. The conference work narrows the scope to model evaluation, lightweight student selection, and MCU deployment.

## Conference

**2026 IEEE 15th Global Conference on Consumer Electronics (IEEE GCCE 2026)**  
Kobe, Japan · October 26–29, 2026  
**First author · Accepted for Oral Presentation**

---

This repository is a curated research archive for documentation and portfolio review. The original experiment workspace contained additional intermediate files and board-integration assets that are not all included here.
