# Experiment Protocol

## Data setting

- Minute-level water-level series: 2,878 cleaned rows.
- Four working semantic states: `S0_STABLE`, `S1_RISING`, `S2_FLOOD`, `S3_RECEDING`.
- The three detected events are kept as event units rather than randomly splitting adjacent time windows.
- Event 3 is the complete flood event and is kept as unseen real-only evaluation data.
- Synthetic/event-level augmentation is restricted to the training side.

## Leakage controls

The model input policy excludes labels, event IDs, split/source metadata, teacher prediction columns, future information, and debug/trace fields. Student features are computed only from the current/past water-level stream and fixed thresholds.

## Teacher / student setup

- Teacher: CatBoost using causal water-level features.
- Student: depth-3 Decision Tree.
- Teacher guidance: teacher hard predictions are used as training targets and teacher confidence is used as a sample weight.
- This is teacher-guided confidence-weighted supervision, not standard KL-divergence soft-label knowledge distillation.

## Evaluation note

The repository preserves a single held-out complete flood event. Results should not be interpreted as cross-event or cross-domain generalization.
