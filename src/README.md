# Source code

The scripts are renamed and ordered to reflect the main research workflow.

1. `01_prepare_water_level_data.py` — clean and validate the minute-level water-level stream.
2. `02_build_semantic_fsm_labels.py` — construct the four working semantic states.
3. `03_build_causal_features.py` — create causal water-level features.
4. `04_select_balanced_synthetic_events.py` — select balanced synthetic event sequences for the training side.
5. `05_freeze_catboost_teacher.py` — freeze the selected CatBoost teacher and export evaluation probabilities.
6. `06_finalize_teacher_selection.py` — summarize teacher selection and leakage evidence.
7. `07_define_student_protocol.py` — define student feature policy and teacher-guided training variants.
8. `08_train_student_candidates.py` — train and compare compact Decision Tree students.
9. `09_audit_student_candidates.py` — audit depth-3 vs depth-4 candidates on control stability and flood/receding trade-offs.
10. `10_freeze_final_student.py` — freeze the selected depth-3 student and export rule representations.

## Important

These files are curated snapshots from the research workspace rather than a single-click reproduction package. Some scripts reference intermediate artifacts generated in earlier experiment stages that are not included in this public-facing archive.
