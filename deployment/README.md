# STM32 Deployment

The IEEE GCCE 2026 study exports the selected depth-3 Decision Tree as C99 and deploys it on an **STM32L476RG**.

## Reported deployment results

- Average inference latency: approximately **79 microseconds**
- PC–MCU implementation agreement: **100%** over the replayed cleaned stream

The 100% figure refers to implementation equivalence between the PC reference and MCU implementation; it is **not classification accuracy**.

## C99 reference implementation

This directory includes:

- `model_inference.h` — compact feature/state definitions and inference interface
- `model_inference.c` — C99 implementation reconstructed directly from the frozen depth-3 tree rules stored in `artifacts/student/tree_rules.txt`

The reference implementation exposes the four semantic states:

- `S0` — Stable
- `S1` — Rising
- `S2` — Flood
- `S3` — Receding

The uploaded research bundle used to curate this repository did not contain the complete STM32Cube firmware project or the exact final board-side source tree used during the conference experiment. Therefore, the C files here are provided as a transparent reference implementation of the frozen decision tree rather than being presented as the original full firmware project.
