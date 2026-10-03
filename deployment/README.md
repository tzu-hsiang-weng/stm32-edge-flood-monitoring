# STM32 deployment

The IEEE GCCE 2026 study exports the compact depth-3 student as C99 and deploys it on an STM32L476RG.

Reported deployment results in the conference manuscript:

- average inference latency: approximately **79 microseconds**
- PC–MCU implementation agreement: **100%** over the replayed cleaned stream

The 100% figure refers to implementation equivalence between the PC reference and MCU implementation; it is **not** classification accuracy.

The uploaded project bundle used to build this repository did not contain the complete STM32 firmware project or final C99 board source. The frozen tree rules and feature specification are therefore provided under `artifacts/student/` as the deployment reference.
