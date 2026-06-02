# Phase 7B v2: Corrected Section 4.2.2-Style Effectiveness Figures

This folder contains corrected Section 4.2.2-style effectiveness figures for the ArduPilot SITL software-sensor recovery project.

## Generated status

- Figure A prediction accuracy: generated
- Figure B clean residual threshold: generated
- Figure C W10_N3 detection examples: generated
- Figure D W10_N3 recovery examples: generated
- Figure E recovery improvement summary: generated

## Corrections in v2

- Detection examples are forced to use the final detector: W10_N3_mean1x.
- Clean residual threshold figure is generated directly from residual_results_v9 files.
- Recovery examples are generated separately for Roll, Pitch, and Yaw using official Phase 5 recovered CSVs.
- RMSE/MAE improvement summary uses the official W10_N3_mean1x Phase 5 recovery summary.
- Output filenames end with _v2 to avoid confusion with the earlier draft figures.

## Mapping to Section 4.2.2

- Figure A: software sensor prediction accuracy.
- Figure B: clean residual/error behavior and p99_abs threshold.
- Figure C: final window detector behavior under attack.
- Figure D: official recovery behavior under attack.
- Figure E: quantitative RMSE/MAE recovery improvement.

## Final configuration

- Software sensor: V9 trained hybrid
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule: use V9 prediction during detected attacks, otherwise keep the measured signal.

## Recommended paper wording

These are Section 4.2.2-style effectiveness figures generated using my ArduPilot SITL dataset. They do not duplicate the original paper's data; instead, they reproduce the same evaluation logic with my final V9/p99_abs/W10_N3_mean1x configuration.
