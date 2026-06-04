# Phase 2B Refined Target-Axis V9 Component Audit Report

## Purpose

This refined audit checks only the attacked/target axis inferred from each detection CSV filename. Missing non-target-axis columns are not treated as failures.

## Decision Logic

- `PASS_COMPONENT_ALIGNMENT` means the target-axis V9 predicted future column matches the V9 prediction file's hybrid prediction and the required component columns exist.

- `CleanFuture` mismatch is reported separately as a provenance warning. It does not automatically block Phase 5B component use if V9 prediction alignment passes.

## Summary

- Total detection CSVs audited: `351`

- Component-alignment pass: `351`

- Component-alignment fail: `0`

- CleanFuture warning rows: `351`

## Final Decision

PASS: Phase 5B may safely use target-axis V9 component columns through the `PredictionFile` mapping. Use the target axis only for each detection CSV.

## CleanFuture Warning Preview

| detection_file                                               | target_axis   |   clean_max_abs_mismatch |   clean_rmse_mismatch | clean_future_status   |
|:-------------------------------------------------------------|:--------------|-------------------------:|----------------------:|:----------------------|
| baseline_01_hover_Pitch_bias_W10_N3_mean1.25x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W10_N3_mean1.5x_detection.csv   | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W10_N3_mean1x_detection.csv     | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W20_N5_mean1.25x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W20_N5_mean1.5x_detection.csv   | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W20_N5_mean1x_detection.csv     | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W30_N8_mean1.25x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W30_N8_mean1.5x_detection.csv   | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_bias_W30_N8_mean1x_detection.csv     | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W10_N3_mean1.25x_detection.csv | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W10_N3_mean1.5x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W10_N3_mean1x_detection.csv    | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W20_N5_mean1.25x_detection.csv | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W20_N5_mean1.5x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W20_N5_mean1x_detection.csv    | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W30_N8_mean1.25x_detection.csv | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W30_N8_mean1.5x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_pulse_W30_N8_mean1x_detection.csv    | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W10_N3_mean1.25x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W10_N3_mean1.5x_detection.csv   | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W10_N3_mean1x_detection.csv     | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W20_N5_mean1.25x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W20_N5_mean1.5x_detection.csv   | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W20_N5_mean1x_detection.csv     | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W30_N8_mean1.25x_detection.csv  | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W30_N8_mean1.5x_detection.csv   | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Pitch_ramp_W30_N8_mean1x_detection.csv     | Pitch         |               0.0110741  |            0.00215877 | WARNING_MISMATCH      |
| baseline_01_hover_Roll_bias_W10_N3_mean1.25x_detection.csv   | Roll          |               0.00797138 |            0.00161915 | WARNING_MISMATCH      |
| baseline_01_hover_Roll_bias_W10_N3_mean1.5x_detection.csv    | Roll          |               0.00797138 |            0.00161915 | WARNING_MISMATCH      |
| baseline_01_hover_Roll_bias_W10_N3_mean1x_detection.csv      | Roll          |               0.00797138 |            0.00161915 | WARNING_MISMATCH      |

