# Phase 7 Methodology Figures

This folder contains paper-style methodology diagrams for the ArduPilot SITL software-sensor attack detection and recovery project.

These figures are architecture and logic diagrams, not result plots. They are intended to complement the Phase 6 result figures.

## Final configuration represented

- Software sensor: V9 trained hybrid
- Prediction target: future state x_hat[k+1] using information available at time k
- Roll/Pitch residual: attacked_future[k+1] - V9_prediction[k+1]
- Yaw residual: circular angular difference
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule:

    if attack_detected:
        recovered[k+1] = V9_prediction[k+1]
    else:
        recovered[k+1] = attacked_measurement[k+1]

## Figure 1: Software-Sensor-Based Attack Detection and Recovery Pipeline

Files:

- figure1_detection_recovery_pipeline.png
- figure1_detection_recovery_pipeline.pdf

This is the main Section 4.2.2-style methodology figure. It shows the runtime pipeline from sensor measurement to residual generation, threshold comparison, window detection, attack decision, recovery switch, and recovered output.

## Figure 2: Training and Deployment Workflow

Files:

- figure2_training_deployment_workflow.png
- figure2_training_deployment_workflow.pdf

This figure shows the full workflow. Clean baseline logs are used for feature extraction, candidate predictor evaluation, V9 selection, clean residual generation, and p99_abs threshold computation. Attacked logs are then used for detection, recovery, and RMSE/MAE evaluation.

## Figure 3: V9 Hybrid Software Sensor Internal Structure

Files:

- figure3_v9_hybrid_internal_structure.png
- figure3_v9_hybrid_internal_structure.pdf

This figure explains the internal V9 predictor. It combines naive, constant-velocity, and ARX-style predictors using validation-based weights.

## Figure 4: Residual and Window Detection Logic

Files:

- figure4_residual_window_detection_logic.png
- figure4_residual_window_detection_logic.pdf

This figure explains how residuals become attack decisions. The absolute residual is compared with p99_abs, and the W10_N3_mean1x detector checks whether at least three threshold violations occur inside a ten-sample window.

## Figure 5: Recovery Decision Logic

Files:

- figure5_recovery_decision_logic.png
- figure5_recovery_decision_logic.pdf

This figure explains the recovery switch. If an attack is detected, the recovered signal uses the V9 prediction. Otherwise, the measured signal is preserved.

## Recommended use

Use Figure 1 as the main Section 4.2.2-style architecture diagram. Use Figures 2-5 as supporting methodology diagrams.
