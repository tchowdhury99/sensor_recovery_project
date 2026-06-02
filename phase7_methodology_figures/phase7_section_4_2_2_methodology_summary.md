# Phase 7 Section 4.2.2 Methodology Summary

The original paper's Section 4.2.2 methodology figure explains the software-sensor-based recovery architecture. Your Phase 7 figures serve the same purpose for your ArduPilot SITL reproduction.

Your Phase 6 package already provides result figures: RMSE/MAE plots, histograms, scatter plots, per-target improvement plots, and representative recovery time-series plots. Phase 7 adds the missing methodology diagrams that explain how the system works.

## Connection to the original methodology

The core idea is that a software sensor predicts the expected future state of the system. The measured sensor value is compared against this prediction. If the residual becomes abnormal according to a clean-data threshold and a window detector, the system treats the measurement as attacked. During detected attack periods, the attacked measurement is replaced by the software-sensor prediction.

Your implementation follows this structure:

1. Clean logs define normal behavior.
2. V9 predicts the future state x_hat[k+1] using information available at time k.
3. The measured future state is compared with the V9 prediction.
4. The residual is checked against the p99_abs threshold.
5. The W10_N3_mean1x window detector makes the attack decision.
6. During detected attacks, recovery replaces the attacked signal with the V9 prediction.
7. Recovery quality is evaluated with RMSE and MAE.

## Project-specific realization

Your project uses ArduPilot SITL data, not the original paper's dataset. Therefore, the numerical results and plots do not need to match the original paper exactly. The important point is that the methodology is aligned with the paper's software-sensor recovery concept.

Your final configuration is:

- Software sensor: V9 trained hybrid
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule: use V9 prediction when attack_detected is true; otherwise keep the measured value

## How to describe the figures

Figure 1 is the main Section 4.2.2-style diagram. It shows the complete runtime detection and recovery pipeline.

Figure 2 explains how clean baseline data is used for training and threshold calibration, and how the trained system is deployed on attacked logs.

Figure 3 explains the V9 hybrid predictor. This is important because V9 is not just a previous-value predictor. It fuses naive, constant-velocity, and ARX-style predictions using validation-based weights.

Figure 4 explains the residual and window detector. It shows that the attack decision is based on repeated threshold violations inside a ten-sample window.

Figure 5 explains the final recovery switch. It shows exactly when the system uses the V9 prediction and when it keeps the measured value.

## Suggested professor-facing explanation

These figures are methodology diagrams. They explain the system architecture and logic behind the quantitative results. The Phase 6 figures show that recovery improved the attacked signals. The Phase 7 figures explain how that recovery was performed: future-state prediction, residual generation, clean thresholding, window-based attack detection, and conditional signal replacement.
