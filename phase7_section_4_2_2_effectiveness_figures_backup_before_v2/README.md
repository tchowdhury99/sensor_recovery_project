# Phase 7B: Section 4.2.2-Style Effectiveness Figures

This folder contains result/effectiveness figures modeled after Section 4.2.2 of the reference paper, but generated using this project's ArduPilot SITL V9 software-sensor recovery results.

## Generated status

- Figure A V9 prediction accuracy: generated
- Figure B clean residual threshold: skipped
- Figure C window detection example: generated
- Figure D recovery time series: generated
- Figure E recovery improvement summary: generated

## Intended mapping to the paper

- Figure A corresponds to the paper's software sensor prediction figure.
- Figure B corresponds to clean residual and threshold behavior.
- Figure C corresponds to residual/window-based attack detection.
- Figure D corresponds to attack recovery time-series behavior.
- Figure E summarizes quantitative recovery improvement using RMSE/MAE.

## Project-specific configuration

- Software sensor: V9 trained hybrid
- Prediction target: future state x_hat[k+1] using information available at time k
- Threshold: p99_abs
- Detector: W10_N3_mean1x
- Recovery rule:

    if attack_detected:
        recovered[k+1] = V9_prediction[k+1]
    else:
        recovered[k+1] = attacked_measurement[k+1]

## Important note

These figures should be described as Section 4.2.2-style reproduction figures, not exact duplicates of the original paper. The original paper used its own robotic vehicle datasets and state-space software sensors. This project uses ArduPilot SITL logs and the selected V9 trained hybrid software sensor.
