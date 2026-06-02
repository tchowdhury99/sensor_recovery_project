# Phase 7B Section 4.2.2-Style Effectiveness Summary

The original paper's Section 4.2.2 is an effectiveness evaluation section. It does not only present a block diagram. It shows whether the software sensor predicts real readings, whether error correction reduces prediction error, whether recovery parameters are selected appropriately, whether the system recovers from sensor attacks, and how the technique behaves under environmental and attack-scale variation.

This project's Phase 7B figures follow the same evaluation logic using the ArduPilot SITL reproduction results.

## Figure A: V9 prediction accuracy

This figure shows whether the V9 trained hybrid software sensor tracks the measured or clean attitude signal. It is the closest equivalent to the paper's software-sensor prediction figure.

## Figure B: Clean residual threshold

This figure shows normal residual behavior and the p99_abs threshold. It explains how the detection threshold is calibrated from clean behavior.

## Figure C: Window detection example

This figure shows how residual threshold violations become an attack decision under the W10_N3_mean1x detector.

## Figure D: Recovery time series

This figure shows the core recovery effect: attacked signal versus recovered signal, with detected/recovery regions if available.

## Figure E: Recovery improvement summary

This figure summarizes quantitative improvement using mean attacked and recovered RMSE/MAE values when the required summary columns are available.

## Professor-facing wording

These are Section 4.2.2-style effectiveness figures. They do not duplicate the original paper's data. Instead, they reproduce the same evaluation structure using my ArduPilot SITL dataset and my final configuration: V9 trained hybrid predictor, p99_abs threshold, W10_N3_mean1x detector, and conditional replacement recovery.
