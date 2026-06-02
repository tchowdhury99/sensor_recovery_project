# Final Recovery Results

This section reports the final recovery results for the ArduPilot SITL reproduction of the software-based realtime recovery methodology. The final configuration used the **V9 trained hybrid software sensor**, the corrected **W10_N3_mean1x** window detector, and the **p99_abs** residual threshold. The official Phase 5 evaluation contained **39 attacked files**.

## Aggregate Recovery Performance

The recovery stage improved **39/39** files under the RMSE criterion and **39/39** files under the MAE criterion.

The mean attacked RMSE was **0.016423050611058**, while the mean recovered RMSE was **0.001332188625576**. This corresponds to a mean RMSE improvement of **92.566251383206%**.

The mean attacked MAE was **0.006979895644135**, while the mean recovered MAE was **0.000446609915515**. This corresponds to a mean MAE improvement of **94.367048917331%**.

## Final Configuration

| Component | Selected Method |
|---|---|
| Software sensor | V9 trained hybrid |
| Detector | W10_N3_mean1x |
| Threshold | p99_abs |
| Official files evaluated | 39 |
| Files improved by RMSE | 39/39 |
| Files improved by MAE | 39/39 |

## Interpretation

These results show that software-sensor-based recovery substantially reduces the error caused by attacked sensor measurements. The recovered signal has much lower error than the attacked signal across the official recovery set. This supports the central reproduction claim that a learned software sensor can act as a backup estimator when the physical sensor stream is compromised.

In this reproduction, the V9 trained hybrid model was selected because earlier ARX-style predictors were unstable for some attitude channels, especially pitch and yaw. The hybrid model combines simple temporal baselines with a trained component and therefore provides a more stable future-state estimate. When paired with the corrected residual detector and threshold-based recovery logic, it produced consistent improvement across the official attacked files.
