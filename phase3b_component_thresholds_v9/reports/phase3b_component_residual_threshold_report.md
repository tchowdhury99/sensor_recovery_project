# Phase 3B Component Residual and Threshold Report

## Status

Phase 3B all-baseline threshold repair completed successfully.

The repair inferred baseline name and axis from the residual CSV filenames.

## All-baseline recommended thresholds

| axis   | component   |    n |        mae |       rmse |   p99_abs_residual |   p995_abs_residual |   mean_abs_plus_3std_abs_threshold |   recommended_threshold |
|:-------|:------------|-----:|-----------:|-----------:|-------------------:|--------------------:|-----------------------------------:|------------------------:|
| Roll   | hybrid      | 1605 | 0.00226985 | 0.00340316 |         0.0107798  |          0.0169816  |                         0.00987902 |              0.0169816  |
| Roll   | naive       | 1605 | 0.00176219 | 0.00311336 |         0.0115145  |          0.0131838  |                         0.00946455 |              0.0131838  |
| Roll   | cv          | 1605 | 0.00165478 | 0.0028374  |         0.010067   |          0.0111234  |                         0.00857162 |              0.0111234  |
| Roll   | arx         | 1605 | 0.00446544 | 0.00604385 |         0.0165949  |          0.020976   |                         0.0166878  |              0.020976   |
| Pitch  | hybrid      | 1605 | 0.00245814 | 0.00341382 |         0.0121921  |          0.0150518  |                         0.00956707 |              0.0150518  |
| Pitch  | naive       | 1605 | 0.00164886 | 0.00288985 |         0.0108235  |          0.0119371  |                         0.00877094 |              0.0119371  |
| Pitch  | cv          | 1605 | 0.00156009 | 0.00265327 |         0.00909989 |          0.0112695  |                         0.00800054 |              0.0112695  |
| Pitch  | arx         | 1605 | 0.0150009  | 0.0273587  |         0.078832   |          0.110714   |                         0.0836606  |              0.110714   |
| Yaw    | hybrid      | 1605 | 0.00105845 | 0.00186413 |         0.00664264 |          0.00934201 |                         0.00566338 |              0.00934201 |
| Yaw    | naive       | 1605 | 0.00145643 | 0.00279493 |         0.0121973  |          0.0204877  |                         0.00861506 |              0.0204877  |
| Yaw    | cv          | 1605 | 0.00069862 | 0.00138485 |         0.00437744 |          0.00462967 |                         0.00428689 |              0.00462967 |
| Yaw    | arx         | 1605 | 0.00906504 | 0.0128837  |         0.0306733  |          0.0335771  |                         0.0365388  |              0.0365388  |

## Decision

Phase 3B passes if this report contains 12 all-baseline rows: Roll/Pitch/Yaw × hybrid/naive/CV/ARX.

Phase 4/5 should not be rerun from this repair alone.
Phase 5B can use these component thresholds as diagnostic support for adaptive-weight recovery.