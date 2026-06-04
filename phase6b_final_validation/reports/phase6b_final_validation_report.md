# Phase 6B Final Validation and Comparison Report

## Purpose

Phase 6B validates the B-phase extensions and decides whether they improve the scientific quality of the project.

This report is read-only with respect to official Phase 1/2/3/4/5 outputs.

## Phase 2B result

| phase    |   total_detection_csvs |   component_alignment_pass |   component_alignment_fail |   clean_future_warning_rows | decision   | interpretation                                                                                          |
|:---------|-----------------------:|---------------------------:|---------------------------:|----------------------------:|:-----------|:--------------------------------------------------------------------------------------------------------|
| Phase 2B |                    351 |                        351 |                          0 |                         351 | PASS       | Target-axis V9 component columns are safe for Phase 5B use; CleanFuture mismatch is provenance warning. |

## Phase 3B component thresholds

| phase    |   all_baseline_rows |   expected_all_baseline_rows | decision   | interpretation                                                       |
|:---------|--------------------:|-----------------------------:|:-----------|:---------------------------------------------------------------------|
| Phase 3B |                  12 |                           12 | PASS       | Component thresholds exist for Roll/Pitch/Yaw × hybrid/naive/CV/ARX. |

| axis   | component   |    n |        mae |       rmse |   p99_abs_residual |   recommended_threshold |
|:-------|:------------|-----:|-----------:|-----------:|-------------------:|------------------------:|
| Pitch  | arx         | 1605 | 0.0150009  | 0.0273587  |         0.078832   |              0.110714   |
| Pitch  | cv          | 1605 | 0.00156009 | 0.00265327 |         0.00909989 |              0.0112695  |
| Pitch  | hybrid      | 1605 | 0.00245814 | 0.00341382 |         0.0121921  |              0.0150518  |
| Pitch  | naive       | 1605 | 0.00164886 | 0.00288985 |         0.0108235  |              0.0119371  |
| Roll   | arx         | 1605 | 0.00446544 | 0.00604385 |         0.0165949  |              0.020976   |
| Roll   | cv          | 1605 | 0.00165478 | 0.0028374  |         0.010067   |              0.0111234  |
| Roll   | hybrid      | 1605 | 0.00226985 | 0.00340316 |         0.0107798  |              0.0169816  |
| Roll   | naive       | 1605 | 0.00176219 | 0.00311336 |         0.0115145  |              0.0131838  |
| Yaw    | arx         | 1605 | 0.00906504 | 0.0128837  |         0.0306733  |              0.0365388  |
| Yaw    | cv          | 1605 | 0.00069862 | 0.00138485 |         0.00437744 |              0.00462967 |
| Yaw    | hybrid      | 1605 | 0.00105845 | 0.00186413 |         0.00664264 |              0.00934201 |
| Yaw    | naive       | 1605 | 0.00145643 | 0.00279493 |         0.0121973  |              0.0204877  |

## Phase 4B threshold-method comparison

| phase    |   rows | best_method       |   best_mean_f1 |   best_mean_recall |   best_mean_fpr |   best_median_delay | decision       | interpretation                                                                                       |
|:---------|-------:|:------------------|---------------:|-------------------:|----------------:|--------------------:|:---------------|:-----------------------------------------------------------------------------------------------------|
| Phase 4B |    117 | scenario_specific |       0.841846 |            0.88264 |       0.0320029 |                   2 | PASS_EXTENSION | Threshold-method comparison is acceptable as an extension; do not replace official Phase 4 silently. |

| threshold_method   |   mean_precision |   mean_recall |   mean_f1 |   mean_fpr |   median_delay |   n_rows |
|:-------------------|-----------------:|--------------:|----------:|-----------:|---------------:|---------:|
| scenario_specific  |         0.817808 |      0.88264  |  0.841846 |  0.0320029 |              2 |       39 |
| dynamic_xkf1       |         0.802128 |      0.886659 |  0.834314 |  0.0357059 |              2 |       24 |
| static_global      |         0.813566 |      0.816708 |  0.793358 |  0.0306827 |              2 |       39 |
| yaw_static_global  |         0.815881 |      0.749544 |  0.74239  |  0.0276258 |              2 |       15 |

## Phase 5B recovery comparison

| method                        |   n_files |   mean_attacked_rmse |   mean_recovered_rmse |   mean_improvement_over_attacked_percent |   better_than_original_count | suspicion_flag       | paper_role                                        |   improvement_over_official_original_percent |
|:------------------------------|----------:|---------------------:|----------------------:|-----------------------------------------:|-----------------------------:|:---------------------|:--------------------------------------------------|---------------------------------------------:|
| official_original_v9_recovery |        24 |             0.017911 |           0.00159639  |                                  91.3606 |                          nan | NO                   | MAIN_RESULT_BASELINE                              |                                     nan      |
| phase5b_xkf1_adaptive_weights |        24 |             0.017911 |           0.000952636 |                                  95.2236 |                           22 | LOW_TO_MODERATE      | EXPERIMENTAL_EXTENSION_OR_SUPPLEMENTARY           |                                      40.3256 |
| phase5b_recent_error_adaptive |        24 |             0.017911 |           0.000437691 |                                  97.5973 |                           24 | HIGH_ORACLE_ASSISTED | OFFLINE_DIAGNOSTIC_ONLY_NOT_MAIN_REAL_TIME_RESULT |                                      72.5825 |

## Phase 5B by-target summary

| target   |   n |   mean_attacked_rmse |   mean_original_rmse |   mean_xkf1_adaptive_rmse |   xkf1_better_count |   mean_xkf1_improvement_over_original |   mean_recent_error_adaptive_rmse |   recent_error_better_count |   mean_recent_error_improvement_over_original |
|:---------|----:|---------------------:|---------------------:|--------------------------:|--------------------:|--------------------------------------:|----------------------------------:|----------------------------:|----------------------------------------------:|
| Pitch    |  12 |            0.0208422 |           0.00198804 |               0.00128771  |                  10 |                               39.3653 |                       0.000563361 |                          12 |                                       76.504  |
| Roll     |  12 |            0.0149799 |           0.00120475 |               0.000617565 |                  12 |                               52.3733 |                       0.000312022 |                          12 |                                       80.5748 |

## Suspicion audit

| item                           | finding                                                                     | risk                                                                                                                  | decision                                                                                                          |
|:-------------------------------|:----------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|
| Phase 5B XKF1 adaptive         | Improves 22/24 cases and uses XKF1 flight-context features.                 | Moderate. Improvement is strong but not perfect; likely plausible.                                                    | Keep as experimental extension/supplementary unless additional real-time validation is done.                      |
| Phase 5B recent-error adaptive | Improves 24/24 cases and uses recent clean-window prediction error.         | High. This is oracle-assisted because clean future/reference error is not available in real-time attacked deployment. | Do not present as main real-time result. Use only as offline diagnostic or upper-bound component-selection study. |
| Phase 3B Pitch ARX             | Pitch ARX recommended threshold is much larger than other Pitch components. | High if ARX is over-trusted for Pitch.                                                                                | Do not use Pitch ARX dominance as main recovery logic without justification.                                      |

## Final decisions

| question                                  | decision                      | reason                                                                                                                           |
|:------------------------------------------|:------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| Does Phase 6B pass?                       | YES                           | B-phase evidence exists and passes core validation checks.                                                                       |
| Do official Phase 4/5 need rerun?         | NO                            | Phase 2B/3B do not reveal a blocking bug; Phase 3B explicitly supports not rerunning Phase 4/5 from repair alone.                |
| Should Phase 4B be included?              | YES_AS_EXTENSION              | Threshold-method comparison is acceptable; scenario-specific is best overall and dynamic XKF1 is close for Roll/Pitch.           |
| Should Phase 5B XKF1 be included?         | YES_AS_EXPERIMENTAL_EXTENSION | XKF1 adaptive improves 22/24 cases and reduces mean recovered RMSE.                                                              |
| Should Phase 5B XKF1 be main result?      | NO                            | Official V9 recovery should remain main; XKF1 adaptive is not universally better and was tested only on Roll/Pitch W10/N3 cases. |
| Should Phase 5B recent-error be included? | ONLY_AS_OFFLINE_DIAGNOSTIC    | It uses recent clean-reference error and produces too-good 24/24 improvement.                                                    |
| Is project ready for final writing?       | YES_WITH_CAUTION              | Main pipeline is complete; report B-phases carefully as extensions, not replacements.                                            |

## Recommended paper presentation

- Present official V9 trained hybrid recovery as the main Phase 5 result.
- Present Phase 4B as a threshold-method ablation/extension.
- Present Phase 5B XKF1 adaptive weighting as an experimental extension or supplementary result.
- Present recent-error adaptive weighting only as an offline diagnostic or upper-bound component-selection study.
- Do not claim recent-error adaptive weighting is deployable in real time unless it is redesigned without clean-reference future values.

## Final conclusion

Phase 6B PASSES. The project is ready for final writing/figures, with the caution that B-phase extensions must be separated from official results.