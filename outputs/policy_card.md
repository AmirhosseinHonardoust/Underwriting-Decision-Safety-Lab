# Policy Card — Underwriting Decision Safety

## Decision
Use a calibrated model to approve/reject automatically when confident; otherwise route to manual review.

## Rule
- Compute p(approve)
- confidence = max(p(approve), 1 - p(approve))
- If confidence >= 0.85: AUTO-DECIDE
- Else: REVIEW

## Target
- Target auto-decision coverage: 0.70
- Expected coverage at recommended threshold: 0.70

## Expected auto-decision performance (test)
- Accuracy (auto-decisions): 0.977
- F1 (auto-decisions): 0.986

## Calibration and ranking health (test)
- ECE: 0.0514
- Brier: 0.0804
- Average precision: 0.9814
 
## Monitoring triggers (suggested)
- If coverage shifts by > 10% vs baseline, re-check score distribution.
- If ECE doubles vs baseline, re-calibrate or retrain.
- If approval rate shifts sharply, check for population/mix shift.
