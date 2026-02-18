# LSTM Gradual Unfreezing Transfer Analysis

*Generated: 2026-02-18 12:43:55*

## Experimental Setup

- **Model scope:** LSTM only (no CNN in this analysis).
- **Stage 1 (head warmup):** Freeze LSTM backbone and train classification head.
- **Stage 2 (gradual unfreezing):** Unfreeze the upper LSTM layer and fine-tune with lower LR.
- **Class balancing:** Use target-domain class-weighted BCE during adaptation.

## Performance Comparison (F1-Score)

| Target | Base LSTM FT (RQ2) | LSTM Gradual Unfreezing | Gain | % Gain |
|--------|--------------------|--------------------------|------|--------|
| FD001 | 0.7256 | 0.7105 | -0.0151 | -2.1% |
| FD003 | 0.1085 | 0.5200 | +0.4115 | +379.1% |
| FD004 | 0.5006 | 0.5576 | +0.0571 | +11.4% |

## Key Takeaways

- Average F1-score improvement across target domains: **+0.1512**
- Maximum gain observed: **+0.4115** (on FD003)

This result isolates the impact of **LSTM gradual unfreezing** against standard LSTM fine-tuning on the same target splits.
