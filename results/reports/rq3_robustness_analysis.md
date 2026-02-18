# RQ3: Model Architecture Robustness Comparison

*Generated: 2026-02-18 12:33:00*

## Research Question

**RQ3:** Which model architecture (CNN vs LSTM) demonstrates higher robustness to domain shift?

## Direct Transfer Robustness

| Domain | CNN F1 | LSTM F1 | Difference (C-L) |
|--------|--------|---------|------------------|
| FD001 | 0.1858 | 0.3034 | -0.1176 |
| FD003 | 0.0704 | 0.0000 | +0.0704 |
| FD004 | 0.4129 | 0.5916 | -0.1787 |

## Fine-Tuned Robustness (20% Labels)

| Target Domain | CNN F1 | LSTM F1 | Difference (C-L) |
|---------------|--------|---------|------------------|
| FD001 | 0.4857 | 0.7256 | -0.2399 |
| FD003 | 0.0604 | 0.1085 | -0.0481 |
| FD004 | 0.1231 | 0.5006 | -0.3774 |

## Statistical Analysis (H03 vs H13)

**H03:** There is no statistically significant difference in robustness between CNN and LSTM models.

**H13:** There is a statistically significant difference in robustness between the two architectures.

### Results Summary

- **Mean F1 (Direct Transfer):** CNN = 0.2231, LSTM = 0.2983
- **Mean F1 (Fine-Tuned):** CNN = 0.2231, LSTM = 0.4449

The **LSTM architecture** appears more robust to domain shift in this study, demonstrating better stability or mean performance across domains.
