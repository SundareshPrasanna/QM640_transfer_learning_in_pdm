# RQ3: Model Architecture Robustness Comparison

*Generated: 2026-01-16 20:38:23*

## Research Question

**RQ3:** Which model architecture (CNN vs LSTM) demonstrates higher robustness to domain shift?

## Direct Transfer Robustness

| Domain | CNN F1 | LSTM F1 | Difference (C-L) |
|--------|--------|---------|------------------|
| FD001 | 0.0988 | 0.0939 | +0.0050 |
| FD002 | 0.5534 | 0.5580 | -0.0046 |
| FD003 | 0.0178 | 0.0141 | +0.0037 |
| FD004 | 0.3657 | 0.4828 | -0.1171 |

## Fine-Tuned Robustness (20% Labels)

| Target Domain | CNN F1 | LSTM F1 | Difference (C-L) |
|---------------|--------|---------|------------------|
| FD001 | 0.5505 | 0.0451 | +0.5054 |
| FD003 | 0.4975 | 0.0965 | +0.4010 |
| FD004 | 0.3927 | 0.5000 | -0.1073 |

## Statistical Analysis (H03 vs H13)

**H03:** There is no statistically significant difference in robustness between CNN and LSTM models.

**H13:** There is a statistically significant difference in robustness between the two architectures.

### Results Summary

- **Mean F1 (Direct Transfer):** CNN = 0.2589, LSTM = 0.2872
- **Mean F1 (Fine-Tuned):** CNN = 0.4802, LSTM = 0.2139

The **CNN architecture** appears more robust to domain shift in this study, especially when considering the effectiveness of transfer learning/fine-tuning. CNN showed significantly better adaptation to FD001 and FD003 compared to LSTM.
