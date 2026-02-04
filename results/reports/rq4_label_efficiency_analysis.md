# RQ4: Label Efficiency Analysis Report

*Generated: 2026-01-16 20:39:29*

## Research Question

**RQ4:** How many target domain labels are required for a fine-tuned model to surpass the performance of a source-trained model?

## Label Efficiency Results (CNN on FD001)

| Label Fraction | Accuracy | F1-Score | ROC-AUC |
|----------------|----------|----------|----------|
| 1.0% | 0.9235 | 0.2338 | 0.9809 |
| 5.0% | 0.9312 | 0.2487 | 0.9804 |
| 10.0% | 0.9813 | 0.4195 | 0.9718 |
| 20.0% | 0.9873 | 0.5505 | 0.9747 |
| 50.0% | 0.9920 | 0.6435 | 0.9844 |

## Statistical Analysis (H04 vs H14)

**H04:** Small amounts of target data (<10%) do not provide significant performance improvements.

**H14:** Significant improvement is possible even with <10% data.

**Baseline (0% Target Labels):** F1 = 0.0988

- **1% Data Improvement:** +0.1350
- **5% Data Improvement:** +0.1499
- **10% Data Improvement:** +0.3207

**Conclusion:** We REJECT H04. Substantial performance gains (e.g., >50% improvement) are achievable with as little as 5% of target domain labeled data.
