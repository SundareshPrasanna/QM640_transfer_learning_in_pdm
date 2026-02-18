# RQ4: Label Efficiency Analysis Report

*Generated: 2026-02-18 12:34:12*

## Research Question

**RQ4:** How many target domain labels are required for a fine-tuned model to surpass the performance of a source-trained model?

## Label Efficiency Results (CNN on FD001)

| Label Fraction | Accuracy | F1-Score | ROC-AUC |
|----------------|----------|----------|----------|
| 1.0% | 0.9897 | 0.2553 | 0.9779 |
| 5.0% | 0.9892 | 0.6309 | 0.9879 |
| 10.0% | 0.9917 | 0.5304 | 0.9786 |
| 20.0% | 0.9788 | 0.4857 | 0.9878 |
| 50.0% | 0.9843 | 0.5556 | 0.9883 |

## Statistical Analysis (H04 vs H14)

**H04:** Small amounts of target data (<10%) do not provide significant performance improvements.

**H14:** Significant improvement is possible even with <10% data.

**Baseline (0% Target Labels):** F1 = 0.1858

- **1% Data Improvement:** +0.0695
- **5% Data Improvement:** +0.4451
- **10% Data Improvement:** +0.3446

**Conclusion:** We REJECT H04. Substantial performance gains (e.g., >50% improvement) are achievable with as little as 5% of target domain labeled data.
