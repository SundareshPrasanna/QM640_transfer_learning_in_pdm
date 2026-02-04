# RQ1: Domain Shift Analysis Report

*Generated: 2026-01-16 20:31:08*

## Research Question

**RQ1:** How does the performance of a fault detection model degrade when transferred directly from a large source dataset to smaller target datasets without adaptation?

## Experimental Setup

- **Source Domain:** FD002
- **Target Domains:** FD001, FD003, FD004
- **Transfer Method:** Direct transfer (no adaptation)
- **Significance Level:** α = 0.05

## Results

### Performance by Domain

| Model | Domain | Type | Accuracy | F1-Score | ROC-AUC |
|-------|--------|------|----------|----------|----------|
| random_forest | FD002 | source | 0.9828 | 0.6305 | 0.9897 |
| random_forest | FD001 | target | 0.8261 | 0.1201 | 0.9801 |
| random_forest | FD003 | target | 0.7819 | 0.0178 | 0.7583 |
| random_forest | FD004 | target | 0.9501 | 0.2281 | 0.9409 |
| cnn | FD002 | source | 0.9811 | 0.5534 | 0.9713 |
| cnn | FD001 | target | 0.7800 | 0.0988 | 0.9013 |
| cnn | FD003 | target | 0.7253 | 0.0178 | 0.6510 |
| cnn | FD004 | target | 0.9755 | 0.3657 | 0.9165 |
| lstm | FD002 | source | 0.9865 | 0.5580 | 0.9641 |
| lstm | FD001 | target | 0.7671 | 0.0939 | 0.9386 |
| lstm | FD003 | target | 0.7244 | 0.0141 | 0.7350 |
| lstm | FD004 | target | 0.9850 | 0.4828 | 0.8389 |

### Performance Degradation Summary

| Model | Source F1 | Mean Target F1 | Degradation | p-value | Significant |
|-------|-----------|----------------|-------------|---------|-------------|
| random_forest | 0.6305 | 0.1220 | +0.5085 | 0.0070 | ✓ |
| cnn | 0.5534 | 0.1608 | +0.3926 | 0.0324 | ✓ |
| lstm | 0.5580 | 0.1969 | +0.3611 | 0.0650 | ✗ |

## Hypothesis Test Results

**H01:** There is no statistically significant difference in fault detection accuracy between source-trained model and its performance on target datasets.

**H11:** There is a statistically significant decrease in fault detection accuracy under direct transfer.

**Conclusion:** We REJECT H01 for random_forest, cnn. There is statistically significant performance degradation under domain shift.
