# RQ2: Fine-Tuning Analysis Report

*Generated: 2026-02-18 12:32:50*

## Research Question

**RQ2:** To what extent does fine-tuning improve fault detection performance compared to direct transfer?

## Experimental Setup

- **Source Domain:** FD002
- **Target Domains:** FD001, FD003, FD004
- **Fine-tuning Strategy:** Freeze base layers, train classification head
- **Labeled Target Data:** 20%
- **Significance Level:** α = 0.05

## Results

### Performance Comparison

| Model | Target | Direct F1 | Fine-Tuned F1 | Improvement |
|-------|--------|-----------|---------------|-------------|
| cnn | FD001 | 0.1858 | 0.4857 | +0.2999 |
| cnn | FD003 | 0.0704 | 0.0604 | -0.0100 |
| cnn | FD004 | 0.4129 | 0.1231 | -0.2898 |
| lstm | FD001 | 0.3034 | 0.7256 | +0.4221 |
| lstm | FD003 | 0.0000 | 0.1085 | +0.1085 |
| lstm | FD004 | 0.5916 | 0.5006 | -0.0910 |

### Statistical Test Results

| Model | Mean Direct F1 | Mean Fine-Tuned F1 | Improvement | p-value | Significant |
|-------|----------------|---------------------|-------------|---------|-------------|
| cnn | 0.2231 | 0.2231 | +0.0000 | 0.5000 | ✗ |
| lstm | 0.2983 | 0.4449 | +0.1465 | 0.2150 | ✗ |

## Hypothesis Test Results

**H02:** Fine-tuning does not significantly improve performance.

**H12:** Fine-tuning significantly improves performance.

**Conclusion:** We FAIL TO REJECT H02. Fine-tuning does not provide statistically significant improvement.
