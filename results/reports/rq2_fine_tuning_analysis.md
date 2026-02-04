# RQ2: Fine-Tuning Analysis Report

*Generated: 2026-01-16 20:36:09*

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
| cnn | FD001 | 0.0988 | 0.5505 | +0.4517 |
| cnn | FD003 | 0.0178 | 0.4975 | +0.4798 |
| cnn | FD004 | 0.3657 | 0.3927 | +0.0270 |
| lstm | FD001 | 0.0939 | 0.0451 | -0.0488 |
| lstm | FD003 | 0.0141 | 0.0965 | +0.0824 |
| lstm | FD004 | 0.4828 | 0.5000 | +0.0172 |

### Statistical Test Results

| Model | Mean Direct F1 | Mean Fine-Tuned F1 | Improvement | p-value | Significant |
|-------|----------------|---------------------|-------------|---------|-------------|
| cnn | 0.1608 | 0.4802 | +0.3195 | 0.0805 | ✗ |
| lstm | 0.1969 | 0.2139 | +0.0170 | 0.3490 | ✗ |

## Hypothesis Test Results

**H02:** Fine-tuning does not significantly improve performance.

**H12:** Fine-tuning significantly improves performance.

**Conclusion:** We FAIL TO REJECT H02. Fine-tuning does not provide statistically significant improvement.
