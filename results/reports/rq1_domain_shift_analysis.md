# RQ1: Domain Shift Analysis Report

*Generated: 2026-02-18 12:31:49*

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
| random_forest | FD002 | source | 0.9814 | 0.5375 | 0.9789 |
| random_forest | FD001 | target | 0.9879 | 0.0000 | 0.9855 |
| random_forest | FD003 | target | 0.9911 | 0.0000 | 0.9901 |
| random_forest | FD004 | target | 0.9902 | 0.3626 | 0.9743 |
| cnn | FD002 | source | 0.9677 | 0.4822 | 0.9701 |
| cnn | FD001 | target | 0.9020 | 0.1858 | 0.9787 |
| cnn | FD003 | target | 0.9614 | 0.0704 | 0.8789 |
| cnn | FD004 | target | 0.9862 | 0.4129 | 0.9074 |
| lstm | FD002 | source | 0.9889 | 0.7433 | 0.9959 |
| lstm | FD001 | target | 0.9901 | 0.3034 | 0.9978 |
| lstm | FD003 | target | 0.9911 | 0.0000 | 0.9609 |
| lstm | FD004 | target | 0.9920 | 0.5916 | 0.9659 |

### Performance Degradation Summary

| Model | Source F1 | Mean Target F1 | Degradation | p-value | Significant |
|-------|-----------|----------------|-------------|---------|-------------|
| random_forest | 0.5375 | 0.1209 | +0.4166 | 0.1250 | ✗ |
| cnn | 0.4822 | 0.2231 | +0.2591 | 0.1250 | ✗ |
| lstm | 0.7433 | 0.2983 | +0.4449 | 0.1250 | ✗ |

## Hypothesis Test Results

**H01:** There is no statistically significant difference in fault detection accuracy between source-trained model and its performance on target datasets.

**H11:** There is a statistically significant decrease in fault detection accuracy under direct transfer.

**Conclusion:** We FAIL TO REJECT H01. No statistically significant performance degradation was observed.
