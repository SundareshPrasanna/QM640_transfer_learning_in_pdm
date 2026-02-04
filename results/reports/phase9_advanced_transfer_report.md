# Phase 9: Advanced Transfer Learning Comparison

*Generated: 2026-01-16 21:18:03*

## 🚀 Experimental Strategies Implemented

1. **Gradual Unfreezing:** Trained head first, then unfrozen last feature extraction block with 10x lower LR.
2. **Domain-Adaptive BatchNorm:** Kept BN layers in `train()` mode to adapt normalization statistics to target distribution.
3. **Dynamic Loss Weighting:** Calculated class weights based specifically on the target sample distribution.

## 📊 Performance Comparison (F1-Score)

| Model | Target | Base FT (RQ2) | Advanced FT (P9) | Gain | % Gain |
|-------|--------|---------------|------------------|------|--------|
| CNN | FD001 | 0.5505 | 0.5131 | -0.0374 | -6.8% |
| CNN | FD003 | 0.4975 | 0.5742 | +0.0767 | +15.4% |
| CNN | FD004 | 0.3927 | 0.4648 | +0.0721 | +18.4% |
| LSTM | FD001 | 0.0451 | 0.7173 | +0.6722 | +1490.9% |
| LSTM | FD003 | 0.0965 | 0.6420 | +0.5455 | +565.2% |
| LSTM | FD004 | 0.5000 | 0.4779 | -0.0221 | -4.4% |

## 🎯 Key Takeaways

- Average F1-score improvement across all domains: **+0.2178**
- Maximum gain observed: **+0.6722** (LSTM on FD001)

The combination of **Adaptive BN** and **Gradual Unfreezing** successfully allowed the model to bridge the domain gap more effectively than freezing the entire base extractor. This validates that some 'fine' adjustment of temporal features is necessary for optimal transfer.
