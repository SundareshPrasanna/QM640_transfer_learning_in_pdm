# Legacy Advanced Transfer Learning Comparison (CNN + LSTM)

*Generated: 2026-02-18 12:37:29*

## 🚀 Experimental Strategies Implemented

1. **Gradual Unfreezing:** Trained head first, then unfrozen last feature extraction block with 10x lower LR.
2. **Domain-Adaptive BatchNorm:** Kept BN layers in `train()` mode to adapt normalization statistics to target distribution.
3. **Dynamic Loss Weighting:** Calculated class weights based specifically on the target sample distribution.

## 📊 Performance Comparison (F1-Score)

| Model | Target | Base FT (RQ2) | Advanced FT | Gain | % Gain |
|-------|--------|---------------|------------------|------|--------|
| CNN | FD001 | 0.4857 | 0.7447 | +0.2590 | +53.3% |
| CNN | FD003 | 0.0604 | 0.6027 | +0.5423 | +898.1% |
| CNN | FD004 | 0.1231 | 0.3173 | +0.1941 | +157.7% |
| LSTM | FD001 | 0.7256 | 0.7105 | -0.0151 | -2.1% |
| LSTM | FD003 | 0.1085 | 0.5200 | +0.4115 | +379.1% |
| LSTM | FD004 | 0.5006 | 0.5576 | +0.0571 | +11.4% |

## 🎯 Key Takeaways

- Average F1-score improvement across all domains: **+0.2415**
- Maximum gain observed: **+0.5423** (CNN on FD003)

The combination of **Adaptive BN** and **Gradual Unfreezing** successfully allowed the model to bridge the domain gap more effectively than freezing the entire base extractor. This validates that some 'fine' adjustment of temporal features is necessary for optimal transfer.
