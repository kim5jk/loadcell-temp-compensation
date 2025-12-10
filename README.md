# Load Cell Temperature Compensation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Achieve industrial-grade stability from a $10 load cell using software temperature compensation**

A Python library for compensating temperature-induced drift in load cell measurements. Transform consumer-grade sensors into precision instruments with **5-6x improvement** in measurement stability.

![Temperature Compensation Demo](docs/demo.png)

<!-- If image doesn't load, view it at: https://raw.githubusercontent.com/kim5jk/loadcell-temp-compensation/main/docs/demo.png -->

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Test Duration | 21 days |
| Temperature Range | 13.8°C to 22.5°C (8.7°C swing) |
| **Raw Weight Stability** | 0.620g std |
| **Compensated Stability** | 0.105g std |
| **Improvement Factor** | **5.9x** |
| Median Improvement | 95.3% |

### Hardware Tested
- **Sensor**: CZL611CD Micro Load Cell (0-5kg, ~$10)
- **ADC**: HX711 24-bit ADC
- **Typical sensor spec**: 1-2.5g drift per 10°C
- **Our result**: 0.105g stability over 8.7°C swing (**10-25x better than spec**)

## 📦 Installation

```bash
pip install loadcell-temp-compensation
```

Or install from source:
```bash
git clone https://github.com/kim5jk/loadcell-temp-compensation.git
cd loadcell-temp-compensation
pip install -e .
```

## 🚀 Quick Start

```python
from loadcell_temp_comp import TemperatureCompensator

# Initialize compensator
compensator = TemperatureCompensator(
    sigma_temp=30,      # Gaussian smoothing for temperature
    sigma_weight=30,    # Gaussian smoothing for weight
    drift_threshold=0.4 # Weight change threshold for segment detection (g)
)

# Fit and transform your data
df_compensated = compensator.fit_transform(
    timestamps=df['datetime'],
    weights=df['mean_weight_g'],
    temperatures=df['temperature']
)

# Access results
print(f"Raw stability: {df['mean_weight_g'].std():.3f}g")
print(f"Compensated stability: {df_compensated['adjusted_weight'].std():.3f}g")
```

## 📖 How It Works

### The Problem
Load cells exhibit temperature-dependent drift due to:
- Thermal expansion of strain gauge elements
- Temperature sensitivity of adhesives
- Thermal gradients within the cell

Even "temperature-compensated" cells show measurable drift requiring software correction for high-precision applications.

### Our Solution: Segment-Based Linear Regression

```
┌─────────────────────────────────────────────────────────────┐
│  1. SEGMENT DETECTION                                        │
│     - Detect "stable periods" between weight changes         │
│     - Threshold-based: ΔW > 0.4g triggers new segment       │
├─────────────────────────────────────────────────────────────┤
│  2. GAUSSIAN SMOOTHING                                       │
│     - Smooth both temperature and weight signals             │
│     - Reduces high-frequency noise                           │
├─────────────────────────────────────────────────────────────┤
│  3. PER-SEGMENT LINEAR REGRESSION                           │
│     - For each segment: ΔW = slope × ΔT + intercept         │
│     - Adapts to non-linear sensor behavior                   │
├─────────────────────────────────────────────────────────────┤
│  4. TEMPERATURE NORMALIZATION                                │
│     - Adjust weight to constant reference temperature        │
│     - W_adj = W_fitted - slope × (T_actual - T_reference)   │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight
Unlike thermal inertia models (e.g., Hiveeyes), our approach:
- **Adapts per-segment**: Each stable period gets its own calibration
- **No time constants needed**: Assumes quasi-static temperature changes
- **Simple & effective**: Linear model achieves 95%+ improvement

## 📊 Comparison with Other Approaches

| Approach | Complexity | Typical Improvement | Notes |
|----------|------------|---------------------|-------|
| Hardware compensation | $$$$ | Built-in | Expensive, limited |
| Hiveeyes thermal inertia | Medium | 8-70x | Requires τ calibration |
| **This library** | **Simple** | **5-6x** | **Segment-adaptive** |

### vs. Hiveeyes Beehive Scale Project

| Aspect | Hiveeyes | This Library |
|--------|----------|--------------|
| Model | Thermal inertia (RC circuit) | Per-segment linear regression |
| Result | ±2g after compensation | **0.105g std** |
| Calibration | Fixed α per cell | Auto-adapts per segment |
| Time constant | τ ≈ 17 minutes | Not needed |

## 🔧 API Reference

### `TemperatureCompensator`

```python
class TemperatureCompensator:
    def __init__(
        self,
        sigma_temp: float = 30,
        sigma_weight: float = 30,
        drift_threshold: float = 0.4,
        reference_temperature: float = 20.0,
        detrend_threshold: float = 0.5
    ):
        """
        Parameters
        ----------
        sigma_temp : float
            Gaussian smoothing sigma for temperature signal
        sigma_weight : float
            Gaussian smoothing sigma for weight signal
        drift_threshold : float
            Weight change threshold (g) to detect new segment
        reference_temperature : float
            Target temperature for normalization (°C)
        detrend_threshold : float
            Long-term drift rate threshold (g/day)
        """
```

### Methods

| Method | Description |
|--------|-------------|
| `fit(timestamps, weights, temperatures)` | Fit compensation model to data |
| `transform(timestamps, weights, temperatures)` | Apply fitted model to new data |
| `fit_transform(...)` | Fit and transform in one step |
| `get_segment_stats()` | Return per-segment calibration statistics |

## 📁 Project Structure

```
loadcell-temp-compensation/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── loadcell_temp_comp/
│   ├── __init__.py
│   ├── compensator.py      # Main TemperatureCompensator class
│   ├── preprocessing.py    # Smoothing, segmentation utilities
│   └── visualization.py    # Plotting functions
├── examples/
│   ├── basic_usage.ipynb
│   └── sample_data.csv
└── tests/
    └── test_compensator.py
```

## 🧪 Validation

### Test Protocol
1. **Duration**: 21 days continuous measurement
2. **Environment**: Indoor, natural temperature variation
3. **Load**: ~340g static load on 5kg cell
4. **Sampling**: ~5.5 minute intervals

### Results by Segment

```
Analyzed 39 stable segments (>5 data points each)

Improvement Distribution:
  >90% improvement: 22 segments (56%)
  >80% improvement: 31 segments (79%)
  >50% improvement: 36 segments (92%)
  Degradation:       0 segments (0%)
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- [ ] Support for rapid temperature changes (thermal inertia model)
- [ ] Auto-tuning of hyperparameters
- [ ] Real-time streaming support
- [ ] Additional sensor types (pH, pressure, etc.)

## 📚 References

- [Hiveeyes Beehive Scale Project](https://community.hiveeyes.org/t/more-on-load-cell-temperature-compensation/4391)
- [Yoctopuce Temperature Drift Compensation](https://www.yoctopuce.com/EN/article/load-cell-temperature-drift-compensation)
- [Phidgets Load Cell Correction](https://www.phidgets.com/?view=articles&article=LoadCellCorrection)

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Hiveeyes community for thermal inertia model inspiration
- P&G for the original use case development

---

**Author**: Jueseok Kim  
**Affiliation**: University of Cincinnati, Mechanical Engineering Ph.D. Candidate
