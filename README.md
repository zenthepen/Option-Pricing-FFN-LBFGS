# Double Heston Model with Jumps: L-BFGS Calibration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements **ultra-precise calibration** of the Double Heston model with jump diffusion to market option prices using the **L-BFGS-B optimization algorithm**. The model captures complex volatility dynamics through two correlated variance factors plus price jumps.

### Key Results

| Method | Mean Error | Runtime | Use Case |
|--------|------------|---------|----------|
| **Pure L-BFGS** | **0.0236%** | 117.8s | ✓ Ultra-precision |
| Hybrid (FFN→L-BFGS) | 0.0257% | 113.9s | Marginal speedup |
| FFN-Only | 11.25% | 0.039s | Fast screening |

**Main Finding**: L-BFGS achieves **477× better accuracy** than FFN-only methods, making it ideal for ultra-high precision calibration scenarios.

## Model Specification

The Double Heston + Jump Diffusion model combines:

1. **Two stochastic variance processes** (fast and slow mean-reverting)
2. **Correlated Brownian motions** (leverage effect)
3. **Poisson jump process** (market crashes/spikes)

### Characteristic Function

The model is priced using the Fourier-cosine (COS) method with the characteristic function:

```
φ(u) = E[exp(iu·log(S_T))] 
     = φ_Heston1(u) · φ_Heston2(u) · φ_Jump(u)
```

**13 parameters** to calibrate:
- Factor 1: `v1_0`, `κ1`, `θ1`, `σ1`, `ρ1`
- Factor 2: `v2_0`, `κ2`, `θ2`, `σ2`, `ρ2`  
- Jumps: `λ`, `μ_j`, `σ_j`

## Installation

```bash
git clone https://github.com/yourusername/Option-Pricing-FFN-LBFGS.git
cd Option-Pricing-FFN-LBFGS
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- TensorFlow >= 2.8.0 (optional, for FFN comparison)

## Usage

### Basic Calibration

```python
from src.calibration.lbfgs_calibrator import DoubleHestonJumpCalibrator

# Define market options (strike, maturity, price, type)
market_options = [
    {'strike': 95, 'maturity': 0.25, 'price': 8.5, 'option_type': 'call'},
    {'strike': 100, 'maturity': 0.25, 'price': 5.2, 'option_type': 'call'},
    # ... more options
]

# Create calibrator
calibrator = DoubleHestonJumpCalibrator(
    spot=100.0,
    risk_free_rate=0.03,
    market_options=market_options
)

# Run calibration (multi-start L-BFGS-B)
result = calibrator.calibrate(maxiter=300, multi_start=3)

print(f"Calibration complete in {result.calibration_time:.1f}s")
print(f"Final error: {result.final_loss * 100:.4f}%")
print(f"Parameters: {result.parameters}")
```

### Option Pricing

```python
from src.models.double_heston import DoubleHeston

# Create model with calibrated parameters
model = DoubleHeston(
    S0=100, K=105, T=0.5, r=0.03,
    v01=0.04, kappa1=2.5, theta1=0.04, sigma1=0.3, rho1=-0.7,
    v02=0.04, kappa2=0.8, theta2=0.04, sigma2=0.2, rho2=-0.5,
    lambda_j=0.15, mu_j=-0.04, sigma_j=0.08,
    option_type='call'
)

price = model.pricing(N=128)  # COS method with 128 terms
print(f"Option price: ${price:.4f}")
```

### Running Tests

```bash
python tests/test_suite.py
```

The comprehensive test suite validates:
- Double Heston pricing correctness (put-call parity)
- Calibration convergence
- Parameter reasonableness
- Data structure integrity

## Repository Structure

```
Option-Pricing-FFN-LBFGS/
├── src/
│   ├── models/
│   │   └── double_heston.py          # COS pricing implementation
│   ├── calibration/
│   │   └── lbfgs_calibrator.py       # L-BFGS-B calibrator
│   └── data/
│       └── synthetic_generator.py    # Training data generator
├── docs/
│   ├── THEORY.md                     # Mathematical foundations
│   ├── METHODOLOGY.md                # Calibration methodology
│   └── LIMITATIONS.md                # Known limitations
├── tests/
│   └── test_suite.py                 # Comprehensive validation
├── results/
│   ├── COMPARISON_TABLE.txt          # Method comparison
│   ├── hybrid_actual_results.json    # Hybrid method results
│   └── lbfgs_actual_results.json     # Pure L-BFGS results
├── requirements.txt
└── README.md
```

## Methodology

### Calibration Pipeline

1. **Parameter Transformation**
   - Positive constraints: `x → exp(x)`
   - Correlation constraints: `x → tanh(x)`

2. **Multi-Start Optimization**
   - 3 initial guesses (literature, perturbed, market-implied)
   - L-BFGS-B with box constraints
   - Best result across all starts

3. **Loss Function**
   ```
   Loss = MSE(relative_errors) + penalty(Feller_condition)
   ```

4. **Convergence Criteria**
   - `ftol = 1e-9` (function tolerance)
   - `gtol = 1e-6` (gradient tolerance)
   - `maxiter = 300`

### Performance

Tested on 15 synthetic option contracts (5 strikes × 3 maturities):

| Metric | Value |
|--------|-------|
| Mean pricing error | 0.0236% |
| Mean runtime | 117.8s |
| Success rate | 100% |
| Mean iterations | 82 |

**Hardware**: Apple M1, 8GB RAM

## Documentation

- **[THEORY.md](docs/THEORY.md)**: Mathematical derivation of characteristic function and COS method
- **[METHODOLOGY.md](docs/METHODOLOGY.md)**: Detailed calibration procedure and parameter constraints
- **[LIMITATIONS.md](docs/LIMITATIONS.md)**: Known issues and recommended best practices

## Results

The `results/` folder contains evaluation data:

- `COMPARISON_TABLE.txt`: Comparison across all methods
- `lbfgs_actual_results.json`: Pure L-BFGS results (5 samples)
- `hybrid_actual_results.json`: Hybrid FFN→L-BFGS results (10 samples)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{double_heston_lbfgs_2026,
  title = {Double Heston Model with Jumps: L-BFGS Calibration},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/Option-Pricing-FFN-LBFGS}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- COS method implementation based on Fang & Oosterlee (2008)
- Double Heston model follows Christoffersen et al. (2009)
- L-BFGS-B algorithm from SciPy's optimize module

## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is research code. Always validate results independently before using in production environments.
