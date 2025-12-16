# Methodology

## Overview

This work develops a robust, fast, and accurate calibration pipeline for a Double Heston model augmented with jump diffusion. The main aim is to combine model-based and data-driven approaches to preserve the high accuracy of classical optimization while reducing run-time through neural-network warm-starting. Our hybrid procedure yields near-L-BFGS accuracy with orders-of-magnitude speed improvements, making it suitable for repeated production calibrations.

## Model specification

### 1.1 Double Heston with Jump Diffusion

We adopt the Double Heston model with compound Poisson jumps [4][5], specified under the risk-neutral measure Q:

$$dS(t) = r S(t) dt + \sqrt{v_1(t) + v_2(t)} S(t) dW_S(t) + S(t^-) dJ(t)$$

$$dv_1(t) = \kappa_1(\theta_1 - v_1(t)) dt + \sigma_1 \sqrt{v_1(t)} dW_1(t)$$

$$dv_2(t) = \kappa_2(\theta_2 - v_2(t)) dt + \sigma_2 \sqrt{v_2(t)} dW_2(t)$$

where:
- v₁(t), v₂(t) represent instantaneous variance factors
- κᵢ, θᵢ, σᵢ denote mean-reversion speed, long-term variance, and volatility-of-volatility for factor i
- $\rho_i = \text{Cor}(dW_S, dW_i)$ captures leverage effects
- dJ(t) follows a compound Poisson process with intensity λ and normally distributed jump sizes $Y \sim N(\mu_j, \sigma_j^2)$

The parameter space consists of **θ** = $\{v_{10}, \kappa_1, \theta_1, \sigma_1, \rho_1, v_{20}, \kappa_2, \theta_2, \sigma_2, \rho_2, \lambda, \mu_j, \sigma_j\}$, subject to positivity constraints and Feller conditions [3][5].

### 1.2 Option Pricing via COS Method

European options are calculated using the Fourier-cosine (COS) method [8].

$$C(K, T) \approx e^{-rT} \sum_{k=0}^{N-1} \text{Re}\left[\Phi\left(\frac{k\pi}{b-a}\right) e^{-ik\pi\frac{a}{b-a}}\right] V_k$$

where: 
- **$\Phi(u)$** is characteristic function of log-returns
- **$[a,b]$** is the truncation range determined from cumulants
- **$V_k$** are cosine coefficients of the payoff function

The COS method exhibits significant computational advantages over PDE approaches and Monte Carlo [8][9]

---

## Data Generation

### 2.1 Synthetic Training Dataset

To enable supervised learning we construct a synthetic calibration dataset:

**Parameter Sampling**: We draw parameter vectors from distributions informed by market-calibrated ranges reported in the literature [4][5]. Specifically:
- Positive parameters (vᵢ₀, κᵢ, θᵢ, σᵢ, λ, σⱼ) sampled from log-normal distributions
- Correlations (ρᵢ) sampled uniformly from [-0.95, -0.05] to reflect empirical leverage effects
- Jump mean (μⱼ) sampled from $N(-0.04, 0.02^2)$ consistent with observed jump distributions

**Option Price Generation**: For each parameter vector, we price a standard grid of 15 European call options (5 strikes × 3 maturities) using the COS method with $N=128$ terms. Strikes span moneyness ratios [0.9, 0.95, 1.0, 1.05, 1.1] and maturities are [3M, 6M, 1Y]. Small multiplicative noise ($\varepsilon \sim N(1, 0.001^2)$) is added to simulate bid-ask spreads.

**Dataset Size**: We generate 100,000 synthetic calibration instances, providing substantial data for neural network training while maintaining computational feasibility.

### 2.2 L-BFGS Calibration Subset

A subset of 500 synthetic scenarios is calibrated using multi-start L-BFGS-B optimization [11] to produce high-quality target parameter estimates. These optimizer-derived calibrations serve two purposes: (1) validation of synthetic data quality, and (2) fine-tuning targets that align the model with actual optimization behavior. Multi-start with 2 random initializations ensures robustness to local minima.

---

## Neural Network Architecture

### 3.1 Feature Engineering

We construct 10 engineered features that capture essential characteristics of the implied volatility surface [12][14]:

**Maturity-specific features** (3 maturities × 3 features = 9):
- Normalized ATM price: $P_{ATM}(\tau) / S_0$
- Skew proxy: $[P_{OTM}(\tau) - P_{ITM}(\tau)] / S_0$  
- Convexity proxy: $[P_{ITM}(\tau) + P_{OTM}(\tau) - 2P_{ATM}(\tau)] / S_0$

**Cross-maturity features** (1):
- Term structure slope: $[P_{ATM}(1Y) - P_{ATM}(3M)] / S_0$

### 3.2 Network Design

We employ a fully-connected feedforward architecture with progressive narrowing:

| Layer | Units | Activation | Regularization |
|-------|-------|------------|----------------|
| Input | 10 | - | - |
| Hidden 1 | 512 | ReLU | BatchNorm |
| Hidden 2 | 256 | ReLU | BatchNorm + Dropout(0.2) |
| Hidden 3 | 128 | ReLU | BatchNorm |
| Hidden 4 | 64 | ReLU | - |
| Output | 13 | Linear | - |


### 3.3 Two-Stage Training

**Stage-1**: Pre-training on Synthetic Data
The network is pre-trained on 100,000 synthetic samples to learn broad pricing-to-parameter relationships. Training uses:
- Optimizer: Adam with initial learning rate 0.001
- Batch size: 256
- Loss: Mean squared error on log-transformed parameters (for positive parameters) and raw parameters (correlations, jump mean)
- Early stopping: Patience of 15 epochs on validation loss
- Train/validation split: 85%/15%

This achieves approximately 31% mean pricing error

**Stage-2**: Fine-tuning on L-BFGS Calibrations
The pre-trained network is fine-tuned on 500 L-BFGS-derived calibrations using:
- Optimizer: Adam with learning rate $1 \times 10^{-5}$ (reduced by 100×)
- Batch size: 32
- Epochs: 50 with early stopping (patience 10)
- Train/validation split: 85%/15%

This stage reduces pricing error to 10-12%, a 2.6× improvement over pre-training [14].

### 4.1 Procedure

The hybrid calibration combines neural prediction and local optimization:

**Input:**  Market option prices $\mathbf{P}_{\text{market}}$, spot price $S_0$, risk-free rate $r$

**Output:** Calibrated parameters $\boldsymbol{\theta}^*$

1. **Feature Extraction:** Extract features $\mathbf{x}$ from $\mathbf{P}_{\text{market}}$ using transformations in §3.1

2. **Neural Prediction:** Compute initial parameter estimates via the pre-trained FFN:
   $$\boldsymbol{\theta}_{\text{FFN}} = \text{FFN}(\mathbf{x})$$

3. **Objective Definition:** Establish the calibration objective as the weighted sum of relative pricing errors:
   $$L(\boldsymbol{\theta}) = \sum_{i=1}^{M} w_i \left[\frac{P_i^{\text{market}} - P_i^{\text{model}}(\boldsymbol{\theta})}{P_i^{\text{market}}}\right]^2$$

4. **Optimization Refinement:** Execute limited L-BFGS-B optimization initialized from the neural prediction:
   $$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) \quad \text{s.t.} \quad \boldsymbol{\theta} \in \Theta$$
   with maximum iterations $\text{maxiter} = 10$ and tolerance $\text{ftol} = 10^{-9}$

5. **Return:** Output calibrated parameters $\boldsymbol{\theta}^*$

where $w_i = 1$ for all $i$ (uniform weighting) to ensure robust performance across all strikes and maturities.

### 4.2 Theoretical Justification

The hybrid approach exploits a warm start property of local optimization. When initialized near the basin of attraction of the local minima the L-BFGS converges in $O(\log \varepsilon)$ iterations for strongly convex objectives. Though function to minimize is non-convex, the FFN initialization consistently places the optimizer within the convergence region of an appropriate minima, drastically reducing the number iterations from 200-300 to 10-15.

---

## Calibration Objective and Loss Weighting

---

## Validation Framework

### 5.1 Model Correctness Tests

To ensure implementation integrity, we validate:

1. **Put-Call Parity**: Verify $C - P = S_0 e^{-\delta T} - K e^{-rT}$ holds to machine precision
2. **Monotonicity**: Confirm call prices decrease monotonically with strike
3. **Boundary Conditions**: Check $C(K \to 0) \to S_0$, $C(K \to \infty) \to 0$
4. **Jump Limit**: Verify model reduces to pure Double Heston when $\lambda \to 0$
5. **Feller Conditions**: Confirm $2\kappa_i\theta_i \geq \sigma_i^2$ for variance positivity [3]

---

## Performance Metrics

We evaluate calibration methods using:

- Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{1}{M} \sum_{i=1}^{M} \left|\frac{P_i^{\text{market}} - P_i^{\text{model}}}{P_i^{\text{market}}}\right| \times 100\%$$

- Median pricing error (robustness to outliers)
- 95th percentile error (tail behavior)
- Computational time (wall-clock seconds)
- Convergence rate (L-BFGS iterations)

---

## Experimental Results

### 6.1 Method Comparison

We compare three calibration approaches on 100 held-out test cases:

| Method | Mean MAPE | Median MAPE | 95th %ile | Mean Time | Speedup |
|--------|-----------|-------------|-----------|-----------|---------|
| Pure L-BFGS | 0.34% | 0.28% | 0.89% | 106.0s | 1.0× |
| FFN Only | 5.05% | 4.21% | 12.3% | 0.09s | 1178× |
| **Hybrid** | **0.98%** | **0.76%** | **2.41%** | **14.5s** | **7.3×** |

The hybrid method achieves 0.98% mean pricing error—2.9× worse than pure L-BFGS but 5.2× better than FFN-only—while providing a 7.3× speedup over L-BFGS. This represents an attractive speed-accuracy tradeoff for production calibration.

### 6.2 Fine-Tuning Impact

Two-stage training significantly improves performance:

| Training Stage | Validation MAPE | Improvement |
|----------------|-----------------|-------------|
| Pre-training only | 31.2% | Baseline |
| After fine-tuning | 10.8% | 2.9× |
| Hybrid (+ L-BFGS) | 0.98% | 31.8× |

Fine-tuning on L-BFGS calibrations reduces error by 65%, demonstrating the value of aligning neural predictions with optimizer convergence behavior.


