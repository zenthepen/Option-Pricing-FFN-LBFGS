# Methodology

The fundamental problem in quantitative finance is pricing derivatives accurately. Starting with the ground breaking work of Black and Scholes, we know that under certain assumptions we can find a closed form soultion of the option prices. But real market shows significant departure from these assumptions.

### 1.1 Black-Scholes Framework

The Black-Scholes model assumes the stock price S(t) follows:

$$dS(t) = \mu S(t) dt + \sigma S(t) dW(t)$$

Under the risk-neutral measure Q, this becomes:

$$dS(t) = r S(t) dt + \sigma S(t) dW(t)$$

where r is the risk-free rate and σ is constant volatility.

The resulting European call option price is given by the famous Black-Scholes formula [1]:

$$C(S, K, T) = S N(d_1) - K e^{-rT} N(d_2)$$

**Limitation**: The constant volatility assumption is unrealistic. Empirical studies show that implied volatility varies with strike price and maturity, forming the volatility smile/smirk [2].

## 2. Stochastic Volatility Models

To capture volatility smiles, researchers developed stochastic volatility models, where volatility is a random process.

### 2.1 The Heston Model

Heston[3] proposed a single factor SVM where the stock returns and the volatility are correlated

$$dS(t) = \mu S(t) dt + \sqrt{v(t)} S(t) dW_S(t)$$

$$dv(t) = \kappa(\theta - v(t)) dt + \sigma_v \sqrt{v(t)} dW_v(t)$$

where:
- **v(t)**: instantaneous variance (stochastic)
- **κ**: mean-reversion speed
- **θ**: long-term variance
- **σ_v**: volatility of volatility
- **ρ**: correlation between asset and variance shocks

**Parameters to calibrate**: {v₀, κ, θ, σ_v, ρ} (5 parameters)

**Advantages**:
✓ Closed-form characteristic function [3]
✓ Captures volatility smile
✓ Tractable for calibration

**Limitation**: Single volatility factor cannot capture the full complexity of the volatility surface, particularly its term structure dynamics [4].

### 2.2 Multi-factor SVM

Christoffersen et al. [4] shoe that **two-factor stochastic volatility models work particularly well** in explaining term structure and shape of the index option.

The **Double Heston model** adds a second volatility factor [4][5]:

$$dS(t) = r S(t) dt + \sqrt{v_1(t) + v_2(t)} S(t) dW_S(t)$$

$$dv_1(t) = \kappa_1 (\theta_1 - v_1(t)) dt + \sigma_1 \sqrt{v_1(t)} dW_1(t)$$

$$dv_2(t) = \kappa_2 (\theta_2 - v_2(t)) dt + \sigma_2 \sqrt{v_2(t)} dW_2(t)$$

where:
- **Factor 1**: Fast mean-reverting (typically κ₁ >> κ₂)
- **Factor 2**: Slow mean-reverting (controls long-term volatility levels)

**Parameters**: {v₁₀, κ₁, θ₁, σ₁, ρ₁, v₂₀, κ₂, θ₂, σ₂, ρ₂} (10 parameters)

**Benefits**:
✓ Better captures volatility surface dynamics [4]
✓ Improved fit across strikes and maturities
✓ Still tractable with semi-analytical methods

## 3. Jump-Diffusion Models

### 3.1 Adding Jumps to Stock Prices

Merton[6] recognized that prices are not continous, rather they experience discontinous jumps.

$$dS(t) = \mu S(t) dt + \sigma S(t) dW(t) + S(t-) dJ(t)$$

where **dJ(t)** is a **compound Poisson process** [6]:

$$dJ(t) = \sum_{i=1}^{N(t)} (Y_i - 1) \delta(t - \tau_i)$$

where:
- **N(t)**: Poisson process with intensity λ (jump frequency)
- **Y_i**: jump size, typically lognormal or normal
- **τᵢ**: jump arrival times

**Jump Parameters**:
- **λ**: jump intensity (jumps per year)
- **μⱼ**: mean jump size
- **σⱼ**: jump volatility

Jump diffusion models capture the fat tails and steep skew observed in equity option prices, especially for out-of-the-money puts.

### 3.2 Double Heston + Jump Diffusion

Combining multifactor SVM and Jump diffusion model, we get:

$$dS(t) = r S(t) dt + \sqrt{v_1(t) + v_2(t)} S(t) dW_S(t) + S(t-) dJ(t)$$

$$dv_1(t) = \kappa_1 (\theta_1 - v_1(t)) dt + \sigma_1 \sqrt{v_1(t)} dW_1(t)$$

$$dv_2(t) = \kappa_2 (\theta_2 - v_2(t)) dt + \sigma_2 \sqrt{v_2(t)} dW_2(t)$$

**Total Parameters**: **13 parameters** [5][7]

**Constraints** (Feller conditions) [3][5]:

$$2\kappa_i \theta_i \geq \sigma_i^2, \quad i = 1,2$$

This ensures variance remains positive.

---

## 4. Option Pricing Methods

### 4.1 Characteristic Function Approach

For the Heston and Double Heston models, closed-form solutions don't exist, but the **characteristic function** is semi-analytical [3]:

$$\Phi(u; S, v_1, v_2, \tau) = E[e^{iu \ln S(\tau)}] = e^{A(\tau,u) + B_1(\tau,u)v_1 + B_2(\tau,u)v_2 + iu\ln S}$$

where A, B₁, B₂ satisfy a system of ODEs [3][4].

Option prices can be recovered via Fourier inversion:

$$C(S, K, T) = e^{-rT} \int_0^\infty v(x) f(x) dx$$

where f(x) is the log-return density and v(x) is the payoff function.

### 4.2 Fourier-Cosine (COS) Method

Instead of FFT, we use the more efficient **COS method** of Fang and Oosterlee [8]:

$$C(S, K, T) \approx e^{-rT} \sum_{k=0}^{N-1} \text{Re}\left[\Phi\left(\frac{k\pi}{b-a}\right) e^{-ik\pi\frac{a}{b-a}}\right] V_k$$

where:

- **[a,b]**: Truncation Range
- **Φ(u)**: characteristic function
- **Vₖ**: cosine coefficients of the payoff
- **N**: number of terms

**Advantages** [8][9]:

Exponential convergence rate (O(e^{-αN}))
Handles European and American options
Natural truncation of integration domain
Much faster than FFT [8][9]

We use N = 128 for our project 

---

## 5. The Calibration Problem

### 5.1 Problem Formulation

Given the market option prices $\{P_i^{market}\}_{i=1}^M$ across M options, we need to find paramters **θ** which minimizes pricing error:

$$\min_{\theta} L(\theta) = \sum_{i=1}^M w_i \left[\frac{P_i^{market} - P_i^{model}(\theta)}{P_i^{market}}\right]^2$$

subject to:

$$\theta \in \Theta = \{\text{parameter space satisfying Feller + bounds}\}$$

where:

**θ** = 13 paramters

### 5.2 Traditional Optimization: L-BFGS

This approach uses **L-BFGS-B** (quasi-Newton method with bounds) [11]:

$$\theta_{t+1} = \theta_t - \alpha_t H_t^{-1} \nabla L(\theta_t)$$

where:
- **H_t**: Limited-memory Hessian approximation
- **α_t**: Step size from line search

**Performance**: Achieves <1% pricing error but requires **100-300 function evaluations**, taking **2-5 minutes** [11].

---

## 6. Deep Learning for Model Calibration

### 6.1 Neural Networks as Surrogate Models

Recent research shows that neural networks can learn to map market observables to parameters [12][13][14].

**Key Insight from Hernandez (2016)** [12]:
> "Neural networks can effectively learn the inverse mapping from option prices to model parameters, achieving ~1000x speedup with acceptable accuracy loss."

**Approach**:
1. Train FFN on synthetic data: option prices → parameters
2. Input: 10 features extracted from option prices
3. Output: 13 calibrated parameters

### 6.2 Feature Engineering

We extract 10 robust features that capture option surface structure [12][14]:

**Maturity-Specific Features** (for each of 3 maturities):
- **ATM volatility**: Normalized ATM option price
- **Skew**: Risk reversal (OTM call - OTM put)
- **Curvature**: Butterfly (steep wings relative to ATM)

**Across All Maturities**:
- **Term structure slope**: ATM price difference between longest and shortest maturity
- **Total premium**: Sum of ATM prices across maturities

These 10 features capture the essential **volatility smile shape and term structure** [4].

### 6.3 Network Architecture

From Bloch (2019) and Hernandez (2016) [12][14]:

graph TD
    A["<b>Input Layer</b><br/>10 Features<br/>━━━━━━━━━<br/>ATM Vol × 3<br/>Skew × 3<br/>Curvature × 3<br/>Term Slope<br/>Total Premium"]
    
    B["<b>Dense Layer 1</b><br/>Units: 512<br/>Activation: ReLU<br/>BatchNormalization"]
    
    C["<b>Dense Layer 2</b><br/>Units: 256<br/>Activation: ReLU<br/>BatchNormalization<br/>Dropout: 0.2"]
    
    D["<b>Dense Layer 3</b><br/>Units: 128<br/>Activation: ReLU<br/>BatchNormalization"]
    
    E["<b>Dense Layer 4</b><br/>Units: 64<br/>Activation: ReLU"]
    
    F["<b>Output Layer</b><br/>Units: 13<br/>Activation: Linear<br/>━━━━━━━━━<br/>v₁₀, κ₁, θ₁, σ₁, ρ₁<br/>v₂₀, κ₂, θ₂, σ₂, ρ₂<br/>λ, μⱼ, σⱼ"]
    
    A -->|Feed Forward| B
    B -->|Feed Forward| C
    C -->|Feed Forward| D
    D -->|Feed Forward| E
    E -->|Feed Forward| F
    
    style A fill:#e1f5ff,stroke:#01579b,stroke-width:2px,color:#000
    style B fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style C fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style E fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000
    style F fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000



### 6.4 Two-Stage Training Strategy

Following Zadgar et al. (2025) [14]:

1) **Pre-training on Synthetic Data**
- Generate 100,000 synthetic market scenarios
- Train FFN on full parameter space
- Result: Learn general pricing relationships
- Error: 31%

2) **Fine-Tuning on real calibrations**
- Generate 500 L-BFGS calibrations on real market data patterns
- Fine-tune with very low learning rate (1e-5)
- Result: Learns market-specific parameter distributions
- Error: ~10-12% (2.5x improvement)

### 6.5 Parameter Identifiability and Deep Learning

Multiple parameters giving the same price is called the Parameter Identifiability problem.
If many parameter combinations price identically, the FFN learns to map prices to a **representative point** in the parameter equivalence class. When refined with L-BFGS, this warm start is already near the true calibration [14].

This explains why FFN pricing error is low (5%) even when parameter recovery error is high (100%+).

---

## 7. Hybrid Calibration: FFN + L-BFGS

### 7.1 The Warm-Start Idea

Combining FFN's speed with L-BFGS's accuracy [14]:

1. **FFN generates initial guess** 
2. **L-BFGS refines from this guess** 

**vs Pure L-BFGS**:
- Cold start: 200-300 iterations
- Warm start: 10-15 iterations

### 7.2 Mathematical Justification

When starting from a good initial guess in a smooth optimization landscape:

$$\theta^* = \theta_0 - \nabla^2 L(\theta_0)^{-1} \nabla L(\theta_0) + O(\|\theta_0 - \theta^*\|^2)$$

A high-quality initialization (from FFN) greatly reduces the number of Newton-Raphson iterations needed [14].

### 7.3 Expected Performance

Based on Hernandez (2016), Bloch (2019), and Zadgar et al. (2025) [12][13][14]:

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| Pure L-BFGS | 106s | <0.5% | Validation, research |
| Pure FFN | 0.09s | 5-6% | Real-time screening |
| **Hybrid** | **14.5s** | **0.98%** | **Production (RECOMMENDED)** |

---

## 8. Implementation Details

### 8.1 COS Method for Double Heston + Jumps

The characteristic function for our model [8][9]:

$$\Phi(u) = \exp\left(A(u) + B_1(u)v_1 + B_2(u)v_2 + C(u) + iu\ln S\right)$$

where:
- A(u), B₁(u), B₂(u): Solve ODEs from Heston dynamics [3]
- C(u): Adjustment for jump component [6]

Truncation range [a,b] selected using generalized method of moments [8].

### 8.2 Parameter Bounds and Constraints

**Box constraints**:
- v₁₀, v₂₀, κ₁, κ₂, θ₁, θ₂ ∈ (0, ∞)
- σ₁, σ₂ ∈ (0, ∞)
- ρ₁, ρ₂ ∈ [-0.99, 0.99]
- λ ∈ (0, ∞)
- μⱼ ∈ (-∞, ∞)
- σⱼ ∈ (0, ∞)

**Feller conditions** [3][5]:
- 2κ₁θ₁ ≥ σ₁² (strict positivity of variance)
- 2κ₂θ₂ ≥ σ₂²

---

## 9. Prior Work and Our Contribution

### 9.1 Evolution of the Field

1. **Black-Scholes (1973)** [1]: Constant volatility
2. **Heston (1993)** [3]: Single-factor stochastic vol
3. **Christoffersen et al. (2009)** [4]: Multi-factor importance
4. **Merton (1976) / Agazzotti et al. (2025)** [6][7]: Jumps in equity models
5. **Fang & Oosterlee (2008)** [8]: Fast COS pricing
6. **Hernandez (2016)** [12]: Early neural network calibration
7. **Bloch (2019)** [14]: Deep learning IV surfaces
8. **Zadgar et al. (2025)** [14]: Deep learning + Heston framework

### 9.2 Our Contribution

We extend Zadgar et al. (2025) [14] from single-factor Heston to **Double Heston + Jump Diffusion** with a **hybrid FFN→L-BFGS approach**

## 10. References

[1] Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.

[2] Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. John Wiley & Sons.

[3] Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *The Review of Financial Studies*, 6(2), 327–343.

[4] Christoffersen, P., Heston, S., & Jacobs, K. (2009). "The Shape and Term Structure of the Index Option Smirk: Why Multifactor Stochastic Volatility Models Work So Well." *Management Science*, 55(12), 1914–1932.

[5] Mehrdoust, F., Noorani, I., & Hamdi, A. (2021). "Calibration of the Double Heston Model and an Analytical Formula in Pricing American Put Option." *Journal of Computational and Applied Mathematics*, 392, 113422.

[6] Merton, R. C. (1976). "Option Pricing When Underlying Stock Returns are Discontinuous." *Journal of Financial Economics*, 3(1-2), 125–144.

[7] Agazzotti, G., Aglieri Rinella, C., Aguilar, J.-P., & Kirkby, J. L. (2025). "Calibration and Option Pricing with Stochastic Volatility and Double Exponential Jumps." *arXiv preprint* arXiv:2502.13824.

[8] Fang, F., & Oosterlee, C. W. (2008). "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions." *SIAM Journal on Scientific Computing*, 31(2), 826–848.

[9] Le Floc'h, F. (2014). "Fourier Integration and Stochastic Volatility Calibration." Working Paper.

[10] Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

[11] Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). "A Limited Memory Algorithm for Bound Constrained Optimization." *SIAM Journal on Scientific Computing*, 16(5), 1190–1208.

[12] Hernandez, A. (2016). "Model Calibration with Neural Networks." Available at SSRN 2812140.

[13] Bloch, D. A. (2019). "Neural Networks Based Dynamic Implied Volatility Surface." Available at SSRN 3492662.

[14] Zadgar, A., Fallah, S., & Mehrdoust, F. (2025). "Deep Learning-Enhanced Calibration of the Heston Model: A Unified Framework." Working Paper, University of Guilan.

