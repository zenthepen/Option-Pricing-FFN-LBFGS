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

