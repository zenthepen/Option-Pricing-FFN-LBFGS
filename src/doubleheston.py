import numpy as np

# Helper for normal CDF without scipy
def norm_cdf(x):
    """Standard normal CDF approximation"""
    return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


class DoubleHeston():
    """
    Double Heston Model for European Option Pricing
    Based on the characteristic function formulation from academic literature
    
    Parameters match standard notation:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - q: Dividend yield (set to 0 if none)
    - v01, v02: Initial variances for factors 1 and 2
    - kappa1, kappa2: Mean reversion speeds
    - theta1, theta2: Long-term variance levels
    - sigma1, sigma2: Volatility of variance (vol-of-vol)
    - rho1, rho2: Correlations between asset and variance
    """
    
    def __init__(self, S0, K, T, r, v01, kappa1, theta1, sigma1, rho1,
                 v02, kappa2, theta2, sigma2, rho2, lambda_j, mu_j, sigma_j, option_type="C", q=0.0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.v01 = v01
        self.kappa1 = kappa1
        self.theta1 = theta1
        self.sigma1 = sigma1
        self.rho1 = rho1
        self.v02 = v02
        self.kappa2 = kappa2
        self.theta2 = theta2
        self.sigma2 = sigma2
        self.rho2 = rho2
        self.option_type = option_type
        self.lambda_j = lambda_j
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def characteristic_function(self, phi, tau):
        """
        Double Heston characteristic function from paper
        
        Parameters:
        -----------
        phi : complex
            Frequency parameter
        tau : float
            Time to maturity
            
        Returns:
        --------
        cf : complex
            Characteristic function value
        """
        i = 1j  # Imaginary unit
        
        # Process 1
        d1 = np.sqrt((self.kappa1 - self.rho1 * self.sigma1 * i * phi)**2 + 
                     self.sigma1**2 * phi * (phi + i))
        
        g1 = (self.kappa1 - self.rho1 * self.sigma1 * i * phi - d1) / \
             (self.kappa1 - self.rho1 * self.sigma1 * i * phi + d1)
        
        B1 = ((self.kappa1 - self.rho1 * self.sigma1 * i * phi - d1) / self.sigma1**2) * \
             ((1 - np.exp(-d1 * tau)) / (1 - g1 * np.exp(-d1 * tau)))
        
        # Process 2
        d2 = np.sqrt((self.kappa2 - self.rho2 * self.sigma2 * i * phi)**2 + 
                     self.sigma2**2 * phi * (phi + i))
        
        g2 = (self.kappa2 - self.rho2 * self.sigma2 * i * phi - d2) / \
             (self.kappa2 - self.rho2 * self.sigma2 * i * phi + d2)
        
        B2 = ((self.kappa2 - self.rho2 * self.sigma2 * i * phi - d2) / self.sigma2**2) * \
             ((1 - np.exp(-d2 * tau)) / (1 - g2 * np.exp(-d2 * tau)))
        
        # Compute A(τ, φ) with jump compensation for risk-neutral measure
        # Jump compensator: exp(μⱼ + 0.5*σⱼ²) - 1 ensures E[S_T] = S_0*exp(rT)
        jump_compensator = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        A = (self.r - self.q - self.lambda_j * jump_compensator) * i * phi * tau
        
        A += (self.kappa1 * self.theta1 / self.sigma1**2) * \
             ((self.kappa1 - self.rho1 * self.sigma1 * i * phi - d1) * tau - 
              2 * np.log((1 - g1 * np.exp(-d1 * tau)) / (1 - g1)))
        
        A += (self.kappa2 * self.theta2 / self.sigma2**2) * \
             ((self.kappa2 - self.rho2 * self.sigma2 * i * phi - d2) * tau - 
              2 * np.log((1 - g2 * np.exp(-d2 * tau)) / (1 - g2)))
        
        # Characteristic function of log(S_T/S_0), not log(S_T)
        # The COS method expects CF of the log-return
        # Jump component: Merton jump-diffusion CF
        cf_jump = np.exp(self.lambda_j * tau * (np.exp(i * phi * self.mu_j - 0.5 * self.sigma_j**2 * phi**2) - 1))
        cf_heston = np.exp(A + B1 * self.v01 + B2 * self.v02)
        
        cf = cf_heston * cf_jump
        return cf

    
    def truncationRange(self, L=10):
        def single_factor_cumulants( v0, kappa, theta, sigma, rho):
                tau = self.T
                lm = kappa
                v_bar = theta
                volvol = sigma
                
                c1 = self.r * tau + (1 - np.exp(-lm * tau)) * (v_bar - v0)/(2 * lm) - v_bar * tau / 2
                
                c2 = 1/(8 * np.power(lm, 3)) * (
                    volvol * tau * lm * np.exp(-lm * tau) * (v0 - v_bar) * (8 * lm * rho - 4 * volvol) +
                    lm * rho * volvol * (1 - np.exp(-lm * tau)) * (16 * v_bar - 8 * v0) +
                    2 * v_bar * lm * tau * (-4 * lm * rho * volvol + np.power(volvol, 2) + 4 * np.power(lm, 2)) +
                    np.power(volvol, 2) * ((v_bar - 2 * v0) * np.exp(-2 * lm * tau) + 
                                        v_bar * (6 * np.exp(-lm * tau) - 7) + 2 * v0) +
                    8 * np.power(lm, 2) * (v0 - v_bar) * (1 - np.exp(-lm * tau))
                )

                return c1, c2
                
        c1_f1, c2_f1 = single_factor_cumulants(self.v01, self.kappa1, self.theta1, self.sigma1, self.rho1)
        c1_f2, c2_f2 = single_factor_cumulants(self.v02, self.kappa2, self.theta2, self.sigma2, self.rho2)

        c1_jump = self.lambda_j * self.T * self.mu_j
        c2_jump = self.lambda_j * self.T * (self.sigma_j**2 + self.mu_j**2)


            
            # Combine cumulants (independent factors add)
        c1_total = c1_f1 + c1_f2 + c1_jump
        c2_total = c2_f1 + c2_f2 + c2_jump
            
            # Truncation range
        a = c1_total - L * np.sqrt(np.abs(c2_total))
        b = c1_total + L * np.sqrt(np.abs(c2_total))
            
            # Ensure strike is within range (in log-return space)
        log_K = np.log(self.K / self.S0)
        a = min(a, log_K - 0.1)
        b = max(b, log_K + 0.1)
            
        return a, b

    def chi_k(self,k,c,d,a,b):
        if k == 0:
            return np.exp(d) - np.exp(c)
        
        u = k * np.pi / (b - a)
        term1 = np.cos(u * (d - a)) * np.exp(d)
        term2 = np.cos(u * (c - a)) * np.exp(c)
        term3 = u * np.sin(u * (d - a)) * np.exp(d)
        term4 = u * np.sin(u * (c - a)) * np.exp(c)
        
        return (1.0 / (1 + u**2)) * (term1 - term2 + term3 - term4)
        
    def psi_k(self,k,c,d,a,b):
        if k == 0:
            return d - c
        
        u = k * np.pi / (b - a)
        return (1.0 / u) * (np.sin(u * (d - a)) - np.sin(u * (c - a)))

    def pricing(self, N=128):
        # For COS method, work in log-return space x = log(S_T/S_0)
        # so log(K) becomes log(K/S0)
        log_K = np.log(self.K / self.S0)
        a, b = self.truncationRange()

        k_values = np.arange(N)
        u_values = k_values * np.pi / (b - a)

        # CF of log-return x = log(S_T/S_0)
        phi_k = np.array([self.characteristic_function(u, self.T) for u in u_values])

        V_k = np.zeros(N)
        if self.option_type == "C":
            # Call payoff: max(S_T - K, 0) = max(S0*exp(x) - K, 0)
            # = S0*exp(x) - K for x > log(K/S0)
            for k in range(N):
                chi = self.chi_k(k, c=log_K, d=b, a=a, b=b)
                psi = self.psi_k(k, c=log_K, d=b, a=a, b=b)
                V_k[k] = (2.0 / (b - a)) * (self.S0 * chi - self.K * psi)
        else:
            # Put payoff: max(K - S_T, 0) = max(K - S0*exp(x), 0)
            # = K - S0*exp(x) for x < log(K/S0)
            for k in range(N):
                chi = self.chi_k(k, c=a, d=log_K, a=a, b=b)
                psi = self.psi_k(k, c=a, d=log_K, a=a, b=b)
                V_k[k] = (2.0 / (b - a)) * (self.K * psi - self.S0 * chi)
        
        # COS summation
        summands = np.real(phi_k * np.exp(-1j * u_values * a)) * V_k
        summands[0] *= 0.5
        
        option_price = np.exp(-self.r * self.T) * np.sum(summands)
        
        return option_price



if __name__ == "__main__":
    print("="*70)
    print("DOUBLE HESTON + JUMP DIFFUSION MODEL - PRICING DEMONSTRATION")
    print("="*70)
    
    # Market parameters
    s = 100
    k = 100
    t = 1.0
    r = 0.05
    
    # Double Heston parameters
    v1 = 0.04
    kappa1 = 2.0
    theta1 = 0.04
    sigma1 = 0.3
    rho1 = -0.5
    
    v2 = 0.04
    kappa2 = 1.5
    theta2 = 0.04
    sigma2 = 0.2
    rho2 = -0.3
    
    print(f"\nMarket Parameters:")
    print(f"  Spot Price (S):          ${s}")
    print(f"  Strike Price (K):        ${k}")
    print(f"  Time to Maturity (T):    {t} year(s)")
    print(f"  Risk-Free Rate (r):      {r*100:.2f}%")
    
    print(f"\nDouble Heston Parameters:")
    print(f"  Factor 1 (Fast Mean-Reverting):")
    print(f"    v1={v1}, κ1={kappa1}, θ1={theta1}, σ1={sigma1}, ρ1={rho1}")
    print(f"  Factor 2 (Slow Mean-Reverting):")
    print(f"    v2={v2}, κ2={kappa2}, θ2={theta2}, σ2={sigma2}, ρ2={rho2}")
    
    # ============================================
    # JUMP DIFFUSION PARAMETERS (NEW)
    # ============================================
    lambda_j = 0.5      # 0.5 jumps per year on average
    mu_j = -0.05        # Negative mean jump (downward jumps)
    sigma_j = 0.10      # Jump volatility
    
    print(f"\nJump Diffusion Parameters:")
    print(f"  λ (Jump Intensity):     {lambda_j} jumps/year")
    print(f"  μⱼ (Mean Jump Size):    {mu_j*100:.2f}%")
    print(f"  σⱼ (Jump Volatility):   {sigma_j*100:.2f}%")
    
    # ============================================
    # PRICING WITH JUMPS
    # ============================================
    print(f"\n" + "-"*70)
    print("PRICING WITH JUMP DIFFUSION:")
    print("-"*70)
    
    call_model = DoubleHeston(s, k, t, r, v1, kappa1, theta1, sigma1, rho1,
                              v2, kappa2, theta2, sigma2, rho2,
                              lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j,
                              option_type='C')
    call_price_jumps = call_model.pricing(N=128)
    
    put_model = DoubleHeston(s, k, t, r, v1, kappa1, theta1, sigma1, rho1,
                            v2, kappa2, theta2, sigma2, rho2,
                            lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j,
                            option_type='P')
    put_price_jumps = put_model.pricing(N=128)
    
    print(f"  Call Price: ${call_price_jumps:.4f}")
    print(f"  Put Price:  ${put_price_jumps:.4f}")
    
    # ============================================
    # COMPARE WITH NO-JUMP MODEL
    # ============================================
    print(f"\n" + "-"*70)
    print("IMPACT OF JUMPS:")
    print("-"*70)
    
    call_no_jump = DoubleHeston(s, k, t, r, v1, kappa1, theta1, sigma1, rho1,
                                v2, kappa2, theta2, sigma2, rho2,
                                lambda_j=0.0, mu_j=0.0, sigma_j=0.0,
                                option_type='C')
    call_price_no_jump = call_no_jump.pricing(N=128)
    
    put_no_jump = DoubleHeston(s, k, t, r, v1, kappa1, theta1, sigma1, rho1,
                              v2, kappa2, theta2, sigma2, rho2,
                              lambda_j=0.0, mu_j=0.0, sigma_j=0.0,
                              option_type='P')
    put_price_no_jump = put_no_jump.pricing(N=128)
    
    print(f"  Without Jumps:")
    print(f"    Call: ${call_price_no_jump:.4f}")
    print(f"    Put:  ${put_price_no_jump:.4f}")
    
    print(f"\n  With Jumps:")
    print(f"    Call: ${call_price_jumps:.4f}")
    print(f"    Put:  ${put_price_jumps:.4f}")
    
    print(f"\n  Jump Premium:")
    print(f"    Call: ${call_price_jumps - call_price_no_jump:.4f}")
    print(f"    Put:  ${put_price_jumps - put_price_no_jump:.4f}")
    
    # ============================================
    # PUT-CALL PARITY CHECK
    # ============================================
    print(f"\n" + "-"*70)
    print("PUT-CALL PARITY CHECK (WITH JUMPS):")
    print("-"*70)
    pcp_left = call_price_jumps - put_price_jumps
    pcp_right = s - k * np.exp(-r * t)
    pcp_diff = abs(pcp_left - pcp_right)
    
    print(f"  C - P            = ${pcp_left:.4f}")
    print(f"  S - K*e^(-rT)    = ${pcp_right:.4f}")
    print(f"  Difference       = ${pcp_diff:.4f}")
    print(f"  Status           : {'✓ PASS' if pcp_diff < 0.01 else '✗ FAIL'}")
    
    print(f"\n" + "="*70)
    print("END OF DEMONSTRATION")
    print("="*70)
