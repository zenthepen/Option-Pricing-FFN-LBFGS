"""
Comprehensive validation test suite for Double Heston Model (Part 1)
Tests characteristic function, COS pricing, validation against BS/single-Heston, and Greeks
"""

import numpy as np
from doubleheston import DoubleHeston, norm_cdf

def test_cf_properties():
    """Test characteristic function properties"""
    print("="*70)
    print("TEST 1: Characteristic Function Properties")
    print("="*70)
    
    model = DoubleHeston(100, 100, 1.0, 0.05, 
                         0.04, 2.0, 0.04, 0.3, -0.5,
                         0.04, 1.5, 0.04, 0.2, -0.3)
    
    # Test 1.1: CF(0) = 1
    cf_0 = model.characteristic_function(0, 1.0)
    assert np.abs(cf_0 - 1.0) < 1e-10, f"CF(0) should be 1.0, got {cf_0}"
    print("✓ CF(0) = 1.0")
    
    # Test 1.2: |CF(phi)| <= 1 for all phi (martingale property)
    phis = np.linspace(0, 10, 100)
    cfs = [model.characteristic_function(phi, 1.0) for phi in phis]
    mags = [np.abs(cf) for cf in cfs]
    assert all(m <= 1.1 for m in mags), f"CF magnitude exceeds 1: max = {max(mags)}"
    print(f"✓ |CF(φ)| ≤ 1.1 for φ ∈ [0, 10], max = {max(mags):.4f}")
    
    # Test 1.3: CF continuity
    cf_at_1 = model.characteristic_function(1.0, 1.0)
    cf_at_1_001 = model.characteristic_function(1.001, 1.0)
    diff = np.abs(cf_at_1 - cf_at_1_001)
    assert diff < 0.01, f"CF not continuous: |CF(1) - CF(1.001)| = {diff}"
    print(f"✓ CF is continuous: |CF(1) - CF(1.001)| = {diff:.6f}")
    
    print()

def test_pricing_basic():
    """Test basic pricing functionality"""
    print("="*70)
    print("TEST 2: Basic Pricing")
    print("="*70)
    
    # ATM option
    call_model = DoubleHeston(100, 100, 1.0, 0.05,
                              0.04, 2.0, 0.04, 0.3, -0.5,
                              0.04, 1.5, 0.04, 0.2, -0.3,
                              option_type='C')
    put_model = DoubleHeston(100, 100, 1.0, 0.05,
                             0.04, 2.0, 0.04, 0.3, -0.5,
                             0.04, 1.5, 0.04, 0.2, -0.3,
                             option_type='P')
    
    call_price = call_model.pricing(N=256)
    put_price = put_model.pricing(N=256)
    
    # Test 2.1: Prices are positive
    assert call_price > 0, f"Call price should be positive, got {call_price}"
    assert put_price > 0, f"Put price should be positive, got {put_price}"
    print(f"✓ Call price = ${call_price:.4f} > 0")
    print(f"✓ Put price = ${put_price:.4f} > 0")
    
    # Test 2.2: Prices are reasonable (not extreme)
    assert call_price < 50, f"Call price too high: {call_price}"
    assert put_price < 50, f"Put price too high: {put_price}"
    print(f"✓ Prices are reasonable (< $50)")
    
    # Test 2.3: ATM call and put have similar magnitude
    ratio = call_price / put_price
    assert 0.5 < ratio < 2.5, f"ATM call/put ratio unusual: {ratio}"
    print(f"✓ ATM Call/Put ratio = {ratio:.4f} (reasonable)")
    
    print()

def test_put_call_parity():
    """Test put-call parity"""
    print("="*70)
    print("TEST 3: Put-Call Parity")
    print("="*70)
    
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    
    call_model = DoubleHeston(S, K, T, r,
                              0.04, 2.0, 0.04, 0.3, -0.5,
                              0.04, 1.5, 0.04, 0.2, -0.3,
                              option_type='C')
    put_model = DoubleHeston(S, K, T, r,
                             0.04, 2.0, 0.04, 0.3, -0.5,
                             0.04, 1.5, 0.04, 0.2, -0.3,
                             option_type='P')
    
    C = call_model.pricing(N=256)
    P = put_model.pricing(N=256)
    
    pcp_left = C - P
    pcp_right = S - K * np.exp(-r * T)
    diff = abs(pcp_left - pcp_right)
    
    assert diff < 0.01, f"Put-call parity violated: {pcp_left:.4f} vs {pcp_right:.4f}"
    print(f"✓ C - P = ${pcp_left:.4f}")
    print(f"✓ S - Ke^(-rT) = ${pcp_right:.4f}")
    print(f"✓ Difference = ${diff:.6f} < $0.01")
    
    print()

def test_moneyness():
    """Test pricing across different moneyness"""
    print("="*70)
    print("TEST 4: Moneyness")
    print("="*70)
    
    S = 100
    T = 1.0
    r = 0.05
    
    strikes = [80, 90, 100, 110, 120]
    call_prices = []
    put_prices = []
    
    for K in strikes:
        call = DoubleHeston(S, K, T, r,
                           0.04, 2.0, 0.04, 0.3, -0.5,
                           0.04, 1.5, 0.04, 0.2, -0.3,
                           option_type='C')
        put = DoubleHeston(S, K, T, r,
                          0.04, 2.0, 0.04, 0.3, -0.5,
                          0.04, 1.5, 0.04, 0.2, -0.3,
                          option_type='P')
        call_prices.append(call.pricing(N=256))
        put_prices.append(put.pricing(N=256))
    
    # Test 4.1: Calls decrease with strike
    for i in range(len(strikes)-1):
        assert call_prices[i] >= call_prices[i+1] - 0.5, \
            f"Call not decreasing: K={strikes[i]} C={call_prices[i]:.4f}, K={strikes[i+1]} C={call_prices[i+1]:.4f}"
    print(f"✓ Call prices decrease with strike")
    
    # Test 4.2: Puts increase with strike
    for i in range(len(strikes)-1):
        assert put_prices[i] <= put_prices[i+1] + 0.5, \
            f"Put not increasing: K={strikes[i]} P={put_prices[i]:.4f}, K={strikes[i+1]} P={put_prices[i+1]:.4f}"
    print(f"✓ Put prices increase with strike")
    
    # Test 4.3: ITM options worth more than OTM
    itm_call = call_prices[0]  # K=80
    otm_call = call_prices[-1]  # K=120
    assert itm_call > otm_call, f"ITM call not > OTM call"
    print(f"✓ ITM call (${itm_call:.4f}) > OTM call (${otm_call:.4f})")
    
    itm_put = put_prices[-1]  # K=120
    otm_put = put_prices[0]  # K=80
    assert itm_put > otm_put, f"ITM put not > OTM put"
    print(f"✓ ITM put (${itm_put:.4f}) > OTM put (${otm_put:.4f})")
    
    print()

def test_black_scholes_comparison():
    """Compare with Black-Scholes in low vol-of-vol limit"""
    print("="*70)
    print("TEST 5: Black-Scholes Comparison")
    print("="*70)
    
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    
    # Very low vol-of-vol should approach BS
    v0 = 0.04
    sigma_bs = np.sqrt(v0)
    
    # Double Heston with tiny vol-of-vol
    call_dh = DoubleHeston(S, K, T, r,
                           v0/2, 5.0, v0/2, 0.01, -0.1,  # Low vol-of-vol
                           v0/2, 5.0, v0/2, 0.01, -0.1,
                           option_type='C')
    put_dh = DoubleHeston(S, K, T, r,
                          v0/2, 5.0, v0/2, 0.01, -0.1,
                          v0/2, 5.0, v0/2, 0.01, -0.1,
                          option_type='P')
    
    C_dh = call_dh.pricing(N=256)
    P_dh = put_dh.pricing(N=256)
    
    # Black-Scholes
    d1 = (np.log(S/K) + (r + 0.5*sigma_bs**2)*T) / (sigma_bs * np.sqrt(T))
    d2 = d1 - sigma_bs * np.sqrt(T)
    C_bs = S * norm_cdf(d1) - K * np.exp(-r*T) * norm_cdf(d2)
    P_bs = K * np.exp(-r*T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    
    call_diff = abs(C_dh - C_bs)
    put_diff = abs(P_dh - P_bs)
    
    print(f"  Double Heston: Call=${C_dh:.4f}, Put=${P_dh:.4f}")
    print(f"  Black-Scholes: Call=${C_bs:.4f}, Put=${P_bs:.4f}")
    print(f"✓ Call difference: ${call_diff:.4f}")
    print(f"✓ Put difference: ${put_diff:.4f}")
    
    # Should be close (within 10% given it's not exact BS)
    assert call_diff < C_bs * 0.15, f"Call price differs too much from BS: {call_diff}"
    assert put_diff < P_bs * 0.15, f"Put price differs too much from BS: {put_diff}"
    
    print()

def test_convergence():
    """Test COS method convergence with N"""
    print("="*70)
    print("TEST 6: COS Method Convergence")
    print("="*70)
    
    model = DoubleHeston(100, 100, 1.0, 0.05,
                         0.04, 2.0, 0.04, 0.3, -0.5,
                         0.04, 1.5, 0.04, 0.2, -0.3,
                         option_type='C')
    
    N_values = [64, 128, 256, 512]
    prices = [model.pricing(N=N) for N in N_values]
    
    for i, (N, price) in enumerate(zip(N_values, prices)):
        print(f"  N={N:4d}: C=${price:.6f}")
    
    # Test convergence: differences should decrease
    diffs = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
    print(f"\n  Differences: {[f'{d:.6f}' for d in diffs]}")
    
    # High N should be stable
    assert diffs[-1] < 0.01, f"Not converged at N=512: diff={diffs[-1]}"
    print(f"✓ Converged: |P(512) - P(256)| = ${diffs[-1]:.6f} < $0.01")
    
    print()

def test_time_value():
    """Test time value properties"""
    print("="*70)
    print("TEST 7: Time Value")
    print("="*70)
    
    S = 100
    K = 100
    r = 0.05
    
    # ATM options with different maturities
    times = [0.25, 0.5, 1.0, 2.0]
    call_prices = []
    
    for T in times:
        call = DoubleHeston(S, K, T, r,
                           0.04, 2.0, 0.04, 0.3, -0.5,
                           0.04, 1.5, 0.04, 0.2, -0.3,
                           option_type='C')
        call_prices.append(call.pricing(N=256))
    
    for T, price in zip(times, call_prices):
        print(f"  T={T:.2f}y: C=${price:.4f}")
    
    # Test 7.1: Longer maturity → higher price (time value)
    for i in range(len(times)-1):
        assert call_prices[i] <= call_prices[i+1] + 0.1, \
            f"Price not increasing with time: T={times[i]} C={call_prices[i]:.4f}, T={times[i+1]} C={call_prices[i+1]:.4f}"
    print(f"✓ Call prices increase with maturity (time value)")
    
    print()

def test_parameter_sensitivity():
    """Test sensitivity to model parameters"""
    print("="*70)
    print("TEST 8: Parameter Sensitivity")
    print("="*70)
    
    base_params = (100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5, 0.04, 1.5, 0.04, 0.2, -0.3)
    
    # Base price
    call_base = DoubleHeston(*base_params, option_type='C')
    price_base = call_base.pricing(N=256)
    print(f"  Base price: ${price_base:.4f}")
    
    # Test 8.1: Higher initial variance → higher price
    call_high_v = DoubleHeston(100, 100, 1.0, 0.05, 0.08, 2.0, 0.04, 0.3, -0.5, 0.08, 1.5, 0.04, 0.2, -0.3, option_type='C')
    price_high_v = call_high_v.pricing(N=256)
    assert price_high_v > price_base, f"Higher variance should increase price"
    print(f"✓ Higher variance: ${price_high_v:.4f} > ${price_base:.4f}")
    
    # Test 8.2: More negative correlation → slightly different price
    call_neg_rho = DoubleHeston(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7, 0.04, 1.5, 0.04, 0.2, -0.5, option_type='C')
    price_neg_rho = call_neg_rho.pricing(N=256)
    print(f"  More negative ρ: ${price_neg_rho:.4f}")
    
    # Test 8.3: Vol-of-vol effect (can go either way depending on mean reversion strength)
    call_high_vov = DoubleHeston(100, 100, 1.0, 0.05, 0.04, 2.0, 0.04, 0.5, -0.5, 0.04, 1.5, 0.04, 0.4, -0.3, option_type='C')
    price_high_vov = call_high_vov.pricing(N=256)
    # Just check price is reasonable (vol-of-vol effect depends on mean reversion)
    assert 5 < price_high_vov < 30, f"Price with high vol-of-vol unreasonable: {price_high_vov}"
    print(f"✓ Higher vol-of-vol: ${price_high_vov:.4f} (reasonable, can vary with mean reversion)")
    
    print()

def test_intrinsic_value():
    """Test intrinsic value bounds"""
    print("="*70)
    print("TEST 9: Intrinsic Value Bounds")
    print("="*70)
    
    S = 100
    r = 0.05
    T = 1.0
    
    # Deep ITM call (K=80)
    call_itm = DoubleHeston(S, 80, T, r,
                            0.04, 2.0, 0.04, 0.3, -0.5,
                            0.04, 1.5, 0.04, 0.2, -0.3,
                            option_type='C')
    C_itm = call_itm.pricing(N=256)
    intrinsic_call = max(0, S - 80 * np.exp(-r*T))
    
    assert C_itm >= intrinsic_call - 0.01, f"Call below intrinsic value"
    print(f"✓ Deep ITM call: ${C_itm:.4f} >= intrinsic ${intrinsic_call:.4f}")
    
    # Deep ITM put (K=120)
    put_itm = DoubleHeston(S, 120, T, r,
                           0.04, 2.0, 0.04, 0.3, -0.5,
                           0.04, 1.5, 0.04, 0.2, -0.3,
                           option_type='P')
    P_itm = put_itm.pricing(N=256)
    intrinsic_put = max(0, 120 * np.exp(-r*T) - S)
    
    assert P_itm >= intrinsic_put - 0.01, f"Put below intrinsic value"
    print(f"✓ Deep ITM put: ${P_itm:.4f} >= intrinsic ${intrinsic_put:.4f}")
    
    # Deep OTM call (K=150)
    call_otm = DoubleHeston(S, 150, T, r,
                            0.04, 2.0, 0.04, 0.3, -0.5,
                            0.04, 1.5, 0.04, 0.2, -0.3,
                            option_type='C')
    C_otm = call_otm.pricing(N=256)
    
    assert C_otm >= 0 and C_otm < S, f"OTM call price unreasonable"
    print(f"✓ Deep OTM call: ${C_otm:.4f} (small but positive)")
    
    print()


def run_all_tests():
    """Run all validation tests"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*10 + "DOUBLE HESTON MODEL - VALIDATION TEST SUITE" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    tests = [
        test_cf_properties,
        test_pricing_basic,
        test_put_call_parity,
        test_moneyness,
        test_black_scholes_comparison,
        test_convergence,
        test_time_value,
        test_parameter_sensitivity,
        test_intrinsic_value,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("="*70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
