"""
Quick test of L-BFGS calibrator using synthetic data.

This bypasses the need for historical market data and validates
that the calibration system works correctly.

Usage:
    python3 test_lbfgs_quick.py
"""

import numpy as np
import pickle
from datetime import datetime
from lbfgs_calibrator import DoubleHestonJumpCalibrator, CalibrationResult
from doubleheston import DoubleHeston


def create_synthetic_market_data():
    """
    Create synthetic 'market' data by pricing options with known parameters.
    
    This simulates having real market data for testing purposes.
    """
    print("="*70)
    print("CREATING SYNTHETIC MARKET DATA")
    print("="*70)
    
    # True parameters (what we'll try to recover)
    true_params = {
        'v1_0': 0.04,
        'kappa1': 2.5,
        'theta1': 0.04,
        'sigma1': 0.3,
        'rho1': -0.7,
        'v2_0': 0.04,
        'kappa2': 0.5,
        'theta2': 0.04,
        'sigma2': 0.2,
        'rho2': -0.5,
        'lambda_j': 0.15,
        'mu_j': -0.04,
        'sigma_j': 0.08
    }
    
    # Market setup
    spot = 100.0
    risk_free = 0.03
    
    # Create option grid
    strikes = [90, 95, 100, 105, 110]
    maturities = [0.25, 0.5, 1.0]
    
    market_options = []
    
    print(f"\nTrue Parameters:")
    for name, value in true_params.items():
        print(f"  {name:10s} = {value:.6f}")
    
    print(f"\nGenerating market prices for {len(strikes) * len(maturities)} options...")
    
    for T in maturities:
        for K in strikes:
            # Price using true parameters
            dh = DoubleHeston(
                S0=spot,
                K=K,
                T=T,
                r=risk_free,
                v01=true_params['v1_0'],
                kappa1=true_params['kappa1'],
                theta1=true_params['theta1'],
                sigma1=true_params['sigma1'],
                rho1=true_params['rho1'],
                v02=true_params['v2_0'],
                kappa2=true_params['kappa2'],
                theta2=true_params['theta2'],
                sigma2=true_params['sigma2'],
                rho2=true_params['rho2'],
                lambda_j=true_params['lambda_j'],
                mu_j=true_params['mu_j'],
                sigma_j=true_params['sigma_j'],
                option_type='call'
            )
            
            price = dh.pricing()
            
            # Add small noise to make it realistic (±0.5%)
            noise = np.random.normal(0, 0.005) * price
            price_noisy = price + noise
            
            market_options.append({
                'strike': K,
                'maturity': T,
                'price': price_noisy,
                'option_type': 'call',
                'true_price': price  # Store true price for comparison
            })
    
    print(f"✓ Created {len(market_options)} synthetic market options")
    
    return {
        'spot': spot,
        'risk_free': risk_free,
        'options': market_options,
        'true_params': true_params
    }


def test_calibration():
    """
    Test L-BFGS calibration on synthetic data.
    """
    print("\n" + "="*70)
    print("TESTING L-BFGS CALIBRATOR")
    print("="*70)
    
    # Create synthetic market data
    market_data = create_synthetic_market_data()
    
    # Create calibrator
    print(f"\nInitializing calibrator...")
    calibrator = DoubleHestonJumpCalibrator(
        spot=market_data['spot'],
        risk_free_rate=market_data['risk_free'],
        market_options=market_data['options']
    )
    
    # Run calibration
    print(f"\nRunning calibration with 3 multi-starts...")
    print(f"(This should take 2-5 minutes)")
    
    result = calibrator.calibrate(maxiter=300, multi_start=3)
    
    # Print results
    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70)
    
    print(f"\nStatus:")
    print(f"  Success: {result.success}")
    print(f"  Time: {result.calibration_time:.1f}s")
    print(f"  Loss: {result.final_loss:.6f}")
    print(f"  Iterations: {result.iterations}")
    
    if result.success:
        # Compare calibrated vs true parameters
        print(f"\nParameter Recovery:")
        print(f"{'Parameter':<12} {'True':<12} {'Calibrated':<12} {'Error %':<10}")
        print("-" * 50)
        
        true_params = market_data['true_params']
        
        for name in calibrator.param_names:
            true_val = true_params[name]
            calib_val = result.parameters[name]
            error_pct = abs((calib_val - true_val) / true_val) * 100
            
            print(f"{name:<12} {true_val:<12.6f} {calib_val:<12.6f} {error_pct:<10.2f}")
        
        # Pricing accuracy
        rel_errors = np.abs((result.model_prices - result.market_prices) / result.market_prices) * 100
        
        print(f"\nPricing Accuracy:")
        print(f"  Mean absolute error: {np.mean(rel_errors):.2f}%")
        print(f"  Median absolute error: {np.median(rel_errors):.2f}%")
        print(f"  Max error: {np.max(rel_errors):.2f}%")
        
        # Sample option comparison
        print(f"\nSample Option Prices (first 5):")
        print(f"{'K':<6} {'T':<6} {'Market':<10} {'Model':<10} {'Error %':<10}")
        print("-" * 50)
        
        for i in range(min(5, len(market_data['options']))):
            opt = market_data['options'][i]
            market_price = result.market_prices[i]
            model_price = result.model_prices[i]
            error = abs((model_price - market_price) / market_price) * 100
            
            print(f"{opt['strike']:<6.0f} {opt['maturity']:<6.2f} "
                  f"{market_price:<10.4f} {model_price:<10.4f} {error:<10.2f}")
    
    print("\n" + "="*70)
    
    return result


def test_parameter_recovery_accuracy():
    """
    Run multiple calibrations to measure parameter recovery accuracy.
    """
    print("\n" + "="*70)
    print("PARAMETER RECOVERY TEST (10 SAMPLES)")
    print("="*70)
    print("\nTesting if calibrator can recover true parameters consistently...")
    
    n_tests = 10
    all_errors = {name: [] for name in [
        'v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
        'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
        'lambda_j', 'mu_j', 'sigma_j'
    ]}
    
    for i in range(n_tests):
        print(f"\nTest {i+1}/{n_tests}...", end=' ')
        
        # Create fresh synthetic data with different noise
        market_data = create_synthetic_market_data()
        
        # Calibrate
        calibrator = DoubleHestonJumpCalibrator(
            spot=market_data['spot'],
            risk_free_rate=market_data['risk_free'],
            market_options=market_data['options']
        )
        
        result = calibrator.calibrate(maxiter=200, multi_start=2)
        
        if result.success:
            # Compute errors
            true_params = market_data['true_params']
            for name in all_errors.keys():
                error_pct = abs((result.parameters[name] - true_params[name]) / true_params[name]) * 100
                all_errors[name].append(error_pct)
            
            print(f"✓ Loss: {result.final_loss:.6f}")
        else:
            print(f"✗ Failed")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("PARAMETER RECOVERY STATISTICS")
    print("="*70)
    print(f"\n{'Parameter':<12} {'Mean Error %':<15} {'Std %':<15} {'Max %':<15}")
    print("-" * 60)
    
    for name, errors in all_errors.items():
        if errors:
            print(f"{name:<12} {np.mean(errors):<15.2f} {np.std(errors):<15.2f} {np.max(errors):<15.2f}")
    
    print("\n" + "="*70)
    print("✓ Recovery test complete!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'recovery':
        # Run parameter recovery test
        test_parameter_recovery_accuracy()
    else:
        # Run single calibration test
        result = test_calibration()
        
        if result.success:
            print("\n✓ L-BFGS calibrator validated successfully!")
            print("\nNext steps:")
            print("  1. Test with real market data: python3 lbfgs_calibrator.py test")
            print("  2. Run recovery test: python3 test_lbfgs_quick.py recovery")
            print("  3. Full calibration: python3 lbfgs_calibrator.py")
        else:
            print("\n✗ Calibration failed. Check implementation.")
