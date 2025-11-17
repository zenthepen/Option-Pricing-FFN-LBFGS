"""
Generate synthetic "historical" calibrations for testing FFN fine-tuning workflow.

This creates 500 CalibrationResult objects as if L-BFGS was run on 500 historical dates.
Use this when you don't have access to real historical option data but want to test
the FFN fine-tuning pipeline.

Usage:
    python3 generate_synthetic_calibrations.py
    
Output:
    lbfgs_calibrations_synthetic.pkl - 500 synthetic calibration results
"""

import numpy as np
import pickle
from datetime import datetime, timedelta
from lbfgs_calibrator import CalibrationResult
from doubleheston import DoubleHeston


def generate_synthetic_calibrations(n_samples: int = 500, 
                                    save_path: str = 'lbfgs_calibrations_synthetic.pkl'):
    """
    Generate synthetic calibration results that mimic real L-BFGS output.
    
    Parameters:
    -----------
    n_samples : int
        Number of synthetic calibrations to generate
    save_path : str
        Path to save results
        
    Returns:
    --------
    list : List of CalibrationResult objects
    """
    print("="*70)
    print("GENERATING SYNTHETIC HISTORICAL CALIBRATIONS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of calibrations: {n_samples}")
    print(f"  Date range: 2022-01-03 to 2024-12-31 (simulated)")
    print(f"  Save path: {save_path}")
    print()
    
    # Generate dates
    start_date = datetime(2022, 1, 3)
    dates = []
    current = start_date
    
    for i in range(n_samples):
        # Skip weekends
        while current.weekday() >= 5:
            current += timedelta(days=1)
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print(f"Generated {len(dates)} trading dates")
    print(f"\nGenerating calibration results...")
    
    calibrations = []
    
    # Define reasonable parameter ranges (market realistic)
    param_ranges = {
        'v1_0': (0.025, 0.080),
        'kappa1': (1.5, 4.5),
        'theta1': (0.025, 0.065),
        'sigma1': (0.20, 0.50),
        'rho1': (-0.85, -0.40),
        'v2_0': (0.020, 0.070),
        'kappa2': (0.30, 1.20),
        'theta2': (0.025, 0.070),
        'sigma2': (0.10, 0.35),
        'rho2': (-0.70, -0.20),
        'lambda_j': (0.05, 0.25),
        'mu_j': (-0.08, -0.01),
        'sigma_j': (0.03, 0.12)
    }
    
    # Market configuration
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    spot_base = 100.0
    risk_free = 0.03
    
    param_names = list(param_ranges.keys())
    
    for i, date in enumerate(dates):
        # Generate random parameters within realistic ranges
        params = {}
        for name, (min_val, max_val) in param_ranges.items():
            params[name] = np.random.uniform(min_val, max_val)
        
        # Add time-series structure (parameters drift slowly)
        if i > 0:
            prev_params = calibrations[-1].parameters
            # Parameters change slowly (90% correlation with previous day)
            for name in param_names:
                alpha = 0.9  # Persistence
                params[name] = alpha * prev_params[name] + (1 - alpha) * params[name]
        
        # Spot price with volatility
        if i == 0:
            spot = spot_base
        else:
            # Random walk with drift
            spot_return = np.random.normal(0.0003, 0.01)  # ~30% annualized vol
            spot = calibrations[-1].spot * (1 + spot_return)
        
        # Generate "market" options
        market_options = []
        market_prices = []
        model_prices = []
        
        for T in maturities:
            for K_relative in strikes:
                K = K_relative * spot / 100.0  # Scale strikes with spot
                
                # Price with Double Heston
                dh = DoubleHeston(
                    S0=spot,
                    K=K,
                    T=T,
                    r=risk_free,
                    v01=params['v1_0'],
                    kappa1=params['kappa1'],
                    theta1=params['theta1'],
                    sigma1=params['sigma1'],
                    rho1=params['rho1'],
                    v02=params['v2_0'],
                    kappa2=params['kappa2'],
                    theta2=params['theta2'],
                    sigma2=params['sigma2'],
                    rho2=params['rho2'],
                    lambda_j=params['lambda_j'],
                    mu_j=params['mu_j'],
                    sigma_j=params['sigma_j'],
                    option_type='call'
                )
                
                price = dh.pricing()
                
                # Add small market noise (realistic bid-ask spread + estimation error)
                market_noise = np.random.normal(0, 0.02) * price  # 2% std
                market_price = price + market_noise
                
                market_options.append({
                    'strike': K,
                    'maturity': T,
                    'price': market_price,
                    'option_type': 'call'
                })
                
                market_prices.append(market_price)
                model_prices.append(price)
        
        # Compute loss (MSPE)
        market_prices = np.array(market_prices)
        model_prices = np.array(model_prices)
        relative_errors = (model_prices - market_prices) / market_prices
        loss = np.mean(relative_errors**2)
        
        # Create CalibrationResult
        result = CalibrationResult(
            date=date,
            spot=spot,
            risk_free=risk_free,
            parameters=params,
            market_prices=market_prices,
            model_prices=model_prices,
            market_options=market_options,
            final_loss=loss,
            calibration_time=np.random.uniform(120, 250),  # Realistic timing
            success=True,
            iterations=np.random.randint(80, 200),
            message='Converged'
        )
        
        calibrations.append(result)
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%)")
    
    # Save results
    print(f"\nSaving to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(calibrations, f)
    
    # Print statistics
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print("="*70)
    
    times = [r.calibration_time for r in calibrations]
    losses = [r.final_loss for r in calibrations]
    spots = [r.spot for r in calibrations]
    
    print(f"\nCalibration Statistics:")
    print(f"  Total calibrations: {len(calibrations)}")
    print(f"  Success rate: 100.0% (synthetic data)")
    print(f"  Mean time: {np.mean(times):.1f}s")
    print(f"  Total time (simulated): {np.sum(times)/3600:.1f} hours")
    
    print(f"\nLoss Statistics:")
    print(f"  Mean loss: {np.mean(losses):.6f}")
    print(f"  Median loss: {np.median(losses):.6f}")
    print(f"  Min loss: {np.min(losses):.6f}")
    print(f"  Max loss: {np.max(losses):.6f}")
    
    print(f"\nSpot Price Evolution:")
    print(f"  Start: ${spots[0]:.2f}")
    print(f"  End: ${spots[-1]:.2f}")
    print(f"  Return: {(spots[-1]/spots[0] - 1)*100:+.2f}%")
    print(f"  Min: ${min(spots):.2f}")
    print(f"  Max: ${max(spots):.2f}")
    
    print(f"\nParameter Statistics:")
    for param in param_names:
        values = [r.parameters[param] for r in calibrations]
        print(f"  {param:10s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
              f"min={np.min(values):.4f}, max={np.max(values):.4f}")
    
    # Pricing error statistics
    all_errors = []
    for r in calibrations:
        rel_errors = np.abs((r.model_prices - r.market_prices) / r.market_prices) * 100
        all_errors.extend(rel_errors)
    
    all_errors = np.array(all_errors)
    print(f"\nPricing Error Statistics:")
    print(f"  Mean absolute error: {np.mean(all_errors):.2f}%")
    print(f"  Median absolute error: {np.median(all_errors):.2f}%")
    print(f"  95th percentile: {np.percentile(all_errors, 95):.2f}%")
    print(f"  Max error: {np.max(all_errors):.2f}%")
    
    print(f"\n{'='*70}")
    print(f"✓ Synthetic calibrations saved to: {save_path}")
    print(f"✓ Ready for FFN fine-tuning")
    print(f"{'='*70}\n")
    
    return calibrations


def compare_with_real_statistics():
    """
    Print expected vs synthetic statistics for validation.
    """
    print("\n" + "="*70)
    print("COMPARISON: SYNTHETIC vs REAL MARKET CALIBRATIONS")
    print("="*70)
    print("""
Expected Real Market Statistics (SPY 2022-2024):
  v1_0:     0.025 - 0.080  (varies with VIX regime)
  kappa1:   1.500 - 4.500  (fast mean reversion)
  theta1:   0.025 - 0.065  (long-term variance)
  sigma1:   0.200 - 0.500  (vol-of-vol for fast component)
  rho1:    -0.850 - -0.400 (negative correlation dominant)
  
  v2_0:     0.020 - 0.070  (slow component initial variance)
  kappa2:   0.300 - 1.200  (slow mean reversion)
  theta2:   0.025 - 0.070  (long-term variance)
  sigma2:   0.100 - 0.350  (vol-of-vol for slow component)
  rho2:    -0.700 - -0.200 (negative correlation)
  
  lambda_j: 0.050 - 0.250  (5-25 jumps per year)
  mu_j:    -0.080 - -0.010 (negative jumps, crashes)
  sigma_j:  0.030 - 0.120  (jump size volatility)

Synthetic Data Characteristics:
  ✓ Parameters sampled from realistic ranges above
  ✓ Time-series structure: 90% autocorrelation (parameters drift slowly)
  ✓ Spot price: Random walk with 30% annualized volatility
  ✓ Market noise: 2% bid-ask spread + estimation error
  ✓ Success rate: 100% (no data issues)

Differences from Real Data:
  ⚠️ No regime switches (real markets have crisis/calm regimes)
  ⚠️ No seasonality (real markets have monthly option expiration effects)
  ⚠️ No jumps in synthetic spot (real markets have discrete jumps)
  ⚠️ Constant noise (real markets have time-varying liquidity)

Use Case:
  ✓ Perfect for TESTING FFN fine-tuning workflow
  ✓ Validates code works before using expensive real data
  ✓ Reproducible results for debugging
  ✗ Not suitable for production deployment (use real data)
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Parse arguments
    n_samples = 500
    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of samples: {sys.argv[1]}")
            print("Usage: python3 generate_synthetic_calibrations.py [n_samples]")
            sys.exit(1)
    
    # Generate synthetic calibrations
    calibrations = generate_synthetic_calibrations(
        n_samples=n_samples,
        save_path='lbfgs_calibrations_synthetic.pkl'
    )
    
    # Show comparison
    compare_with_real_statistics()
    
    # Usage example
    print("\n" + "="*70)
    print("USAGE EXAMPLE: FFN FINE-TUNING")
    print("="*70)
    print("""
import pickle
import numpy as np

# Load synthetic calibrations
with open('lbfgs_calibrations_synthetic.pkl', 'rb') as f:
    calibrations = pickle.load(f)

print(f"Loaded {len(calibrations)} calibrations")

# Extract parameters for FFN training
param_names = ['v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
               'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
               'lambda_j', 'mu_j', 'sigma_j']

# Convert to arrays
params_array = np.array([
    [calib.parameters[name] for name in param_names]
    for calib in calibrations
])

prices_array = np.array([
    calib.market_prices for calib in calibrations
])

print(f"Parameters shape: {params_array.shape}")  # (500, 13)
print(f"Prices shape: {prices_array.shape}")      # (500, 15)

# Fine-tune FFN
from ffn import DoubleHestonFFN

model = DoubleHestonFFN.load('ffn_double_heston_jumps.pth')
model.fine_tune(params_array, prices_array, epochs=50, lr=1e-5)
model.save('ffn_finetuned_synthetic.pth')

print("✓ Fine-tuning complete!")
""")
    print("="*70 + "\n")
