"""
Synthetic Calibration Data Generator for FFN Training

IMPORTANT: This generates SYNTHETIC data with randomly sampled parameters.
           It is NOT real calibration data and should NOT be used for
           performance benchmarking or comparison purposes.
           
Purpose: Generate training data for feedforward neural network that predicts
         Double Heston parameters from option prices.
"""

import numpy as np
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'calibration'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from lbfgs_calibrator import CalibrationResult
from double_heston import DoubleHeston


def generate_synthetic_calibrations(n_samples: int = 500, 
                                    save_path: str = 'lbfgs_calibrations_synthetic.pkl'):
    """
    Generate synthetic calibration data for FFN training.
    
    WARNING: This is NOT real market calibration data!
    - Parameters are randomly sampled from realistic ranges
    - Market prices are synthetic with added noise
    - No actual L-BFGS optimization is performed
    - calibration_time and iterations are set to None
    
    Args:
        n_samples: Number of synthetic calibration samples to generate
        save_path: Path to save the generated data
        
    Returns:
        List of CalibrationResult objects with synthetic data
    """
    print("="*70)
    print("GENERATING SYNTHETIC HISTORICAL CALIBRATIONS")
    print("="*70)
    print("\n⚠️  WARNING: This generates SYNTHETIC data, not real calibrations!")
    print("   - Parameters are randomly sampled, not from actual L-BFGS optimization")
    print("   - Market prices are synthetic with added noise")
    print("   - Timing and iteration data are None (not actually calibrated)")
    print("   - This data is for FFN training only, NOT for performance benchmarks")
    print()
    print(f"Configuration:")
    print(f"  Number of calibrations: {n_samples}")
    print(f"  Date range: 2022-01-03 to 2024-12-31 (simulated)")
    print(f"  Save path: {save_path}")
    print()
    
    # Generate trading dates
    start_date = datetime(2022, 1, 3)
    dates = []
    current = start_date
    
    for i in range(n_samples):
        while current.weekday() >= 5:  # Skip weekends
            current += timedelta(days=1)
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    print(f"Generated {len(dates)} trading dates")
    print(f"\nGenerating calibration results...")
    
    calibrations = []

    # Parameter ranges based on empirical market data
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
    
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    spot_base = 100.0
    risk_free = 0.03
    
    param_names = list(param_ranges.keys())
    
    for i, date in enumerate(dates):
        # Sample parameters from realistic ranges
        params = {}
        for name, (min_val, max_val) in param_ranges.items():
            params[name] = np.random.uniform(min_val, max_val)
        
        # Add time-series structure (90% autocorrelation)
        if i > 0:
            prev_params = calibrations[-1].parameters
            for name in param_names:
                alpha = 0.9  # Persistence
                params[name] = alpha * prev_params[name] + (1 - alpha) * params[name]
        
        # Evolve spot price (random walk)
        if i == 0:
            spot = spot_base
        else:
            spot_return = np.random.normal(0.0003, 0.01)  # ~30% annualized vol
            spot = calibrations[-1].spot * (1 + spot_return)
        
        # Generate synthetic market prices
        market_options = []
        market_prices = []
        model_prices = []
        
        for T in maturities:
            for K_relative in strikes:
                K = K_relative * spot / 100.0  # Maintain moneyness
                
                # Compute theoretical price
                dh = DoubleHeston(
                    S0=spot, K=K, T=T, r=risk_free,
                    v01=params['v1_0'], kappa1=params['kappa1'],
                    theta1=params['theta1'], sigma1=params['sigma1'], rho1=params['rho1'],
                    v02=params['v2_0'], kappa2=params['kappa2'],
                    theta2=params['theta2'], sigma2=params['sigma2'], rho2=params['rho2'],
                    lambda_j=params['lambda_j'], mu_j=params['mu_j'],
                    sigma_j=params['sigma_j'], option_type='call'
                )
                
                price = dh.pricing()
                
                # Add market noise (bid-ask spread + estimation error)
                market_noise = np.random.normal(0, 0.02) * price
                market_price = price + market_noise
                
                market_options.append({
                    'strike': K,
                    'maturity': T,
                    'price': market_price,
                    'option_type': 'call'
                })
                
                market_prices.append(market_price)
                model_prices.append(price)
        
        market_prices = np.array(market_prices)
        model_prices = np.array(model_prices)
        relative_errors = (model_prices - market_prices) / market_prices
        loss = np.mean(relative_errors**2)
        
        # Create result object (calibration_time and iterations are None)
        result = CalibrationResult(
            date=date,
            spot=spot,
            risk_free=risk_free,
            parameters=params,
            market_prices=market_prices,
            model_prices=model_prices,
            market_options=market_options,
            final_loss=loss,
            calibration_time=None,  # NOT from actual calibration
            success=True,
            iterations=None,  # NOT from actual calibration
            message='Synthetic data (not from real calibration)'
        )
        
        calibrations.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%)")
    
    # Save to disk
    print(f"\nSaving to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(calibrations, f)
    
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print("="*70)
    
    # Display statistics
    losses = [r.final_loss for r in calibrations]
    spots = [r.spot for r in calibrations]
    
    print(f"\nCalibration Statistics:")
    print(f"  Total calibrations: {len(calibrations)}")
    print(f"  Success rate: 100.0% (synthetic data)")
    print(f"  Timing: N/A (not actually calibrated)")
    
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


if __name__ == "__main__":
    n_samples = 500
    if len(sys.argv) > 1:
        try:
            n_samples = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of samples: {sys.argv[1]}")
            print("Usage: python synthetic_generator.py [n_samples]")
            sys.exit(1)
    
    calibrations = generate_synthetic_calibrations(
        n_samples=n_samples,
        save_path='lbfgs_calibrations_synthetic.pkl'
    )
