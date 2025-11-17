"""
Evaluate Fine-Tuned FFN Pricing Accuracy

This script evaluates the fine-tuned FFN model's pricing accuracy by:
1. Loading the fine-tuned model and test calibrations
2. Predicting parameters for each test sample
3. Pricing options with predicted parameters
4. Computing pricing errors vs true market prices
5. Generating comprehensive performance statistics

Expected: 10-15% pricing error (improvement from 31% pre-training)
"""

import numpy as np
import pickle
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from doubleheston import DoubleHeston

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)


def extract_features_single_sample(prices, strikes, maturities, spot=100.0):
    """
    Extract 11 features from 15 option prices (MUST match training).
    
    Parameters:
    -----------
    prices : np.ndarray, shape (15,)
        Option prices for 5 strikes × 3 maturities
    strikes : np.ndarray
        Strike prices [90, 95, 100, 105, 110]
    maturities : np.ndarray
        Maturities [0.25, 0.5, 1.0]
    spot : float
        Spot price (typically 100.0)
        
    Returns:
    --------
    features : np.ndarray
        11 features matching training feature extraction
    """
    # Reshape to (5 strikes, 3 maturities)
    prices_2d = prices.reshape(len(strikes), len(maturities))
    
    features = []
    
    # Find ATM index (should be index 2 for strike=100)
    atm_idx = np.argmin(np.abs(strikes - spot))
    
    # Features 1-9: 3 features per maturity (ATM, skew, butterfly)
    for mat_idx in range(len(maturities)):
        prices_at_mat = prices_2d[:, mat_idx]
        atm_price = prices_at_mat[atm_idx]
        
        # Feature: ATM price (normalized by spot)
        features.append(atm_price / spot)
        
        # Feature: Skew (25-delta risk reversal approximation)
        # OTM call (105) vs OTM put (95)
        otm_call_idx = np.argmin(np.abs(strikes - spot*1.05))
        otm_put_idx = np.argmin(np.abs(strikes - spot*0.95))
        skew = (prices_at_mat[otm_call_idx] - prices_at_mat[otm_put_idx]) / spot
        features.append(skew)
        
        # Feature: Curvature (butterfly)
        itm_idx = np.argmin(np.abs(strikes - spot*0.95))
        otm_idx = np.argmin(np.abs(strikes - spot*1.05))
        butterfly = (prices_at_mat[itm_idx] + prices_at_mat[otm_idx] - 2*atm_price) / spot
        features.append(butterfly)
    
    # Feature 10: Term structure slope (ATM long - ATM short)
    atm_short = prices_2d[atm_idx, 0]  # First maturity
    atm_long = prices_2d[atm_idx, -1]   # Last maturity
    term_slope = (atm_long - atm_short) / spot
    features.append(term_slope)
    
    # Feature 11: Total ATM premium across maturities
    total_atm = np.sum(prices_2d[atm_idx, :]) / spot
    features.append(total_atm)
    
    # Should have exactly 11 features
    assert len(features) == 11, f"Expected 11 features, got {len(features)}"
    
    return np.array(features)


class FinetunedFFNEvaluator:
    """Evaluates fine-tuned FFN model pricing accuracy"""
    
    # Parameter names in correct order (matching calibration data)
    PARAM_NAMES = ['v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
                   'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
                   'lambda_j', 'mu_j', 'sigma_j']
    
    # Indices that were log-transformed during training
    LOG_TRANSFORM_INDICES = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]
    
    def __init__(self, model_path, scalers_path, test_data_path):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to fine-tuned Keras model
            scalers_path: Path to feature/target scalers
            test_data_path: Path to L-BFGS calibrations
        """
        print("Loading fine-tuned model and data...")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Loaded model: {model_path}")
        
        # Load scalers
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"✓ Loaded scalers: {scalers_path}")
        
        # Load all calibrations
        with open(test_data_path, 'rb') as f:
            all_data = pickle.load(f)
        print(f"✓ Loaded {len(all_data)} calibrations")
        
        # Use last 100 as test set (not used in fine-tuning)
        self.test_data = all_data[-100:]
        print(f"✓ Using last {len(self.test_data)} samples as test set\n")
    
    def extract_features(self, option_prices, strikes, maturities, spot):
        """
        Extract 11 features from option prices
        
        Args:
            option_prices: Array of option prices
            strikes: List of strikes (absolute values)
            maturities: List of maturities
            spot: Spot price
            
        Returns:
            Array of 11 features
        """
        return extract_features_single_sample(option_prices, strikes, maturities, spot)
    
    def inverse_transform_predictions(self, pred_scaled):
        """
        Inverse transform predicted parameters from normalized space
        
        Args:
            pred_scaled: Scaled predictions from model
            
        Returns:
            Original parameter space predictions
        """
        # Inverse scale
        pred_unscaled = self.scalers['target_scaler'].inverse_transform(pred_scaled)[0]
        
        # Reverse log transform for specific indices
        pred_params = pred_unscaled.copy()
        for idx in self.LOG_TRANSFORM_INDICES:
            pred_params[idx] = np.exp(pred_params[idx])
        
        return pred_params
    
    def price_options_with_params(self, params, spot, risk_free, strikes, maturities):
        """
        Price all options using predicted parameters
        
        Args:
            params: Array of 13 parameters in order:
                    [v1_0, kappa1, theta1, sigma1, rho1,
                     v2_0, kappa2, theta2, sigma2, rho2,
                     lambda_j, mu_j, sigma_j]
            spot: Spot price (actual spot from calibration)
            risk_free: Risk-free rate
            strikes: List of strikes (absolute values, scaled with spot)
            maturities: List of maturities
            
        Returns:
            Array of predicted option prices (ordered by T, then K)
        """
        prices = []
        
        # Order must match how options are stored: for each T, iterate through all K
        for T in maturities:
            for K in strikes:
                model_instance = DoubleHeston(
                    S0=spot,  # Use actual spot from calibration
                    K=K,      # Use actual strike (already scaled with spot)
                    T=T,
                    r=risk_free,
                    v01=params[0],    # v1_0
                    kappa1=params[1],
                    theta1=params[2],
                    sigma1=params[3],
                    rho1=params[4],
                    v02=params[5],    # v2_0
                    kappa2=params[6],
                    theta2=params[7],
                    sigma2=params[8],
                    rho2=params[9],
                    lambda_j=params[10],
                    mu_j=params[11],
                    sigma_j=params[12],
                    option_type='call'  # Must match data generation
                )
                
                # Price with COS method (N=128 to match data generation)
                price = model_instance.pricing(N=128)
                prices.append(price)
        
        return np.array(prices)
    
    def evaluate_single_sample(self, calib):
        """
        Evaluate FFN on single calibration sample
        
        Args:
            calib: CalibrationResult object with market_prices, parameters, spot, etc.
            
        Returns:
            Dictionary with pricing_error, parameter_error, pred_params, etc.
        """
        # Extract strikes and maturities from market_options
        strikes = sorted(list(set([opt['strike'] for opt in calib.market_options])))
        maturities = sorted(list(set([opt['maturity'] for opt in calib.market_options])))
        
        # Extract features (strikes are absolute values, spot needed for normalization)
        features = self.extract_features(
            calib.market_prices,
            np.array(strikes),
            np.array(maturities),
            calib.spot
        )
        
        # Predict parameters
        features_scaled = self.scalers['feature_scaler'].transform([features])
        pred_scaled = self.model.predict(features_scaled, verbose=0)
        pred_params = self.inverse_transform_predictions(pred_scaled)
        
        # Get true parameters
        true_params = np.array([calib.parameters[name] for name in self.PARAM_NAMES])
        
        # Price options with predicted parameters
        pred_prices = self.price_options_with_params(
            pred_params,
            calib.spot,
            calib.risk_free,
            strikes,  # Use absolute strikes for pricing
            maturities
        )
        
        # Compute pricing error
        true_prices = np.array(calib.market_prices)
        abs_errors = np.abs(pred_prices - true_prices)
        pct_errors = (abs_errors / true_prices) * 100
        pricing_error = np.mean(pct_errors)
        
        # Compute parameter error (for reference)
        param_abs_errors = np.abs(pred_params - true_params)
        param_pct_errors = (param_abs_errors / np.abs(true_params)) * 100
        parameter_error = np.mean(param_pct_errors)
        
        return {
            'pricing_error': pricing_error,
            'parameter_error': parameter_error,
            'pred_params': pred_params,
            'true_params': true_params,
            'pred_prices': pred_prices,
            'true_prices': true_prices,
            'individual_price_errors': pct_errors
        }
    
    def evaluate_all(self, verbose=True):
        """
        Evaluate FFN on all test samples
        
        Args:
            verbose: Print progress
            
        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("EVALUATING FINE-TUNED FFN ON TEST SET")
        print("="*80)
        print(f"Test set size: {len(self.test_data)} calibrations")
        print(f"Model: {self.model.count_params():,} parameters\n")
        
        results = []
        start_time = time.time()
        
        for i, calib in enumerate(self.test_data):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processing sample {i+1}/{len(self.test_data)}...", end='\r')
            
            result = self.evaluate_single_sample(calib)
            results.append(result)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Completed in {elapsed:.2f}s ({len(self.test_data)/elapsed:.1f} samples/sec)\n")
        
        return self._compute_statistics(results, elapsed)
    
    def _compute_statistics(self, results, elapsed_time):
        """Compute comprehensive statistics from results"""
        
        pricing_errors = [r['pricing_error'] for r in results]
        parameter_errors = [r['parameter_error'] for r in results]
        
        # Individual price errors across all samples
        all_price_errors = np.concatenate([r['individual_price_errors'] for r in results])
        
        stats = {
            'pricing_errors': {
                'mean': np.mean(pricing_errors),
                'median': np.median(pricing_errors),
                'std': np.std(pricing_errors),
                'min': np.min(pricing_errors),
                'max': np.max(pricing_errors),
                'p25': np.percentile(pricing_errors, 25),
                'p75': np.percentile(pricing_errors, 75),
                'p95': np.percentile(pricing_errors, 95),
                'p99': np.percentile(pricing_errors, 99)
            },
            'parameter_errors': {
                'mean': np.mean(parameter_errors),
                'median': np.median(parameter_errors),
                'std': np.std(parameter_errors),
                'min': np.min(parameter_errors),
                'max': np.max(parameter_errors)
            },
            'individual_price_errors': {
                'mean': np.mean(all_price_errors),
                'median': np.median(all_price_errors),
                'std': np.std(all_price_errors),
                'max': np.max(all_price_errors)
            },
            'timing': {
                'total_time': elapsed_time,
                'per_sample': elapsed_time / len(results),
                'samples_per_sec': len(results) / elapsed_time
            },
            'improvement': {
                'baseline_error': 31.0,  # Pre-training error
                'current_error': np.mean(pricing_errors),
                'reduction_pct': (1 - np.mean(pricing_errors) / 31.0) * 100
            },
            'raw_results': results
        }
        
        return stats
    
    def print_report(self, stats):
        """Print comprehensive evaluation report"""
        
        print("="*80)
        print("FINE-TUNED FFN EVALUATION RESULTS")
        print("="*80)
        
        print("\n📊 PRICING ACCURACY (Mean Error per Calibration)")
        print("-" * 80)
        pe = stats['pricing_errors']
        print(f"  Mean Error:          {pe['mean']:>8.2f}%")
        print(f"  Median Error:        {pe['median']:>8.2f}%")
        print(f"  Std Deviation:       {pe['std']:>8.2f}%")
        print(f"  Min Error:           {pe['min']:>8.2f}%")
        print(f"  Max Error:           {pe['max']:>8.2f}%")
        print(f"  25th Percentile:     {pe['p25']:>8.2f}%")
        print(f"  75th Percentile:     {pe['p75']:>8.2f}%")
        print(f"  95th Percentile:     {pe['p95']:>8.2f}%")
        print(f"  99th Percentile:     {pe['p99']:>8.2f}%")
        
        print("\n📈 INDIVIDUAL OPTION PRICING ERRORS")
        print("-" * 80)
        ipe = stats['individual_price_errors']
        print(f"  Mean Error:          {ipe['mean']:>8.2f}%")
        print(f"  Median Error:        {ipe['median']:>8.2f}%")
        print(f"  Std Deviation:       {ipe['std']:>8.2f}%")
        print(f"  Max Error:           {ipe['max']:>8.2f}%")
        
        print("\n🎯 PARAMETER RECOVERY (Reference Only)")
        print("-" * 80)
        print("  Note: Multiple parameter sets can produce identical prices")
        pe_param = stats['parameter_errors']
        print(f"  Mean Error:          {pe_param['mean']:>8.2f}%")
        print(f"  Median Error:        {pe_param['median']:>8.2f}%")
        print(f"  Std Deviation:       {pe_param['std']:>8.2f}%")
        
        print("\n⚡ PERFORMANCE TIMING")
        print("-" * 80)
        timing = stats['timing']
        print(f"  Total Time:          {timing['total_time']:>8.2f}s")
        print(f"  Per Calibration:     {timing['per_sample']*1000:>8.2f}ms")
        print(f"  Throughput:          {timing['samples_per_sec']:>8.1f} samples/sec")
        
        print("\n✨ IMPROVEMENT SUMMARY")
        print("-" * 80)
        imp = stats['improvement']
        print(f"  Before Fine-Tuning:  {imp['baseline_error']:>8.1f}% pricing error")
        print(f"  After Fine-Tuning:   {imp['current_error']:>8.2f}% pricing error")
        print(f"  Improvement:         {imp['reduction_pct']:>8.1f}% reduction")
        
        print("\n🎯 ASSESSMENT")
        print("-" * 80)
        mean_error = pe['mean']
        if mean_error <= 15:
            print(f"  ✅ TARGET ACHIEVED: {mean_error:.2f}% ≤ 15% target")
            print("  Model suitable for fast initial predictions")
        elif mean_error <= 20:
            print(f"  ⚠️  CLOSE TO TARGET: {mean_error:.2f}% (target: 15%)")
            print("  Acceptable for rapid screening, refinement recommended")
        else:
            print(f"  ❌ BELOW TARGET: {mean_error:.2f}% > 15% target")
            print("  Consider additional fine-tuning or hybrid approach")
        
        print("\n" + "="*80)
        
        return stats


def main():
    """Main evaluation script"""
    
    # File paths (adjust if running from different directory)
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
    scalers_path = base_dir / 'data' / 'scalers.pkl'
    test_data_path = base_dir / 'data' / 'lbfgs_calibrations_synthetic.pkl'
    
    # Check files exist
    for path in [model_path, scalers_path, test_data_path]:
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
    
    # Create evaluator
    evaluator = FinetunedFFNEvaluator(
        str(model_path),
        str(scalers_path),
        str(test_data_path)
    )
    
    # Run evaluation
    stats = evaluator.evaluate_all(verbose=True)
    
    # Print report
    evaluator.print_report(stats)
    
    # Save results
    results_path = base_dir / 'results' / 'ffn_evaluation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
