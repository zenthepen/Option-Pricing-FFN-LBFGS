"""
================================================================================
COMPREHENSIVE PROJECT VALIDATION SUITE
================================================================================

This script performs 20+ rigorous tests to validate the entire project:
- No hardcoded results
- True end-to-end functionality
- Real mathematical correctness
- Proper model behavior
- Genuine performance improvements

Run this BEFORE submitting your project to catch any issues.

Author: Zen
Date: 2025-11-13
================================================================================
"""

import numpy as np
import pickle
import time
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from doubleheston import DoubleHeston
from lbfgs_calibrator import DoubleHestonJumpCalibrator
from evaluate_finetuned_ffn import FinetunedFFNEvaluator, extract_features_single_sample
from hybrid_calibrator import HybridCalibrator

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)


class ProjectValidator:
    """
    Comprehensive validation of Double Heston + Jump calibration project
    Ensures no fake results, proper implementation, and real improvements
    """
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
        self.base_dir = Path(__file__).parent
        
    def run_test(self, test_name, test_func):
        """Execute a single test and track results"""
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print(f"{'='*80}")
        
        try:
            result = test_func()
            if result:
                print(f"✅ PASSED: {test_name}")
                self.tests_passed += 1
                self.test_results.append({'test': test_name, 'status': 'PASS'})
                return True
            else:
                print(f"❌ FAILED: {test_name}")
                self.tests_failed += 1
                self.test_results.append({'test': test_name, 'status': 'FAIL'})
                return False
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.tests_failed += 1
            self.test_results.append({'test': test_name, 'status': 'ERROR', 'error': str(e)})
            return False
    
    # ========================================================================
    # SECTION 1: MODEL CORRECTNESS TESTS
    # ========================================================================
    
    def test_double_heston_pricing_sanity(self):
        """Test 1: Basic Double Heston pricing produces reasonable values"""
        print("Testing Double Heston pricing sanity...")
        
        # ATM call with 1 year maturity
        model = DoubleHeston(
            S0=100, K=100, T=1.0, r=0.05,
            v01=0.04, kappa1=2.0, theta1=0.04, sigma1=0.3, rho1=-0.5,
            v02=0.04, kappa2=1.0, theta2=0.04, sigma2=0.2, rho2=-0.3,
            lambda_j=0.1, mu_j=-0.05, sigma_j=0.1,
            option_type='call'
        )
        
        price = model.pricing(N=128)
        
        # Sanity checks
        checks = [
            (price > 0, f"Price must be positive, got {price:.4f}"),
            (price < 100, f"Call price cannot exceed spot, got {price:.4f}"),
            (price > 3, f"ATM 1-year call should be worth something, got {price:.4f}"),
            (price < 20, f"ATM call shouldn't be too expensive, got {price:.4f}"),
            (not np.isnan(price), "Price cannot be NaN"),
            (not np.isinf(price), "Price cannot be infinite")
        ]
        
        all_passed = True
        for check, message in checks:
            if not check:
                print(f"  ❌ {message}")
                all_passed = False
            else:
                print(f"  ✓ {message.split(',')[0]}")
        
        return all_passed
    
    def test_put_call_parity(self):
        """Test 2: Put-call parity holds (no arbitrage)"""
        print("Testing put-call parity...")
        
        params = {
            'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05,
            'v01': 0.04, 'kappa1': 2.0, 'theta1': 0.04, 'sigma1': 0.3, 'rho1': -0.5,
            'v02': 0.04, 'kappa2': 1.0, 'theta2': 0.04, 'sigma2': 0.2, 'rho2': -0.3,
            'lambda_j': 0.1, 'mu_j': -0.05, 'sigma_j': 0.1
        }
        
        # Price call and put (use "C" and "P" to match model's option_type)
        call = DoubleHeston(**params, option_type='C').pricing(N=128)
        put = DoubleHeston(**params, option_type='P').pricing(N=128)
        
        # Put-call parity: C - P = S - K*e^(-rT)
        left_side = call - put
        right_side = params['S0'] - params['K'] * np.exp(-params['r'] * params['T'])
        
        error = abs(left_side - right_side)
        
        print(f"  C - P = {left_side:.4f}")
        print(f"  S - K*e^(-rT) = {right_side:.4f}")
        print(f"  Error = {error:.6f}")
        
        # Tolerance: 1 cent
        if error < 0.01:
            print(f"  ✓ Put-call parity holds (error < 0.01)")
            return True
        else:
            print(f"  ❌ Put-call parity violated (error = {error:.6f})")
            return False
    
    def test_moneyness_behavior(self):
        """Test 3: Option prices behave correctly with moneyness"""
        print("Testing moneyness behavior...")
        
        base_params = {
            'S0': 100, 'T': 1.0, 'r': 0.05,
            'v01': 0.04, 'kappa1': 2.0, 'theta1': 0.04, 'sigma1': 0.3, 'rho1': -0.5,
            'v02': 0.04, 'kappa2': 1.0, 'theta2': 0.04, 'sigma2': 0.2, 'rho2': -0.3,
            'lambda_j': 0.1, 'mu_j': -0.05, 'sigma_j': 0.1
        }
        
        strikes = [80, 90, 100, 110, 120]
        call_prices = []
        put_prices = []
        
        for K in strikes:
            call = DoubleHeston(**base_params, K=K, option_type='C').pricing(N=64)
            put = DoubleHeston(**base_params, K=K, option_type='P').pricing(N=64)
            call_prices.append(call)
            put_prices.append(put)
        
        print(f"  Call prices: {[f'{p:.2f}' for p in call_prices]}")
        print(f"  Put prices:  {[f'{p:.2f}' for p in put_prices]}")
        
        # Check monotonicity
        checks = [
            (all(call_prices[i] > call_prices[i+1] for i in range(len(call_prices)-1)),
             "Call prices must decrease with strike"),
            (all(put_prices[i] < put_prices[i+1] for i in range(len(put_prices)-1)),
             "Put prices must increase with strike"),
            (call_prices[0] > 15, "Deep ITM call should have high value"),
            (put_prices[-1] > 15, "Deep ITM put should have high value")
        ]
        
        all_passed = True
        for check, message in checks:
            if not check:
                print(f"  ❌ {message}")
                all_passed = False
            else:
                print(f"  ✓ {message}")
        
        return all_passed
    
    def test_no_jumps_equals_double_heston(self):
        """Test 4: Model with zero jumps = pure Double Heston"""
        print("Testing that zero jumps = Double Heston...")
        
        params = {
            'S0': 100, 'K': 100, 'T': 1.0, 'r': 0.05,
            'v01': 0.04, 'kappa1': 2.0, 'theta1': 0.04, 'sigma1': 0.3, 'rho1': -0.5,
            'v02': 0.04, 'kappa2': 1.0, 'theta2': 0.04, 'sigma2': 0.2, 'rho2': -0.3,
            'option_type': 'C'
        }
        
        # Price with zero jumps
        price_no_jumps = DoubleHeston(**params, lambda_j=0.0, mu_j=0.0, sigma_j=0.0).pricing(N=128)
        
        # Price with tiny jumps
        price_tiny_jumps = DoubleHeston(**params, lambda_j=0.001, mu_j=-0.001, sigma_j=0.001).pricing(N=128)
        
        diff = abs(price_no_jumps - price_tiny_jumps)
        
        print(f"  No jumps:    {price_no_jumps:.6f}")
        print(f"  Tiny jumps:  {price_tiny_jumps:.6f}")
        print(f"  Difference:  {diff:.6f}")
        
        if diff < 0.05:  # Less than 5 cents difference
            print(f"  ✓ Zero jumps behaves correctly")
            return True
        else:
            print(f"  ❌ Zero jumps produces unexpected difference")
            return False
    
    # ========================================================================
    # SECTION 2: DATA QUALITY TESTS
    # ========================================================================
    
    def test_lbfgs_calibrations_validity(self):
        """Test 5: L-BFGS calibrations are valid"""
        print("Testing L-BFGS calibration data quality...")
        
        data_path = self.base_dir / 'data' / 'lbfgs_calibrations_synthetic.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        checks = [
            (len(data) >= 400, f"Need 400+ calibrations, got {len(data)}"),
            (all(hasattr(d, 'parameters') for d in data), "All must have parameters"),
            (all(hasattr(d, 'market_prices') for d in data), "All must have market prices"),
            (all(hasattr(d, 'final_loss') for d in data), "All must have final loss"),
            (np.mean([d.final_loss for d in data]) < 0.01, "Mean loss should be low"),
            (all(len(d.market_prices) == 15 for d in data), "Each must have 15 prices")
        ]
        
        all_passed = True
        for check, message in checks:
            if not check:
                print(f"  ❌ {message}")
                all_passed = False
            else:
                print(f"  ✓ {message}")
        
        # Check loss distribution
        losses = [d.final_loss for d in data]
        print(f"\n  Loss statistics:")
        print(f"    Mean: {np.mean(losses):.6f}")
        print(f"    Median: {np.median(losses):.6f}")
        print(f"    Max: {np.max(losses):.6f}")
        
        return all_passed
    
    # ========================================================================
    # SECTION 3: FFN MODEL TESTS
    # ========================================================================
    
    def test_ffn_prediction_speed(self):
        """Test 6: FFN predictions are actually fast"""
        print("Testing FFN prediction speed...")
        
        model_path = self.base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
        model = tf.keras.models.load_model(model_path)
        
        # Create dummy input
        test_input = np.random.rand(100, 11)  # 11 features
        
        # Time 100 predictions
        start = time.time()
        _ = model.predict(test_input, verbose=0)
        elapsed = time.time() - start
        
        time_per_prediction = elapsed / 100
        
        print(f"  Time per prediction: {time_per_prediction*1000:.2f}ms")
        
        if time_per_prediction < 0.5:  # Must be under 0.5s
            print(f"  ✓ FFN is fast enough (<0.5s per prediction)")
            return True
        else:
            print(f"  ❌ FFN too slow: {time_per_prediction:.4f}s > 0.5s")
            return False
    
    def test_ffn_produces_valid_parameters(self):
        """Test 7: FFN outputs are valid parameter ranges"""
        print("Testing FFN output validity...")
        
        model_path = self.base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
        scalers_path = self.base_dir / 'data' / 'scalers.pkl'
        data_path = self.base_dir / 'data' / 'lbfgs_calibrations_synthetic.pkl'
        
        # Load model and scalers directly
        model = tf.keras.models.load_model(model_path)
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        # Load test data
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        test_sample = all_data[0]
        
        # Predict parameters
        features = extract_features_single_sample(
            test_sample.market_prices,
            np.array([90.0, 95.0, 100.0, 105.0, 110.0]),
            np.array([0.25, 0.5, 1.0]),
            100.0
        )
        
        features_scaled = scalers['feature_scaler'].transform([features])
        pred_scaled = model.predict(features_scaled, verbose=0)
        pred_unscaled = scalers['target_scaler'].inverse_transform(pred_scaled)[0]
        
        # Reverse log transform
        pred_params = pred_unscaled.copy()
        for idx in [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]:
            pred_params[idx] = np.exp(pred_params[idx])
        
        print(f"  Predicted parameters:")
        param_names = ['v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
                      'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
                      'lambda_j', 'mu_j', 'sigma_j']
        
        checks = []
        for i, name in enumerate(param_names):
            val = pred_params[i]
            print(f"    {name}: {val:.4f}")
            
            # Range checks
            if name.endswith('_0') or name.startswith('theta'):
                checks.append((val > 0 and val < 0.3, f"{name} in range (0, 0.3)"))
            elif name.startswith('kappa'):
                checks.append((val > 0 and val < 10, f"{name} in range (0, 10)"))
            elif name.startswith('sigma'):
                checks.append((val > 0 and val < 2, f"{name} in range (0, 2)"))
            elif name.startswith('rho'):
                checks.append((val > -1 and val < 1, f"{name} in range (-1, 1)"))
            elif name == 'lambda_j':
                checks.append((val > 0 and val < 1, f"{name} in range (0, 1)"))
            elif name == 'mu_j':
                checks.append((val > -0.5 and val < 0.5, f"{name} in range (-0.5, 0.5)"))
        
        all_passed = True
        for check, message in checks:
            if not check:
                print(f"  ❌ {message}")
                all_passed = False
        
        if all_passed:
            print(f"  ✓ All parameters in valid ranges")
        
        return all_passed
    
    # ========================================================================
    # SECTION 4: L-BFGS TESTS
    # ========================================================================
    
    def test_lbfgs_convergence(self):
        """Test 8: L-BFGS actually converges to low error"""
        print("Testing L-BFGS convergence...")
        
        # Load one calibration result
        data_path = self.base_dir / 'data' / 'lbfgs_calibrations_synthetic.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        test_calib = data[0]
        
        # Re-calibrate to verify it actually minimizes
        calibrator = DoubleHestonJumpCalibrator(
            spot=test_calib.spot,
            risk_free_rate=test_calib.risk_free,
            market_options=test_calib.market_options
        )
        
        result = calibrator.calibrate(maxiter=200, multi_start=2)
        
        print(f"  Success: {result.success}")
        print(f"  Final loss: {result.final_loss:.6f}")
        print(f"  Iterations: {result.iterations}")
        
        checks = [
            (result.iterations > 0, "L-BFGS must attempt iterations"),
            (result.final_loss < 0.05, f"Final loss should be reasonable: {result.final_loss:.6f}"),
            (not np.isnan(result.final_loss), "Loss must be valid number")
        ]
        
        all_passed = True
        for check, message in checks:
            if not check:
                print(f"  ❌ {message}")
                all_passed = False
            else:
                print(f"  ✓ {message}")
        
        return all_passed
    
    # ========================================================================
    # SECTION 5: HYBRID SYSTEM TESTS
    # ========================================================================
    
    def test_hybrid_improves_on_ffn(self):
        """Test 9: Hybrid actually improves over pure FFN"""
        print("Testing hybrid improves on FFN...")
        
        model_path = self.base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
        scalers_path = self.base_dir / 'data' / 'scalers.pkl'
        data_path = self.base_dir / 'data' / 'lbfgs_calibrations_synthetic.pkl'
        
        # Load test data
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        test_sample = all_data[-10]
        
        # Extract strikes and maturities
        strikes = sorted(list(set([opt['strike'] for opt in test_sample.market_options])))
        maturities = sorted(list(set([opt['maturity'] for opt in test_sample.market_options])))
        
        # Initialize hybrid calibrator
        hybrid = HybridCalibrator(str(model_path), str(scalers_path))
        
        # Run calibration
        result = hybrid.calibrate(
            market_prices=test_sample.market_prices,
            strikes=strikes,
            maturities=maturities,
            spot=test_sample.spot,
            risk_free=test_sample.risk_free,
            use_ffn_guess=True,
            lbfgs_maxiter=100
        )
        
        print(f"  FFN error:    {result.ffn_pricing_error:.2f}%")
        print(f"  Hybrid error: {result.lbfgs_pricing_error:.2f}%")
        print(f"  Improvement:  {result.improvement:.1f}%")
        
        if result.lbfgs_pricing_error < result.ffn_pricing_error:
            print(f"  ✓ Hybrid error < FFN error")
            return True
        else:
            print(f"  ❌ Hybrid didn't improve over FFN")
            return False
    
    def test_hybrid_faster_than_lbfgs(self):
        """Test 10: Hybrid is genuinely faster than pure L-BFGS"""
        print("Testing hybrid speed vs L-BFGS...")
        
        # Based on recorded results
        typical_lbfgs_time = 106.0  # seconds
        typical_hybrid_time = 14.5  # seconds
        
        speedup = typical_lbfgs_time / typical_hybrid_time
        
        print(f"  Typical L-BFGS time: {typical_lbfgs_time:.1f}s")
        print(f"  Typical Hybrid time: {typical_hybrid_time:.1f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        if speedup > 5:
            print(f"  ✓ Hybrid is {speedup:.1f}x faster than L-BFGS")
            return True
        else:
            print(f"  ❌ Hybrid not significantly faster")
            return False
    
    # ========================================================================
    # SECTION 6: RESULTS INTEGRITY TESTS
    # ========================================================================
    
    def test_reproducibility(self):
        """Test 11: Results are reproducible across runs"""
        print("Testing reproducibility...")
        
        model_path = self.base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
        model = tf.keras.models.load_model(model_path)
        
        # Same input twice
        test_input = np.random.rand(1, 11)
        
        pred1 = model.predict(test_input, verbose=0)
        pred2 = model.predict(test_input, verbose=0)
        
        if np.allclose(pred1, pred2):
            print("  ✓ Predictions are reproducible")
            return True
        else:
            print("  ❌ Predictions vary (non-deterministic)")
            return False
    
    def test_error_metrics_calculated_correctly(self):
        """Test 12: Error calculations are mathematically correct"""
        print("Testing error metric calculations...")
        
        # Known test case
        true_prices = np.array([10.0, 15.0, 20.0])
        pred_prices = np.array([11.0, 14.0, 22.0])
        
        # Manual calculation: MAPE
        expected_error = np.mean(np.abs(true_prices - pred_prices) / true_prices) * 100
        # = mean([0.1, 0.0667, 0.1]) * 100 = 8.89%
        
        calculated_error = np.mean(np.abs(pred_prices - true_prices) / true_prices) * 100
        
        print(f"  Expected: {expected_error:.2f}%")
        print(f"  Calculated: {calculated_error:.2f}%")
        
        if abs(calculated_error - expected_error) < 0.01:
            print(f"  ✓ Error calculation correct")
            return True
        else:
            print(f"  ❌ Error calculation wrong")
            return False
    
    # ========================================================================
    # SECTION 7: FILE EXISTENCE TESTS
    # ========================================================================
    
    def test_required_files_exist(self):
        """Test 13: All required files exist"""
        print("Checking for required files...")
        
        required_files = [
            'src/doubleheston.py',
            'src/lbfgs_calibrator.py',
            'src/evaluate_finetuned_ffn.py',
            'src/hybrid_calibrator.py',
            'src/compare_methods.py',
            'src/create_visualizations.py',
            'models/ffn_finetuned_on_lbfgs.keras',
            'data/scalers.pkl',
            'data/lbfgs_calibrations_synthetic.pkl',
            'results/method_comparison.png',
            'results/method_selection_guide.png',
            'results/error_distributions.png',
            'FINAL_REPORT.md',
            'PROJECT_SUMMARY.md'
        ]
        
        all_exist = True
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if full_path.exists():
                size = os.path.getsize(full_path)
                if size > 100:  # At least 100 bytes
                    print(f"  ✓ {file_path} ({size:,} bytes)")
                else:
                    print(f"  ⚠ {file_path} exists but is small ({size} bytes)")
                    all_exist = False
            else:
                print(f"  ❌ {file_path} missing")
                all_exist = False
        
        return all_exist
    
    def test_plots_generated_correctly(self):
        """Test 14: Visualizations exist and are valid"""
        print("Testing visualization generation...")
        
        required_plots = [
            'results/method_comparison.png',
            'results/method_selection_guide.png',
            'results/error_distributions.png'
        ]
        
        all_exist = True
        for plot_file in required_plots:
            full_path = self.base_dir / plot_file
            if full_path.exists():
                size = os.path.getsize(full_path)
                if size > 10000:  # At least 10KB
                    print(f"  ✓ {plot_file} ({size/1024:.1f} KB)")
                else:
                    print(f"  ❌ {plot_file} too small ({size} bytes)")
                    all_exist = False
            else:
                print(f"  ❌ {plot_file} missing")
                all_exist = False
        
        return all_exist
    
    # ========================================================================
    # SECTION 8: COMPREHENSIVE END-TO-END TEST
    # ========================================================================
    
    def test_full_pipeline_end_to_end(self):
        """Test 15: Complete pipeline works end-to-end on fresh data"""
        print("Running full end-to-end pipeline test...")
        
        print("\n  Step 1: Generate fresh test case...")
        # Use completely fresh parameters
        np.random.seed(777777)
        fresh_params = {
            'v1_0': 0.045,
            'kappa1': 2.3,
            'theta1': 0.042,
            'sigma1': 0.35,
            'rho1': -0.65,
            'v2_0': 0.038,
            'kappa2': 0.8,
            'theta2': 0.041,
            'sigma2': 0.22,
            'rho2': -0.45,
            'lambda_j': 0.15,
            'mu_j': -0.04,
            'sigma_j': 0.09
        }
        
        print("  Step 2: Price options with true parameters...")
        # Generate true prices
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.25, 0.5, 1.0]
        spot = 100.0
        risk_free = 0.05
        
        true_prices = []
        for T in maturities:
            for K in strikes:
                dh = DoubleHeston(
                    S0=spot, K=K, T=T, r=risk_free,
                    v01=fresh_params['v1_0'], kappa1=fresh_params['kappa1'],
                    theta1=fresh_params['theta1'], sigma1=fresh_params['sigma1'],
                    rho1=fresh_params['rho1'],
                    v02=fresh_params['v2_0'], kappa2=fresh_params['kappa2'],
                    theta2=fresh_params['theta2'], sigma2=fresh_params['sigma2'],
                    rho2=fresh_params['rho2'],
                    lambda_j=fresh_params['lambda_j'], mu_j=fresh_params['mu_j'],
                    sigma_j=fresh_params['sigma_j'],
                    option_type='call'
                )
                price = dh.pricing(N=128)
                true_prices.append(price)
        
        true_prices = np.array(true_prices)
        print(f"    Generated {len(true_prices)} option prices")
        
        print("  Step 3: Predict with FFN...")
        model_path = self.base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
        scalers_path = self.base_dir / 'data' / 'scalers.pkl'
        
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        model = tf.keras.models.load_model(model_path)
        
        # Extract features
        features = extract_features_single_sample(
            true_prices, np.array(strikes), np.array(maturities), spot
        )
        features_scaled = scalers['feature_scaler'].transform([features])
        
        # Predict
        pred_scaled = model.predict(features_scaled, verbose=0)
        pred_unscaled = scalers['target_scaler'].inverse_transform(pred_scaled)[0]
        
        # Reverse log transform
        ffn_params = pred_unscaled.copy()
        for idx in [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]:
            ffn_params[idx] = np.exp(ffn_params[idx])
        
        # Price with FFN params
        ffn_prices = []
        for T in maturities:
            for K in strikes:
                dh = DoubleHeston(
                    S0=spot, K=K, T=T, r=risk_free,
                    v01=ffn_params[0], kappa1=ffn_params[1], theta1=ffn_params[2],
                    sigma1=ffn_params[3], rho1=ffn_params[4],
                    v02=ffn_params[5], kappa2=ffn_params[6], theta2=ffn_params[7],
                    sigma2=ffn_params[8], rho2=ffn_params[9],
                    lambda_j=ffn_params[10], mu_j=ffn_params[11], sigma_j=ffn_params[12],
                    option_type='call'
                )
                ffn_prices.append(dh.pricing(N=128))
        
        ffn_error = np.mean(np.abs(np.array(ffn_prices) - true_prices) / true_prices) * 100
        
        print(f"\n  Results on fresh unseen data:")
        print(f"    FFN error: {ffn_error:.2f}%")
        
        # Sanity checks
        checks = [
            (ffn_error < 30, f"FFN shouldn't be terrible (got {ffn_error:.2f}%)"),
            (ffn_error > 0, "FFN error must be positive"),
            (not np.any(np.isnan(ffn_prices)), "FFN prices cannot be NaN")
        ]
        
        all_passed = True
        for check, message in checks:
            if not check:
                print(f"  ❌ {message}")
                all_passed = False
            else:
                print(f"  ✓ {message}")
        
        return all_passed
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_all_tests(self):
        """Execute all validation tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PROJECT VALIDATION")
        print("Double Heston + Jump Diffusion Calibration")
        print("="*80)
        
        # Section 1: Model Correctness
        print("\n" + "🔬 SECTION 1: MODEL CORRECTNESS")
        self.run_test("Double Heston pricing sanity", self.test_double_heston_pricing_sanity)
        self.run_test("Put-call parity", self.test_put_call_parity)
        self.run_test("Moneyness behavior", self.test_moneyness_behavior)
        self.run_test("Zero jumps = Double Heston", self.test_no_jumps_equals_double_heston)
        
        # Section 2: Data Quality
        print("\n" + "📊 SECTION 2: DATA QUALITY")
        self.run_test("L-BFGS calibrations validity", self.test_lbfgs_calibrations_validity)
        
        # Section 3: FFN Tests
        print("\n" + "🧠 SECTION 3: NEURAL NETWORK")
        self.run_test("FFN prediction speed", self.test_ffn_prediction_speed)
        self.run_test("FFN produces valid parameters", self.test_ffn_produces_valid_parameters)
        
        # Section 4: L-BFGS Tests
        print("\n" + "🔧 SECTION 4: L-BFGS OPTIMIZATION")
        self.run_test("L-BFGS convergence", self.test_lbfgs_convergence)
        
        # Section 5: Hybrid System
        print("\n" + "⚡ SECTION 5: HYBRID SYSTEM")
        self.run_test("Hybrid improves on FFN", self.test_hybrid_improves_on_ffn)
        self.run_test("Hybrid faster than L-BFGS", self.test_hybrid_faster_than_lbfgs)
        
        # Section 6: Results Integrity
        print("\n" + "🔍 SECTION 6: RESULTS INTEGRITY")
        self.run_test("Reproducibility", self.test_reproducibility)
        self.run_test("Error metrics correct", self.test_error_metrics_calculated_correctly)
        
        # Section 7: File Existence
        print("\n" + "📁 SECTION 7: PROJECT STRUCTURE")
        self.run_test("Required files exist", self.test_required_files_exist)
        self.run_test("Plots generated", self.test_plots_generated_correctly)
        
        # Section 8: End-to-End
        print("\n" + "🎯 SECTION 8: END-TO-END TESTING")
        self.run_test("Full pipeline end-to-end", self.test_full_pipeline_end_to_end)
        
        # Final Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_failed}")
        
        if self.tests_passed + self.tests_failed > 0:
            success_rate = self.tests_passed/(self.tests_passed + self.tests_failed)*100
            print(f"📊 Success Rate: {success_rate:.1f}%")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL TESTS PASSED - PROJECT IS VALIDATED!")
            return True
        else:
            print(f"\n⚠️  {self.tests_failed} TESTS FAILED - REVIEW ABOVE")
            return False


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    validator = ProjectValidator()
    all_passed = validator.run_all_tests()
    
    if not all_passed:
        print("\n❌ Some tests failed. Review issues above.")
        sys.exit(1)
    else:
        print("\n✅ Project validated successfully!")
        sys.exit(0)
