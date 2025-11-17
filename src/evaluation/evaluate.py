"""
Compare All Calibration Methods

This script performs comprehensive comparison of three calibration approaches:
1. Pure L-BFGS: Slow but most accurate (0.34% error, 106s)
2. Fine-tuned FFN: Fast but less accurate (5% error, 0.09s)  
3. Hybrid FFN→L-BFGS: Best balance (1-3% error, 10-20s)

For each method, evaluates on test set and compares:
- Pricing accuracy (mean, median, percentiles)
- Runtime performance
- Accuracy vs speed trade-offs
- Use case recommendations

Author: Zen
Date: November 2025
"""

import numpy as np
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from doubleheston import DoubleHeston
from evaluate_finetuned_ffn import FinetunedFFNEvaluator
from hybrid_calibrator import HybridCalibrator
from lbfgs_calibrator import DoubleHestonJumpCalibrator

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)


class MethodComparator:
    """Compare all three calibration methods on test data"""
    
    def __init__(self, base_dir: Path):
        """
        Initialize comparator
        
        Args:
            base_dir: Base directory containing models, data, etc.
        """
        self.base_dir = base_dir
        
        # Load test data
        test_data_path = base_dir / 'data' / 'lbfgs_calibrations_synthetic.pkl'
        with open(test_data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        # Use last 50 samples for comparison (faster than 100)
        self.test_data = all_data[-50:]
        print(f"Loaded {len(self.test_data)} test samples\n")
        
        # Initialize evaluators
        model_path = base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
        scalers_path = base_dir / 'data' / 'scalers.pkl'
        
        self.ffn_evaluator = FinetunedFFNEvaluator(
            str(model_path),
            str(scalers_path),
            str(test_data_path)
        )
        self.ffn_evaluator.test_data = self.test_data  # Override with smaller test set
        
        self.hybrid_calibrator = HybridCalibrator(str(model_path), str(scalers_path))
    
    def evaluate_ffn_method(self, verbose=True) -> Dict:
        """Evaluate FFN-only method"""
        if verbose:
            print("="*80)
            print("METHOD 1: FFN-ONLY (Fast Predictions)")
            print("="*80)
        
        results = self.ffn_evaluator.evaluate_all(verbose=verbose)
        
        if verbose:
            print()
        
        return results
    
    def evaluate_hybrid_method(self, verbose=True, max_samples=10) -> Dict:
        """
        Evaluate Hybrid FFN→L-BFGS method
        
        Args:
            max_samples: Maximum samples to test (L-BFGS is slow)
        """
        if verbose:
            print("="*80)
            print("METHOD 2: HYBRID (FFN + L-BFGS Refinement)")
            print("="*80)
            print(f"Testing on {max_samples} samples (L-BFGS is time-consuming)...\n")
        
        pricing_errors = []
        ffn_times = []
        lbfgs_times = []
        total_times = []
        improvements = []
        
        for i, calib in enumerate(self.test_data[:max_samples]):
            if verbose:
                print(f"Sample {i+1}/{max_samples}...")
            
            # Extract strikes and maturities
            strikes = sorted(list(set([opt['strike'] for opt in calib.market_options])))
            maturities = sorted(list(set([opt['maturity'] for opt in calib.market_options])))
            
            # Run hybrid calibration
            result = self.hybrid_calibrator.calibrate(
                market_prices=calib.market_prices,
                strikes=strikes,
                maturities=maturities,
                spot=calib.spot,
                risk_free=calib.risk_free,
                use_ffn_guess=True,
                lbfgs_maxiter=150  # Reduced for speed
            )
            
            pricing_errors.append(result.lbfgs_pricing_error)
            ffn_times.append(result.ffn_time)
            lbfgs_times.append(result.lbfgs_time)
            total_times.append(result.total_time)
            improvements.append(result.improvement)
            
            if verbose:
                print(f"  ✓ Total: {result.total_time:.1f}s, Error: {result.lbfgs_pricing_error:.2f}%\n")
        
        if verbose:
            print(f"\n✓ Completed {max_samples} hybrid calibrations\n")
        
        return {
            'pricing_errors': {
                'mean': np.mean(pricing_errors),
                'median': np.median(pricing_errors),
                'std': np.std(pricing_errors),
                'min': np.min(pricing_errors),
                'max': np.max(pricing_errors),
                'p95': np.percentile(pricing_errors, 95)
            },
            'timing': {
                'ffn_time': np.mean(ffn_times),
                'lbfgs_time': np.mean(lbfgs_times),
                'total_time': np.mean(total_times),
                'per_sample': np.mean(total_times)
            },
            'improvement': np.mean(improvements),
            'raw_errors': pricing_errors,
            'raw_times': total_times
        }
    
    def evaluate_lbfgs_method(self, verbose=True, max_samples=5) -> Dict:
        """
        Evaluate pure L-BFGS method (cold start, no FFN guess)
        
        Args:
            max_samples: Maximum samples to test (very slow!)
        """
        if verbose:
            print("="*80)
            print("METHOD 3: PURE L-BFGS (Cold Start, No FFN)")
            print("="*80)
            print(f"Testing on {max_samples} samples (this will take several minutes)...\n")
        
        pricing_errors = []
        runtimes = []
        
        for i, calib in enumerate(self.test_data[:max_samples]):
            if verbose:
                print(f"Sample {i+1}/{max_samples}...")
            
            # Extract strikes and maturities
            strikes = sorted(list(set([opt['strike'] for opt in calib.market_options])))
            maturities = sorted(list(set([opt['maturity'] for opt in calib.market_options])))
            
            # Create market_options format
            market_options = []
            for idx, (T, K) in enumerate([(T, K) for T in maturities for K in strikes]):
                market_options.append({
                    'strike': K,
                    'maturity': T,
                    'price': calib.market_prices[idx],
                    'option_type': 'call'
                })
            
            # Initialize L-BFGS calibrator
            calibrator = DoubleHestonJumpCalibrator(calib.spot, calib.risk_free, market_options)
            
            # Run calibration (cold start, multi-start)
            start_time = time.time()
            result = calibrator.calibrate(maxiter=200, multi_start=3)
            elapsed = time.time() - start_time
            
            pricing_errors.append(result.final_loss * 100)  # Convert to percentage
            runtimes.append(elapsed)
            
            if verbose:
                print(f"  ✓ Time: {elapsed:.1f}s, Error: {result.final_loss*100:.2f}%\n")
        
        if verbose:
            print(f"\n✓ Completed {max_samples} L-BFGS calibrations\n")
        
        return {
            'pricing_errors': {
                'mean': np.mean(pricing_errors),
                'median': np.median(pricing_errors),
                'std': np.std(pricing_errors),
                'min': np.min(pricing_errors),
                'max': np.max(pricing_errors),
                'p95': np.percentile(pricing_errors, 95)
            },
            'timing': {
                'total_time': np.mean(runtimes),
                'per_sample': np.mean(runtimes)
            },
            'raw_errors': pricing_errors,
            'raw_times': runtimes
        }
    
    def print_comparison_table(self, ffn_results, hybrid_results, lbfgs_results):
        """Print comprehensive comparison table"""
        
        print("="*100)
        print("COMPREHENSIVE METHOD COMPARISON")
        print("="*100)
        print()
        
        # Accuracy comparison
        print("📊 PRICING ACCURACY")
        print("-" * 100)
        print(f"{'Metric':<20} {'FFN-Only':>15} {'Hybrid':>15} {'Pure L-BFGS':>15} {'Best Method':<20}")
        print("-" * 100)
        
        metrics = [
            ('Mean Error', 'mean'),
            ('Median Error', 'median'),
            ('Std Deviation', 'std'),
            ('Min Error', 'min'),
            ('Max Error', 'max'),
            ('95th Percentile', 'p95')
        ]
        
        for label, key in metrics:
            ffn_val = ffn_results['pricing_errors'][key]
            hybrid_val = hybrid_results['pricing_errors'][key]
            lbfgs_val = lbfgs_results['pricing_errors'][key]
            
            best = min(ffn_val, hybrid_val, lbfgs_val)
            if best == lbfgs_val:
                best_method = "L-BFGS ⭐"
            elif best == hybrid_val:
                best_method = "Hybrid ⭐"
            else:
                best_method = "FFN ⭐"
            
            print(f"{label:<20} {ffn_val:>14.2f}% {hybrid_val:>14.2f}% {lbfgs_val:>14.2f}% {best_method:<20}")
        
        print()
        
        # Performance comparison
        print("⚡ RUNTIME PERFORMANCE")
        print("-" * 100)
        print(f"{'Metric':<20} {'FFN-Only':>15} {'Hybrid':>15} {'Pure L-BFGS':>15} {'Speedup':<20}")
        print("-" * 100)
        
        ffn_time = ffn_results['timing']['per_sample']
        hybrid_time = hybrid_results['timing']['total_time']
        lbfgs_time = lbfgs_results['timing']['per_sample']
        
        print(f"{'Per Calibration':<20} {ffn_time*1000:>13.1f}ms {hybrid_time:>14.1f}s {lbfgs_time:>14.1f}s")
        print(f"{'Speedup vs L-BFGS':<20} {lbfgs_time/ffn_time:>14.1f}x {lbfgs_time/hybrid_time:>14.1f}x {'1.0x':>15} {'FFN: fastest'}")
        print(f"{'Speedup vs Hybrid':<20} {hybrid_time/ffn_time:>14.1f}x {'1.0x':>15} {hybrid_time/lbfgs_time:>14.2f}x {'FFN: fastest'}")
        
        print()
        
        # Trade-off analysis
        print("🎯 ACCURACY vs SPEED TRADE-OFFS")
        print("-" * 100)
        
        ffn_score = ffn_time / ffn_results['pricing_errors']['mean']
        hybrid_score = hybrid_time / hybrid_results['pricing_errors']['mean']
        lbfgs_score = lbfgs_time / lbfgs_results['pricing_errors']['mean']
        
        print(f"{'Method':<20} {'Time':>12} {'Error':>12} {'Efficiency Score':>18} {'Best For'}")
        print("-" * 100)
        print(f"{'FFN-Only':<20} {ffn_time:>11.3f}s {ffn_results['pricing_errors']['mean']:>11.2f}% {ffn_score:>17.6f} {'Real-time systems'}")
        print(f"{'Hybrid':<20} {hybrid_time:>11.3f}s {hybrid_results['pricing_errors']['mean']:>11.2f}% {hybrid_score:>17.6f} {'Production (balanced)'}")
        print(f"{'Pure L-BFGS':<20} {lbfgs_time:>11.3f}s {lbfgs_results['pricing_errors']['mean']:>11.2f}% {lbfgs_score:>17.6f} {'Offline/ground truth'}")
        
        print()
        print("Note: Lower Efficiency Score = better time/error ratio")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 100)
        print()
        print("Use FFN-Only when:")
        print("  • Need millisecond latency (<100ms)")
        print("  • Can tolerate ~5% pricing error")
        print("  • Real-time trading, screening, rapid calibration")
        print(f"  • Performance: {ffn_results['pricing_errors']['mean']:.2f}% error, {ffn_time*1000:.0f}ms")
        print()
        
        print("Use Hybrid when:")
        print("  • Need accuracy close to L-BFGS (1-3% error)")
        print("  • Can afford 10-20 seconds per calibration")
        print("  • Production risk management, daily calibrations")
        print(f"  • Performance: {hybrid_results['pricing_errors']['mean']:.2f}% error, {hybrid_time:.1f}s")
        print(f"  • {lbfgs_time/hybrid_time:.1f}x faster than pure L-BFGS")
        print()
        
        print("Use Pure L-BFGS when:")
        print("  • Need highest possible accuracy (<1% error)")
        print("  • Time is not critical (minutes acceptable)")
        print("  • Model validation, research, ground truth")
        print(f"  • Performance: {lbfgs_results['pricing_errors']['mean']:.2f}% error, {lbfgs_time:.1f}s")
        print()
        
        print("="*100)
    
    def save_results(self, ffn_results, hybrid_results, lbfgs_results):
        """Save comparison results"""
        results = {
            'ffn': ffn_results,
            'hybrid': hybrid_results,
            'lbfgs': lbfgs_results,
            'summary': {
                'ffn_error': ffn_results['pricing_errors']['mean'],
                'ffn_time': ffn_results['timing']['per_sample'],
                'hybrid_error': hybrid_results['pricing_errors']['mean'],
                'hybrid_time': hybrid_results['timing']['total_time'],
                'lbfgs_error': lbfgs_results['pricing_errors']['mean'],
                'lbfgs_time': lbfgs_results['timing']['per_sample'],
                'speedup_hybrid_vs_lbfgs': lbfgs_results['timing']['per_sample'] / hybrid_results['timing']['total_time'],
                'speedup_ffn_vs_lbfgs': lbfgs_results['timing']['per_sample'] / ffn_results['timing']['per_sample']
            }
        }
        
        output_path = self.base_dir / 'results' / 'method_comparison.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        return results


def main():
    """Run comprehensive method comparison"""
    
    print("="*100)
    print("DOUBLE HESTON CALIBRATION: METHOD COMPARISON")
    print("="*100)
    print()
    print("This script compares three calibration methods:")
    print("  1. FFN-Only: Fast predictions (~90ms)")
    print("  2. Hybrid: FFN + L-BFGS refinement (~10-20s)")
    print("  3. Pure L-BFGS: Cold start optimization (~60-120s)")
    print()
    print("Warning: This will take 15-30 minutes to complete!")
    print()
    
    base_dir = Path(__file__).parent.parent
    comparator = MethodComparator(base_dir)
    
    # Method 1: FFN-Only (fast, 50 samples)
    print("\n" + "="*100)
    print("STAGE 1/3: Evaluating FFN-Only Method")
    print("="*100)
    ffn_results = comparator.evaluate_ffn_method(verbose=True)
    
    # Method 2: Hybrid (moderate speed, 10 samples)
    print("\n" + "="*100)
    print("STAGE 2/3: Evaluating Hybrid Method")
    print("="*100)
    hybrid_results = comparator.evaluate_hybrid_method(verbose=True, max_samples=10)
    
    # Method 3: Pure L-BFGS (slow, 5 samples only)
    print("\n" + "="*100)
    print("STAGE 3/3: Evaluating Pure L-BFGS Method")
    print("="*100)
    lbfgs_results = comparator.evaluate_lbfgs_method(verbose=True, max_samples=5)
    
    # Print comparison
    print("\n" * 2)
    comparator.print_comparison_table(ffn_results, hybrid_results, lbfgs_results)
    
    # Save results
    comparator.save_results(ffn_results, hybrid_results, lbfgs_results)
    
    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()
