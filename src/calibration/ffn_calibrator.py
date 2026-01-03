

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
    
    prices_2d = prices.reshape(len(strikes), len(maturities))
    
    features = []
    
   
    atm_idx = np.argmin(np.abs(strikes - spot))
    
    
    for mat_idx in range(len(maturities)):
        prices_at_mat = prices_2d[:, mat_idx]
        atm_price = prices_at_mat[atm_idx]
        
       
        features.append(atm_price / spot)
        
        otm_call_idx = np.argmin(np.abs(strikes - spot*1.05))
        otm_put_idx = np.argmin(np.abs(strikes - spot*0.95))
        skew = (prices_at_mat[otm_call_idx] - prices_at_mat[otm_put_idx]) / spot
        features.append(skew)
        
       
        itm_idx = np.argmin(np.abs(strikes - spot*0.95))
        otm_idx = np.argmin(np.abs(strikes - spot*1.05))
        butterfly = (prices_at_mat[itm_idx] + prices_at_mat[otm_idx] - 2*atm_price) / spot
        features.append(butterfly)
    
   
    atm_short = prices_2d[atm_idx, 0]  
    atm_long = prices_2d[atm_idx, -1]   
    term_slope = (atm_long - atm_short) / spot
    features.append(term_slope)
    
   
    total_atm = np.sum(prices_2d[atm_idx, :]) / spot
    features.append(total_atm)
    
   
    assert len(features) == 11, f"Expected 11 features, got {len(features)}"
    
    return np.array(features)


class FinetunedFFNEvaluator:
    """Evaluates fine-tuned FFN model pricing accuracy"""
    
    
    PARAM_NAMES = ['v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
                   'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
                   'lambda_j', 'mu_j', 'sigma_j']
    

    LOG_TRANSFORM_INDICES = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]
    
    def __init__(self, model_path, scalers_path, test_data_path):
        print("Loading fine-tuned model and data...")
        
       
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Loaded model: {model_path}")
        
        
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"✓ Loaded scalers: {scalers_path}")
        
       
        with open(test_data_path, 'rb') as f:
            all_data = pickle.load(f)
        print(f"✓ Loaded {len(all_data)} calibrations")
        
       
        self.test_data = all_data[-100:]
        print(f"✓ Using last {len(self.test_data)} samples as test set\n")
    
    def extract_features(self, option_prices, strikes, maturities, spot):
       
        return extract_features_single_sample(option_prices, strikes, maturities, spot)
    
    def inverse_transform_predictions(self, pred_scaled):
    
        pred_unscaled = self.scalers['target_scaler'].inverse_transform(pred_scaled)[0]
        
      
        pred_params = pred_unscaled.copy()
        for idx in self.LOG_TRANSFORM_INDICES:
            pred_params[idx] = np.exp(pred_params[idx])
        
        return pred_params
    
    def price_options_with_params(self, params, spot, risk_free, strikes, maturities):
    
        prices = []
        
        for T in maturities:
            for K in strikes:
                model_instance = DoubleHeston(
                    S0=spot,  
                    K=K,    
                    T=T,
                    r=risk_free,
                    v01=params[0],    
                    kappa1=params[1],
                    theta1=params[2],
                    sigma1=params[3],
                    rho1=params[4],
                    v02=params[5],   
                    kappa2=params[6],
                    theta2=params[7],
                    sigma2=params[8],
                    rho2=params[9],
                    lambda_j=params[10],
                    mu_j=params[11],
                    sigma_j=params[12],
                    option_type='call' 
                )
                
               
                price = model_instance.pricing(N=128)
                prices.append(price)
        
        return np.array(prices)
    
    def evaluate_single_sample(self, calib):
    
        strikes = sorted(list(set([opt['strike'] for opt in calib.market_options])))
        maturities = sorted(list(set([opt['maturity'] for opt in calib.market_options])))
        
        features = self.extract_features(
            calib.market_prices,
            np.array(strikes),
            np.array(maturities),
            calib.spot
        )
        
       
        features_scaled = self.scalers['feature_scaler'].transform([features])
        pred_scaled = self.model.predict(features_scaled, verbose=0)
        pred_params = self.inverse_transform_predictions(pred_scaled)
        
       
        true_params = np.array([calib.parameters[name] for name in self.PARAM_NAMES])
        
        pred_prices = self.price_options_with_params(
            pred_params,
            calib.spot,
            calib.risk_free,
            strikes,  
            maturities
        )
        
       
        true_prices = np.array(calib.market_prices)
        abs_errors = np.abs(pred_prices - true_prices)
        pct_errors = (abs_errors / true_prices) * 100
        pricing_error = np.mean(pct_errors)
        
        
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
        
        pricing_errors = [r['pricing_error'] for r in results]
        parameter_errors = [r['parameter_error'] for r in results]
        
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
                'baseline_error': 31.0,
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
    
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'ffn_finetuned_on_lbfgs.keras'
    scalers_path = base_dir / 'data' / 'scalers.pkl'
    test_data_path = base_dir / 'data' / 'lbfgs_calibrations_synthetic.pkl'
    
    
    for path in [model_path, scalers_path, test_data_path]:
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
    
   
    evaluator = FinetunedFFNEvaluator(
        str(model_path),
        str(scalers_path),
        str(test_data_path)
    )
    
    
    stats = evaluator.evaluate_all(verbose=True)
    
    
    evaluator.print_report(stats)
    
  
    results_path = base_dir / 'results' / 'ffn_evaluation_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
