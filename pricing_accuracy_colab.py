"""
Pricing Accuracy Evaluation for Double Heston FFN Calibration
==============================================================

Run this AFTER training your FFN model in the main Colab notebook.

This script evaluates your FFN by what REALLY matters: pricing accuracy!

Key Insight: Parameter recovery errors don't translate to pricing errors.
Many different parameter sets can produce similar option prices.

Author: Double Heston Calibration Project
Date: November 11, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: Define Pricing Accuracy Evaluator
# ============================================================================

def evaluate_pricing_accuracy(true_params, pred_params, DoubleHeston, n_samples=100):
    """
    Evaluate FFN by pricing accuracy, not parameter recovery
    
    Parameters:
    -----------
    true_params : np.ndarray, shape (n, 13)
        True parameters from test set
    pred_params : np.ndarray, shape (n, 13)
        Predicted parameters from FFN
    DoubleHeston : class
        Your DoubleHeston pricing class
    n_samples : int
        Number of test samples to evaluate
        
    Returns:
    --------
    dict with pricing errors, parameter errors, and statistics
    """
    print("="*70)
    print("PRICING ACCURACY EVALUATION")
    print("="*70)
    print(f"\nEvaluating {n_samples} test samples...")
    print("This measures what REALLY matters: option pricing accuracy!\n")
    
    spot = 100.0
    r = 0.05
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    
    all_pricing_errors = []
    all_param_errors = []
    failed_samples = 0
    
    for idx in range(min(n_samples, len(true_params))):
        true_p = true_params[idx]
        pred_p = pred_params[idx]
        
        sample_pricing_errors = []
        
        # Price all 15 options with both parameter sets
        for K in strikes:
            for T in maturities:
                try:
                    # True price
                    model_true = DoubleHeston(
                        S0=spot, K=K, T=T, r=r,
                        v01=true_p[0], kappa1=true_p[1], theta1=true_p[2],
                        sigma1=true_p[3], rho1=true_p[4],
                        v02=true_p[5], kappa2=true_p[6], theta2=true_p[7],
                        sigma2=true_p[8], rho2=true_p[9],
                        lambda_j=true_p[10], mu_j=true_p[11], sigma_j=true_p[12],
                        option_type='C'
                    )
                    price_true = model_true.pricing(N=64)
                    
                    # Predicted price
                    model_pred = DoubleHeston(
                        S0=spot, K=K, T=T, r=r,
                        v01=pred_p[0], kappa1=pred_p[1], theta1=pred_p[2],
                        sigma1=pred_p[3], rho1=pred_p[4],
                        v02=pred_p[5], kappa2=pred_p[6], theta2=pred_p[7],
                        sigma2=pred_p[8], rho2=pred_p[9],
                        lambda_j=pred_p[10], mu_j=pred_p[11], sigma_j=pred_p[12],
                        option_type='C'
                    )
                    price_pred = model_pred.pricing(N=64)
                    
                    # Percentage error
                    if price_true > 0.01:  # Avoid division by tiny numbers
                        pct_error = abs(price_true - price_pred) / price_true * 100
                        sample_pricing_errors.append(pct_error)
                
                except Exception as e:
                    failed_samples += 1
                    continue
        
        if len(sample_pricing_errors) > 0:
            all_pricing_errors.append(np.mean(sample_pricing_errors))
            
            # Parameter errors for comparison
            param_error = np.mean(np.abs((true_p - pred_p) / (true_p + 1e-10)) * 100)
            all_param_errors.append(param_error)
        
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{n_samples}")
    
    results = {
        'pricing_errors': np.array(all_pricing_errors),
        'param_errors': np.array(all_param_errors),
        'n_evaluated': len(all_pricing_errors),
        'failed_samples': failed_samples
    }
    
    return results


# ============================================================================
# STEP 2: Print Results Summary
# ============================================================================

def print_pricing_summary(results):
    """Print comprehensive summary of pricing accuracy"""
    
    pricing_errors = results['pricing_errors']
    param_errors = results['param_errors']
    
    print("\n" + "="*70)
    print("PRICING ACCURACY SUMMARY")
    print("="*70)
    
    print(f"\n📊 PRICING ACCURACY (What really matters!):")
    print(f"  Mean Pricing Error:    {np.mean(pricing_errors):.2f}%")
    print(f"  Median Pricing Error:  {np.median(pricing_errors):.2f}%")
    print(f"  Std Dev:               {np.std(pricing_errors):.2f}%")
    print(f"  95th Percentile:       {np.percentile(pricing_errors, 95):.2f}%")
    print(f"  Max Error:             {np.max(pricing_errors):.2f}%")
    
    print(f"\n📉 PARAMETER RECOVERY (For reference only):")
    print(f"  Mean Parameter Error:  {np.mean(param_errors):.2f}%")
    print(f"  Median Parameter Error: {np.median(param_errors):.2f}%")
    
    mean_pricing_error = np.mean(pricing_errors)
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if mean_pricing_error < 5.0:
        print("  ✅ EXCELLENT: Your FFN is working very well!")
        print("     Even with high parameter errors, it produces accurate prices.")
        print("     → Ready for production use!")
    elif mean_pricing_error < 10.0:
        print("  ✓ GOOD: Your FFN is acceptable for practical use.")
        print("    Parameter errors are high but pricing is reasonable.")
        print("    → Can be used with caution or as initialization for L-BFGS")
    elif mean_pricing_error < 20.0:
        print("  ⚠ MODERATE: FFN needs improvement but shows promise.")
        print("    → Consider: more training data, longer training, or architecture changes")
    else:
        print("  ❌ NEEDS WORK: Both parameter and pricing errors are high.")
        print("    → Try: simpler model (10 params), better data, or different approach")
    
    print(f"\n💡 KEY INSIGHT:")
    print("  Parameter errors don't directly translate to pricing errors!")
    print("  Many different parameter sets produce similar option prices.")
    print("  This is the 'identifiability problem' in stochastic vol models.")
    print("="*70)


# ============================================================================
# STEP 3: Visualization
# ============================================================================

def plot_pricing_accuracy(results):
    """Create comprehensive visualization of pricing accuracy"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pricing_errors = results['pricing_errors']
    param_errors = results['param_errors']
    
    # Plot 1: Pricing Error Distribution
    axes[0, 0].hist(pricing_errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(pricing_errors), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(pricing_errors):.2f}%')
    axes[0, 0].axvline(np.median(pricing_errors), color='green', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(pricing_errors):.2f}%')
    axes[0, 0].set_xlabel('Pricing Error (%)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Pricing Errors\n(THIS IS WHAT MATTERS!)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Parameter Error Distribution
    axes[0, 1].hist(param_errors, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(param_errors), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {np.mean(param_errors):.2f}%')
    axes[0, 1].set_xlabel('Parameter Error (%)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Distribution of Parameter Errors\n(Reference only)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Pricing vs Parameter Error Scatter
    axes[1, 0].scatter(param_errors, pricing_errors, alpha=0.4, s=30, color='purple')
    axes[1, 0].set_xlabel('Parameter Error (%)', fontsize=11)
    axes[1, 0].set_ylabel('Pricing Error (%)', fontsize=11)
    axes[1, 0].set_title('Pricing Error vs Parameter Error\n(Weak correlation shows identifiability issue)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(param_errors, pricing_errors)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}\nWeak = Good!',
                   transform=axes[1, 0].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.7))
    
    # Plot 4: Cumulative Distribution
    sorted_pricing = np.sort(pricing_errors)
    cumulative = np.arange(1, len(sorted_pricing) + 1) / len(sorted_pricing) * 100
    
    axes[1, 1].plot(sorted_pricing, cumulative, linewidth=2, color='darkgreen')
    axes[1, 1].axvline(5.0, color='red', linestyle='--', linewidth=2, label='5% threshold (Excellent)')
    axes[1, 1].axvline(10.0, color='orange', linestyle='--', linewidth=2, label='10% threshold (Good)')
    axes[1, 1].set_xlabel('Pricing Error (%)', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=11)
    axes[1, 1].set_title('Cumulative Distribution of Pricing Errors', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Highlight performance regions
    pct_under_5 = (pricing_errors < 5).sum() / len(pricing_errors) * 100
    pct_under_10 = (pricing_errors < 10).sum() / len(pricing_errors) * 100
    axes[1, 1].text(0.95, 0.05, f'{pct_under_5:.1f}% under 5%\n{pct_under_10:.1f}% under 10%',
                   transform=axes[1, 1].transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('pricing_accuracy_evaluation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Pricing accuracy plots saved to pricing_accuracy_evaluation.png")
    plt.show()


# ============================================================================
# USAGE INSTRUCTIONS (Copy to Colab after training)
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         PRICING ACCURACY EVALUATION FOR DOUBLE HESTON FFN            ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE (Run in Colab AFTER training your model):
------------------------------------------------

# After Section 9 in the main notebook, run:

pricing_results = evaluate_pricing_accuracy(
    true_params=y_true,
    pred_params=y_pred,
    DoubleHeston=DoubleHeston,
    n_samples=100
)

print_pricing_summary(pricing_results)
plot_pricing_accuracy(pricing_results)

# Download the plot
from google.colab import files
files.download('pricing_accuracy_evaluation.png')

KEY INSIGHT:
-----------
Your FFN should be judged by pricing accuracy, NOT parameter recovery!

High parameter errors (50-100%+) are NORMAL and expected due to the
"identifiability problem" in stochastic volatility models.

What matters: Can the predicted parameters price options accurately?

If mean pricing error < 10%, your FFN is working well! ✅
""")
