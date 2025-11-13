from doubleheston import DoubleHeston
import numpy as np
import pickle

def sample_beta_correlation(a=2, b=5):
    """
    Sample from Beta(a,b) distribution scaled to [-1, 1] for correlation
    Using accept-reject method to avoid scipy dependency
    """
    # Simple method: use uniform and transform
    # For correlation, we want bias toward negative values
    # Use a simpler approach: sample from truncated normal
    rho = np.random.normal(loc=-0.3, scale=0.3)
    return np.clip(rho, -0.99, 0.99)

def generate_synthetic_data(n_samples=10000, save_path="synthetic_data.pkl"):
    """
    Generate synthetic training data for Double Heston + Jump Diffusion calibration
    
    Parameters:
    -----------
    n_samples : int
        Number of parameter sets to generate
    save_path : str
        Path to save the pickle file
        
    Returns:
    --------
    dataset : dict
        Dictionary containing parameters, prices, strikes, maturities
    """
    print("="*70)
    print("SYNTHETIC DATA GENERATION - DOUBLE HESTON + JUMPS")
    print("="*70)
    
    dataset = {
        'parameters': [],
        'option_prices': []
    }
    
    spot = 100.0
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    
    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Spot: ${spot}")
    print(f"  Strikes: {strikes}")
    print(f"  Maturities: {maturities}")
    print(f"  Options per sample: {len(strikes) * len(maturities)}")
    
    print(f"\nGenerating samples...")
    
    rejected_jump_dominance = 0
    samples_generated = 0
    
    while samples_generated < n_samples:
        # Sample Heston parameters from realistic distributions (UNCHANGED)
        params = {
            'v1_0': np.random.lognormal(mean=np.log(0.04), sigma=0.5),
            'kappa1': np.random.lognormal(mean=np.log(2.0), sigma=0.8),
            'theta1': np.random.lognormal(mean=np.log(0.04), sigma=0.5),
            'sigma1': np.random.lognormal(mean=np.log(0.3), sigma=0.5),
            'rho1': sample_beta_correlation(),
            
            'v2_0': np.random.lognormal(mean=np.log(0.04), sigma=0.5),
            'kappa2': np.random.lognormal(mean=np.log(0.5), sigma=0.8),
            'theta2': np.random.lognormal(mean=np.log(0.04), sigma=0.5),
            'sigma2': np.random.lognormal(mean=np.log(0.2), sigma=0.5),
            'rho2': sample_beta_correlation(),
            
            # BOUNDED JUMP PARAMETERS (MARKET-REALISTIC)
            # Jump intensity: 0.05-0.30 jumps/year (typical equity markets: 0.1-0.2)
            'lambda_j': np.random.uniform(0.05, 0.30),
            
            # Jump mean: -8% to -1% (small negative jumps, crashes not booms)
            'mu_j': np.random.uniform(-0.08, -0.01),
            
            # Jump volatility: 3-12% (typical jump sizes 5-10%)
            'sigma_j': np.random.uniform(0.03, 0.12)
        }
        
        # Enforce Heston parameter constraints
        params['v1_0'] = np.clip(params['v1_0'], 0.001, 0.5)
        params['v2_0'] = np.clip(params['v2_0'], 0.001, 0.5)
        params['kappa1'] = np.clip(params['kappa1'], 0.1, 10.0)
        params['kappa2'] = np.clip(params['kappa2'], 0.1, 10.0)
        params['theta1'] = np.clip(params['theta1'], 0.001, 0.5)
        params['theta2'] = np.clip(params['theta2'], 0.001, 0.5)
        params['sigma1'] = np.clip(params['sigma1'], 0.01, 2.0)
        params['sigma2'] = np.clip(params['sigma2'], 0.01, 2.0)
        params['rho1'] = np.clip(params['rho1'], -0.99, 0.99)
        params['rho2'] = np.clip(params['rho2'], -0.99, 0.99)
        
        # JUMP DOMINANCE CHECK: Ensure jumps don't dominate total volatility
        # Jump contribution to variance = lambda * (mu^2 + sigma^2)
        jump_vol_contribution = params['lambda_j'] * (params['mu_j']**2 + params['sigma_j']**2)
        total_heston_vol = params['v1_0'] + params['v2_0']
        
        # Reject if jumps contribute more than 30% of total volatility
        if jump_vol_contribution > 0.3 * total_heston_vol:
            rejected_jump_dominance += 1
            continue  # Skip this sample and generate a new one
        
        # Price options for all strike-maturity combinations
        prices = []
        for K in strikes:
            for T in maturities:
                model = DoubleHeston(
                    S0=spot, K=K, T=T, r=0.05,
                    v01=params['v1_0'], kappa1=params['kappa1'],
                    theta1=params['theta1'], sigma1=params['sigma1'],
                    rho1=params['rho1'], v02=params['v2_0'],
                    kappa2=params['kappa2'], theta2=params['theta2'],
                    sigma2=params['sigma2'], rho2=params['rho2'],
                    lambda_j=params['lambda_j'], mu_j=params['mu_j'],
                    sigma_j=params['sigma_j'], option_type='C'
                )
                
                price = model.pricing(N=64)  # N=64 for speed
                
                # Skip if price is invalid (can happen with extreme parameters)
                if price < 0 or np.isnan(price) or np.isinf(price):
                    price = 0.01  # Minimum price
                
                # Add realistic market noise (1-2% bid-ask spread)
                noise_pct = np.random.uniform(0.005, 0.02)
                noise = np.random.normal(0, abs(price) * noise_pct)
                prices.append(max(0.01, price + noise))  # Ensure positive
        
        # Store valid sample
        dataset['parameters'].append(list(params.values()))
        dataset['option_prices'].append(prices)
        samples_generated += 1
        
        # Progress
        if samples_generated % 1000 == 0:
            print(f"  Progress: {samples_generated}/{n_samples} ({samples_generated/n_samples*100:.1f}%)")
    
    # Convert to arrays
    dataset['parameters'] = np.array(dataset['parameters'])
    dataset['option_prices'] = np.array(dataset['option_prices'])
    dataset['param_names'] = [
        'v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
        'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
        'lambda_j', 'mu_j', 'sigma_j'
    ]
    dataset['strikes'] = strikes
    dataset['maturities'] = maturities
    
    # Save
    print(f"\nSaving to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n✓ Synthetic data generation complete!")
    print(f"  Total samples generated: {n_samples}")
    print(f"  Rejected due to jump dominance: {rejected_jump_dominance} ({rejected_jump_dominance/(n_samples+rejected_jump_dominance)*100:.1f}%)")
    print(f"  Parameters per sample: 13")
    print(f"  Options per sample: {len(strikes) * len(maturities)}")
    print(f"  Dataset size: {dataset['parameters'].shape}")
    print(f"  Price array size: {dataset['option_prices'].shape}")
    
    # Validation statistics
    params_array = dataset['parameters']
    print(f"\n📊 Jump Parameter Statistics (Bounded):")
    print(f"  lambda_j: [{params_array[:, 10].min():.3f}, {params_array[:, 10].max():.3f}] (target: [0.05, 0.30])")
    print(f"  mu_j:     [{params_array[:, 11].min():.3f}, {params_array[:, 11].max():.3f}] (target: [-0.08, -0.01])")
    print(f"  sigma_j:  [{params_array[:, 12].min():.3f}, {params_array[:, 12].max():.3f}] (target: [0.03, 0.12])")
    
    return dataset


if __name__ == "__main__":
    # Generate a small test dataset first
    print("Generating test dataset (100 samples)...\n")
    test_data = generate_synthetic_data(n_samples=100, save_path="synthetic_data_test.pkl")
    
    print("\n" + "="*70)
    print("Test successful! Ready to generate full dataset.")
    print("To generate full dataset, run:")
    print("  dataset = generate_synthetic_data(n_samples=10000)")
    print("="*70)

