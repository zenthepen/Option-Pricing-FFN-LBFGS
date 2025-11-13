"""
Fine-tune existing FFN model on L-BFGS calibration data.

This script:
1. Loads pre-trained FFN model (trained on 100k synthetic samples)
2. Loads 500 L-BFGS calibrations from real market data
3. Fine-tunes with very low learning rate (1e-5)
4. Expected: Reduce pricing error from 31% → 10-15%

Author: Zen
Date: November 2025
"""

import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Check TensorFlow/Keras availability
try:
    from tensorflow import keras
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not found. Install with: pip install tensorflow")
    sys.exit(1)


def check_files_exist():
    """Verify all required files are present."""
    required_files = {
        'best_ffn_model.keras': 'Pre-trained FFN model',
        'scalers.pkl': 'Feature and target scalers',
        'lbfgs_calibrations_synthetic.pkl': 'L-BFGS calibration data'
    }
    
    print("="*70)
    print("CHECKING REQUIRED FILES")
    print("="*70)
    
    missing = []
    for file, desc in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✓ {file:<40} ({size:.1f} KB) - {desc}")
        else:
            print(f"✗ {file:<40} MISSING - {desc}")
            missing.append(file)
    
    if missing:
        print(f"\n❌ ERROR: {len(missing)} required file(s) missing!")
        print("Please ensure all files are in the current directory.")
        sys.exit(1)
    
    print()


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
    features : list
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
    
    # Should have exactly 11 features (9 from maturity loop + 2 aggregate features)
    assert len(features) == 11, f"Expected 11 features, got {len(features)}"
    
    return features


def transform_targets(targets):
    """
    Apply log transform to positive parameters (MUST match training).
    
    Parameters:
    -----------
    targets : np.ndarray, shape (n_samples, 13)
        Raw parameter values
        
    Returns:
    --------
    transformed : np.ndarray, shape (n_samples, 13)
        Log-transformed targets
    """
    transformed = targets.copy()
    
    # Indices for log transform: v1_0, kappa1, theta1, sigma1, v2_0, kappa2, theta2, sigma2, lambda_j, sigma_j
    # Indices: 0, 1, 2, 3, 5, 6, 7, 8, 10, 12
    # Leave as-is: rho1 (4), rho2 (9), mu_j (11)
    log_indices = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]
    
    # Apply log with small epsilon to avoid log(0)
    transformed[:, log_indices] = np.log(transformed[:, log_indices] + 1e-10)
    
    return transformed


def prepare_lbfgs_data(lbfgs_calibrations, feature_scaler, target_scaler):
    """
    Prepare L-BFGS calibration data for fine-tuning.
    
    Parameters:
    -----------
    lbfgs_calibrations : list
        List of CalibrationResult objects
    feature_scaler : StandardScaler
        Fitted feature scaler from training
    target_scaler : StandardScaler
        Fitted target scaler from training
        
    Returns:
    --------
    X_scaled : np.ndarray, shape (n_samples, 10)
        Scaled features
    y_scaled : np.ndarray, shape (n_samples, 13)
        Scaled targets
    """
    print("="*70)
    print("PREPARING L-BFGS CALIBRATION DATA")
    print("="*70)
    
    # Fixed configuration (must match training)
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    spot = 100.0
    
    # Parameter names in order
    param_names = ['v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
                   'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
                   'lambda_j', 'mu_j', 'sigma_j']
    
    X_raw = []
    y_raw = []
    
    print(f"\nProcessing {len(lbfgs_calibrations)} calibrations...")
    
    for i, calib in enumerate(lbfgs_calibrations):
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(lbfgs_calibrations)} ({(i+1)/len(lbfgs_calibrations)*100:.1f}%)")
        
        # Extract features from market prices
        try:
            # Get market prices (should be 15 values)
            market_prices = calib.market_prices
            if len(market_prices) != 15:
                print(f"  ⚠️  Warning: Sample {i} has {len(market_prices)} prices (expected 15), skipping")
                continue
            
            # Extract features
            features = extract_features_single_sample(market_prices, strikes, maturities, spot)
            
            # Get parameters
            params = [calib.parameters[name] for name in param_names]
            
            # Check for invalid values
            if np.any(np.isnan(features)) or np.any(np.isnan(params)):
                print(f"  ⚠️  Warning: Sample {i} contains NaN values, skipping")
                continue
            
            if np.any(np.isinf(features)) or np.any(np.isinf(params)):
                print(f"  ⚠️  Warning: Sample {i} contains Inf values, skipping")
                continue
            
            # Check positive parameters are positive
            positive_indices = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]
            if np.any(np.array(params)[positive_indices] <= 0):
                print(f"  ⚠️  Warning: Sample {i} has non-positive parameters, skipping")
                continue
            
            X_raw.append(features)
            y_raw.append(params)
            
        except Exception as e:
            print(f"  ⚠️  Error processing sample {i}: {str(e)}, skipping")
            continue
    
    print(f"\n✓ Successfully processed {len(X_raw)} / {len(lbfgs_calibrations)} samples")
    
    if len(X_raw) < 50:
        print(f"\n❌ ERROR: Too few valid samples ({len(X_raw)} < 50)")
        print("Cannot proceed with fine-tuning.")
        sys.exit(1)
    
    # Convert to arrays
    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)
    
    print(f"\nData shapes:")
    print(f"  Features (raw): {X_raw.shape}")
    print(f"  Targets (raw):  {y_raw.shape}")
    
    # Apply transformations (MUST match training)
    print(f"\nApplying transformations...")
    
    # 1. Transform features using fitted scaler
    X_scaled = feature_scaler.transform(X_raw)
    print(f"  ✓ Features scaled")
    
    # 2. Log transform targets
    y_transformed = transform_targets(y_raw)
    print(f"  ✓ Targets log-transformed")
    
    # 3. Scale transformed targets
    y_scaled = target_scaler.transform(y_transformed)
    print(f"  ✓ Targets scaled")
    
    # Validation
    print(f"\nTransformed data shapes:")
    print(f"  X_scaled: {X_scaled.shape}")
    print(f"  y_scaled: {y_scaled.shape}")
    
    # Check for NaN/Inf after transformation
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        print("  ⚠️  WARNING: X_scaled contains NaN or Inf values!")
    
    if np.any(np.isnan(y_scaled)) or np.any(np.isinf(y_scaled)):
        print("  ⚠️  WARNING: y_scaled contains NaN or Inf values!")
    
    print(f"\nFeature statistics (scaled):")
    print(f"  Mean: {X_scaled.mean(axis=0)}")
    print(f"  Std:  {X_scaled.std(axis=0)}")
    
    print(f"\nTarget statistics (scaled):")
    print(f"  Mean: {y_scaled.mean(axis=0)}")
    print(f"  Std:  {y_scaled.std(axis=0)}")
    print()
    
    return X_scaled, y_scaled


def finetune_model(model, X_train, y_train, X_val, y_val):
    """
    Fine-tune pre-trained model with low learning rate.
    
    Parameters:
    -----------
    model : keras.Model
        Pre-trained FFN model
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
        
    Returns:
    --------
    model : keras.Model
        Fine-tuned model
    history : dict
        Training history
    """
    print("="*70)
    print("FINE-TUNING MODEL")
    print("="*70)
    
    # Recompile with very low learning rate
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print(f"\nFine-tuning configuration:")
    print(f"  Learning rate: 1e-5 (very low for fine-tuning)")
    print(f"  Optimizer: Adam")
    print(f"  Loss: MSE")
    print(f"  Batch size: 32")
    print(f"  Max epochs: 50")
    print(f"  Early stopping: patience=10")
    print(f"  ReduceLROnPlateau: patience=5, factor=0.5")
    
    print(f"\nTraining data:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples: {len(X_val)}")
    print(f"  Train/Val split: {len(X_train)/(len(X_train)+len(X_val))*100:.1f}% / {len(X_val)/(len(X_train)+len(X_val))*100:.1f}%")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'ffn_finetuned_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    print(f"\nStarting fine-tuning...")
    print("-" * 70)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("-" * 70)
    print(f"✓ Fine-tuning complete!")
    print()
    
    return model, history


def print_summary(history):
    """Print training summary statistics."""
    print("="*70)
    print("FINE-TUNING SUMMARY")
    print("="*70)
    
    # Get best epoch
    val_losses = history.history['val_loss']
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = val_losses[best_epoch - 1]
    initial_val_loss = val_losses[0]
    final_val_loss = val_losses[-1]
    
    print(f"\nTraining History:")
    print(f"  Initial val_loss: {initial_val_loss:.6f}")
    print(f"  Best val_loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Final val_loss: {final_val_loss:.6f}")
    print(f"  Epochs trained: {len(val_losses)}")
    print(f"  Improvement: {(initial_val_loss - best_val_loss) / initial_val_loss * 100:.1f}%")
    
    # Training loss
    if 'loss' in history.history:
        train_losses = history.history['loss']
        print(f"\n  Initial train_loss: {train_losses[0]:.6f}")
        print(f"  Final train_loss: {train_losses[-1]:.6f}")
    
    # MAE if available
    if 'val_mae' in history.history:
        val_mae = history.history['val_mae']
        print(f"\n  Initial val_MAE: {val_mae[0]:.6f}")
        print(f"  Final val_MAE: {val_mae[-1]:.6f}")
    
    print(f"\nExpected Improvement:")
    print(f"  Before fine-tuning: ~31% mean pricing error")
    print(f"  After fine-tuning: ~10-15% mean pricing error (2-3x improvement)")
    print(f"  Actual improvement: Run pricing accuracy evaluation to measure")
    
    print(f"\nFiles Created:")
    print(f"  ✓ ffn_finetuned_on_lbfgs.keras (fine-tuned model)")
    print(f"  ✓ ffn_finetuned_checkpoint.keras (backup)")
    
    print(f"\nNext Steps:")
    print(f"  1. Run pricing accuracy evaluation")
    print(f"  2. Compare: Synthetic-only FFN vs Fine-tuned FFN")
    print(f"  3. Test on held-out validation data")
    print(f"  4. Build hybrid FFN→L-BFGS system")
    
    print("="*70)
    print()


def main():
    """Main fine-tuning workflow."""
    print("\n" + "="*70)
    print("FFN FINE-TUNING ON L-BFGS CALIBRATION DATA")
    print("="*70)
    print("\nObjective: Fine-tune FFN to reduce pricing error from 31% → 10-15%")
    print()
    
    # Step 1: Check files
    check_files_exist()
    
    # Step 2: Load existing model
    print("="*70)
    print("LOADING EXISTING FFN MODEL")
    print("="*70)
    
    try:
        model = keras.models.load_model('best_ffn_model.keras')
        print(f"✓ Model loaded successfully")
        print(f"  Architecture: {len(model.layers)} layers")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # Count parameters
        trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
        print(f"  Trainable parameters: {trainable_params:,}")
        print()
    except Exception as e:
        print(f"❌ ERROR loading model: {str(e)}")
        sys.exit(1)
    
    # Step 3: Load scalers
    print("="*70)
    print("LOADING SCALERS")
    print("="*70)
    
    try:
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        
        print(f"✓ Feature scaler loaded")
        print(f"  Type: {type(feature_scaler).__name__}")
        print(f"  Features: {len(feature_scaler.mean_)} dimensions")
        
        print(f"✓ Target scaler loaded")
        print(f"  Type: {type(target_scaler).__name__}")
        print(f"  Targets: {len(target_scaler.mean_)} parameters")
        print()
    except Exception as e:
        print(f"❌ ERROR loading scalers: {str(e)}")
        sys.exit(1)
    
    # Step 4: Load L-BFGS calibrations
    print("="*70)
    print("LOADING L-BFGS CALIBRATIONS")
    print("="*70)
    
    try:
        with open('lbfgs_calibrations_synthetic.pkl', 'rb') as f:
            lbfgs_data = pickle.load(f)
        
        print(f"✓ Loaded {len(lbfgs_data)} calibrations")
        
        # Show sample
        if len(lbfgs_data) > 0:
            sample = lbfgs_data[0]
            print(f"\nSample calibration:")
            print(f"  Date: {sample.date}")
            print(f"  Spot: ${sample.spot:.2f}")
            print(f"  Market prices: {len(sample.market_prices)} options")
            print(f"  Parameters: {len(sample.parameters)} values")
            print(f"  Loss: {sample.final_loss:.6f}")
        print()
    except Exception as e:
        print(f"❌ ERROR loading L-BFGS data: {str(e)}")
        sys.exit(1)
    
    # Step 5: Prepare fine-tuning data
    X_scaled, y_scaled = prepare_lbfgs_data(lbfgs_data, feature_scaler, target_scaler)
    
    # Step 6: Split into train/validation
    print("="*70)
    print("SPLITTING DATA")
    print("="*70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled,
        test_size=0.15,
        random_state=42,
        shuffle=True
    )
    
    print(f"\nData split (85% train / 15% validation):")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print()
    
    # Step 7: Fine-tune model
    model, history = finetune_model(model, X_train, y_train, X_val, y_val)
    
    # Step 8: Save fine-tuned model
    print("="*70)
    print("SAVING FINE-TUNED MODEL")
    print("="*70)
    
    try:
        model.save('ffn_finetuned_on_lbfgs.keras')
        print(f"✓ Model saved to: ffn_finetuned_on_lbfgs.keras")
        
        # Also save history
        with open('finetuning_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        print(f"✓ Training history saved to: finetuning_history.pkl")
        print()
    except Exception as e:
        print(f"⚠️  Warning: Error saving model: {str(e)}")
        print(f"   Model is still in memory, manually save if needed.")
        print()
    
    # Step 9: Print summary
    print_summary(history)
    
    print("✓ FINE-TUNING COMPLETE!")
    print("\nYou can now:")
    print("  1. Load the fine-tuned model: keras.models.load_model('ffn_finetuned_on_lbfgs.keras')")
    print("  2. Run pricing accuracy evaluation to measure improvement")
    print("  3. Compare with the original model (best_ffn_model.keras)")
    print()


if __name__ == "__main__":
    main()
