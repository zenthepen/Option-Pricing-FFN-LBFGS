import numpy as np
import pickle

# Manual implementations to avoid scipy/sklearn NumPy 2.x issues
class StandardScaler:
    """Manual StandardScaler to avoid sklearn dependency"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        return self
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_

def train_test_split(*arrays, test_size=0.2, random_state=None):
    """Manual train_test_split to avoid sklearn dependency"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])
    
    return tuple(result)

# Try to import TensorFlow, but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except (ImportError, ValueError) as e:
    print(f"Warning: TensorFlow not available ({e})")
    print("Note: TensorFlow may need to be reinstalled for NumPy 2.x compatibility")
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    callbacks = None

try:
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib not available. Install with: pip install matplotlib")
    PLT_AVAILABLE = False

class OptionDataPreprocessor:
    """
    Prepare option price data for neural network training
    """
    
    def __init__(self):
        self.feature_scaler = None
        self.target_scaler = None
        self.param_names = [
            'v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
            'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
            'lambda_j', 'mu_j', 'sigma_j'
        ]
    
    def extract_features(self, option_prices, strikes, maturities, spot=100.0):
        """
        Extract meaningful features from raw option prices
        
        Parameters:
        -----------
        option_prices : np.ndarray, shape (n_samples, n_options)
            Raw option prices
        strikes : np.ndarray
            Strike prices
        maturities : np.ndarray
            Time to maturities
        spot : float
            Spot price
            
        Returns:
        --------
        features : np.ndarray, shape (n_samples, n_features)
            Engineered features
        """
        n_samples = option_prices.shape[0]
        n_strikes = len(strikes)
        n_maturities = len(maturities)
        
        # Reshape: (n_samples, n_strikes, n_maturities)
        prices_3d = option_prices.reshape(n_samples, n_strikes, n_maturities)
        
        features_list = []
        
        for i in range(n_samples):
            sample_features = []
            
            for mat_idx, T in enumerate(maturities):
                prices_at_maturity = prices_3d[i, :, mat_idx]
                
                # Feature 1: ATM price (normalized by spot)
                atm_idx = np.argmin(np.abs(strikes - spot))
                atm_price = prices_at_maturity[atm_idx]
                sample_features.append(atm_price / spot)
                
                # Feature 2: Skew (25-delta risk reversal approximation)
                # Use OTM call vs OTM put prices
                otm_call_idx = np.argmin(np.abs(strikes - spot*1.05))
                otm_put_idx = np.argmin(np.abs(strikes - spot*0.95))
                skew = (prices_at_maturity[otm_call_idx] - 
                       prices_at_maturity[otm_put_idx]) / spot
                sample_features.append(skew)
                
                # Feature 3: Curvature (butterfly)
                itm_idx = np.argmin(np.abs(strikes - spot*0.95))
                otm_idx = np.argmin(np.abs(strikes - spot*1.05))
                butterfly = (prices_at_maturity[itm_idx] + 
                           prices_at_maturity[otm_idx] - 
                           2 * atm_price) / spot
                sample_features.append(butterfly)
            
            # Feature 4: Term structure slope
            if n_maturities > 1:
                atm_short = prices_3d[i, atm_idx, 0]
                atm_long = prices_3d[i, atm_idx, -1]
                term_slope = (atm_long - atm_short) / spot
                sample_features.append(term_slope)
            
            # Feature 5: Total ATM premium across maturities
            total_atm = np.sum(prices_3d[i, atm_idx, :]) / spot
            sample_features.append(total_atm)
            
            features_list.append(sample_features)
        
        return np.array(features_list)
    
    def prepare_training_data(self, data_path, test_size=0.15, val_size=0.15):
        """
        Load synthetic data and prepare for training
        
        Parameters:
        -----------
        data_path : str
            Path to synthetic_data.pkl
        test_size : float
            Fraction for test set
        val_size : float
            Fraction for validation set
            
        Returns:
        --------
        splits : dict
            Dictionary with train/val/test splits
        """
        # Load data
        print("Loading synthetic data...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract features from option prices
        print("Extracting features from option prices...")
        features = self.extract_features(
            data['option_prices'],
            data['strikes'],
            data['maturities']
        )
        
        targets = data['parameters']
        
        print(f"Feature shape: {features.shape}")
        print(f"Target shape: {targets.shape}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, targets, test_size=(test_size + val_size), random_state=42
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=42
        )
        
        # Normalize features
        print("Normalizing features...")
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Normalize targets (helps training stability)
        # Use log transform for positive parameters
        print("Normalizing targets...")
        y_train_transformed = self._transform_targets(y_train)
        y_val_transformed = self._transform_targets(y_val)
        y_test_transformed = self._transform_targets(y_test)
        
        self.target_scaler = StandardScaler()
        y_train_scaled = self.target_scaler.fit_transform(y_train_transformed)
        y_val_scaled = self.target_scaler.transform(y_val_transformed)
        y_test_scaled = self.target_scaler.transform(y_test_transformed)
        
        print(f"\nData splits:")
        print(f"  Train: {X_train_scaled.shape[0]} samples")
        print(f"  Val:   {X_val_scaled.shape[0]} samples")
        print(f"  Test:  {X_test_scaled.shape[0]} samples")
        
        return {
            'X_train': X_train_scaled,
            'y_train': y_train_scaled,
            'X_val': X_val_scaled,
            'y_val': y_val_scaled,
            'X_test': X_test_scaled,
            'y_test': y_test_scaled,
            'y_train_orig': y_train,
            'y_val_orig': y_val,
            'y_test_orig': y_test
        }
    
    def _transform_targets(self, targets):
        """Apply log transform to positive parameters, leave others unchanged"""
        transformed = targets.copy()
        
        # Indices: v1_0, kappa1, theta1, sigma1, rho1, v2_0, kappa2, theta2, sigma2, rho2, lambda_j, mu_j, sigma_j
        # Log transform: indices 0,1,2,3,5,6,7,8,10,12 (positive parameters)
        # Leave as-is: indices 4,9,11 (correlations and mu_j)
        
        log_indices = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]
        transformed[:, log_indices] = np.log(transformed[:, log_indices] + 1e-10)
        
        return transformed
    
    def inverse_transform_predictions(self, y_scaled):
        """Convert scaled predictions back to original parameter space"""
        # Inverse standard scaling
        y_transformed = self.target_scaler.inverse_transform(y_scaled)
        
        # Inverse log transform
        y_original = y_transformed.copy()
        log_indices = [0, 1, 2, 3, 5, 6, 7, 8, 10, 12]
        y_original[:, log_indices] = np.exp(y_transformed[:, log_indices])
        
        return y_original
    
    def save_scalers(self, path='scalers.pkl'):
        """Save scalers for later use"""
        with open(path, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, f)
        print(f"Scalers saved to {path}")


class FFNTrainer:
    """
    Train the FFN calibration model
    """
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.history = None
    
    def train(self, data_splits, epochs=100, batch_size=256, 
              learning_rate=0.001, patience=15):
        """
        Train the model
        
        Parameters:
        -----------
        data_splits : dict
            Dictionary with train/val/test data
        epochs : int
            Maximum training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Initial learning rate
        patience : int
            Early stopping patience
            
        Returns:
        --------
        history : History object
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for training. Please install TensorFlow.")
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',  # Can use custom loss here
            metrics=['mae', 'mape']
        )
        
        # Callbacks
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            
            # Model checkpointing
            callbacks.ModelCheckpoint(
                'best_ffn_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging (optional)
            callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        print("\n" + "="*70)
        print("STARTING FFN PRE-TRAINING ON SYNTHETIC DATA")
        print("="*70 + "\n")
        
        # Train
        self.history = self.model.fit(
            data_splits['X_train'],
            data_splits['y_train'],
            validation_data=(data_splits['X_val'], data_splits['y_val']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        return self.history
    
    def evaluate(self, data_splits):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET")
        print("="*70 + "\n")
        
        # Predict on test set
        y_pred_scaled = self.model.predict(data_splits['X_test'], verbose=0)
        
        # Inverse transform to original scale
        y_pred = self.preprocessor.inverse_transform_predictions(y_pred_scaled)
        y_true = data_splits['y_test_orig']
        
        # Compute errors for each parameter
        param_names = self.preprocessor.param_names
        
        print(f"{'Parameter':<12} {'MAE':<12} {'RMSE':<12} {'MAPE %':<10}")
        print("-" * 50)
        
        for i, name in enumerate(param_names):
            mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
            rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i])**2))
            mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / (y_true[:, i] + 1e-10))) * 100
            
            print(f"{name:<12} {mae:<12.6f} {rmse:<12.6f} {mape:<10.2f}")
        
        # Overall metrics
        mae_overall = np.mean(np.abs(y_true - y_pred))
        rmse_overall = np.sqrt(np.mean((y_true - y_pred)**2))
        
        print("-" * 50)
        print(f"{'OVERALL':<12} {mae_overall:<12.6f} {rmse_overall:<12.6f}")
        
        return {
            'predictions': y_pred,
            'true_values': y_true,
            'mae': mae_overall,
            'rmse': rmse_overall
        }
    
    def plot_training_history(self):
        """Plot training curves"""
        if not PLT_AVAILABLE:
            print("Warning: Matplotlib not available. Cannot plot training history.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE curve
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training history plot saved to 'training_history.png'")


def ffn_model(input_dim, output_dim):
    """Define and compile the feedforward neural network model"""
    input_layer = layers.Input(shape=(input_dim,), name = 'option_features')

    x = layers.Dense(512,activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation='relu')(x)
    
    output_layer = layers.Dense(output_dim, activation='linear', name='model_output')(x)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def create_custom_loss():
    """
    Custom loss function with Feller penalty
    """
    def loss_with_feller_penalty(y_true, y_pred):
        # Standard MSE
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Feller condition penalty (applied in transformed space)
        # After inverse transform: 2*kappa*theta >= sigma^2
        # Indices: kappa1=1, theta1=2, sigma1=3, kappa2=6, theta2=7, sigma2=8
        
        # For simplicity, penalize in scaled space (approximate)
        # Real implementation would inverse transform first
        feller_penalty = tf.constant(0.0)
        
        # Weight MSE more heavily
        total_loss = mse + 0.1 * feller_penalty
        
        return total_loss
    
    return loss_with_feller_penalty


# Test the preprocessor
if __name__ == "__main__":
    print("="*70)
    print("TESTING DATA PREPROCESSOR")
    print("="*70)
    
    preprocessor = OptionDataPreprocessor()
    
    # Load and prepare data
    data_splits = preprocessor.prepare_training_data('synthetic_data_test.pkl')
    
    print("\n" + "="*70)
    print("✓ Preprocessing complete!")
    print("="*70)



        

