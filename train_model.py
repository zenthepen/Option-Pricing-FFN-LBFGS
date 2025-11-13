"""
Training script for Double Heston + Jump Diffusion calibration neural network

Usage:
    python train_model.py --data synthetic_10k.pkl --epochs 100 --batch_size 256
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from ffn import OptionDataPreprocessor, FFNTrainer, ffn_model

def main():
    parser = argparse.ArgumentParser(description='Train Double Heston calibration neural network')
    parser.add_argument('--data', type=str, default='synthetic_10k.pkl',
                       help='Path to synthetic data file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Training batch size')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--model_save_path', type=str, default='best_model.keras',
                       help='Path to save best model')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DOUBLE HESTON + JUMPS CALIBRATION - NEURAL NETWORK TRAINING")
    print("="*70)
    
    # 1. Prepare data
    print("\n[1/4] Loading and preprocessing data...")
    preprocessor = OptionDataPreprocessor()
    data_splits = preprocessor.prepare_training_data(args.data)
    
    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['val']
    X_test, y_test = data_splits['test']
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # 2. Create model
    print("\n[2/4] Creating neural network model...")
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = ffn_model(input_dim=input_dim, output_dim=output_dim)
    
    print(f"  Architecture: {input_dim} → [512, 256, 128, 64] → {output_dim}")
    print(f"  Total parameters: {model.count_params():,}")
    
    # 3. Train
    print("\n[3/4] Training model...")
    trainer = FFNTrainer(model, preprocessor)
    
    history = trainer.train(
        data_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        model_save_path=args.model_save_path
    )
    
    # 4. Evaluate
    print("\n[4/4] Evaluating model on test set...")
    metrics = trainer.evaluate(data_splits)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest model saved to: {args.model_save_path}")
    print(f"\nTest Set Performance:")
    print(f"  Overall MAE:  {metrics['overall']['mae']:.6f}")
    print(f"  Overall RMSE: {metrics['overall']['rmse']:.6f}")
    print(f"  Overall MAPE: {metrics['overall']['mape']:.2f}%")
    
    print("\nPer-Parameter MAE (scaled space):")
    param_names = ['v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
                   'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
                   'lambda_j', 'mu_j', 'sigma_j']
    for i, name in enumerate(param_names):
        print(f"  {name:10s}: {metrics['per_parameter']['mae'][i]:.6f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
