

import numpy as np
import pandas as pd
import pickle
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from double_heston import DoubleHeston


@dataclass
class CalibrationResult:
    """
    Container for calibration results.
    
    Note: calibration_time and iterations may be None for synthetic data
    that was not actually calibrated (parameters randomly sampled).
    Only trust these values when from actual L-BFGS optimization runs.
    """
    date: str
    spot: float
    risk_free: float
    parameters: Dict[str, float]
    market_prices: np.ndarray
    model_prices: np.ndarray
    market_options: List[Dict]
    final_loss: float
    calibration_time: float = None  # Optional: None for synthetic data
    success: bool = True
    iterations: int = None  # Optional: None for synthetic data
    message: str = ""


class DoubleHestonJumpCalibrator:

    
    def __init__(self, spot: float, risk_free_rate: float, market_options: List[Dict]):
        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.market_options = market_options
        self.market_prices = np.array([opt['price'] for opt in market_options])
        
        self.param_names = [
            'v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
            'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
            'lambda_j', 'mu_j', 'sigma_j'
        ]
        
        self.n_calls = 0
        self.best_loss = np.inf
        
    def transform_params(self, x: np.ndarray) -> Dict[str, float]:
        """Transform unconstrained optimization variables to model parameters."""
        params = {}
        
        # Positive parameters (exponential transformation)
        params['v1_0'] = np.exp(x[0])
        params['kappa1'] = np.exp(x[1])
        params['theta1'] = np.exp(x[2])
        params['sigma1'] = np.exp(x[3])
        
        # Correlation parameters (tanh transformation for [-1, 1] range)
        params['rho1'] = np.tanh(x[4])
        
        # Second factor parameters
        params['v2_0'] = np.exp(x[5])
        params['kappa2'] = np.exp(x[6])
        params['theta2'] = np.exp(x[7])
        params['sigma2'] = np.exp(x[8])
        params['rho2'] = np.tanh(x[9])
        
        # Jump parameters
        params['lambda_j'] = np.exp(x[10])  # Jump intensity (positive)
        params['mu_j'] = x[11]  # Jump mean (unconstrained)
        params['sigma_j'] = np.exp(x[12])  # Jump volatility (positive)
        
        return params
    
    def inverse_transform_params(self, params: Dict[str, float]) -> np.ndarray:
        """Transform model parameters to unconstrained optimization variables."""
        x = np.zeros(13)
        
        x[0] = np.log(params['v1_0'])
        x[1] = np.log(params['kappa1'])
        x[2] = np.log(params['theta1'])
        x[3] = np.log(params['sigma1'])
        x[4] = np.arctanh(np.clip(params['rho1'], -0.999, 0.999))
        
        x[5] = np.log(params['v2_0'])
        x[6] = np.log(params['kappa2'])
        x[7] = np.log(params['theta2'])
        x[8] = np.log(params['sigma2'])
        x[9] = np.arctanh(np.clip(params['rho2'], -0.999, 0.999))
        
        x[10] = np.log(params['lambda_j'])
        x[11] = params['mu_j']
        x[12] = np.log(params['sigma_j'])
        
        return x
    
    def compute_feller_penalty(self, params: Dict[str, float]) -> float:
        """Compute penalty for violating Feller condition."""
        penalty1 = max(0, params['sigma1']**2 - 2*params['kappa1']*params['theta1'])
        penalty2 = max(0, params['sigma2']**2 - 2*params['kappa2']*params['theta2'])
        
        return 1000.0 * (penalty1 + penalty2)
    
    def compute_loss(self, x: np.ndarray) -> float:
        """Compute calibration loss (relative MSE + penalties)."""
        self.n_calls += 1
        
        try:
            # Transform parameters
            params = self.transform_params(x)

            # Compute model prices for all options
            model_prices = []
            for opt in self.market_options:
                try:
                    dh = DoubleHeston(
                        S0=self.spot,
                        K=opt['strike'],
                        T=opt['maturity'],
                        r=self.risk_free_rate,
                        v01=params['v1_0'],
                        kappa1=params['kappa1'],
                        theta1=params['theta1'],
                        sigma1=params['sigma1'],
                        rho1=params['rho1'],
                        v02=params['v2_0'],
                        kappa2=params['kappa2'],
                        theta2=params['theta2'],
                        sigma2=params['sigma2'],
                        rho2=params['rho2'],
                        lambda_j=params['lambda_j'],
                        mu_j=params['mu_j'],
                        sigma_j=params['sigma_j'],
                        option_type=opt['option_type']
                    )
                    price = dh.pricing()
                    
                    if np.isnan(price) or np.isinf(price) or price <= 0:
                        return 1e10
                    
                    model_prices.append(price)
                    
                except Exception as e:
                    return 1e10
            
            model_prices = np.array(model_prices)
            
            # Relative MSE
            relative_errors = (model_prices - self.market_prices) / self.market_prices
            mse = np.mean(relative_errors**2)

            # Add Feller condition penalty
            feller_penalty = self.compute_feller_penalty(params)
            
            total_loss = mse + feller_penalty
            
            if total_loss < self.best_loss:
                self.best_loss = total_loss
            
            return total_loss
            
        except Exception as e:
            return 1e10
    
    def get_initial_guess(self, guess_type: int = 0) -> np.ndarray:
        """Generate initial parameter guess for optimization."""
        
        if guess_type == 0:
            # Standard initial guess (literature values)
            params = {
                'v1_0': 0.04, 'kappa1': 2.5, 'theta1': 0.04, 'sigma1': 0.3, 'rho1': -0.7,
                'v2_0': 0.04, 'kappa2': 0.5, 'theta2': 0.04, 'sigma2': 0.2, 'rho2': -0.5,
                'lambda_j': 0.15, 'mu_j': -0.04, 'sigma_j': 0.08
            }
            
        elif guess_type == 1:
            # Perturbed guess (add ±20% noise to standard values)
            base_params = {
                'v1_0': 0.04, 'kappa1': 2.5, 'theta1': 0.04, 'sigma1': 0.3, 'rho1': -0.7,
                'v2_0': 0.04, 'kappa2': 0.5, 'theta2': 0.04, 'sigma2': 0.2, 'rho2': -0.5,
                'lambda_j': 0.15, 'mu_j': -0.04, 'sigma_j': 0.08
            }
            
            # Perturb positive parameters by ±20%
            params = {}
            for name, value in base_params.items():
                if name in ['rho1', 'rho2', 'mu_j']:
                    # Keep correlations and jump mean close to base
                    params[name] = value * (1 + np.random.uniform(-0.15, 0.15))
                else:
                    # Perturb positive parameters
                    params[name] = value * (1 + np.random.uniform(-0.20, 0.20))
            
            # Clip correlations to valid range
            params['rho1'] = np.clip(params['rho1'], -0.95, -0.3)
            params['rho2'] = np.clip(params['rho2'], -0.95, -0.3)
            
        else:  # guess_type == 2
            # Market-implied initial guess
            atm_options = [opt for opt in self.market_options 
                          if 0.95 < opt['strike']/self.spot < 1.05]
            
            if atm_options:
                avg_price = np.mean([opt['price'] for opt in atm_options])
                avg_maturity = np.mean([opt['maturity'] for opt in atm_options])
                # Rough implied variance estimate
                implied_var = (avg_price / self.spot) / np.sqrt(avg_maturity)
                implied_var = max(0.01, min(0.1, implied_var))
            else:
                implied_var = 0.04
            
            params = {
                'v1_0': implied_var, 'kappa1': 2.0, 'theta1': implied_var, 
                'sigma1': 0.4, 'rho1': -0.6,
                'v2_0': implied_var, 'kappa2': 0.7, 'theta2': implied_var, 
                'sigma2': 0.25, 'rho2': -0.4,
                'lambda_j': 0.12, 'mu_j': -0.03, 'sigma_j': 0.07
            }
        
        return self.inverse_transform_params(params)
    
    def calibrate(self, maxiter: int = 300, multi_start: int = 3) -> CalibrationResult:
        """
        Calibrate Double Heston model parameters to market option prices.
        
        Args:
            maxiter: Maximum L-BFGS iterations
            multi_start: Number of random starting points to try
            
        Returns:
            CalibrationResult with best parameters found
        """
        start_time = time.time()
        
        best_result = None
        best_loss = np.inf
        
        for start_idx in range(multi_start):
            self.n_calls = 0
            self.best_loss = np.inf
            
            x0 = self.get_initial_guess(guess_type=start_idx % 3)
            
            try:
                result = minimize(
                    fun=self.compute_loss,
                    x0=x0,
                    method='L-BFGS-B',
                    options={
                        'maxiter': maxiter,
                        'ftol': 1e-9,
                        'gtol': 1e-6,
                        'disp': False
                    }
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    
                    params = self.transform_params(result.x)
                    
                    # Compute final model prices
                    model_prices = []
                    for opt in self.market_options:
                        dh = DoubleHeston(
                            S0=self.spot,
                            K=opt['strike'],
                            T=opt['maturity'],
                            r=self.risk_free_rate,
                            v01=params['v1_0'],
                            kappa1=params['kappa1'],
                            theta1=params['theta1'],
                            sigma1=params['sigma1'],
                            rho1=params['rho1'],
                            v02=params['v2_0'],
                            kappa2=params['kappa2'],
                            theta2=params['theta2'],
                            sigma2=params['sigma2'],
                            rho2=params['rho2'],
                            lambda_j=params['lambda_j'],
                            mu_j=params['mu_j'],
                            sigma_j=params['sigma_j'],
                            option_type=opt['option_type']
                        )
                        model_prices.append(dh.pricing())
                    
                    best_result = CalibrationResult(
                        date='',  
                        spot=self.spot,
                        risk_free=self.risk_free_rate,
                        parameters=params,
                        market_prices=self.market_prices,
                        model_prices=np.array(model_prices),
                        market_options=self.market_options,
                        final_loss=result.fun,
                        calibration_time=time.time() - start_time,
                        success=result.success,
                        iterations=result.nit,
                        message=result.message
                    )
                    
            except Exception as e:
                continue
        
        # If all attempts failed
        if best_result is None:
            best_result = CalibrationResult(
                date='',
                spot=self.spot,
                risk_free=self.risk_free_rate,
                parameters={name: 0.0 for name in self.param_names},
                market_prices=self.market_prices,
                model_prices=np.zeros_like(self.market_prices),
                market_options=self.market_options,
                final_loss=np.inf,
                calibration_time=time.time() - start_time,
                success=False,
                iterations=0,
                message="All optimization starts failed"
            )
        
        return best_result


if __name__ == "__main__":
    print("="*70)
    print("DOUBLE HESTON + JUMP DIFFUSION L-BFGS CALIBRATOR")
    print("="*70)
    print("\nThis is the calibration module.")
    print("For usage examples, see the tests directory.")
    print("="*70)
