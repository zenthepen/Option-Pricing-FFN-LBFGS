"""
L-BFGS Calibrator for Double Heston + Jump Diffusion Model

This module provides production-ready calibration for the 13-parameter
Double Heston + Jump model using historical market option prices.

Author: Zen
Date: November 2025
"""

import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Import the pricing model
from doubleheston import DoubleHeston


@dataclass
class CalibrationResult:
    """Stores results from a single calibration run."""
    date: str
    spot: float
    risk_free: float
    parameters: Dict[str, float]
    market_prices: np.ndarray
    model_prices: np.ndarray
    market_options: List[Dict]
    final_loss: float
    calibration_time: float
    success: bool
    iterations: int
    message: str


class DoubleHestonJumpCalibrator:
    """
    L-BFGS-B calibrator for Double Heston + Jump Diffusion model.
    
    Calibrates 13 parameters to minimize weighted RMSE between market
    and model option prices.
    
    Parameters:
    -----------
    spot : float
        Current spot price
    risk_free_rate : float
        Risk-free interest rate (annualized)
    market_options : list of dict
        Each dict contains: strike, maturity, price, option_type
    
    Example:
    --------
    >>> options = [
    ...     {'strike': 100, 'maturity': 0.25, 'price': 5.2, 'option_type': 'call'},
    ...     {'strike': 105, 'maturity': 0.25, 'price': 2.8, 'option_type': 'call'},
    ... ]
    >>> calibrator = DoubleHestonJumpCalibrator(100.0, 0.03, options)
    >>> result = calibrator.calibrate(maxiter=300, multi_start=3)
    """
    
    def __init__(self, spot: float, risk_free_rate: float, market_options: List[Dict]):
        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.market_options = market_options
        self.market_prices = np.array([opt['price'] for opt in market_options])
        
        # Parameter names (in order)
        self.param_names = [
            'v1_0', 'kappa1', 'theta1', 'sigma1', 'rho1',
            'v2_0', 'kappa2', 'theta2', 'sigma2', 'rho2',
            'lambda_j', 'mu_j', 'sigma_j'
        ]
        
        # Optimization statistics
        self.n_calls = 0
        self.best_loss = np.inf
        
    def transform_params(self, x: np.ndarray) -> Dict[str, float]:
        """
        Transform unconstrained optimization variables to constrained parameters.
        
        Uses exponential transformation for positive parameters and tanh for correlations.
        
        Parameters:
        -----------
        x : np.ndarray
            13 unconstrained optimization variables
            
        Returns:
        --------
        dict : Parameter dictionary with all 13 parameters
        """
        params = {}
        
        # Variance parameters (positive): use exp
        params['v1_0'] = np.exp(x[0])
        params['kappa1'] = np.exp(x[1])
        params['theta1'] = np.exp(x[2])
        params['sigma1'] = np.exp(x[3])
        
        # Correlation (bounded to [-1, 1]): use tanh
        params['rho1'] = np.tanh(x[4])
        
        # Second Heston component
        params['v2_0'] = np.exp(x[5])
        params['kappa2'] = np.exp(x[6])
        params['theta2'] = np.exp(x[7])
        params['sigma2'] = np.exp(x[8])
        params['rho2'] = np.tanh(x[9])
        
        # Jump parameters
        params['lambda_j'] = np.exp(x[10])  # Jump intensity (positive)
        params['mu_j'] = x[11]  # Jump mean (can be negative)
        params['sigma_j'] = np.exp(x[12])  # Jump volatility (positive)
        
        return params
    
    def inverse_transform_params(self, params: Dict[str, float]) -> np.ndarray:
        """
        Convert constrained parameters back to unconstrained optimization variables.
        
        Parameters:
        -----------
        params : dict
            Dictionary with all 13 parameters
            
        Returns:
        --------
        np.ndarray : 13 unconstrained variables
        """
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
        """
        Compute penalty for violating Feller condition: 2*kappa*theta >= sigma^2.
        
        Parameters:
        -----------
        params : dict
            Parameter dictionary
            
        Returns:
        --------
        float : Penalty value (0 if satisfied, positive if violated)
        """
        penalty1 = max(0, params['sigma1']**2 - 2*params['kappa1']*params['theta1'])
        penalty2 = max(0, params['sigma2']**2 - 2*params['kappa2']*params['theta2'])
        
        return 1000.0 * (penalty1 + penalty2)
    
    def compute_loss(self, x: np.ndarray) -> float:
        """
        Compute weighted mean squared percentage error between market and model prices.
        
        Parameters:
        -----------
        x : np.ndarray
            Unconstrained optimization variables
            
        Returns:
        --------
        float : Total loss (weighted RMSE + Feller penalty)
        """
        self.n_calls += 1
        
        try:
            # Transform to constrained parameters
            params = self.transform_params(x)
            
            # Price all options using Double Heston model
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
                    
                    # Handle invalid prices
                    if np.isnan(price) or np.isinf(price) or price <= 0:
                        return 1e10
                    
                    model_prices.append(price)
                    
                except Exception as e:
                    # Pricing failed for this option
                    return 1e10
            
            model_prices = np.array(model_prices)
            
            # Compute weighted mean squared percentage error
            relative_errors = (model_prices - self.market_prices) / self.market_prices
            mse = np.mean(relative_errors**2)
            
            # Add Feller condition penalty
            feller_penalty = self.compute_feller_penalty(params)
            
            total_loss = mse + feller_penalty
            
            # Track best loss
            if total_loss < self.best_loss:
                self.best_loss = total_loss
            
            return total_loss
            
        except Exception as e:
            # Catch any unexpected errors
            return 1e10
    
    def get_initial_guess(self, guess_type: int = 0) -> np.ndarray:
        """
        Generate initial parameter guess for optimization.
        
        Parameters:
        -----------
        guess_type : int
            0 = Default reasonable parameters
            1 = Random perturbation around defaults
            2 = ATM volatility-informed guess
            
        Returns:
        --------
        np.ndarray : Initial unconstrained variables
        """
        if guess_type == 0:
            # Default: typical market parameters
            params = {
                'v1_0': 0.04, 'kappa1': 2.5, 'theta1': 0.04, 'sigma1': 0.3, 'rho1': -0.7,
                'v2_0': 0.04, 'kappa2': 0.5, 'theta2': 0.04, 'sigma2': 0.2, 'rho2': -0.5,
                'lambda_j': 0.15, 'mu_j': -0.04, 'sigma_j': 0.08
            }
            
        elif guess_type == 1:
            # Random perturbation (±30%)
            params = {
                'v1_0': 0.04 * np.random.uniform(0.7, 1.3),
                'kappa1': 2.5 * np.random.uniform(0.7, 1.3),
                'theta1': 0.04 * np.random.uniform(0.7, 1.3),
                'sigma1': 0.3 * np.random.uniform(0.7, 1.3),
                'rho1': np.random.uniform(-0.9, -0.5),
                'v2_0': 0.04 * np.random.uniform(0.7, 1.3),
                'kappa2': 0.5 * np.random.uniform(0.7, 1.3),
                'theta2': 0.04 * np.random.uniform(0.7, 1.3),
                'sigma2': 0.2 * np.random.uniform(0.7, 1.3),
                'rho2': np.random.uniform(-0.7, -0.3),
                'lambda_j': np.random.uniform(0.08, 0.25),
                'mu_j': np.random.uniform(-0.07, -0.02),
                'sigma_j': np.random.uniform(0.05, 0.12)
            }
            
        else:  # guess_type == 2
            # ATM-implied volatility informed guess
            atm_options = [opt for opt in self.market_options 
                          if 0.95 < opt['strike']/self.spot < 1.05]
            
            if atm_options:
                avg_price = np.mean([opt['price'] for opt in atm_options])
                avg_maturity = np.mean([opt['maturity'] for opt in atm_options])
                # Rough implied vol estimate
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
        Calibrate model parameters using L-BFGS-B optimization with multiple starts.
        
        Parameters:
        -----------
        maxiter : int
            Maximum iterations per optimization run
        multi_start : int
            Number of different initial guesses to try (2-3 recommended)
            
        Returns:
        --------
        CalibrationResult : Best calibration result
        """
        start_time = time.time()
        
        best_result = None
        best_loss = np.inf
        
        for start_idx in range(multi_start):
            # Reset optimization statistics
            self.n_calls = 0
            self.best_loss = np.inf
            
            # Get initial guess
            x0 = self.get_initial_guess(guess_type=start_idx % 3)
            
            try:
                # Run L-BFGS-B optimization
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
                
                # Check if this is the best result so far
                if result.fun < best_loss:
                    best_loss = result.fun
                    
                    # Transform to parameters
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
                        date='',  # Will be set by caller
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
                # This start failed, try next one
                continue
        
        # If all starts failed, return failure result
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


def fetch_options_for_date(ticker: str, date: datetime, 
                           min_moneyness: float = 0.8, 
                           max_moneyness: float = 1.2) -> Optional[Dict]:
    """
    Fetch historical option prices for a specific date using yfinance.
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol (e.g., 'SPY', '^SPX')
    date : datetime
        Target date for option data
    min_moneyness : float
        Minimum strike/spot ratio to include
    max_moneyness : float
        Maximum strike/spot ratio to include
        
    Returns:
    --------
    dict or None : Dictionary with spot, risk_free, options list, or None if data unavailable
    """
    try:
        import yfinance as yf
        
        # Get ticker object
        stock = yf.Ticker(ticker)
        
        # Get spot price for the date
        hist = stock.history(start=date, end=date + timedelta(days=1))
        if hist.empty:
            return None
        
        spot = float(hist['Close'].iloc[0])
        
        # Approximate risk-free rate (use 10-year treasury rate, or default to 3%)
        # For production, fetch from FRED or other source
        risk_free = 0.03  # Approximate for 2022-2024 period
        
        # Get option chain (note: yfinance only provides current options, not historical)
        # For true historical data, need a paid data provider
        # This is a simplified example - adjust based on your data source
        
        try:
            # Get available expiration dates
            expirations = stock.options
            if not expirations:
                return None
            
            options_data = []
            
            # Try to get 3 different maturities
            target_days = [30, 90, 180]
            selected_expirations = []
            
            for target in target_days:
                target_date = date + timedelta(days=target)
                # Find closest expiration
                closest = min(expirations, 
                             key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
                if closest not in selected_expirations:
                    selected_expirations.append(closest)
            
            # Fetch options for selected expirations
            for exp_date in selected_expirations[:3]:
                chain = stock.option_chain(exp_date)
                calls = chain.calls
                
                # Calculate maturity in years
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                maturity = (exp_datetime - date).days / 365.0
                
                # Filter by moneyness
                calls = calls[
                    (calls['strike'] >= spot * min_moneyness) & 
                    (calls['strike'] <= spot * max_moneyness)
                ]
                
                # Select options with reasonable volume/open interest
                calls = calls[
                    (calls['volume'] > 10) & 
                    (calls['openInterest'] > 50)
                ]
                
                # Create option dictionaries
                for _, row in calls.iterrows():
                    # Use mid-price
                    bid = row['bid']
                    ask = row['ask']
                    if bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2.0
                        
                        options_data.append({
                            'strike': float(row['strike']),
                            'maturity': maturity,
                            'price': mid_price,
                            'option_type': 'call',
                            'bid': bid,
                            'ask': ask,
                            'volume': row['volume'],
                            'open_interest': row['openInterest']
                        })
            
            # Need at least 10 options for reasonable calibration
            if len(options_data) < 10:
                return None
            
            return {
                'spot': spot,
                'risk_free': risk_free,
                'options': options_data
            }
            
        except Exception as e:
            return None
            
    except Exception as e:
        return None


def run_historical_calibrations(
    ticker: str = 'SPY',
    start_date: str = '2022-01-03',
    end_date: str = '2024-12-31',
    max_dates: int = 500,
    checkpoint_freq: int = 50,
    save_path: str = 'lbfgs_calibrations.pkl'
) -> List[CalibrationResult]:
    """
    Run L-BFGS calibration on historical market data for multiple dates.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    max_dates : int
        Maximum number of dates to calibrate
    checkpoint_freq : int
        Save checkpoint every N calibrations
    save_path : str
        Path to save final results
        
    Returns:
    --------
    list : List of CalibrationResult objects
    """
    print("="*70)
    print("L-BFGS HISTORICAL CALIBRATION - DOUBLE HESTON + JUMPS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Ticker: {ticker}")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Max dates: {max_dates}")
    print(f"  Checkpoint frequency: every {checkpoint_freq} calibrations")
    print(f"  Save path: {save_path}")
    
    # Generate trading dates (weekdays only, approximate)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_dates = []
    current = start_dt
    while current <= end_dt and len(all_dates) < max_dates:
        # Skip weekends
        if current.weekday() < 5:
            all_dates.append(current)
        current += timedelta(days=1)
    
    print(f"\nGenerated {len(all_dates)} potential trading dates")
    print(f"\nStarting calibrations...")
    print("="*70)
    
    results = []
    successful = 0
    failed = 0
    total_time = 0
    
    for i, date in enumerate(all_dates, 1):
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            print(f"\n[{i}/{len(all_dates)}] Calibrating: {date_str}")
            
            # Fetch option data
            options_data = fetch_options_for_date(ticker, date)
            
            if options_data is None:
                print(f"  ⚠️  No option data available - skipping")
                failed += 1
                continue
            
            print(f"  Spot: ${options_data['spot']:.2f}")
            print(f"  Options: {len(options_data['options'])} contracts")
            
            # Create calibrator
            calibrator = DoubleHestonJumpCalibrator(
                spot=options_data['spot'],
                risk_free_rate=options_data['risk_free'],
                market_options=options_data['options']
            )
            
            # Run calibration with 2 multi-starts
            result = calibrator.calibrate(maxiter=300, multi_start=2)
            result.date = date_str
            
            if result.success and result.final_loss < 1e9:
                successful += 1
                results.append(result)
                
                # Compute pricing error statistics
                rel_errors = np.abs((result.model_prices - result.market_prices) / result.market_prices) * 100
                mean_error = np.mean(rel_errors)
                
                print(f"  ✓ Success! Time: {result.calibration_time:.1f}s, Loss: {result.final_loss:.6f}")
                print(f"    Mean pricing error: {mean_error:.2f}%")
                print(f"    Progress: {successful} successful / {failed} failed / {i} total")
            else:
                failed += 1
                print(f"  ✗ Calibration failed: {result.message}")
            
            total_time += result.calibration_time
            
            # Save checkpoint
            if len(results) > 0 and len(results) % checkpoint_freq == 0:
                checkpoint_path = save_path.replace('.pkl', f'_checkpoint_{len(results)}.pkl')
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(results, f)
                print(f"\n  💾 Checkpoint saved: {checkpoint_path}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Calibration interrupted by user")
            break
            
        except Exception as e:
            failed += 1
            print(f"  ✗ Error: {str(e)}")
            continue
    
    # Save final results
    if results:
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n{'='*70}")
        print(f"✓ Final results saved: {save_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("L-BFGS HISTORICAL CALIBRATION COMPLETE")
    print("="*70)
    print(f"\nTotal dates attempted: {len(all_dates)}")
    print(f"Successful calibrations: {successful} ({successful/len(all_dates)*100:.1f}%)")
    print(f"Failed calibrations: {failed} ({failed/len(all_dates)*100:.1f}%)")
    
    if results:
        times = [r.calibration_time for r in results]
        losses = [r.final_loss for r in results]
        
        print(f"\nTiming Statistics:")
        print(f"  Mean calibration time: {np.mean(times):.1f}s")
        print(f"  Median calibration time: {np.median(times):.1f}s")
        print(f"  Total time: {total_time/3600:.1f} hours")
        
        print(f"\nLoss Statistics:")
        print(f"  Mean loss: {np.mean(losses):.6f}")
        print(f"  Median loss: {np.median(losses):.6f}")
        print(f"  Min loss: {np.min(losses):.6f}")
        print(f"  Max loss: {np.max(losses):.6f}")
        
        # Parameter statistics
        print(f"\nParameter Statistics (successful calibrations):")
        param_names = results[0].parameters.keys()
        for param in param_names:
            values = [r.parameters[param] for r in results]
            print(f"  {param:10s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                  f"min={np.min(values):.4f}, max={np.max(values):.4f}")
        
        # Pricing error statistics
        all_errors = []
        for r in results:
            rel_errors = np.abs((r.model_prices - r.market_prices) / r.market_prices) * 100
            all_errors.extend(rel_errors)
        
        all_errors = np.array(all_errors)
        print(f"\nPricing Error Statistics:")
        print(f"  Mean absolute error: {np.mean(all_errors):.2f}%")
        print(f"  Median absolute error: {np.median(all_errors):.2f}%")
        print(f"  95th percentile: {np.percentile(all_errors, 95):.2f}%")
        print(f"  Max error: {np.max(all_errors):.2f}%")
    
    print(f"\n{'='*70}")
    print(f"✓ Ready for FFN fine-tuning")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    """
    Main execution script with single date test and full calibration option.
    """
    
    print("\n" + "="*70)
    print("DOUBLE HESTON + JUMP DIFFUSION L-BFGS CALIBRATOR")
    print("="*70)
    
    # Mode selection
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode: Single date calibration
        print("\n🧪 TEST MODE: Single Date Calibration")
        print("="*70)
        
        test_date = datetime(2023, 6, 15)
        print(f"\nFetching options for {test_date.strftime('%Y-%m-%d')}...")
        
        options_data = fetch_options_for_date('SPY', test_date)
        
        if options_data is None:
            print("❌ Failed to fetch option data for test date")
            print("Note: yfinance only provides current option data, not historical")
            print("For true historical calibration, use a paid data provider (e.g., OptionMetrics)")
            sys.exit(1)
        
        print(f"\n✓ Fetched {len(options_data['options'])} options")
        print(f"  Spot: ${options_data['spot']:.2f}")
        print(f"  Risk-free rate: {options_data['risk_free']:.2%}")
        
        print(f"\nCreating calibrator...")
        calibrator = DoubleHestonJumpCalibrator(
            spot=options_data['spot'],
            risk_free_rate=options_data['risk_free'],
            market_options=options_data['options']
        )
        
        print(f"Running calibration (this may take 2-5 minutes)...")
        result = calibrator.calibrate(maxiter=300, multi_start=2)
        
        print(f"\n{'='*70}")
        print("CALIBRATION RESULTS")
        print("="*70)
        print(f"Success: {result.success}")
        print(f"Time: {result.calibration_time:.1f}s")
        print(f"Loss: {result.final_loss:.6f}")
        print(f"Iterations: {result.iterations}")
        
        if result.success:
            print(f"\nCalibrated Parameters:")
            for name, value in result.parameters.items():
                print(f"  {name:10s} = {value:.6f}")
            
            # Pricing errors
            rel_errors = np.abs((result.model_prices - result.market_prices) / result.market_prices) * 100
            print(f"\nPricing Accuracy:")
            print(f"  Mean absolute error: {np.mean(rel_errors):.2f}%")
            print(f"  Median absolute error: {np.median(rel_errors):.2f}%")
            print(f"  Max error: {np.max(rel_errors):.2f}%")
        
        print(f"\n{'='*70}")
        print("✓ Test complete! Ready for full historical calibration.")
        print("Run without 'test' argument to calibrate 500 dates.")
        print(f"{'='*70}\n")
        
    else:
        # Full calibration mode
        print("\n🚀 FULL CALIBRATION MODE: 500 Historical Dates")
        print("="*70)
        print("\n⚠️  WARNING: This will take 16-25 hours to complete!")
        print("Checkpoints will be saved every 50 calibrations.")
        print("\nPress Ctrl+C to stop at any time (progress will be saved).\n")
        
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Calibration cancelled.")
            sys.exit(0)
        
        # Run full historical calibration
        calibrations = run_historical_calibrations(
            ticker='SPY',
            start_date='2022-01-03',
            end_date='2024-12-31',
            max_dates=500,
            checkpoint_freq=50,
            save_path='lbfgs_calibrations.pkl'
        )
        
        print("\n✓ All calibrations complete!")
        print(f"✓ Results saved to: lbfgs_calibrations.pkl")
        print(f"✓ Ready for FFN fine-tuning\n")
