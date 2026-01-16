# Limitations and Future Directions

## 1. Parameter Identifiability

The Double-Heston + Jump diffusion model shows significant parameter identifiability challenges, especially for correlation and mean-reversion parameters. Multiple parameter combinations can produce similar option prices, making parameter-level recovery uncertain. We therefore emphasize pricing error as our main metric of success.

## 2. Synthetic Data Limitations

The model currently has only been trained on synthetic data. Real market calibration may exhibit characteristics not captured in our dataset, including bid-ask spreads, illiquid strikes, and market microstructure noise.

## 3. Single-Regime Assumption

The model assumes constant parameter dynamics while real markets show regime-switching behavior (e.g., low volatility vs. crisis periods). Parameters calibrated in one regime may not be suitable for another, requiring frequent recalibration. Extending the framework to regime-switching models is planned for future work.

## 4. Limited Scalability

Current testing is performed on only 15 option contracts (5 strikes × 3 maturities). Real-world applications often involve 100+ liquid options across multiple expirations. The computational cost scales linearly with the number of options:
- 15 options: ~118 seconds per calibration
- 100 options: projected ~10+ minutes per calibration

This limits practical applicability for portfolios requiring frequent recalibration or real-time pricing.

## Future Directions

These limitations present opportunities for improvement through:
- Regime-switching model extensions
- Real market data validation
- Computational optimization for larger option sets
- Parameter dimensionality reduction techniques
