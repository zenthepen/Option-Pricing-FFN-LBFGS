# Limitations and Future Directions

## 1. Parameter Identifiability

The Double-Heston + Jump diffusion model shows significant parameter identifiability challenges, especially for correlation and mean-reversion parameters. Multiple parameters give similar option prices due to which parameter-level recovery is uncertain. We therefore emphasize pricing error as our main metric of success.

## 2. Synthetic Data limitations

The model currently has only been trained on synthetic data. Real market calibration may exhibit not captured in our dataset.

## 3. Single-Regime Assumption

The model assumes constant parameter dynamics while real markets show regime switching behaviour. Extending the framework to regime-switching models is planned for future work.


