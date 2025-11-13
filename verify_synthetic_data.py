import pickle
import numpy as np

# Load the test dataset
with open('synthetic_data_test.pkl', 'rb') as f:
    data = pickle.load(f)

print("="*70)
print("SYNTHETIC DATA VERIFICATION")
print("="*70)

print("\nDataset Contents:")
print(f"  Keys: {list(data.keys())}")
print(f"\n  Parameters shape: {data['parameters'].shape}")
print(f"  Option prices shape: {data['option_prices'].shape}")
print(f"  Parameter names: {data['param_names']}")
print(f"  Strikes: {data['strikes']}")
print(f"  Maturities: {data['maturities']}")

print("\n" + "-"*70)
print("Sample 1:")
print("-"*70)
params = data['parameters'][0]
prices = data['option_prices'][0]

print("\nParameters:")
for name, value in zip(data['param_names'], params):
    print(f"  {name:12s}: {value:.6f}")

print("\nOption Prices:")
print(f"  {'Strike':<8} {'T=0.25':<10} {'T=0.5':<10} {'T=1.0':<10}")
print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
for i, K in enumerate(data['strikes']):
    idx_base = i * len(data['maturities'])
    row = f"  {K:<8.0f}"
    for j in range(len(data['maturities'])):
        row += f" ${prices[idx_base + j]:<9.4f}"
    print(row)

print("\n" + "-"*70)
print("Parameter Statistics (across all samples):")
print("-"*70)
for i, name in enumerate(data['param_names']):
    col = data['parameters'][:, i]
    print(f"  {name:12s}: mean={np.mean(col):7.4f}, std={np.std(col):6.4f}, "
          f"min={np.min(col):7.4f}, max={np.max(col):7.4f}")

print("\n" + "-"*70)
print("Price Statistics:")
print("-"*70)
all_prices = data['option_prices'].flatten()
print(f"  Mean:   ${np.mean(all_prices):.4f}")
print(f"  Std:    ${np.std(all_prices):.4f}")
print(f"  Min:    ${np.min(all_prices):.4f}")
print(f"  Max:    ${np.max(all_prices):.4f}")
print(f"  Median: ${np.median(all_prices):.4f}")

# Check for any invalid prices
invalid = np.sum(all_prices <= 0)
print(f"\n  Invalid prices (≤0): {invalid}")
print(f"  Valid prices: {len(all_prices) - invalid} / {len(all_prices)}")

print("\n" + "="*70)
print("✓ Data validation complete!")
print("="*70)
