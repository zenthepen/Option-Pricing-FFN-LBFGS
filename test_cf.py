import numpy as np
from doubleheston import DoubleHeston

# Test parameters
S0 = 100
K = 100
T = 1.0
r = 0.05
v01 = 0.04
kappa1 = 2.0
theta1 = 0.04
sigma1 = 0.3
rho1 = -0.5
v02 = 0.04
kappa2 = 1.5
theta2 = 0.04
sigma2 = 0.2
rho2 = -0.3

model = DoubleHeston(S0, K, T, r, v01, kappa1, theta1, sigma1, rho1,
                     v02, kappa2, theta2, sigma2, rho2)

# Test CF at phi=0 (should be exactly 1.0)
cf_0 = model.characteristic_function(0, T)
print(f"CF(0) = {cf_0}")
print(f"CF(0) should be 1.0, actual: {np.abs(cf_0):.10f}")

# Test a few other values
print("\nCF at various phi values:")
for phi in [0.1, 0.5, 1.0, 2.0]:
    cf = model.characteristic_function(phi, T)
    print(f"  phi={phi:4.1f}: CF = {cf:.6f}, |CF| = {np.abs(cf):.6f}")

# Check truncation range
a, b = model.truncationRange()
print(f"\nTruncation range: [{a:.4f}, {b:.4f}]")
print(f"log(K) = {np.log(K):.4f}")
