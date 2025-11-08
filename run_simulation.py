# Heston Stochastic Volatility Simulation: From Black–Scholes to Heston

print("🚀 Starting Heston Model - Monte Carlo Simulation (Full SDEs)")
print("This simulates price and volatility, demonstrating the Heston model's skew.")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm, skew, kurtosis
    print("✅ Libraries imported successfully!")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Please run: pip install numpy matplotlib scipy")
    input("Press Enter to exit...")
    exit()

# --- 1. Parameter Definitions ---
S0 = 100.0     # Initial Stock Price
r = 0.05       # Risk-Free Rate (used for drift and discounting)
K = 100.0      # Strike Price (for the option valuation)
T = 1.0        # Time to Expiration (1 year)

# Heston Model Parameters
rho = -0.7     # Correlation (enforced skew)
v0 = 0.04      # Initial Variance (sqrt(0.04) = 20% volatility)
kappa = 2.0    # Mean Reversion Rate
theta = 0.04   # Long-term Average Variance
sigma = 0.3    # Volatility of Volatility

# Simulation Parameters
dt = 1/252     # Daily steps (252 trading days)
n_steps = int(T/dt)
# Use a high number of paths for accurate pricing, but only plot a few.
N_PATHS_MC = 100000 
N_PATHS_PLOT = 5

print(f"\n📊 Simulating {N_PATHS_MC} paths for pricing, {n_steps} steps.")
print(f"📈 Parameters: ρ={rho}, κ={kappa}, θ={theta}, σ={sigma}")

# --- Parameter Validation ---
def validate_heston_parameters(kappa, theta, sigma, v0):
    """Check parameter validity and Feller condition"""
    feller_ratio = 2 * kappa * theta / (sigma ** 2)
    feller_satisfied = feller_ratio > 1
    
    print(f"🔍 Parameter Validation:")
    print(f"   Feller Condition (2κθ > σ²): {feller_ratio:.2f} > 1 → {feller_satisfied}")
    print(f"   Mean Reversion Strength (κ): {kappa}")
    print(f"   Vol-of-Vol (σ): {sigma}")
    
    if not feller_satisfied:
        print("   ⚠️ Warning: Feller condition not satisfied - variance may hit zero frequently")
    
    return feller_satisfied

# Validate parameters before simulation
feller_ok = validate_heston_parameters(kappa, theta, sigma, v0)

# --- 2. Initialization ---
np.random.seed(42)  # For reproducible results

v_paths = np.zeros((N_PATHS_MC, n_steps))
s_paths = np.zeros((N_PATHS_MC, n_steps))
v_paths[:, 0] = v0
s_paths[:, 0] = S0

print("🔄 Running Monte Carlo simulation...")

# --- 3. Simulation Loop (Improved Euler-Maruyama Scheme) ---
for t in range(1, n_steps):
    # Generating two independent standard normal variables
    Z1 = np.random.normal(0.0, 1.0, N_PATHS_MC)
    Z2 = np.random.normal(0.0, 1.0, N_PATHS_MC)

    # Create Correlated Wiener Increments (dW) using Cholesky factorization
    sqrt_dt = np.sqrt(dt)
    dW1 = sqrt_dt * Z1  # For the Stock Price (S)
    dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2) # For the Variance (v)
    
    # Previous values
    v_prev = v_paths[:, t-1]
    S_prev = s_paths[:, t-1]

    # --- Variance SDE (v_t) - Improved Full Truncation Scheme ---
    # dv = κ(θ-v)dt + σ√v dW2
    drift_v = kappa * (theta - v_prev) * dt
    diffusion_v = sigma * np.sqrt(v_prev) * dW2
    
    v_current = v_prev + drift_v + diffusion_v
    
    # Ensure variance doesn't go negative (Full Truncation)
    v_paths[:, t] = np.maximum(v_current, 0)

    # --- Stock Price SDE (S_t) - Risk-Neutral Measure ---
    # IMPROVED: Use exact solution for stock price (prevents negative prices)
    # dS = r*S*dt + √v*S*dW1 → Exact solution: S_t = S_0 * exp((r - 0.5*v)*dt + √v*dW1)
    drift_s = (r - 0.5 * v_prev) * dt
    diffusion_s = np.sqrt(v_prev) * dW1
    
    s_paths[:, t] = S_prev * np.exp(drift_s + diffusion_s)

print("✅ Simulation complete! Analyzing results...")

# --- 4. Enhanced Monte Carlo Analysis ---
def enhanced_analysis(s_paths, v_paths, S0, r, T, K):
    """Comprehensive analysis of simulation results"""
    
    # Calculate final prices and returns
    final_prices = s_paths[:, -1]
    log_returns = np.log(final_prices / S0)
    
    # Option pricing with confidence interval
    payoffs = np.maximum(final_prices - K, 0)
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    mc_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(N_PATHS_MC)
    confidence_interval = (mc_price - 1.96 * mc_std, mc_price + 1.96 * mc_std)
    
    print(f"\n📊 Enhanced Simulation Statistics:")
    print(f"   Final Price Mean: ${np.mean(final_prices):.2f}")
    print(f"   Final Price Std: ${np.std(final_prices):.2f}")
    print(f"   Return Skewness: {float(skew(log_returns)):.3f}")
    print(f"   Return Kurtosis: {float(kurtosis(log_returns)):.3f}")
    print(f"   Average Volatility: {np.mean(np.sqrt(v_paths)):.1%}")
    print(f"   Volatility of Volatility: {np.std(np.sqrt(v_paths)):.1%}")
    
    return mc_price, mc_std, confidence_interval

# Run enhanced analysis
mc_price, mc_std, confidence_interval = enhanced_analysis(s_paths, v_paths, S0, r, T, K)

print(f"\n💵 Heston Monte Carlo Call Price (K={K}, T={T}): ${mc_price:.4f}")
print(f"   📊 95% Confidence Interval: (${confidence_interval[0]:.4f}, ${confidence_interval[1]:.4f})")
print(f"   📐 Standard Error: ${mc_std:.4f}")

# --- 5. Visualization A: Volatility Mean Reversion (Enhanced Time-Series Plot) ---

plt.figure(figsize=(12, 8))

# Subplot 1: Volatility paths
plt.subplot(2, 1, 1)
time = np.linspace(0, T, n_steps)

# Plot only a small subset of paths for clarity
colors = ['blue', 'green', 'orange', 'purple', 'brown']
for i in range(N_PATHS_PLOT):
    plt.plot(time, np.sqrt(v_paths[i, :]), color=colors[i], label=f'Volatility Path {i+1}', linewidth=1.5)

# Add the long-term average line
plt.axhline(np.sqrt(theta), color='red', linestyle='--', label=f'Long-term Vol (√θ={np.sqrt(theta):.1%})', linewidth=2)

# Format the plot
plt.xlabel('Time (years)')
plt.ylabel('Volatility')
plt.title('Heston Model: Volatility Mean Reversion ($\kappa$ and $\theta$ in Action)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Volatility distribution at final time
plt.subplot(2, 1, 2)
final_volatilities = np.sqrt(v_paths[:, -1])
plt.hist(final_volatilities, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.sqrt(theta), color='red', linestyle='--', label=f'Long-term Vol (√θ={np.sqrt(theta):.1%})', linewidth=2)
plt.xlabel('Final Volatility')
plt.ylabel('Density')
plt.title('Distribution of Final Volatilities', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('variance_simulation_enhanced.png', dpi=150, bbox_inches='tight')
print("💾 Plot 1 saved as 'variance_simulation_enhanced.png'")
plt.show()

# --- 6. Visualization B: Price Distribution (The Skew/Kurtosis) ---

log_returns = np.log(s_paths[:, -1] / S0)

plt.figure(figsize=(12, 6))

# Subplot 1: Return distribution comparison
plt.subplot(1, 2, 1)
plt.hist(log_returns, bins=50, density=True, alpha=0.6, color='skyblue', label='Heston Log-Return Distribution ($\rho=-0.7$)')
plt.title('Heston vs. Black-Scholes: Demonstrating Skew and Kurtosis', fontsize=14)
plt.xlabel('Log Returns ($\ln(S_T / S_0)$)')
plt.ylabel('Density')

# Add the Black-Scholes (Normal) Distribution for comparison
avg_vol = np.sqrt(np.mean(v_paths))
mu_bs = (r - 0.5 * avg_vol**2) * T
sigma_bs = avg_vol * np.sqrt(T)

x = np.linspace(log_returns.min(), log_returns.max(), 100)
plt.plot(x, norm.pdf(x, mu_bs, sigma_bs), 'r--', linewidth=2, label='Black-Scholes (Normal) Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Convergence monitoring
plt.subplot(1, 2, 2)
payoffs = np.maximum(s_paths[:, -1] - K, 0)
cumulative_means = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)
cumulative_means_discounted = cumulative_means * np.exp(-r * T)

plt.plot(cumulative_means_discounted, alpha=0.7, color='green')
plt.axhline(mc_price, color='red', linestyle='--', label=f'Final Price: ${mc_price:.4f}')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.title('Monte Carlo Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('linear')

plt.tight_layout()
plt.savefig('return_distribution_enhanced.png', dpi=150, bbox_inches='tight')
print("💾 Plot 2 saved as 'return_distribution_enhanced.png'")
plt.show()

# --- 7. Volatility Regime Analysis ---
plt.figure(figsize=(12, 6))

# Calculate volatility percentiles across all paths
vol_percentiles = np.percentile(np.sqrt(v_paths), [10, 25, 50, 75, 90], axis=0)

plt.fill_between(time, vol_percentiles[0], vol_percentiles[4], alpha=0.3, color='lightblue', label='80% Range')
plt.fill_between(time, vol_percentiles[1], vol_percentiles[3], alpha=0.5, color='blue', label='50% Range')
plt.plot(time, vol_percentiles[2], 'k-', linewidth=2, label='Median Volatility')
plt.axhline(np.sqrt(theta), color='r', linestyle='--', label=f'Long-term Vol (√θ={np.sqrt(theta):.1%})')

plt.xlabel('Time (years)')
plt.ylabel('Volatility')
plt.title('Volatility Regimes: Distribution Across All Paths', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('volatility_regimes.png', dpi=150, bbox_inches='tight')
print("💾 Plot 3 saved as 'volatility_regimes.png'")
plt.show()

print("\n🎉 PHASE I COMPLETE: Enhanced Monte Carlo Simulation and Visual Analysis Done!")
print("📖 Key Insights:")
print("   • The Heston distribution shows skewness and fat tails vs normal distribution")
print("   • Volatility demonstrates clear mean reversion behavior") 
print("   • Negative correlation creates the characteristic volatility skew")
print("   • Monte Carlo convergence provides pricing confidence intervals")
print(f"   • Feller condition: {'SATISFIED' if feller_ok else 'NOT SATISFIED'}")

print("\n🚀 Next: Implement the Characteristic Function for semi-analytical pricing (Phase II).")
# input("Press Enter to close...") # Commented out for cleaner script execution
