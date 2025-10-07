# Day 1: Heston Variance Simulation - Beginner Version
print("🚀 Starting Heston Model - Day 1 Simulation")
print("This demonstrates how volatility changes over time...")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    print("✅ Libraries imported successfully!")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Please run: pip install numpy matplotlib")
    input("Press Enter to exit...")
    exit()

# Heston parameters - these control how volatility behaves
v0 = 0.04      # Starting volatility (20% = sqrt(0.04))
kappa = 2.0    # How quickly volatility returns to average
theta = 0.04   # Long-term average volatility  
sigma = 0.3    # How random the volatility changes are
T = 1.0        # 1 year simulation
dt = 1/252     # Daily steps (252 trading days)
n_steps = int(T/dt)
n_paths = 5    # Number of different scenarios

print(f"📊 Simulating {n_paths} volatility paths for {T} year...")
print(f"📈 Parameters: κ={kappa}, θ={theta}, σ={sigma}")

# Initialize arrays
np.random.seed(42)  # For reproducible results
v_paths = np.zeros((n_paths, n_steps))
v_paths[:, 0] = v0  # All paths start at initial volatility

print("🔄 Running simulation...")
# Simulate each day
for t in range(1, n_steps):
    # Random shocks for each path
    dW = np.random.normal(0, np.sqrt(dt), n_paths)
    
    # Heston variance equation: dv = κ(θ-v)dt + σ√v dW
    drift = kappa * (theta - v_paths[:, t-1]) * dt
    diffusion = sigma * np.sqrt(v_paths[:, t-1]) * dW
    v_paths[:, t] = v_paths[:, t-1] + drift + diffusion
    
    # Ensure variance doesn't go negative
    v_paths[:, t] = np.maximum(v_paths[:, t], 0)

print("✅ Simulation complete! Creating visualization...")

# Create the plot
plt.figure(figsize=(12, 6))
time = np.linspace(0, T, n_steps)

# Plot each volatility path
colors = ['blue', 'green', 'orange', 'purple', 'brown']
for i in range(n_paths):
    plt.plot(time, v_paths[i, :], color=colors[i], label=f'Volatility Path {i+1}', linewidth=1.5)

# Add the long-term average line
plt.axhline(theta, color='red', linestyle='--', label='Long-term Average (θ)', linewidth=2)

# Format the plot
plt.xlabel('Time (years)')
plt.ylabel('Variance')
plt.title('Heston Model: Volatility Mean Reversion\nNotice how paths revert toward the red line!', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Save and show
plt.savefig('variance_simulation.png', dpi=150, bbox_inches='tight')
print("💾 Plot saved as 'variance_simulation.png'")
plt.show()

print("\n🎉 DAY 1 COMPLETE!")
print("📖 What you just visualized:")
print("   • Volatility changes randomly (stochastic)")
print("   • But it tends to return to average (mean-reverting)") 
print("   • This is the core innovation of the Heston model!")
print("\n🚀 Tomorrow: We'll implement the pricing formulas!")
input("Press Enter to close...")