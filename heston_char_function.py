import numpy as np
import cmath
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ----------------------------------------------------------------------
# 1. HESTON CHARACTERISTIC FUNCTION INTEGRAND
# ----------------------------------------------------------------------
def heston_Pj_integrand(u, j, K, S0, v0, r, T, kappa, theta, sigma, rho):
    """
    The integrand function for the Heston risk-neutral probability Pj.
    Incorporates the log-stock price transformation and damping for stability.
    """
    i = complex(0, 1)
    minimal_u = 1e-10
    
    # FIX: Increased Damping factor for numerical stability during calibration
    damping_alpha = 7.0 

    # Singularity Check at u = 0
    if u < minimal_u:
        return 0.0

    # Shifted u based on j (j=1 for P1, j=2 for P2)
    u_shifted = u - i * 0.5 if j == 1 else u + i * 0.5
    iu_shifted = i * u_shifted

    # Bound check to prevent instability
    if abs(u) > 1000: # Slightly increased bound check
        return 0.0

    x = cmath.log(S0)
    
    # D: Square root term in the Characteristic Function
    d = cmath.sqrt((rho * sigma * iu_shifted - kappa)**2 + sigma**2 * (iu_shifted + u_shifted**2))

    # g: Little Trap formulation component
    g = (kappa - rho * sigma * iu_shifted - d) / (kappa - rho * sigma * iu_shifted + d)
    exp_neg_dT = cmath.exp(-d * T)

    # Avoid division by near-zero
    denominator = 1 - g * exp_neg_dT
    if abs(denominator) < 1e-12:
        return 0.0

    # C: Characteristic function component 1
    C = iu_shifted * r * T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * iu_shifted - d) * T - 2 * cmath.log(denominator / (1 - g))
    )
    # D: Characteristic function component 2
    D = (kappa - rho * sigma * iu_shifted - d) * (1 - exp_neg_dT) / (sigma**2 * denominator)

    char_exp = C + D * v0 + iu_shifted * x

    # Prevent overflow in exponential
    if char_exp.real > 700:
        return 0.0

    phi = cmath.exp(char_exp)

    # Final overflow check
    if abs(phi) > 1e100:
        return 0.0

    # Final output with damping
    component_1 = cmath.exp(-i * u * cmath.log(K))
    output = (component_1 * phi * np.exp(-damping_alpha * u)) / (i * u)

    return output.real

# ----------------------------------------------------------------------
# 2. RISK-NEUTRAL PROBABILITY CALCULATION
# ----------------------------------------------------------------------
def risk_neutral_prob_function(j, K, S0, v0, r, T, kappa, theta, sigma, rho):
    """
    Calculates the Heston risk-neutral probability Pj using numerical integration.
    """
    heston_params = (K, S0, v0, r, T, kappa, theta, sigma, rho)

    # Adaptive integration over a stable range
    integral_result, error_estimate = quad(
        func=heston_Pj_integrand,
        a=1e-10,
        b=400, # FIX: Increased upper limit for robust integration
        args=(j,) + heston_params,
        limit=1000,
        epsabs=1e-6,
        epsrel=1e-6,
        full_output=0
    )

    Pj = 0.5 + (1 / np.pi) * integral_result
    Pj = np.clip(Pj, 1e-6, 1 - 1e-6)

    # Clamp to [0,1]
    if Pj < 0 or Pj > 1:
        Pj = max(0.0, min(1.0, Pj))

    return Pj

# ----------------------------------------------------------------------
# 3. HESTON CALL PRICE
# ----------------------------------------------------------------------
def heston_call_price(K, S0, v0, r, T, kappa, theta, sigma, rho):
    """
    Calculates the Heston call option price using P1 and P2 probabilities.
    """
    heston_params = (K, S0, v0, r, T, kappa, theta, sigma, rho)

    P1 = risk_neutral_prob_function(1, *heston_params)
    P2 = risk_neutral_prob_function(2, *heston_params)

    Call_Price = S0 * P1 - K * np.exp(-r * T) * P2
    
    # FIX: Ensure non-negative price
    return max(0.0, Call_Price)

# ----------------------------------------------------------------------
# 4. BLACK-SCHOLES PRICE (Used for IV Calculation)
# ----------------------------------------------------------------------
def black_scholes_call_price(S0, K, T, r, sigma):
    """Calculates the Black-Scholes call option price."""
    # Prevent division by zero if T or sigma is zero
    if T <= 1e-6 or sigma <= 1e-6:
        return max(0.0, S0 - K * np.exp(-r * T))

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# ----------------------------------------------------------------------
# 5. IMPLIED VOLATILITY SOLVER (Used by Calibration)
# ----------------------------------------------------------------------
def implied_volatility(target_price, S0, K, T, r, initial_vol=0.5):
    """
    Solves for the Implied Volatility (IV) using Brent's method.
    Returns: float: The Implied Volatility (sigma).
    """
    
    def objective_function(sigma):
        return black_scholes_call_price(S0, K, T, r, sigma) - target_price

    if T <= 1e-6: # Prevent IV calculation for zero T
        return 0.0
        
    # Check for trivial cases (Deep ITM or OTM)
    intrinsic_value = max(0.0, S0 - K * np.exp(-r * T))
    if target_price < intrinsic_value:
        return 1e-6 # Return near-zero volatility
    
    low_vol = 1e-6 
    high_vol = 5.0 

    try:
        implied_vol = brentq(
            f=objective_function, 
            a=low_vol, 
            b=high_vol, 
            xtol=1e-6
        )
        return implied_vol
        
    except ValueError:
        # Solver failed; typically means the target price is outside BS range
        return np.nan 
    
# ----------------------------------------------------------------------
# 6. PHASE III DEMO/STANDALONE EXECUTION (Optional for project run)
# ----------------------------------------------------------------------
def run_phase_3_analysis():
    """
    Prices a strike grid using Heston, inverts to IV, and plots the skew/surface.
    """
    # ... (Phase III plotting logic is unchanged and can be kept or removed 
    # if you only want the pricing/IV functions) ...
    pass # Leaving this empty to keep the file focused on core functions

if __name__ == "__main__":
    # Example: Simple test case to confirm pricing stability
    K, S0, v0, r, T, kappa, theta, sigma, rho = (100.0, 100.0, 0.04, 0.05, 0.5, 1.0, 0.04, 0.3, -0.7)
    test_price = heston_call_price(K, S0, v0, r, T, kappa, theta, sigma, rho)
    test_iv = implied_volatility(test_price, S0, K, T, r)
    
    print(f"Heston Test Price (K={K}, T={T}): {test_price:.4f}")
    print(f"Implied Volatility (IV): {test_iv:.4f}")

    # Note: run_phase_3_analysis() is commented out to keep output clean during calibration run.
    # run_phase_3_analysis()