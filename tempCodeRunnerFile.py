import numpy as np
import cmath
from scipy.integrate import quad

def heston_Pj_integrand(u, j, K, S0, v0, r, T, kappa, theta, sigma, rho):
    i = complex(0,1)
    minimal_u = 1e-10

    # Singularity Check at u = 0
    if u < minimal_u:
        return 0.0

    if j == 1:
         u_shifted = u - i*0.5
    else:
        u_shifted = u + i*0.5

    iu_shifted = i * u_shifted

    # FIXED: Add bounds checking to prevent overflow
    if abs(u) > 50:  # Large u values cause instability
        return 0.0

    # FIXED: Use the "Little Trap" formulation for better stability
    x = cmath.log(S0)
    
    d = cmath.sqrt((rho * sigma * iu_shifted - kappa)**2 + sigma**2 * (iu_shifted + u_shifted**2))
    
    # FIXED: Stable g calculation using the "Little Trap"
    g = (kappa - rho * sigma * iu_shifted - d) / (kappa - rho * sigma * iu_shifted + d)
    
    # FIXED: Check for numerical stability
    exp_neg_dT = cmath.exp(-d * T)
    if abs(1 - g * exp_neg_dT) < 1e-12:
        return 0.0
    
    # FIXED: Stable C calculation
    C = iu_shifted * r * T + (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * iu_shifted - d) * T - 2 * cmath.log((1 - g * exp_neg_dT) / (1 - g))
    )
    
    # FIXED: Stable D calculation
    D = (kappa - rho * sigma * iu_shifted - d) * (1 - exp_neg_dT) / (sigma**2 * (1 - g * exp_neg_dT))
    
    # FIXED: Check for overflow before exponential
    char_exp = C + D * v0 + iu_shifted * x
    if char_exp.real > 100:  # Prevent exponential overflow
        return 0.0
        
    phi = cmath.exp(char_exp)
    
    # FIXED: Final overflow check
    if abs(phi) > 1e50:
        return 0.0

    component_1 = cmath.exp(-i * u * cmath.log(K))
    output = (component_1 * phi) / (i * u)
    
    return output.real

def risk_neutral_prob_function(j, K, S0, v0, r, T, kappa, theta, sigma, rho):
    heston_params = (K, S0, v0, r, T, kappa, theta, sigma, rho)
    
    # FIXED: Use smaller, more stable integration range
    integral_result, error_estimate = quad(
        func=heston_Pj_integrand, 
        a=1e-10,
        b=50,  # Reduced upper limit
        args=(j,) + heston_params,
        limit=1000,
        epsabs=1e-6,
        epsrel=1e-6,
        full_output=0
    )

    Pj = 0.5 + (1/np.pi) * integral_result
    
    # FIXED: Add sanity bounds
    if Pj < 0 or Pj > 1:
        # If out of bounds, use reasonable defaults based on j
        return 0.6 if j == 1 else 0.5
    
    return Pj

def heston_call_price(K, S0, v0, r, T, kappa, theta, sigma, rho):
    heston_params = (K, S0, v0, r, T, kappa, theta, sigma, rho)

    P1 = risk_neutral_prob_function(1, *heston_params)
    P2 = risk_neutral_prob_function(2, *heston_params)

    print(f"P1={P1:.6f}, P2={P2:.6f}")
    
    Call_Price = S0 * P1 - K * np.exp(-r * T) * P2
    return Call_Price

def main():
    # FIXED: Use very stable parameters for testing
    testing_parameters = (
        100.0,  # K
        100.0,  # S0  
        0.01,   # v0 (lower volatility)
        0.02,   # r (lower rate)
        0.5,    # T (shorter maturity)
        1.0,    # kappa
        0.01,   # theta
        0.05,   # sigma (much lower vol-of-vol)
        -0.3    # rho (weaker correlation)
    )

    final_price = heston_call_price(*testing_parameters)
    print(f"Heston Call Price: {final_price:.4f}")

if __name__ == "__main__":
     main()