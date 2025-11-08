import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import minimize 

# IMPORTANT: Ensure heston_char_function.py is in the same directory.
# This imports the pricing functions and the implied volatility solver
from heston_char_function import heston_call_price, implied_volatility 

# ----------------------------------------------------------------------
# 1. MARKET DATA FETCHING FUNCTION
# ----------------------------------------------------------------------
def fetch_and_clean_options_data(ticker_symbol='SPY', expiration_date=None, risk_free_rate=0.05):
    """
    Fetches, cleans, and structures options chain data for calibration, 
    prioritizing a stable maturity (30-90 days).
    Returns: A tuple (market_data_records, S0, r)
    """
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        S0 = ticker.info['regularMarketPrice']
    except Exception:
        print("❌ Could not fetch current spot price (S0).")
        return None, None, None
    
    if expiration_date is None:
        all_expirations = ticker.options
        if not all_expirations:
            print("❌ No options data available for this ticker.")
            return None, None, None
        
        today = pd.Timestamp.now().date()
        selected_exp = None
        
        for exp_str in all_expirations:
            exp_date = pd.to_datetime(exp_str).date()
            time_diff = (exp_date - today).days
            
            if time_diff >= 30 and time_diff <= 90:
                selected_exp = exp_str
                break
        
        if selected_exp is None:
            print("⚠️ Warning: Ideal 30-90 day expiration not found. Falling back to the second near-term date.")
            exp = all_expirations[1]
        else:
            exp = selected_exp
            
    else:
        exp = expiration_date

    try:
        options = ticker.option_chain(exp)
    except Exception as e:
        print(f"❌ Error fetching option chain for {exp}: {e}")
        return None, None, None

    calls = options.calls.copy()
    calls['MidPrice'] = (calls['bid'] + calls['ask']) / 2
    
    expiry_date = pd.to_datetime(exp)
    T = (expiry_date.date() - pd.Timestamp.now().date()).days / 365.25 
    
    calls = calls[(calls['volume'] > 0) & 
                  (calls['MidPrice'] > 1e-6) & 
                  (calls['ask'] > calls['bid'])].reset_index(drop=True)
    
    market_data = calls[['strike', 'MidPrice']].rename(
        columns={'strike': 'K', 'MidPrice': 'MarketPrice'}
    )
    market_data['T'] = T 
    
    return market_data.to_records(index=False), S0, risk_free_rate

# ----------------------------------------------------------------------
# 2. CALIBRATION HELPER FUNCTIONS
# ----------------------------------------------------------------------

def feller_constraint(params):
    """Feller Condition: 2 * kappa * theta - sigma**2 >= 0"""
    kappa, theta, sigma, rho, v0 = params
    return 2 * kappa * theta - sigma**2

def objective_function(params, market_data_records, S0, r):
    """Calculates the Root Mean Squared Error (IV RMSE)."""
    kappa, theta, sigma, rho, v0 = params
    squared_errors = []
    
    # We rely on the scipy 'constraints' tuple for Feller Condition enforcement.

    for row in market_data_records:
        K, T, market_price = row['K'], row['T'], row['MarketPrice']
        
        try:
            # 1. Calculate Heston Price
            heston_price = heston_call_price(K, S0, v0, r, T, kappa, theta, sigma, rho)
            
            # 2. Invert Prices to Implied Volatility (IV)
            market_iv = implied_volatility(market_price, S0, K, T, r)
            heston_iv = implied_volatility(heston_price, S0, K, T, r)
            
            # 3. Calculate Error
            if np.isnan(market_iv) or np.isnan(heston_iv):
                 squared_errors.append(1e5) # Penalty for IV solver failure
            else:
                 squared_errors.append((market_iv - heston_iv) ** 2)
            
        except Exception:
            squared_errors.append(1e10) # Heavy penalty for integration/pricing crash
            
    if not squared_errors:
        return 1e10
         
    return np.sqrt(np.mean(squared_errors))

# ----------------------------------------------------------------------
# 3. MAIN CALIBRATION FUNCTION
# ----------------------------------------------------------------------
def calibrate_heston(initial_params, bounds, constraints, market_data_records, S0, r):
    """Runs the constrained optimization to find the optimal Heston parameters (Ω*)."""
    print("\n\n--- Starting Heston Model Calibration (Phase IV) ---")
    print(f"Initial Parameter Guess: [κ, θ, σ, ρ, v₀] = {initial_params}")
    
    calibration_result = minimize(
        fun=objective_function,     
        x0=initial_params,          
        args=(market_data_records, S0, r), 
        method='SLSQP',             
        bounds=bounds,              
        constraints=constraints,    
        tol=1e-7,                   
        options={'disp': True, 'maxiter': 500}
    )

    optimal_params = calibration_result.x
    min_rmse = calibration_result.fun
    
    print("\n✅ Calibration Complete!")
    print(f"Optimal Parameters (κ, θ, σ, ρ, v₀): {optimal_params}")
    print(f"Minimum Implied Volatility RMSE: {min_rmse:.6f}")
    
    return optimal_params, min_rmse

# ----------------------------------------------------------------------
# 4. PHASE V: VISUALIZATION FUNCTION (Updated to Save File)
# ----------------------------------------------------------------------
def plot_calibration_fit(market_data, optimal_params, S0, r):
    """
    Plots the Market Implied Volatility (IV) against the Heston Fitted IV.
    Also saves the generated plot to the local directory.
    """
    kappa, theta, sigma_vol, rho, v0 = optimal_params

    strikes = np.array([row['K'] for row in market_data])
    T = market_data[0]['T'] 
    
    # Calculate Market IVs (the target points)
    market_ivs = np.array([
        implied_volatility(row['MarketPrice'], S0, row['K'], row['T'], r) 
        for row in market_data
    ])
    
    # Generate Heston IVs (The Fitted Curve)
    heston_ivs = []
    
    # Calculate Heston IVs using the optimal parameters
    for K in strikes:
        try:
            heston_price = heston_call_price(K, S0, v0, r, T, kappa, theta, sigma_vol, rho)
            iv = implied_volatility(heston_price, S0, K, T, r)
            heston_ivs.append(iv)
        except Exception:
            heston_ivs.append(np.nan)

    heston_ivs = np.array(heston_ivs)
    
    # Filter out NaNs for clean plotting
    valid_indices = ~np.isnan(market_ivs) & ~np.isnan(heston_ivs)
    strikes_valid = strikes[valid_indices]
    market_ivs_valid = market_ivs[valid_indices]
    heston_ivs_valid = heston_ivs[valid_indices]
    
    # Calculate final RMSE on the valid subset 
    final_rmse = np.sqrt(np.mean((market_ivs_valid - heston_ivs_valid)**2)) if len(market_ivs_valid) > 0 else np.nan
        
    # Create the Plot
    plt.figure(figsize=(12, 7))
    
    plt.scatter(strikes_valid, market_ivs_valid * 100, 
                color='#FF4500', marker='o', s=25, alpha=0.7, label='Market IV (Target)')
    
    # Sort for a clean line plot
    sort_indices = np.argsort(strikes_valid)
    plt.plot(strikes_valid[sort_indices], heston_ivs_valid[sort_indices] * 100, 
             color='#004D40', linestyle='-', linewidth=2, label='Heston Fitted IV')
    
    plt.axvline(S0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label=f'ATM (S0={S0:.2f})')
    
    plt.title(f'Heston Calibration Fit: Implied Volatility Skew (T={T:.2f} Years)\nFinal IV RMSE: {final_rmse:.5f}', fontsize=16)
    plt.xlabel('Strike Price (K)', fontsize=14)
    plt.ylabel('Implied Volatility (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # --- NEW: SAVE THE GRAPH ---
    # Creates a descriptive filename including T and RMSE
    filename = f"Heston_Skew_T{T:.2f}_RMSE{final_rmse:.4f}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n✅ Plot saved successfully as: {filename}")
    # ---------------------------
    
    plt.show() # Displays the graph

# ----------------------------------------------------------------------
# 5. EXECUTION BLOCK (FINAL)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Get Market Data 
    market_data_records, S0, r = fetch_and_clean_options_data(ticker_symbol='SPY')
    
    if market_data_records is None or S0 is None:
        print("\nFATAL ERROR: Could not retrieve market data. Cannot proceed with calibration.")
        exit()

    print(f"\n--- Market Data Snapshot ---")
    print(f"Spot Price (S0): ${S0:.2f}")
    print(f"Risk-Free Rate (r): {r:.2%}")
    print(f"Options to Calibrate: {len(market_data_records)} contracts for T={market_data_records[0]['T']:.2f} years")

    # 2. Define Initial Guesses and Constraints
    initial_params = np.array([1.0, 0.04, 0.3, -0.5, 0.03])
    
    # Final, relaxed bounds from the last run
    bounds = [
        (0.01, 10.0), # kappa
        (0.001, 5.0), # theta
        (1e-6, 2.0),  # sigma
        (-1.0, 1.0),  # rho
        (0.001, 5.0)  # v0
    ]
    
    constraints = ({'type': 'ineq', 'fun': feller_constraint})

    # 3. Run the Calibration
    optimal_params, min_rmse = calibrate_heston(
        initial_params, 
        bounds, 
        constraints, 
        market_data_records, 
        S0, 
        r
    )
    
    print("\n\n#####################################################")
    print("### PHASE IV COMPLETE: HESTON MODEL CALIBRATION ###")
    print("#####################################################")
    print(f"Optimal Heston Parameters:")
    print(f"  Kappa (κ): {optimal_params[0]:.4f}")
    print(f"  Theta (θ): {optimal_params[1]:.4f}")
    print(f"  Sigma (σ): {optimal_params[2]:.4f}")
    print(f"  Rho (ρ):   {optimal_params[3]:.4f}")
    print(f"  V0 (v₀):   {optimal_params[4]:.4f}")
    print(f"Final IV RMSE: {min_rmse:.6f}")
    
    # 4. Visualization (Phase V)
    print("\nStarting PHASE V: Plotting Calibration Fit... 📈")
    plot_calibration_fit(market_data_records, optimal_params, S0, r)