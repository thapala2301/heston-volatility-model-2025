# Heston Model Research

The Heston Model is a stochastic model widely used for pricing options in equity and foreign exchange markets. It extends the logic of the Black–Scholes model by introducing a more realistic treatment of volatility — allowing it to vary randomly over time rather than assuming it remains constant.

This model has become a standard tool in quantitative finance because it captures several key market phenomena, particularly the **volatility smile** — a pattern observed when implied volatility is plotted against strike prices.

## 1. Motivation

The Black–Scholes model assumes that volatility is constant throughout the life of an option. In practice, this is unrealistic. Market participants observe that implied volatility changes across strike prices and maturities, producing the "smile" or "skew" patterns seen in real data.

The Heston Model resolves this by modelling volatility as a stochastic process that fluctuates over time, yet tends to revert toward a long-run mean. This creates a more flexible and accurate description of market behaviour, especially for pricing options that are deep in or out of the money.

## 2. Core Equations

The model assumes two coupled stochastic differential equations (SDEs):
1) dSₜ = μSₜdt + √vₜSₜdWₜ¹ 
2) dvₜ = κ(θ - vₜ)dt + σ√vₜdWₜ²

where the two Brownian motions are correlated:
dWₜ¹dWₜ² = ρdt


### 2.1 Stock Price Dynamics

The first equation models the asset price **Sₜ**:

- **dSₜ**: infinitesimal change in the asset price
- **μ**: expected rate of return per unit time  
- **Sₜ**: current asset price
- **√vₜ**: instantaneous volatility (standard deviation of returns)
- **dWₜ¹**: Brownian motion representing random shocks to price

The term **μSₜdt** captures the deterministic drift or expected return, while **√vₜSₜdWₜ¹** represents random fluctuations driven by market noise.

### 2.2 Variance (Volatility) Dynamics

The second equation models the variance **vₜ**:

- **dvₜ**: infinitesimal change in variance
- **κ**: mean reversion rate — speed at which variance reverts to its long-run average
- **θ**: long-term mean variance
- **vₜ**: current variance level
- **σ**: volatility of volatility (the standard deviation of variance changes)
- **dWₜ²**: Brownian motion introducing random shocks to variance

This equation treats volatility as its own random process that fluctuates independently of — but is correlated with — the underlying asset's returns.

## 3. Intuitive Interpretation

### Mean Reversion (κ(θ - vₜ)dt)
Volatility can be thought of as a ball tied to a rope fixed at height **θ**. The rope's strength (represented by **κ**) determines how quickly the ball is pulled back when it drifts away. A higher **κ** means volatility reverts to its mean faster.

### Random Shocks (σ√vₜdWₜ²)
Volatility itself experiences random fluctuations. The size of these fluctuations depends on **σ** — the volatility of volatility — and the current variance level **vₜ**.

### Correlation (ρ)
The parameter **ρ** captures the relationship between price and volatility shocks. In equity markets, **ρ** is typically negative, meaning that when prices fall, volatility tends to rise (the "leverage effect").

## 4. Volatility Smile and Market Realism

By allowing variance to be stochastic and mean-reverting, the Heston model naturally produces an implied volatility smile — the curvature in implied volatilities across strike prices.

Instead of enforcing constant volatility (as in Black–Scholes), Heston's dynamics let volatility cluster and drift, leading to patterns that closely match real market data.

When calibrated to market option prices, the Heston model can accurately capture both volatility smiles and skews across maturities.

## 5. Limitations and Breakdown Scenarios

Despite its realism, the Heston model has certain limitations:

- For very short maturities, it can fail to capture the rapid "explosion" in volatility observed empirically.
- It assumes continuous diffusion dynamics and cannot model jumps or discontinuities in asset prices.
- In markets where the implied volatility surface exhibits extreme curvature or "roughness," Heston's smooth variance process struggles to fit observed data.

These discrepancies arise because real-world volatility may follow rough or jump-driven processes rather than the continuous mean-reverting diffusion that Heston assumes.

## 6. Summary Table of Parameters

| Symbol | Parameter | Meaning |
|--------|-----------|---------|
| **κ** | Mean reversion rate | Speed at which variance reverts to its long-run mean |
| **θ** | Long-run variance | Average level of variance over time |
| **σ** | Volatility of volatility | Randomness in the variance process |
| **v₀** | Initial variance | Starting value of variance |
| **ρ** | Correlation | Relationship between price and variance shocks |

## 7. Key Takeaways

- The Heston model extends Black–Scholes by introducing stochastic variance.
- It captures volatility smiles and clustering, making it more realistic for option pricing.
- The variance process follows a Cox–Ingersoll–Ross (CIR) type SDE, ensuring that **vₜ** stays positive.
- Its main mathematical challenge lies in deriving and using the characteristic function for pricing, which we will explore next.
