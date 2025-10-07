# Heston Characteristic Function: Complete Mathematical Foundation

This document explains **how and why** the Heston characteristic function works — bridging the gap between the stochastic model, Fourier analysis, and real implementation.

## 1. The Fundamental Problem

### 1.1 Black–Scholes Limitation
The **Black–Scholes model** assumes constant volatility:

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

**Problem:** Real markets exhibit:
* Volatility that changes over time
* Correlation between price and volatility
* “Volatility smiles” in option data

### 1.2 Heston’s Breakthrough Solution
Heston introduced **stochastic volatility**, allowing volatility to evolve randomly:

$$
\begin{cases}
dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^1 \\
dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^2 \\
dW_t^1 dW_t^2 = \rho\,dt
\end{cases}
$$

Where:
* $\sqrt{v_t}$ — stochastic volatility
* $\kappa$ — mean reversion speed
* $\theta$ — long-term average variance
* $\sigma$ — volatility of volatility
* $\rho$ — correlation between price and volatility shocks

### 1.3 The Mathematical Challenge
* **Black–Scholes:** simple closed-form solution exists.
* **Heston:** no direct formula for option prices — must derive one using **characteristic functions** and **Fourier transforms**.

---

## 2. What is a Characteristic Function?

### 2.1 Probability Foundation
For a random variable $X$, its **characteristic function** is:

$$
\phi(u) = \mathbb{E}[e^{i u X}]
$$

Where:
* $i = \sqrt{-1}$: imaginary unit
* $u$: real (or complex) frequency variable
* $\mathbb{E}$: expected value

### 2.2 Simple Example — Coin Flip
If $X = 1$ (heads) or $0$ (tails) with probability $0.5$ each:

$$
\phi(u) = 0.5\,e^{i u} + 0.5\,e^{0} = 0.5(e^{i u} + 1)
$$

### 2.3 The Fourier Inversion Magic
A probability density $f(x)$ can be recovered from its characteristic function:

$$
f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \phi(u)e^{-iux}\,du
$$

So $\phi(u)$ encodes the **entire distribution** of $X$.

### 2.4 For Stock Prices
For option pricing, we want the characteristic function of the **log-price** $\ln S_T$:

$$
\phi(u) = \mathbb{E}\left[e^{i u \ln S_T}\right]
$$

This compactly stores all information about the future distribution of $S_T$.

---

## 3. Why Fourier? Why Complex Numbers?

### 3.1 The Mathematical Superpower
* Directly integrating over random paths is hard.
* Characteristic functions often have **analytic solutions**, even when densities don’t.

### 3.2 Complex Numbers as Rotation Operators
The Euler formula states:

$$
e^{i\theta} = \cos(\theta) + i\sin(\theta)
$$

In the complex plane, $e^{i\theta}$ means a **rotation** by angle $\theta$.

### 3.3 Fourier Transform as a “Mathematical Prism”
Just as a prism splits light into colors, the Fourier transform decomposes a distribution into frequency components that are easier to analyze.

### 3.4 Carr–Madan Insight (1999)
Carr and Madan showed that **option prices** can be expressed directly as the **Fourier transform of the characteristic function** — a massive simplification for stochastic models like Heston.

---

## 4. The Heston Characteristic Function — Step-by-Step for Beginners

### 4.1 What Is the Characteristic Function in Heston?

The **characteristic function** is a mathematical tool that lets us describe the probability distribution of future stock prices, even when volatility itself is random.

Heston (1993) showed that, thanks to the structure of his stochastic volatility model, we can write this function in a very compact way — called **exponential-affine form**:

$$
\phi(u) = \exp\!\left(C(\tau,u) + D(\tau,u)v_t + i u \ln S_t\right)
$$

Let’s break down what this means:

- $\phi(u)$: The characteristic function itself. For each value of $u$, this function encodes information about possible future prices.
- $\tau = T - t$: Time left until option expiry (in years, for example).
- $S_t$: The current price of the asset (e.g., stock).
- $v_t$: The current variance (volatility squared) of the asset.
- $C(\tau, u)$ and $D(\tau, u)$: Two functions (explained below!) that capture how volatility changes with time, and how it interacts with market parameters.

**Why is this form useful?**  
Because you only need to plug in numbers for $C$, $D$, $v_t$, $S_t$, and $u$ — and you instantly get the characteristic function, which is the key ingredient for option pricing.

---

### 4.2 How Do We Calculate $C(\tau, u)$ and $D(\tau, u)$?

These two functions boil down the complicated math of stochastic volatility into formulas that are surprisingly universal.

#### Step 1: Calculate the "intermediate variables"
These help simplify the main equations:

- $b = \kappa - i \rho \sigma u$
  - $\kappa$: Mean reversion rate (how quickly volatility returns to its normal level)
  - $\rho$: Correlation between price and volatility changes
  - $\sigma$: How volatile the volatility is (“vol of vol”)
  - $i$: Square root of -1 (imaginary unit)
  - $u$: Frequency variable (can be thought of as a “probe” into the distribution)
- $d = \sqrt{b^2 + \sigma^2 (i u + u^2)}$
  - This is a kind of “adjusted volatility,” taking complex effects into account.
- $g = \frac{b - d}{b + d}$
  - This is just a handy ratio that shows up in the formulas.

#### Step 2: Plug into the formulas for $C$ and $D$

- $D(\tau, u) = \frac{b - d}{\sigma^2} \cdot \frac{1 - e^{-d\tau}}{1 - g e^{-d\tau}}$
  - $D$ controls how future volatility affects the characteristic function.
- $C(\tau, u) = i u r \tau + \frac{\kappa \theta}{\sigma^2} \left[(b - d)\tau - 2 \ln \left(\frac{1 - g e^{-d\tau}}{1 - g}\right)\right]$
  - $C$ controls how the model’s long-term average volatility, mean reversion, and risk-free rate all contribute.

**Summary formula:**  
Once you have $C$ and $D$, the full characteristic function is:

$$
\boxed{\phi(u) = \exp\left(C(\tau, u) + D(\tau, u) v_t + i u \ln S_t \right)}
$$

---

### 4.3 What Does Each Parameter Do?

Each parameter in the Heston model has a meaning, and tweaking it changes the shape of the distribution (and thus, option prices):

| Parameter | What It Means | What It Changes in $\phi(u)$ |
| :---: | :--- | :--- |
| $\kappa$ | Speed of mean reversion | How fast volatility returns to “normal” |
| $\theta$ | Long-run average variance | The “typical” level volatility hovers around |
| $\sigma$ | Volatility of volatility | How wild or calm volatility itself is |
| $\rho$ | Correlation | Whether volatility goes up when price drops (or vice versa) |
| $v_t$ | Current variance | The present uncertainty in the market (varies day-to-day) |

---

**In plain English:**  
- The Heston characteristic function is like a “code” for the entire future distribution of prices.  
- You build it by plugging market parameters and current conditions into a formula.  
- This function then lets you price options and understand risk, even in complex markets where volatility itself is unpredictable.

---

## 5. From Math to Code

### 5.1 Implementation Strategy

```python
import cmath

def heston_characteristic_function(u, tau, x, v, kappa, theta, sigma, rho, r):
    """Calculates the Heston characteristic function (log-price)."""
    # Ensure u is treated as a complex number
    u = complex(u)
    
    # 1. Define b, d, g
    b = kappa - 1j * rho * sigma * u
    # Note: d must use the principal square root
    d = cmath.sqrt(b**2 + sigma**2 * (1j*u + u**2))
    g = (b - d) / (b + d)

    # 2. Define D(tau, u)
    exp_dt = cmath.exp(-d * tau)
    D = (b - d) / sigma**2 * (1 - exp_dt) / (1 - g * exp_dt)
    
    # 3. Define C(tau, u)
    # The log-term should use cmath.log for complex arguments
    log_term = cmath.log((1 - g * exp_dt) / (1 - g))
    C = 1j * u * r * tau + (kappa * theta / sigma**2) * ((b - d) * tau - 2 * log_term)
    
    # 4. Final characteristic function
    phi = cmath.exp(C + D * v + 1j * u * x)
    
    return phi
``` 

### 5.2 Implementation Strategy

At $u=0$:

$$
\phi(0) = \mathbb{E}\left[e^{i \times 0 \times \ln S_T}\right] = \mathbb{E}[1] = 1.
$$

✅ Test numerically that $\phi(0) = 1 + 0i$. This is the **mathematical check** for any characteristic function.

---

## 6. Three Core Analogies

### 6.1 Analogy One
| Concept | Medical World | Heston World |
| :--- | :--- | :--- |
| **Object** | Human body | Stock price distribution |
| **Scan Type** | MRI under various settings | Characteristic function with different $u$ |
| **Machine** | MRI scanner | Mathematical transform |
| **Reconstruction** | 3D image | Option price curve |
| **Key Insight** | MRI reconstructs anatomy from frequency data | Fourier methods reconstruct option prices from $\phi(u)$ |

### 6.2 Analogy Two

| Concept | Music Studio | Heston World |
| :--- | :--- | :--- |
| **Performance** | Live orchestra | Stock price movement |
| **Instruments** | Different sound sources | Risk factors (price, volatility) |
| **Mixing Board** | Audio console | $\phi(u)$ formula |
| **Equalizer** | Frequency controls | Choice of $u$ values |
| **Final Track** | Mastered mix | Option price surface |
| **Key Insight** | Adjusting frequencies refines a mix | Evaluating $\phi(u)$ at different $u$ reveals the structure of the probability distribution |

### 6.3 Analogy Three
| Concept | GPS Navigation | Heston World |
| :--- | :--- | :--- |
| **Map** | Road network | Price probability space |
| **Traffic** | Market conditions | Volatility dynamics |
| **GPS Algorithm** | Route optimization | Characteristic function computation |
| **Different Routes** | Highways vs. local roads | Different possible price paths |
| **ETA Calculation** | Travel time | Expected option price |
| **Key Insight** | GPS evaluates all possible routes simultaneously | $\phi(u)$ encodes all possible price paths simultaneously |

---

