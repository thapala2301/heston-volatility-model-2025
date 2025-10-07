# Characteristic Function Learning Notes

## What I Understand So Far:

### 1. In My Own Words:
Previously when we ran simulations, we could see nummerous paths that underlying asset prices could take due to the variations in volatility. The characteristic function is a tool that fully describes the joint distribution of the underlying asset price and its stochastic volatility.

### 2. The Big Picture:
Black-Scholes assumes constant volatility while Heston incorporates stochastic volatility that changes over time and is correlated with the asset price.

### 3. The Fourier Connection:
The problem? can be converted into a characterisc function which can then be efficiently inverted to fin the option`s probablitydistribution and price. This involves dealing with complexities in stochastic volaility by working in the frequency domain instead. Such a manthematical maneouvre allows for faster calculations for a wider range of maturities and strikes whilst dodging issues like the Gibbs phenomenon.

### 4. Questions I Have:
- What is the formula and its components and how does it work in simple terms?
- How is the formuula linked to the 2 SDEs we met earlier?
- What exactly is the 'u' parameter?
- Why do we need complex numbers? Why do we need Fourier?
- How does this connect to my variance simulation?

## Key Concepts to Research:
- [ ] Characteristic functions in probability?
- [ ] Fourier transforms in finance ?
- [ ] Complex numbers in quantitative finance ?
- [ ] Carr-Madan FFT method ?

## Analogies That Help Me Understand:
[No idea - give me good ones that are simple and make sesnse]

## How This Fits With My Variance Simulation:
My simulation showed volatility paths. The characteristic function mathematically describes the probabilities of all possible price paths.
