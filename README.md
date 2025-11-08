# Heston Stochastic Volatility Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Quantitative Finance](https://img.shields.io/badge/Quantitative-Finance-orange)
![Project Status](https://img.shields.io/badge/Status-In%20Development-yellow)

A comprehensive implementation of the Heston stochastic volatility model for option pricing, featuring numerical simulations, Fourier transform pricing methods, and model calibration techniques.

## 📊 Project Overview

This project implements the Heston model, which extends Black-Scholes by modeling volatility as a stochastic process that exhibits mean reversion. Unlike constant volatility assumptions, this approach captures real-market phenomena like volatility smiles and skews.

**Key Features:**
- ✅ **Variance Process Simulation** - Visualize mean reversion in volatility
- 🔄 **Stochastic Volatility** - Realistic modeling of changing market volatility
- 📈 **Professional Visualization** - Clear plots demonstrating model behavior
- 🏗️ **Modular Architecture** - Clean, maintainable code structure

## 🚀 Quick Start

### Run the Variance Simulation
```bash
# Clone the repository
git clone https://github.com/thapala2301/heston-volatility-model
cd heston-volatility-model

# Install dependencies
pip install -r requirements.txt

# Run the simulation
python run_simulation.py
