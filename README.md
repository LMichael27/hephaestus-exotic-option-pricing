# Efficient Pricing of Exotic Derivatives

**Deep learning approximation of Monte Carlo prices under GBM, Heston and Bates models**

A project of the **Hephaestus Applied Artificial Intelligence Association** at Bocconi University (Fall 2025).
The full write-up is in [`Paper.pdf`](Paper.pdf).

## Overview

Exotic, path-dependent options rarely have closed-form prices, especially under stochastic volatility or jumps.
Monte Carlo simulation handles them but is slow when thousands of prices must be recomputed, for example in calibration or risk monitoring.

This project trains feedforward neural networks to learn the pricing map

$$f(\theta) = \mathbb{E}^{\mathbb{Q}}\left[e^{-rT}\,\Phi(S_{[0,T]};\theta)\right]$$

from model and contract parameters to Monte Carlo prices. Once trained, the network returns a price almost instantly.

It covers three asset dynamics of increasing complexity:

| Model | Dynamics | Parameters |
|---|---|---|
| **GBM** | Geometric Brownian motion | `sigma` |
| **Heston** | Stochastic volatility (QE scheme, Andersen 2008) | `v0`, `kappa`, `theta`, `xi`, `rho` |
| **Bates** | Heston + Merton log-normal jumps | Heston parameters + `jump_lambda`, `jump_mu`, `jump_delta` |

It also covers six path-dependent payoffs:
- Asian options with arithmetic averaging (call and put)
- Up-and-out barrier call and down-and-out barrier put
- Floating-strike lookback options (call and put)

## Repository structure

| File | Description |
|---|---|
| `data_generation.py` | Monte Carlo data generator: parameter sampling, path simulation, payoffs and batched pricing |
| `data_generation.ipynb` | Step-by-step notebook version of the generator, with a path visualisation for a sample contract |
| `Creating_Neural_Network_GBM_final.ipynb` | Network training and evaluation under GBM |
| `Creating_Neural_Network_Heston_final.ipynb` | Network training and evaluation under Heston |
| `Creating_Neural_Network_Bates_final.ipynb` | Network training and evaluation under Bates |
| `Creating_Neural_Network_Heston.ipynb` | Earlier, simpler baseline for Heston (smaller network, no batch normalization) |
| `exotic_mc_dataset_*.parquet` | Generated dataset with 10,000 priced contracts. `_2` is an identical copy of `_1`, so the notebooks use only `_1` |
| `exotic_mc_dataset_meta_*.json` | Generation settings: Monte Carlo configuration and parameter ranges |
| `Paper.pdf` | Project report |

## Data generation

Each row of the dataset is one contract. To build a row, the generator does the following:
1. It picks a model and an option type at random.
2. It samples the parameters uniformly from the ranges below.
3. It prices the contract with Monte Carlo: 30,000 paths, 252 time steps, antithetic variates and seed 42.

| Parameter | Range | | Parameter | Range |
|---|---|---|---|---|
| `S0`, `K` | 50 – 150 | | `v0`, `theta` | 0.01 – 0.50 |
| `r` | 0 – 8% | | `kappa` | 0.5 – 6.0 |
| `T` (years) | 0.05 – 2.0 | | `xi` | 0.1 – 2.0 |
| `sigma` | 0.05 – 0.90 | | `rho` | −0.95 – 0 |
| barrier | 0.85 – 1.25 × `S0` | | `jump_lambda` / `jump_mu` / `jump_delta` | 0–1 / −0.25–0.10 / 0.05–0.50 |

The output columns are `model`, `option`, `price` and `stderr` (Monte Carlo standard error), followed by all parameters.
A parameter that does not apply to the row's model is stored as `NaN`.

## Neural network

The network is a fully connected regressor written in PyTorch.
- **Inputs:** the model and contract parameters and a one-hot encoding of the option type, standardized with `StandardScaler`. The Monte Carlo standard error (`stderr`) is not used as an input, because it is only known after running the simulation.
- **Hidden layers:** Linear, BatchNorm, GELU and Dropout, with widths 128 → 128 → 64 → 32, followed by a linear output.
- **Training:** MSE loss with AdamW (lr 1e-3), a `ReduceLROnPlateau` scheduler and early stopping on validation loss.
- **Data split:** 70 % train, 15 % validation, 15 % test.
- **Metrics:** MSE, RMSE, MAE, R² and the share of predictions within ±20 % of the Monte Carlo price.

The paper describes the final experimental setup: one network per *(model, option)* pair, averaged over 10 random seeds. See Section 3 of the paper.

## Results

The table shows validation metrics averaged over seeds, taken from the paper. "Rel. cov." is the share of predictions within ±20 % of the Monte Carlo price.

| Option | GBM RMSE | Heston RMSE | Bates RMSE | GBM Rel. cov. | Heston Rel. cov. | Bates Rel. cov. |
|---|---|---|---|---|---|---|
| Asian call | 1.35 | 2.84 | 3.45 | 0.74 | 0.60 | 0.58 |
| Asian put | 1.40 | 2.56 | 3.60 | 0.73 | 0.70 | 0.63 |
| Barrier down-and-out put | 2.13 | 3.29 | 4.13 | 0.03 | 0.05 | 0.01 |
| Barrier up-and-out call | 4.02 | 6.61 | 7.84 | 0.10 | 0.10 | 0.04 |
| Lookback call | 1.33 | 2.66 | 3.20 | 0.98 | 0.97 | 0.21 |
| Lookback put | 1.41 | 2.66 | 3.24 | 0.98 | 0.97 | 0.18 |

The results show three main patterns:
- **Smooth payoffs are learned well.** Asian and lookback options fit closely under GBM and Heston.
- **Barrier options are the hardest case.** The knock-out feature makes prices jump sharply as parameters change, and the network fails to capture this.
- **Model complexity hurts accuracy.** Errors grow from GBM to Heston to Bates, with the jump-diffusion setting performing worst.

Possible next steps:
- Generate a much larger dataset. With about 10,000 samples spread over 18 *(model, option)* pairs, each network sees only a few hundred examples.
- Sample more densely near the barrier.
- Use a two-stage model for barrier options.

## Getting started

```bash
pip install numpy pandas pyarrow torch scikit-learn matplotlib jupyter
```

- **Train the networks:** open any `Creating_Neural_Network_*_final.ipynb` notebook and run all cells. The notebooks read the included parquet files.
- **Regenerate the data:** run `python data_generation.py`. Set `n_samples` in the `__main__` block first, since each sample runs a full Monte Carlo simulation and the job can take a long time.

## Team

**Project head:** Michael Ladaa
**Team members:** Aleksandar Georgiev, Mattia Arecco, Neil Maaouni, Raffaele Simbari, Salomé Messika
