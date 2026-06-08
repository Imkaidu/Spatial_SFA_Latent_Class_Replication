# Spatial SFA with Latent Classes — Replication Package

Replication code and data for:

> Du, K., Prokhorov, A., & Tran, K. C. (2026). Spatial stochastic frontier analysis with latent classes. *Working paper*.

Public repository: https://github.com/Imkaidu/Spatial_SFA_Latent_Class_Replication

## Contents

| File | Description |
|---|---|
| `Main_Empirical_SFA_Latent_Class_nStarts.m` | Estimation script — runs EM algorithm for 6 model configurations and reproduces Table 5 |
| `SFA_Core.m` | Core estimation engine (EM algorithm, parameter management, SE computation) |
| `Empirical_Data.mat` | Pre-processed balanced panel (n=121 countries, T=20 years, 2000–2019) |

## Requirements

- MATLAB R2023a or later
- Statistics and Machine Learning Toolbox (for `kmeans`, `normcdf`)
- Optimization Toolbox (for `fmincon`)
- Parallel Computing Toolbox (optional; speeds up multi-start estimation via `parfor`)

## Usage

1. Place all three files in the same directory
2. Open MATLAB and `cd` to that directory
3. Run:
```matlab
Main_Empirical_SFA_Latent_Class_nStarts.m
```

The script estimates the spatial SFA latent class model for 6 configurations (2 specifications × 3 country samples) and prints parameter estimates with standard errors, matching Table 5 of the paper.

## Model Configurations

| Specification | Sample | n |
|---|---|---|
| Spec=1 (per-capita): ln(GDP/pop) | Full Sample (WB all income groups) | 121 |
| Spec=1 (per-capita) | High Income (WB High + Upper-Middle) | 80 |
| Spec=1 (per-capita) | Developing (WB Upper-Middle, Lower-Middle, Low) | 71 |
| Spec=2 (per-worker): ln(GDP/emp) | Full Sample | 121 |
| Spec=2 (per-worker) | High Income | 80 |
| Spec=2 (per-worker) | Developing | 71 |

## Authors

- **Kai Du** — University of Wollongong
- **Artem Prokhorov** — University of Sydney
- **Kien C. Tran** — University of Lethbridge
