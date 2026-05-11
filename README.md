# DPAC — Technical Guide

## What is the DPAC?

The **DPAC** (*Decision Point for Adequate Control*) is a Bayesian statistical indicator of occupational overexposure. It is defined as the **70th posterior percentile of the P95 exposure distribution**.

In practice, the DPAC is the concentration below which the 95th percentile of exposure falls — i.e., the value exceeded by no more than 5% of workers or situations — with a posterior probability of 70%. It is expressed in the same unit as the measurements (e.g. ppm or mg/m³).

The method is based on **Expostats Tool 2** (Lavoué et al. 2019, *Ann. Occup. Hyg.*).

---

## Statistical model — principle and code

### Why a hierarchical model?

The data come from **multiple sensors** (Respiratory, Breast L, Stomach R, etc.) worn simultaneously or successively. Each sensor represents a *group*, and concentrations are expected to vary at two distinct scales:

- **Within-group**: from one measurement to the next on the same sensor (measurement noise, temporal fluctuation) → variance σ²_W
- **Between-group**: from one sensor to another (anatomical differences, position, flow rate) → variance σ²_B

A flat model would ignore this structure and under- or over-estimate overall uncertainty. The hierarchical model estimates both variance components simultaneously and combines them to characterise total exposure.

---

### Model structure

**Level 1 — distribution of measurements within each group:**

```
log(X_ij) ~ N(μ_i, σ_W)
```

Each measurement j of group i follows a normal distribution on the log scale, centred on the group log-mean μ_i, with a within-group standard deviation σ_W shared across all groups.

**Level 2 — distribution of group means:**

```
μ_i ~ N(μ, σ_B)
```

Group means μ_i are not estimated independently: they are drawn from a common distribution with mean μ and standard deviation σ_B. This is the core of the hierarchical model — groups *inform each other* (partial pooling).

**Practical effect:** a group with few measurements has its μ_i pulled towards the global mean μ (shrinkage effect), which is statistically more conservative than producing an isolated estimate from sparse data.

**Total distribution and P95:**

The distribution of all exposures — across all sensors — has total variance:

```
σ_total = √(σ_B² + σ_W²)
```

The P95 (value exceeded by 5% of exposures) is:

```
P95 = exp(μ + z_0.95 × σ_total)     where z_0.95 = 1.645
```

---

### Code walkthrough

#### 1. Priors

```python
mu      = pm.Uniform("mu",      lower=-20, upper=20)
sigma_B = pm.LogNormal("sigma_B", mu=-0.8786, sigma=0.7823012)
sigma_W = pm.LogNormal("sigma_W", mu=-0.4106, sigma=0.7254381)
```

- `mu`: non-informative prior on the global log-concentration mean.
- `sigma_B` and `sigma_W`: log-normal priors calibrated by **Lavoué et al. 2019** on a large occupational hygiene database. These hyperpriors bring realistic external information about the expected magnitude of variability.

#### 2. Random effects — non-centred parameterisation

```python
z_i  = pm.Normal("z_i", mu=0, sigma=1, shape=n_groupes)
mu_i = pm.Deterministic("mu_i", mu + z_i * sigma_B)
```

The naive approach would be `mu_i ~ Normal(mu, sigma_B)`. The problem is that when `sigma_B` is small, the posterior of `mu_i` becomes very concentrated and the MCMC sampler must navigate an extremely narrow *funnel* (*Neal's funnel*), which severely slows convergence.

The **non-centred parameterisation** decouples the two: standardised effects `z_i ~ N(0,1)` are sampled from a fixed, easy-to-explore distribution, and `mu_i` is reconstructed as `mu + z_i × sigma_B`. Geometrically, the shape of the sampling space no longer changes with `sigma_B`.

#### 3. Likelihood of detected values

```python
pm.Normal("obs", mu=mu_i[idx_det], sigma=sigma_W, observed=log_y_det)
```

For each detected measurement, the model is told: *this log-value was drawn from N(μ_i, σ_W)* — the standard log-likelihood. `idx_det` holds the group index of each detected measurement.

#### 4. Likelihood of censored values — `pm.Potential`

Censored values do not provide an exact measurement but a **constraint**: the true value lies within a known interval. They cannot be passed to `observed` directly. Instead, their log-likelihood contribution is added manually via `pm.Potential`.

**Left censoring** (`<LoD` — value below the detection limit):

```python
p = normal_cdf(log_LOD[orig_idx_left[i]], mu_i[ci], sigma_W)
pm.Potential(f"cens_left_{i}", pm.math.log(p + 1e-10))
```

The probability that the true value is below LoD is the CDF of the normal distribution evaluated at `log(LoD)`. This is exactly `P(X < LoD)` under the distribution of group `ci`.

**Interval censoring** (`[LoD-LoQ]` — value between LoD and LoQ):

```python
p = normal_cdf(log_LOQ[orig_idx_int[i]], mu_i[ci], sigma_W) \
  - normal_cdf(log_LOD[orig_idx_int[i]], mu_i[ci], sigma_W)
pm.Potential(f"cens_int_{i}", pm.math.log(p + 1e-10))
```

The probability that the true value lies in `[LoD, LoQ]` is the difference of the CDF at both bounds: `P(LoD ≤ X ≤ LoQ)`. Each row can have its own `log_LOD[i]` and `log_LOQ[i]`, which is why per-row LoD/LoQ management is essential when different sensors have different detection thresholds.

#### 5. Derived quantities

```python
sigma_total = pm.Deterministic("sigma_total", pm.math.sqrt(sigma_B**2 + sigma_W**2))
p_critere   = pm.Deterministic("p_critere",   pm.math.exp(mu + z_critere * sigma_total))
```

`pm.Deterministic` does not define an additional random variable — it instructs the sampler to *save* the computed value at each draw. This produces the full posterior distribution of `sigma_total` and `p_critere` (the P95) without any post-hoc calculation.

---

### How PyMC assembles the joint log-posterior for NUTS

When writing `pm.Uniform`, `pm.Normal`, `pm.Potential`, etc., PyMC does no numerical computation. It **builds a symbolic graph** describing the unnormalised joint log-density:

```
log p(θ | data) = log p(μ) + log p(σ_B) + log p(σ_W)     ← log-priors
               + Σ_i log p(z_i)                            ← random effects
               + Σ_j log p(x_j | μ_{g(j)}, σ_W)           ← detected values
               + Σ_k log P(X_k < LoD_k | ...)             ← left-censored
               + Σ_l log P(LoD_l ≤ X_l ≤ LoQ_l | ...)     ← interval-censored
```

where `θ = (μ, σ_B, σ_W, z_1, …, z_n)` is the vector of **all free parameters** — in our case 3 + n_groups scalars.

Each term is a function of θ. PyMC sums them to produce a single function `ℓ(θ)`. This is **all that NUTS receives**: a black box capable of evaluating `ℓ(θ)` and its **gradient ∂ℓ/∂θ** at any point θ.

#### How NUTS explores this space

NUTS (*No-U-Turn Sampler*) is a variant of *Hamiltonian Monte Carlo* (HMC). The idea is physical: imagine a ball rolling on a surface whose height is `-ℓ(θ)`. High-density posterior regions are valleys; low-probability regions are hills.

At each iteration:

1. **Random momentum**: the ball is given a random initial velocity `p ~ N(0, M)`, where `M` is a mass matrix (diagonal, adapted during tuning).
2. **Hamiltonian integration**: the ball's trajectory is simulated in small *leapfrog* steps, using the gradient `∂ℓ/∂θ` to compute forces. The trajectory explores the surface efficiently — unlike naive Metropolis-Hastings, which takes short random steps.
3. **U-turn criterion**: NUTS extends the trajectory as long as it is making progress. As soon as it starts doubling back on itself (U-turn), it stops and proposes the final position as the next sample.
4. **Acceptance**: a Metropolis correction ensures the chain converges to the correct distribution even if the integration is not exact.

The gradient is computed by **automatic differentiation** (via JAX in the numpyro backend). This is why `pm.math.erf`, `pm.math.sqrt`, etc. must be used inside PyMC expressions rather than `np.erf`: NumPy operations are not differentiable by JAX.

#### The tuning phase (2 000 steps)

During tuning (the 2 000 discarded warm-up steps), the sampler learns:
- the **step size** ε (so that the acceptance rate stays close to `target_accept=0.95`)
- the **mass matrix** M (to normalise the scale of each parameter dimension)

After tuning, the 4 000 retained draws are near-independent samples from the posterior distribution.

#### The 4 chains

Four chains start from different initial positions (generated automatically). Convergence is verified by comparing variance *between* chains and *within* each chain: if all chains explored the same space, r̂ will be close to 1.

---

### From MCMC to DPAC

At the end of sampling, `trace.posterior["p_critere"]` contains 4 chains × 4 000 = 16 000 values of P95, each consistent with the data and the priors. The distribution of these values *is* the uncertainty on P95.

```
DPAC = percentile(p_critere_posterior, 70)
```

The choice of the 70th percentile (rather than the median or mean) follows the Expostats convention: it corresponds to a *cautious but not ultra-conservative* decision — there is a 70% probability that the true P95 is below the DPAC.

---

## Data format

The Excel file must contain at least two columns (case-insensitive names):

| Column | Content | Examples |
|---|---|---|
| `Valeur` | Measured concentration | `0.45`, `<0.01`, `[0.001-0.02]` |
| `Capteur` | Sensor / group identifier | `Respiratory`, `Breast L` |

### Censored value encoding

| Notation | Meaning | LoD / LoQ used |
|---|---|---|
| `0.45` | Detected value | — |
| `<0.01` | Left-censored (below LoD) | LoD = 0.01 |
| `[0.001-0.02]` | Interval-censored (between LoD and LoQ) | LoD = 0.001, LoQ = 0.02 |

LoD and LoQ are extracted automatically from the value strings. Explicit columns `LoD effectif` / `LoQ effectif` override the parsed values if present (useful when thresholds vary by row or by instrument).

---

## Using DPAC.ipynb

### Quick start

1. Open `DPAC.ipynb` in JupyterLab or VS Code
2. Run the **Imports** cell
3. Run the **Configuration** cell — fill in the form (file, output folder, acceptable risk, criterion percentile) and click **✓ Apply**
4. Run the **Utility functions** cell (once per session)
5. Go to the desired section

### Available sections

#### Section 1 — Interactive analysis (single file)

Complete analysis in five sequential steps:

| Step | Description |
|---|---|
| Step 1 | Load and inspect data (groups, censoring summary) |
| Step 2 | Build the hierarchical PyMC model |
| Step 3 | MCMC sampling (≈ 1–5 min depending on dataset size) |
| Step 4 | Compute DPAC, IC90, IC50 and plot posterior distributions |
| Step 5 | Export to `DPAC_logfile.xlsx` (`DPAC Log` and `*_risk` sheets) |

At the end of Step 5, an **interactive menu** offers to immediately start a new analysis, plot risk curves, or open the comparative chart.

#### Section 3 — Risk curves

Plots the **overexposure fraction (%) vs OEL** for each analysis stored in the logfile. Each curve is shown with its credible intervals:

- Light grey band: IC90 (90% credible interval)
- Dark grey band: IC50 (50% credible interval)
- Green dot: DPAC value at the target risk level

#### Section 4 — Comparative DPAC chart

Horizontal comparison of IC90 and IC50 across all logfile analyses. Useful for comparing multiple workstations, conditions, or studies in a single report.

---

## Interpreting results

### DPAC value

| Value | Interpretation |
|---|---|
| DPAC < OEL | Exposure likely under control |
| DPAC ≈ OEL | Uncertain zone — decision requires contextualisation |
| DPAC > OEL | Overexposure risk — corrective measures should be considered |

### Credible intervals IC90 / IC50

Credible intervals reflect **statistical uncertainty** around the DPAC. A narrow IC90 indicates that the data constrain the model well; a wide IC90 signals high uncertainty (typically due to a large proportion of censored values or a small sample size).

### MCMC diagnostics

| Indicator | Acceptable value | Meaning |
|---|---|---|
| r̂ (r_hat) | < 1.01 | Chain convergence |
| ESS (bulk/tail) | > 400 | Effective number of independent samples |

r̂ > 1.01 on `sigma_B` or `sigma_W` may indicate a Neal's funnel geometry. The model uses non-centred parameterisation to mitigate this.

---

## Output files

| File | Content |
|---|---|
| `DPAC_logfile.xlsx` | `DPAC Log` sheet: summary table (GM, GSD, DPAC, IC90, IC50) — one row per analysis |
| `DPAC_logfile.xlsx` | `*_risk` sheets: risk curves (100 OEL × overexposure fraction points) |

---

## Reference

Lavoué J, Vadali M, Ramachandran G (2019). *Expostats: A Bayesian Toolkit to Assist the Interpretation of Occupational Exposure Measurements.* Annals of Work Exposures and Health, 63(3), 267–279. https://doi.org/10.1093/annweh/wxy110
