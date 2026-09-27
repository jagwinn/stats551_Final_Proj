# Bayesian Ordinal Regression for Smartphone Price Classification

This project uses Bayesian ordered logistic regression to study how smartphone hardware specifications relate to four ordered price tiers. Rather than treating the tiers as unrelated classes, the model uses their natural ordering and produces full posterior distributions that quantify uncertainty in each feature's effect.

The analysis was completed as a University of Michigan STATS 551 final project and implemented in R and Stan.

## Project Goals

The project addresses three questions:

1. Which hardware features are most strongly associated with a smartphone's price tier?
2. Does a Bayesian ordered logistic model adequately reproduce the structure observed in the data?
3. How much uncertainty is associated with the model's coefficients and predicted price categories?

## Dataset

The analysis uses the [Mobile Price Classification dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) from Kaggle. The training data contain 2,000 smartphones, 20 hardware predictors, and an ordinal response representing four price tiers:

1. Low cost
2. Medium cost
3. High cost
4. Very high cost

Each tier contains 500 observations, so class imbalance is not a concern in the training data. Predictors include:

- Battery capacity, processor speed, and number of cores
- RAM and internal memory
- Front and primary camera resolution
- Pixel height, pixel width, and screen dimensions
- Phone weight, depth, and talk time
- Bluetooth, dual-SIM, 3G, 4G, touchscreen, and Wi-Fi indicators

Before fitting the model, binary features were converted to logical variables, the target was recoded from `0-3` to ordered categories `1-4`, and nonbinary predictors were standardized.

## Why Ordered Logistic Regression?

The response is categorical, but its categories have a meaningful order. A standard multiclass classifier would ignore that structure. Ordered logistic regression instead assigns each phone a latent price score,

$$
\eta_i = \mathbf{x}_i^\top \boldsymbol{\beta},
$$

and uses three ordered cutpoints to divide that score into four price categories. Under the proportional-odds assumption, each predictor shifts the probability distribution toward either lower or higher price tiers in a consistent direction.

## Bayesian Model

The model was written directly in Stan and estimated with `rstan`. Weakly informative priors were assigned to the coefficients and ordered cutpoints:

$$
\beta_j \sim \mathcal{N}(0,1), \qquad c_k \sim \mathcal{N}(0,5), \qquad c_1 < c_2 < c_3.
$$

Sampling used four Markov chains with 5,000 iterations per chain, including 2,500 warmup iterations, for 10,000 retained posterior draws in total. The Stan program also generated replicated price categories for posterior predictive checking.

## Analysis Workflow

1. Explore feature distributions, correlations, and price-tier balance.
2. Examine the relationship between RAM and price category.
3. Standardize numeric predictors and construct the Stan design matrix.
4. Fit the Bayesian ordered logistic model with Hamiltonian Monte Carlo.
5. Evaluate convergence using trace plots, effective sample sizes, and $\hat{R}$.
6. Interpret posterior means and 95% credible intervals.
7. Assess model fit with marginal and conditional posterior predictive checks.

## Key Findings

- **RAM was the dominant predictor of price tier.** Higher RAM strongly shifted probability toward more expensive categories.
- **Battery capacity, pixel height, pixel width, and internal memory also had positive effects.** Their 95% credible intervals excluded zero after accounting for the other predictors.
- **Phone weight had a negative effect.** Heavier phones were less likely to fall into a higher price tier, holding other specifications constant.
- **Many remaining features had uncertain effects.** Their 95% credible intervals overlapped zero after the stronger hardware predictors were included.
- **The sampler converged successfully.** Trace plots mixed well, $\hat{R}$ values were approximately 1, and effective sample sizes were sufficiently large.
- **Posterior predictive checks reproduced the main observed structure.** Simulated outcomes closely matched the four category frequencies, overall mean category, and monotonic relationship between RAM and price tier.

These results show the value of an ordinal Bayesian model: it preserves the ordering of the outcome while providing interpretable feature effects and uncertainty estimates.

## Limitations

- The Kaggle data are synthetic, so the estimated relationships may not generalize to real smartphone markets.
- RAM nearly separates the price categories, producing a very large coefficient and making the problem easier than a realistic pricing task may be.
- The available Kaggle test set does not include price labels, so the project could not report genuine out-of-sample predictive accuracy.
- The proportional-odds assumption requires each predictor to have the same directional effect across all cumulative category boundaries.
- Posterior predictive checks focused primarily on marginal category balance and the relationship between RAM and price.

Future work could evaluate the model on labeled real-world data, use cross-validation, add more conditional posterior predictive checks, compare non-proportional-odds models, and apply shrinkage or variable-selection priors.

## Repository Contents

| File | Description |
| --- | --- |
| `Stats551_FinalProject.pdf` | Final written report, including the model, results, diagnostics, and discussion |
| `551_code_FINAL.pdf` | Rendered R and Stan analysis with figures and model output |
| `phone_fin_proj.stan` | Stan specification for the Bayesian ordered logistic model |
| `train.csv` | Kaggle training data used in the analysis |

## Reproducing the Analysis

The original analysis expects `train.csv` and `phone_fin_proj.stan` to be in the working directory. A working C++ toolchain is also required because `rstan` compiles the Stan model locally.

Install the primary R packages with:

```r
install.packages(c(
  "tidyverse",
  "GGally",
  "rstan",
  "bayesplot",
  "gridExtra"
))
```

The model can then be fit from R with:

```r
fit <- rstan::stan(
  file = "phone_fin_proj.stan",
  data = stan_data,
  chains = 4,
  iter = 5000,
  seed = 42
)
```

The full data-cleaning, visualization, model-fitting, diagnostic, and posterior-predictive workflow is shown in `551_code_FINAL.pdf`.

## Technical Skills Demonstrated

- Bayesian ordinal regression
- Stan model specification and MCMC sampling
- Posterior inference and credible-interval interpretation
- Convergence diagnostics and posterior predictive checking
- Data preprocessing and exploratory visualization in R
- Reproducible statistical reporting with R Markdown

## Author

Jake Gwinn
