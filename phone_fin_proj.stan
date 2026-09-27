
// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> N;                   // number of observations
  int<lower=1> D;                   // number of predictors
  int<lower=2> K;                   // number of ordered categories
  matrix[N, D] X;                   // design matrix
  int<lower=1, upper=K> y[N];       // outcome: 1..K
}

parameters {
  vector[D] beta;                   // coefficients
  ordered[K-1] cutpoints;           // thresholds
}

transformed parameters {
  vector[N] lp = X * beta;          // linear predictor
}

model {
  // priors
  beta      ~ normal(0, 1);     
  cutpoints ~ normal(0, 5);

  // likelihood
  y ~ ordered_logistic(lp, cutpoints);
}

generated quantities {
  vector[N] log_lik;
  int<lower=1,upper=K> y_pred[N];

  for (n in 1:N) {
    log_lik[n] = ordered_logistic_lpmf(y[n] | lp[n], cutpoints);
    y_pred[n]  = ordered_logistic_rng(lp[n], cutpoints);
  }
}
