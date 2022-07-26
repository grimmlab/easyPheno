import pyro
import pyro.distributions as dist
import torch

from . import _bayesian_alphabet


class BayesA(_bayesian_alphabet.Bayes):
    """
    Implementation of a class for Bayes A.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easyPheno.model._bayesian_alphabet.Bayes` for more information on the attributes.
    """

    """
    def probability_model(self, X, y):
        # Define our intercept prior
        intercept_prior = dist.Normal(0.0, 1.0)
        linear_combination = pyro.sample(f"beta_intercept", intercept_prior)

        # Also define coefficient priors
        for i in range(X.shape[1]):
            coefficient_prior = dist.Normal(0.0, 1.0)
            beta_coef = pyro.sample(f"beta_{i}", coefficient_prior)
            linear_combination = linear_combination + (X[:, i] * beta_coef)

        # Define a sigma prior for the random error
        sigma = pyro.sample("sigma", dist.HalfNormal(scale=10.0))

        # For a simple linear model, the expected mean is the linear combination of parameters
        mean = linear_combination

        with pyro.plate("data", y.shape[0]):
            # Assume our expected mean comes from a normal distribution with the mean which
            # depends on the linear combination, and a standard deviation "sigma"
            outcome_dist = dist.Normal(mean, sigma)

            # Condition the expected mean on the observed target y
            observation = pyro.sample("obs", outcome_dist, obs=y)
        """
    def __init__(self, task: str, encoding: str = None, iterations: int = 100, warmup: int = 10,
                 deg_freedom: int = 5, r2: float = 0.5, shape0: float = 1.1):
        super().__init__(task=task, encoding=encoding, iterations=iterations, warmup=warmup)
        self.deg_freedom = deg_freedom
        self.r2 = r2
        self.shape0 = shape0
        self.S = None
        self.betas = None
        self.var_betas = None
        self.e = None
        self.sigma2 = None
        self.S0 = None

    def probability_model(self, X, y):
        print('new sample')
        # init at first iteration
        self.e = y - torch.mean(y) if self.e is None else self.e
        self.sigma2 = torch.var(self.e) * (1-self.r2) if self.sigma2 is None else self.sigma2
        self.betas = torch.zeros(X.shape[1]) if self.betas is None else self.betas
        x2 = torch.sum(torch.square(X), dim=0)  # compute x'_j*x_j for each j
        if self.S0 is None:
            sumMean2 = torch.sum(torch.square(torch.mean(X, dim=0)))  # sum over squared means for all j  -> scalar
            MSx = torch.sum(x2) / X.shape[1] - sumMean2
            self.S0 = torch.var(y) * self.r2 / MSx * (self.deg_freedom+2)
        self.var_betas = self.S0/(self.deg_freedom + 2) * torch.ones(X.shape[1]) \
            if self.var_betas is None else self.var_betas
        self.S = self.S0 if self.S is None else self.S

        beta0_prior = dist.Normal(torch.mean(self.e), torch.sqrt(torch.clamp(self.sigma2 / X.shape[0], min=0.001)))
        beta0 = pyro.sample(f'beta0', beta0_prior)
        self.e += beta0
        rate0 = (self.shape0-1) / self.S0
        for i in range(X.shape[1]):
            tau = self.sigma2 / self.var_betas[i].clone()
            old_b = self.betas[i].clone()
            beta_mean = (torch.matmul(X[:, i], self.e) + x2[i] * old_b) / (x2[i] + tau)
            beta_sd = self.sigma2 / (x2[i] + tau)
            betas_prior = dist.Normal(beta_mean, torch.clamp(beta_sd, min=0.001))
            self.betas[i] = pyro.sample(f'beta_{i}', betas_prior)
            self.e += torch.reshape((old_b - self.betas[i]) * X[:, i], (-1, 1))
            self.var_betas[i] = (self.S.clone() + self.betas[i].clone() * self.betas[i].clone()) / \
                                pyro.sample(f'var_beta_{i}', dist.Chi2(df=self.deg_freedom + 1))

        # update shape parameter
        self.S = pyro.sample('shape', dist.Gamma(X.shape[1] * self.deg_freedom / 2 + self.shape0,
                                                 torch.sum(1 / self.var_betas) / 2 + rate0))

        # update sigma prior
        sigma_prior = dist.Chi2(self.deg_freedom + X.shape[0])
        self.sigma2 = (torch.dot(self.e.flatten(), self.e.flatten()) + self.S0) / pyro.sample('sigma', sigma_prior)

        # expected mean
        with pyro.plate('data', y.shape[0]):
            outcome_dist = dist.Normal(self.e, torch.sqrt(torch.clamp(self.sigma2, min=0.001)))
            observation = pyro.sample('obs', outcome_dist, obs=y)
