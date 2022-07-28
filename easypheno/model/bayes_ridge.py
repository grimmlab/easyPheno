import pyro
import pyro.distributions as dist
import torch

from . import _bayesian_linreg


class BayesRidge(_bayesian_linreg.Bayes):
    """
    Implementation of a class for Bayesian Ridge Regression. R implementaiton of Bayes A, B and C is availabile via Docker workflow and weigh faster.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easypheno.model._bayesian_linreg.Bayes` for more information on the attributes.
    """

    def probability_model(self, X: torch.tensor, y: torch.Tensor):
        """
        Implementation of probability model for Bayesian ridge regression

        See :obj:`~easypheno.model._bayesian_linreg.Bayes` for more information on the attributes.
        """
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
            outcome_dist = dist.Normal(mean, torch.clamp(sigma, min=0.0001, max=10**6))

            # Condition the expected mean on the observed target y
            observation = pyro.sample("obs", outcome_dist, obs=y)
