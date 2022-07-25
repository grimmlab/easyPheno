import numpy as np
import pyro
import torch

from . import _bayesian_alphabet


class BayesA(_bayesian_alphabet.Bayes):
    """
    Implementation of a class for Bayes A.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easyPheno.model._bayesian_alphabet.Bayes` for more information on the attributes.
    """
    def probability_model(self, X, y):
        # Define our intercept prior
        intercept_prior = pyro.dist.Normal(0.0, 1.0)
        linear_combination = pyro.sample(f"beta_intercept", intercept_prior)

        # Also define coefficient priors
        for i in range(X.shape[1]):
            coefficient_prior = pyro.dist.Normal(0.0, 1.0)
            beta_coef = pyro.sample(f"beta_{i}", coefficient_prior)
            linear_combination = linear_combination + (X[:, i] * beta_coef)

        # Define a sigma prior for the random error
        sigma = pyro.sample("sigma", pyro.dist.HalfNormal(scale=10.0))

        # For a simple linear model, the expected mean is the linear combination of parameters
        mean = linear_combination

        with pyro.plate("data", y.shape[0]):
            # Assume our expected mean comes from a normal distribution with the mean which
            # depends on the linear combination, and a standard deviatin "sigma"
            outcome_dist = pyro.dist.Normal(mean, sigma)

            # Condition the expected mean on the observed target y
            observation = pyro.sample("obs", outcome_dist, obs=y)

    def fit(self, X: np.array, y: np.array, iterations: int = 6000, warmup: int = 1000) -> np.array:
        """
        Implementation of fit function for Bayesian alphabet.

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        X = torch.tensor(X)
        y = torch.tensor(y)

        # Clear the parameter storage
        pyro.clear_param_store()

        # Initialize our No U-Turn Sampler
        my_kernel = pyro.infer.NUTS(self.model_normal, max_tree_depth=7)  # a shallower tree helps the algorithm run faster

        # Employ the sampler in an MCMC sampling
        # algorithm, and sample 3100 samples.
        # Then discard the first 100
        my_mcmc1 = pyro.MCMC(my_kernel, num_samples=iterations, warmup_steps=warmup)
        # Run the sampler
        my_mcmc1.run(X, y)


        # save results as numpy arrays
        coefficients = np.mean(my_mcmc1.get_samples()[:, :-1])
        self.beta = coefficients[1:]
        self.mu = coefficients[0]

        return self.predict(X_in=X)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of predict function for Bayesisan alphabet model.

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        return self.mu + np.matmul(X_in, self.beta)
