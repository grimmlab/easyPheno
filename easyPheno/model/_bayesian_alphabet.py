import numpy as np
import pyro
import torch

from . import _param_free_base_model


class Bayes(_param_free_base_model.ParamFreeBaseModel):
    """
    Implementation of a class for Bayesian alphabet.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information on the attributes.

        *Additional attributes*

        - mu (*np.array*): intercept
        - beta (*np.array*): effect size
        - iterations (*int*): MCMC sampler iterations
        - warumup (*int*): number of discarded MCMC warmup iterations
    """
    standard_encoding = '012'
    possible_encodings = ['101']

    def __init__(self, task: str, encoding: str = None, iterations: int = 6000, warmup: int = 1000):
        super().__init__(task=task, encoding=encoding)
        self.mu = None
        self.beta = None
        self.iterations = iterations
        self.warmup = warmup

    def probability_model(self, X, y):
        """probability model with priors for each model"""
        raise NotImplementedError

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
        my_kernel = pyro.infer.NUTS(self.probability_model, max_tree_depth=7)  # a shallower tree helps the algorithm run faster

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
