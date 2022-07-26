import numpy as np
import pandas as pd
import pyro
import torch

from . import _param_free_base_model


class Bayes(_param_free_base_model.ParamFreeBaseModel):
    """
    Implementation of a class for Bayesian linear regression.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information on the attributes.

        *Additional attributes*

        - mu (*np.array*): intercept
        - beta (*np.array*): effect size
        - iterations (*int*): MCMC sampler iterations
        - warmup (*int*): number of discarded MCMC warmup iterations
    """
    standard_encoding = '012'
    possible_encodings = ['101']

    def __init__(self, task: str, encoding: str = None, iterations: int = 100, warmup: int = 10):
        super().__init__(task=task, encoding=encoding)
        self.mu = None
        self.beta = None
        self.iterations = iterations
        self.warmup = warmup

    def probability_model(self, X: torch.Tensor, y: torch.Tensor):
        """
        Probability model that needs to be implemented by each child model.

        :param X: feature matrix
        :param y: target vector
        """
        raise NotImplementedError

    def fit(self, X: np.array, y: np.array) -> np.array:
        """
        Implementation of fit function for Bayesian linear regression.

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        torch.autograd.set_detect_anomaly(True)
        X = torch.tensor(X)
        y = torch.tensor(y)

        # Clear the parameter storage
        pyro.clear_param_store()

        # Initialize our No U-Turn Sampler
        my_kernel = pyro.infer.NUTS(self.probability_model, max_tree_depth=7)  # a shallower tree helps the algorithm run faster

        # Employ the sampler in an MCMC sampling
        # algorithm, and sample 3100 samples.
        # Then discard the first 100
        mcmc = pyro.infer.MCMC(my_kernel, num_samples=self.iterations, warmup_steps=self.warmup)
        # Run the sampler
        mcmc.run(X, y)

        # save results as numpy arrays
        coefficients = pd.DataFrame(mcmc.get_samples()).iloc[:, :-1].mean()
        self.beta = coefficients[1:]
        self.mu = coefficients.iloc[0]

        return self.predict(X_in=X)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of predict function for Bayesian linear regression.

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        return self.mu + np.matmul(X_in, self.beta)
