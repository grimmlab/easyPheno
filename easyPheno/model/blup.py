import numpy as np
from scipy.optimize import minimize
from . import _param_free_base_model


class Blup(_param_free_base_model.ParamFreeBaseModel):
    """
    Implementation of a class for BLUP.

    *Attributes*

        *Inherited attributes*

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information on the attributes.

        *Additional attributes*

        - beta (*np.array*): TODO
        - u (*np.array*): TODO
    """
    standard_encoding = '101'
    possible_encodings = ['101']

    @staticmethod
    def reml(delta: float, n: int, eigenvalues: np.array, omega2: np.array):
        return (n-1)*np.log(sum(np.divide(omega2, (eigenvalues+delta)))) + sum(np.log(eigenvalues+delta))

    def fit(self, X: np.array, y: np.array) -> np.array:
        """
        Implementation of fit function for BLUP.

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        n = y.shape[0]
        Z = np.ones((n, 1))
        S = np.eye(n) - 1/n*np.ones((n,n))
        sqn = np.sqrt(n)

        # compute spectral decomposition
        H = np.matmul(X, X.T) + sqn*np.eye(n)
        eigenvalues_H, V_H = np.linalg.eig(H)
        eigenvalues_H = eigenvalues_H - sqn
        SHS = np.matmul(S, np.matmul(H, S))
        eigenvalues_SHS, V_SHS = np.linalg.eig(SHS)
        V_SHS = np.delete(V_SHS, -1, axis=1)
        eigenvalues_SHS = np.delete(eigenvalues_SHS, -1) - sqn
        omega = np.matmul(V_SHS.T, y)
        omega2 = np.multiply(omega, omega)

        # minimize REML
        x0 = np.abs(np.min(eigenvalues_SHS)) + 1e-09
        delta_opt = minimize(self.reml, x0, args=(n, eigenvalues_SHS, omega2)).x

        # calculate inverse of H
        H_inv = np.matmul(V_H, np.divide(V_H, (eigenvalues_H + delta_opt)).T)

        # calculate beta and u
        self.beta = np.linalg.solve(np.matmul(Z.T, np.matmul(H_inv, Z)), np.matmul(Z.T, np.matmul(H_inv, y)))
        self.u = np.matmul(X.T, np.matmul(H_inv, (y - np.matmul(Z, self.beta))))
        return self.predict(X_in=X)

    def predict(self, X_in: np.array) -> np.array:
        """
        Implementation of predict function for BLUP.

        See :obj:`~easyPheno.model._param_free_base_model.ParamFreeBaseModel` for more information.
        """
        return self.beta + np.matmul(X_in, self.u)
