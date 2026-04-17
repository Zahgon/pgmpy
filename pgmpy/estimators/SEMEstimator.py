import numpy as np
import pandas as pd
import statsmodels.api as sm

from pgmpy import config
from pgmpy.inference import CausalInference
from pgmpy.models import SEM, SEMAlg, SEMGraph
from pgmpy.utils import compat_fns, optimize, pinverse


class SEMEstimator:
    """
    Base class of SEM estimators. All the estimators inherit this class.
    """

    def __init__(self, model):
        if config.get_backend() == "numpy":
            msg = (
                f"{type(self)} requires pytorch backend, currently it is "
                "set to numpy."
                "Call pgmpy.config.set_backend('torch') to switch the backend globally."
            )
            raise ValueError(msg)

        if isinstance(model, (SEMGraph, SEM)):
            self.model = model.to_lisrel()
        elif isinstance(model, SEMAlg):
            self.model = model
        else:
            raise ValueError(f"Model should be an instance of either SEMGraph or SEMAlg class. Got type: {type(model)}")

        import torch

        # Initialize trainable and fixed mask tensors
        self.B_mask = torch.tensor(
            self.model.B_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )
        self.zeta_mask = torch.tensor(
            self.model.zeta_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )

        self.B_fixed_mask = torch.tensor(
            self.model.B_fixed_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )
        self.zeta_fixed_mask = torch.tensor(
            self.model.zeta_fixed_mask,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )

        self.wedge_y = torch.tensor(
            self.model.wedge_y,
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )
        self.B_eye = torch.eye(
            self.B_mask.shape[0],
            device=config.DEVICE,
            dtype=config.DTYPE,
            requires_grad=False,
        )

    def _get_implied_cov(self, B, zeta):
        """
        Computes the implied covariance matrix from the given parameters.
        """
        pass

    def ml_loss(self, params, loss_args):
        r"""
        Method to compute the Maximum Likelihood loss function. The optimizer calls this
        method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
        .. math:: F_{ML} = \log |\Sigma(\theta)| + tr(S \Sigma^{-1}(\theta)) - \log S - (p+q)

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        pass

    def uls_loss(self, params, loss_args):
        r"""
        Method to compute the Unweighted Least Squares fitting function. The optimizer calls
        this method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
        .. math:: F_{ULS} = tr[(S - \Sigma(\theta))^2]

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        pass

    def gls_loss(self, params, loss_args):
        r"""
        Method to compute the Weighted Least Squares fitting function. The optimizer calls
        this method after each iteration with updated params to compute the new loss.

        The fitting function for ML is:
        .. math:: F_{ULS} = tr \{ [(S - \Sigma(\theta)) W^{-1}]^2 \}

        Parameters
        ----------
        params: dict
            params contain all the variables which are updated in each iteration of the
            optimization.

        loss_args: dict
            loss_args contain all the variable which are not updated in each iteration but
            are required to compute the loss.

        Returns
        -------
        torch.tensor: The loss value for the given params and loss_args
        """
        pass

    def get_init_values(self, data, method):
        """
        Computes the starting values for the optimizer.

        Reference
        ---------
        .. [1] Table 4C.1: Bollen, K. (2014). Structural Equations with Latent Variables.
                New York, NY: John Wiley & Sons.

        """
        pass

    def fit(
        self,
        data,
        method,
        opt="adam",
        init_values="random",
        exit_delta=1e-4,
        max_iter=1000,
        **kwargs,
    ):
        """
        Estimate the parameters of the model from the data.

        Parameters
        ----------
        data: pandas DataFrame or pgmpy.data.Data instance
            The data from which to estimate the parameters of the model.

        method: str ("ml"|"uls"|"gls"|"2sls")
            The fitting function to use.
            ML : Maximum Likelihood
            ULS: Unweighted Least Squares
            GLS: Generalized Least Squares
            2sls: 2-SLS estimator

        init_values: str or dict
            Options for str: random | std | iv
            dict: dictionary with keys `B` and `zeta`.

        **kwargs: dict
            Extra parameters required in case of some estimators.
            GLS:
                W: np.array (n x n) where n is the number of observe variables.
            2sls:
                x:
                y:

        Returns
        -------
            pgmpy.model.SEM instance: Instance of the model with estimated parameters

        References
        ----------
        .. [1] Bollen, K. A. (2010). Structural equations with latent variables. New York: Wiley.
        """
        pass


class IVEstimator:
    """
    Initialize IVEstimator object.

    Parameters
    ----------
    model: pgmpy.models.SEM
        The model for which estimation need to be done.

    Examples
    --------
    >>> from pgmpy.models import SEM
    >>> from pgmpy.estimators import IVEstimator
    >>> model = SEM.from_graph(
    ...     ebunch=[
    ...         ("Z1", "X", 1.0),
    ...         ("Z2", "X", 1.0),
    ...         ("Z2", "W", 1.0),
    ...         ("W", "U", 1.0),
    ...         ("U", "X", 1.0),
    ...         ("U", "Y", 1.0),
    ...         ("X", "Y", 1.0),
    ...     ],
    ...     latents=["U"],
    ...     err_var={"Z1": 1, "Z2": 1, "W": 1, "X": 1, "U": 1, "Y": 1},
    ... )
    >>> estimator = IVEstimator(model)
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, Y, data, ivs=None, civs=None):
        """
        Estimates the parameter X -> Y.

        Parameters
        ----------
        X: str
            The covariate variable of the parameter being estimated.

        Y: str
            The predictor variable of the parameter being estimated.

        data: pd.DataFrame
            The data from which to learn the parameter.

        ivs: List (default: None)
            List of variable names which should be used as Instrumental Variables (IV).
            If not specified, tries to find the IVs from the model structure, fails if
            can't find either IV or Conditional IV.

        civs: List of tuples (tuple form: (var, coditional_var))
            List of conditional IVs to use for estimation.
            If not specified, tries to find the IVs from the model structure, fails if
            can't find either IV or Conditional IVs.

        Returns
        -------
        tuple: (float, statsmodels.regression.linear_model.RegressionResultsWrapper)
            A tuple where the first element is the estimated causal parameter
            for X -> Y, and the second element is the fitted OLS results object
            from the second stage regression (a RegressionResultsWrapper). Call
            `.summary()` on this object to get the textual summary.

        Examples
        --------
        >>> from pgmpy.models import SEM
        >>> from pgmpy.estimators import IVEstimator
        >>> model = SEM.from_graph(
        ...     ebunch=[
        ...         ("Z1", "X", 1.0),
        ...         ("Z2", "X", 1.0),
        ...         ("Z2", "W", 1.0),
        ...         ("W", "U", 1.0),
        ...         ("U", "X", 1.0),
        ...         ("U", "Y", 1.0),
        ...         ("X", "Y", 1.0),
        ...     ],
        ...     latents=["U"],
        ...     err_var={"Z1": 1, "Z2": 1, "W": 1, "X": 1, "U": 1, "Y": 1},
        ... )
        >>> data = model.to_lisrel().generate_samples(500)
        >>> estimator = IVEstimator(model)
        >>> param, results = estimator.fit(X="X", Y="Y", data=data)
        """
        pass
