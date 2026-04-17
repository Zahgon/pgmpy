from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted, validate_data

from pgmpy.prediction._base import _BaseCausalPrediction


class DoubleMLRegressor(_BaseCausalPrediction):
    """
    Implements the Double Machine Learning Regressor[1] (DML2) with cross-fitting.

    This estimator implements the DoubleML algorithm with cross-fitting in a
    scikit-learn compatible estimator API. It uses user-specified causal graphs
    to extract exposure, outcome, and adjustment variables and uses that to
    fit/predict a Double ML regressor. The model is defined as follows:

    Given data D: (Y, T, X), where:
        Y : outcome variable
        T : treatment (exposure) variable
        X : adjustment (confounder + pretreatment) variables

    The DoubleML fitting procedure consists of three main steps:

    1. Sample splitting into `n_folds` folds for cross-fitting.
    2. Fitting two nuisance estimators on each fold:
        - Outcome Models (`outcome_est_`): Predict Y using X.
        - Treatment Models (`treatment_est_`): Predict T from X.

    2. Computing residuals using nuisance estimators on each fold.
        - Outcome residuals: `Y - outcome_est_.predict(X)`
        - Treatment residuals: `T - treatment_est_.predict(X)`

    3. Stack the residuals from the folds together and fit the effect estimator
    (`effect_est_`) to predict the outcome residuals from the treatment
    residuals.

    Using the fitted models, predictions on new data `(X_new, T_new)` are computed as:

        `res_T_new = T_new - treatment_est_.predict(X_new)`
        `Y_pred =  effect_est_(res_T_new) + outcome_est_.predict(X_new)`

    Parameters
    ----------
    causal_graph : DAG, PDAG, ADMG, MAG, or PAG
        Causal graph with defined variable roles. The causal graph must have
        the following roles: `exposures`, `outcomes`, and `adjustment`.
        Additionally, `pretreatment` can be specified.

    nuisance_estimators: an estimator or a tuple of estimators of size 2 (default=LinearRegression)
        If a single estimator is provided, it is used for both outcome and
        treatment nuisance models.

        If a tuple of two estimators is provided, the first one is used for the
        treatment model and the second for the outcome model.

        If None, defaults to LinearRegression for both models.

    effect_estimator : estimator-like (default=LinearRegression)
        Estimator for the final effect estimation step. Must have a `fit` method
        and a `predict` method. If None, defaults to LinearRegression.

    n_folds : int, default=5
        Number of folds to use for cross-fitting. If 1, doesn't perform
        cross-fitting and computes in-sample residuals.

    seed : int or None
        Random seed for cross-fitting splits.

    Attributes
    ----------
    n_folds_ : int
        Number of folds used in cross-fitting.

    n_features_in_ : int
        Number of features seen during fit.

    n_samples_ : int
        Number of samples seen during fit.

    exposure_var_ : str
        Name of the exposure (treatment) variable.

    outcome_var_ : str
        Name of the outcome variable.

    adjustment_vars_ : list of str
        Names of adjustment (confounder) variables.

    pretreatment_vars_ : list of str
        Names of pretreatment variables.

    feature_columns_fit_ : list of str
        Names of features used in the model.

    outcome_est_ : estimator-like or list of estimator-like
        Fitted outcome nuisance model(s).

    treatment_est_ : estimator-like or list of estimator-like
        Fitted treatment nuisance model(s).

    effect_est_ : estimator-like
        Fitted final effect estimator.

    Examples
    --------
    >>> # Example 1: With adjustments and cross-fitting
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from pgmpy.base.DAG import DAG
    >>> from pgmpy.prediction import DoubleMLRegressor

    >>> # Simulate data from a linear Gaussian BN that we use to estimate the causal effect from.
    >>> lgbn = DAG.from_dagitty(
    ...     "dag { X -> T [beta=0.2] X -> Y [beta=0.3] T -> Y [beta=0.4] }"
    ... )
    >>> data = lgbn.simulate(n_samples=1000, seed=42)
    >>> X = data.loc[:, ["X", "T"]]
    >>> y = data["Y"]

    >>> # construct a DAG (roles must match DataFrame column names)
    >>> dag = DAG(
    ...     lgbn.edges(), roles={"exposures": "T", "adjustment": "X", "outcomes": "Y"}
    ... )
    >>> dml = DoubleMLRegressor(
    ...     causal_graph=dag,
    ...     nuisance_estimators=LinearRegression(),
    ...     effect_estimator=LinearRegression(),
    ...     n_folds=3,
    ... )
    >>> dml = dml.fit(X, y)
    >>> dml.effect_est_
    LinearRegression()
    >>> dml.effect_est_.coef_.round(1)
    array([0.4])

    >>> preds = dml.predict(X.iloc[:5])
    >>> preds.shape
    (5,)

    >>> dml.n_folds_
    3
    >>> dml.n_samples_
    1000

    Notes
    -----
    While the implementations allows the effect estimator to be any sklearn
    compatible estimator, the theoretical guarantees for DoubleML hold when the
    effect estimator is a linear model (such as LinearRegression). Using
    non-linear effect estimators may lead to biased estimates.

    References
    ----------
    .. [1] Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
           C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for
           treatment and structural parameters. The Econometrics Journal, 21(1),
           C1-C68.

    """

    def __init__(
        self,
        causal_graph,
        nuisance_estimators=None,
        effect_estimator=None,
        n_folds: int = 5,
        seed: int | None = None,
    ):

        self.causal_graph = causal_graph
        self.nuisance_estimators = nuisance_estimators
        self.effect_estimator = effect_estimator
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y, sample_weight: Any | None = None):
        """
        Fit the DoubleML model using the provided data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature data containing exposure and adjustment variables.
            If a numpy array is provided, it is converted to a dataframe with column names starting from 0.

        y : pandas.Series, pandas.DataFrame, or numpy.ndarray
            Outcome variable. If a DataFrame is provided, it must have a single column.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights to be used in fitting the nuisance and effect estimators.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass

    def predict(self, X):
        """
        Makes conditional interventional (CATE) predictions using the fitted DoubleML model.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature data containing data for exposure and adjustment variables for which to make predictions.

        Returns
        -------
        outcome_pred : numpy.ndarray
            Predicted outcome values.

        """
        pass
