import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class _BaseCausalPrediction(RegressorMixin, BaseEstimator):
    """
    Base class for causal prediction algorithms in pgmpy. Provides common
    functionality for preparing and validating feature dataframes.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def _prepare_feature_df(self, X, required_features) -> pd.DataFrame:
        """
        Convert input (either numpy array or dataframe) to a DataFrame and
        validate that column names exactly match DAG variables.

        If a numpy array is provided, it is converted to a DataFrame with
        range index column names (0, 1, ..., n_features-1).

        Parameters
        ----------
        X : array-like or DataFrame
            Input features.

        required_features : list[int]
            Column indices expected from the DAG.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only required columns.
        """
        pass
