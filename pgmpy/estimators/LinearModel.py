import statsmodels.api as sm
from statsmodels.api import GLS, OLS, WLS


class LinearEstimator:
    """
    A simple linear model built on statmodels.
    """

    def __init__(self, graph, estimator_type="linear", **kwargs):
        self._supported_models = {"linear": OLS, "OLS": OLS, "GLS": GLS, "WLS": WLS}
        if estimator_type not in self._supported_models.keys():
            raise NotImplementedError(
                "We currently only support OLS, GLS, and WLS. Please specify which you would like to use."
            )
        else:
            self.estimator = self._supported_models[estimator_type]

    def _model(self, X, Y, Z, data, **kwargs):
        pass

    def fit(self, X, Y, Z, data, **kwargs):
        pass

    def _get_ate(self):
        pass

    def summary(self):
        pass
