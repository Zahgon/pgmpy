import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from pgmpy.structure_score._base import BaseStructureScore


class LogLikelihoodCondGauss(BaseStructureScore):
    r"""
    Log-likelihood score for Bayesian networks with mixed discrete and continuous variables.

    This score is based on conditional Gaussian distributions [1]_ and supports local families with both discrete and
    continuous variables.

    For a continuous target :math:`C_1` with continuous parents :math:`C_2` and discrete parents :math:`D`, it computes

    .. math::
        \ell(C_1 \mid C_2, D) = \sum_{t=1}^{n} \log \frac{p(c_{1t}, c_{2t} \mid d_t)}{p(c_{2t} \mid d_t)}.

    For a discrete target :math:`D_1` with continuous parents :math:`C` and discrete parents :math:`D_2`, it computes

    .. math::
        \ell(D_1 \mid C, D_2) = \sum_{t=1}^{n} \log \frac{p(c_t \mid d_{1t}, d_{2t}) p(d_{1t}, d_{2t})} {p(c_t \mid
        d_{2t}) p(d_{2t})}.

    The Gaussian densities are estimated from the corresponding grouped samples.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns may be discrete or continuous variables.
    state_names : dict, optional
        Dictionary mapping discrete variable names to their possible states.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.structure_score import LogLikelihoodCondGauss
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.integers(0, 2, size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = LogLikelihoodCondGauss(data)
    >>> round(score.local_score("A", ("B", "C")), 3)
    np.float64(-137.319)

    Raises
    ------
    ValueError
        If the data or variable types are not suitable for conditional Gaussian modeling.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian Networks of Mixed Variables. International
        Journal of Data Science and Analytics, 6(1), 3-18. https://doi.org/10.1007/s41060-017-0085-7
    """

    _tags = {
        "name": "ll-cg",
        "supported_datatype": "mixed",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    @staticmethod
    def _adjusted_cov(df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _cat_parents_product(self, parents: tuple[str, ...]) -> int:
        pass

    def _get_num_parameters(self, variable: str, parents: tuple[str, ...]) -> int:
        pass

    def _log_likelihood(self, variable: str, parents: tuple[str, ...]) -> float:
        pass

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        pass
