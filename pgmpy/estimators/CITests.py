import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy import logger
from pgmpy.independencies import IndependenceAssertion
from pgmpy.utils import get_dataset_type


class CITestRegistry:
    """
    Registry to manage Conditional Independence (CI) Test Strategies.

    Allows looking up tests by name or inferring suitable tests based on data type.
    """

    def __init__(self):
        self._registry: dict[str, Callable] = {}
        self._tags: dict[str, list[str]] = {}
        self._defaults: dict[str, str] = {
            "continuous": "pearsonr",
            "discrete": "chi_square",
            "mixed": "pillai",
        }

    def register(self, name: str, data_types: list[str]):
        """
        Decorator to register a CI test strategy.

        Parameters
        ----------
        name : str
            The name of the test (case-insensitive).

        data_types : list of str
            List of data types this test supports (e.g., ['continuous', 'discrete']).
        """

        def decorator(func: Callable):
            pass

        return decorator

    def list_all(self, data_type=None) -> list[str]:
        """
        Lists all registered CI test strategies.

        Parameters
        ----------
        data_type : str, optional
            If provided, filters tests that support the given data type.

        Returns
        -------
        list of str
            Names of all registered CI tests.
        """
        pass

    def get_test(self, test: str | None | Callable, data: pd.DataFrame | None = None) -> Callable:
        """
        Retrieves a CI test strategy.

        Parameters
        ----------
        test : str, callable or None
            The name of the test, a callable function, or None.

        data : pandas.DataFrame, optional
            The dataframe used to infer the test type if `test` is None.

        Returns
        -------
        callable
            The CI test function.

        Raises
        ------
        ValueError
            If `test` is None and `data` is None, or if the test name is not found.
        """
        pass


ci_registry = CITestRegistry()


@ci_registry.register(
    name="independence_match",
    data_types=["discrete", "continuous", "mixed"],
)
def independence_match(X, Y, Z, independencies, **kwargs):
    """
    Check if `X \u27c2 Y | Z` is in `independences`.

    This method is implemented to have a uniform API when the independences
    are provided explicitly instead of being inferred from data.

    Parameters
    ----------
    X : str
        The first variable for testing the independence condition X \u27c2 Y | Z.

    Y : str
        The second variable for testing the independence condition X \u27c2 Y | Z.
    Z : list or array-like
        A list of conditional variables for testing the condition X \u27c2 Y | Z.
    independencies : pgmpy.independencies.Independencies
        The object containing the known independences.

    Returns
    -------
    bool
        True if the independence assertion is present in `independences`, else False.
    """
    pass


@ci_registry.register(name="pearsonr", data_types=["continuous"])
def pearsonr(X, Y, Z, data, boolean=True, **kwargs):
    """
    Compute Pearson correlation coefficient and p-value for testing non-correlation.

    Should be used only on continuous data. In case when :math:`Z \\neq \\emptyset` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    X : str
        The first variable for testing the independence condition X \u27c2 Y | Z.

    Y : str
        The second variable for testing the independence condition X \u27c2 Y | Z.

    Z : list or array-like
        A list of conditional variables for testing the condition X \u27c2 Y | Z.

    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean : bool, default=True
        If True, returns a boolean indicating independence (based on `significance_level`).
        If False, returns the test statistic and p-value.

    **kwargs
        Additional arguments. Must contain `significance_level` if `boolean=True`.

    Returns
    -------
    result : bool or tuple
        If boolean=True, returns True if p-value >= significance_level, else False.
        If boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """
    pass


@ci_registry.register(name="power_divergence", data_types=["discrete"])
def power_divergence(X, Y, Z, data, boolean=True, lambda_="cressie-read", **kwargs):
    """
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis
    for the test is X is independent of Y given Z. A lot of the frequency comparision
    based statistics (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:

            * "pearson"             1          "Chi-squared test"
            * "log-likelihood"      0          "G-test or log-likelihood"
            * "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            * "mod-log-likelihood"  -1         "Modified Log-likelihood"
            * "neyman"              -2         "Neyman's statistic"
            * "cressie-read"        2/3        "The value recommended in the paper[1]"

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    **kwargs
        Must contain `significance_level` if `boolean=True`.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

    References
    ----------
    .. [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests."
      Journal of the Royal Statistical Society: Series B (Methodological) 46.3 (1984): 440-464.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> chi_square(X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_

    """
    pass


@ci_registry.register(name="chi_square", data_types=["discrete"])
def chi_square(X, Y, Z, data, boolean=True, **kwargs):
    """
    Perform Chi-square conditional independence test.

    Tests the null hypothesis that X is independent from Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
        be specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.
        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> chi_square(X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> chi_square(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    pass


@ci_registry.register(name="g_sq", data_types=["discrete"])
def g_sq(X, Y, Z, data, boolean=True, **kwargs):
    """
    G squared test for conditional independence. Also commonly known as G-test,
    likelihood-ratio or maximum likelihood statistical significance test.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False. If
        boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> g_sq(X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> g_sq(X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05)
    np.True_
    >>> g_sq(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    pass


@ci_registry.register(name="log_likelihood", data_types=["discrete"])
def log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Log likelihood ratio test for conditional independence. Also commonly known
    as G-test, G-squared test or maximum likelihood statistical significance
    test.  Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.  If
        boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27c2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> log_likelihood(
    ...     X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> log_likelihood(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> log_likelihood(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    pass


@ci_registry.register(name="modified_log_likelihood", data_types=["discrete"])
def modified_log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Modified log likelihood ratio test for conditional independence.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must be
        specified. If p_value of the test is greater than equal to
        `significance_level`, returns True. Otherwise returns False.
        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X \u27c2 Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> modified_log_likelihood(
    ...     X="A", Y="C", Z=[], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> modified_log_likelihood(
    ...     X="A", Y="B", Z=["D"], data=data, boolean=True, significance_level=0.05
    ... )
    np.True_
    >>> modified_log_likelihood(
    ...     X="A", Y="B", Z=["D", "E"], data=data, boolean=True, significance_level=0.05
    ... )
    np.False_
    """
    pass


def _get_predictions(X, Y, Z, data, **kwargs):
    """
    Helper Strategy: Function to get predictions using XGBoost for `ci_pillai`.
    Not registered directly as a CI test.
    """
    pass


@ci_registry.register(name="pillai", data_types=["discrete", "continuous", "mixed"])
def pillai_trace(X, Y, Z, data, boolean=True, **kwargs):
    """
    A mixed-data residualization based conditional independence test[1].

    Uses XGBoost estimator to compute LS residuals[2], and then does an
    association test (Pillai's Trace) on the residuals.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27c2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27c2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27c2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

    References
    ----------
    .. [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional" "conditional
           independences for categorical and ordinal data." Proceedings of the
           AAAI Conference on Artificial Intelligence.
    .. [2] Li, C.; and Shepherd, B. E. 2010. Test of Association Between Two Ordinal Variables While Adjusting for
           Covariates. Journal of the American Statistical Association.
    .. [3] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in testing the multivariate
           general linear hypothesis. Computational Statistics & Data Analysis.
    """
    pass


@ci_registry.register(name="gcm", data_types=["continuous"])
def gcm(X, Y, Z, data, boolean=True, **kwargs):
    """
    The Generalized Covariance Measure(GCM) test for CI.

    It performs linear regressions on the conditioning variable and then tests
    for a vanishing covariance between the resulting residuals. Details of the
    method can be found in [1].

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X \u27c2 Y | Z

    Y: str
        The second variable for testing the independence condition X \u27c2 Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X \u27c2 Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the pearson correlation coefficient and p_value
            of the test.

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

    References
    ----------
    .. [1] Rajen D. Shah, and Jonas Peters. "The Hardness of Conditional Independence Testing and the Generalised
        Covariance Measure".
    """
    pass


@ci_registry.register(name="pearsonr_equivalence", data_types=["continuous"])
def pearsonr_equivalence(X, Y, Z, data, boolean=True, delta_threshold=0.1, **kwargs) -> tuple | bool:
    """
    Computes a two-sided level-alpha equivalent test using partial correlations.

    Tests the Null Hypothesis that the partial correlation is greater than or
    equal to `delta_threshold` (Dependence). Rejection implies Practical Independence.

    Parameters
    ----------
    X: str
        The first variable for testing the independence condition X _|_ Y | Z

    Y: str
        The second variable for testing the independence condition X _|_ Y | Z

    Z: list/array-like
        A list of conditional variable for testing the condition X _|_ Y | Z

    data: pandas.DataFrame
        The dataset in which to test the independence condition.

    boolean: bool
        If True, returns True (Independent) if p_value < significance_level.

    delta_threshold: float
        The equivalence bound (threshold for practical independence).

    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True (Independent) if p-value < significance_level.
        If boolean=False, returns (Partial Correlation, p-value).

    References
    ----------
    .. [1] Malinsky, Daniel. "A cautious approach to constraint-based causal model selection." arXiv preprint
            arXiv:2404.18232 (2024).
    """
    pass
