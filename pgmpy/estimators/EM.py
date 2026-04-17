from itertools import chain, product
from math import log
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config, logger
from pgmpy.base import DAG
from pgmpy.estimators import (
    BayesianEstimator,
    MaximumLikelihoodEstimator,
    ParameterEstimator,
)
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


class ExpectationMaximization(ParameterEstimator):
    """
    Class used to compute parameters for a model using Expectation
    Maximization (EM).

    EM is an iterative algorithm commonly used for
    estimation in the case when there are latent variables in the model.
    The algorithm iteratively improves the parameter estimates, maximizing
    the likelihood of the given data.

    Parameters
    ----------
    model: A pgmpy.models.DiscreteBayesianNetwork instance

    data: pandas DataFrame object
        DataFrame object with column names identical to the variable names
        of the network.  (If some values in the data are missing, the data
        cells should be set to `numpy.nan`.  Note that pandas converts each
        column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states
        that the variable can take. If unspecified, the observed values in
        the data set are taken to be the only possible states.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import ExpectationMaximization
    >>> data = pd.DataFrame(
    ...     np.random.randint(low=0, high=2, size=(1000, 5)),
    ...     columns=["A", "B", "C", "D", "E"],
    ... )
    >>> model = DiscreteBayesianNetwork(
    ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
    ... )
    >>> estimator = ExpectationMaximization(model, data)
    """

    def __init__(
        self,
        model: DAG | DiscreteBayesianNetwork,
        data: pd.DataFrame,
        **kwargs,
    ):
        if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
            raise NotImplementedError("Expectation Maximization is only implemented for DAG or DiscreteBayesianNetwork")

        if isinstance(model, DAG):
            model_bn = DiscreteBayesianNetwork(model.edges())
            model_bn.add_nodes_from(model.nodes())
            model_bn.latents = model.latents
            model = model_bn

        # Drop fully missing columns and treat them as latent if not already
        original_cols = set(data.columns)
        data = data.dropna(axis=1, how="all")
        dropped_cols = original_cols - set(data.columns)
        new_latents = [col for col in dropped_cols if col not in model.latents]

        if new_latents:
            logger.warning(
                f"Columns {new_latents} have all missing values and are not marked as latent. "
                "Treating them as latent variables."
            )
            model.latents.update(new_latents)

        # Drop rows with any missing values in partially observed columns
        original_rows_count = data.shape[0]
        data = data.dropna()
        dropped_rows_count = original_rows_count - data.shape[0]

        if dropped_rows_count:
            logger.warning(
                f"{dropped_rows_count} rows with missing values in partially "
                "missing columns were dropped from the dataset."
            )

        super().__init__(model, data, **kwargs)
        self.model_copy = self.model.copy()

    def _get_log_likelihood(self, datapoint: dict[str, Any]) -> float:
        """
        Computes the likelihood of a given datapoint. Goes through each
        CPD matching the combination of states to get the value and multiplying
        them together.
        """
        pass

    def _parallel_compute_weights(
        self,
        data_unique: pd.DataFrame,
        latent_card: dict[str, int],
        n_counts: dict[tuple, int],
        offset: int,
        batch_size: int,
    ) -> pd.DataFrame:
        pass

    def _compute_weights(
        self,
        n_jobs: int,
        latent_card: dict[str, int],
        batch_size: int,
    ) -> pd.DataFrame:
        """
        For each data point, create extra data points for each possible combination
        of states of latent variables and assigns weights to each of them.
        """
        pass

    def _is_converged(
        self,
        new_cpds: list[TabularCPD],
        atol: float = 1e-08,
    ) -> bool:
        """
        Checks if the values of `new_cpds` are within the tolerance limits of the current
        model cpds.
        """
        pass

    def get_parameters(
        self,
        latent_card: dict[str, int] | None = None,
        apply_smoothing: bool = False,
        max_iter: int = 100,
        atol: float = 1e-08,
        n_jobs: int = 1,
        batch_size: int = 1000,
        seed: int | None = None,
        init_cpds: dict[str, TabularCPD] | str = {},
        show_progress: bool = True,
        **kwargs,
    ) -> list[TabularCPD]:
        """
        Method to estimate all model parameters (CPDs) using Expectation Maximization.

        Parameters
        ----------
        latent_card: dict (default: None)
            A dictionary of the form {latent_var: cardinality} specifying the
            cardinality (number of states) of each latent variable. If None,
            assumes `2` states for each latent variable.

        apply_smoothing: bool (default: False)
            If True, `prior_type` and any additional arguments related to it
            needs to be specified. Please refer
            `BayesianEstimator.get_parameters` method for more details on which
            arguments need to be specified.

        max_iter: int (default: 100)
            The maximum number of iterations the algorithm is allowed to run for.
            If max_iter is reached, return the last value of parameters.

        atol: int (default: 1e-08)
            The absolute accepted tolerance for checking convergence. If the parameters
            change is less than atol in an iteration, the algorithm will exit.

        n_jobs: int (default: 1)
            Number of jobs to run in parallel.
            Using n_jobs > 1 for small models or datasets might be slower.

        batch_size: int (default: 1000)
            Number of data used to compute weights in a batch.

        seed: int
            The random seed to use for generating the initial values.

        init_cpds: dict or str
            dict: A dictionary of the form {variable: instance of TabularCPD}
            specifying the initial CPD values for the EM optimizer to start
            with. If not specified, CPDs involving latent variables are
            initialized randomly, and CPDs involving only observed variables are
            initialized with their MLE estimates.

            str: `uniform`, all CPDs will be initialized to have a uniform distribution.
                 `random`, all CPDs will be initialized randomly.

        show_progress: boolean (default: True)
            Whether to show a progress bar for iterations.

        Returns
        -------
        Estimated parameters (CPDs): list
            A list of estimated CPDs for the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import ExpectationMaximization as EM
        >>> rng = np.random.default_rng(42)
        >>> data = pd.DataFrame(
        ...     rng.integers(low=0, high=2, size=(1000, 3)),
        ...     columns=["A", "C", "D"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D")], latents={"B"}
        ... )
        >>> estimator = EM(model, data)
        >>> params = estimator.get_parameters(latent_card={"B": 3})
        >>> # Sorting the CPDs by variable name to ensure consistent order for doctest comparison
        >>> sorted(
        ...     params, key=lambda cpd: cpd.variable
        ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(A:2) at 0x...>,
        <TabularCPD representing P(B:3 | A:2, C:2) at 0x...>,
        <TabularCPD representing P(C:2) at 0x...>,
        <TabularCPD representing P(D:2 | C:2) at 0x...>]
        """
        pass
