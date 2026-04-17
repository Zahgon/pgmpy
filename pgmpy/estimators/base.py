#!/usr/bin/env python

from collections import defaultdict

import numpy as np

from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference.ExactInference import BeliefPropagation
from pgmpy.utils import build_state_names, get_state_counts, preprocess_data


class BaseEstimator:
    """
    Base class for estimators in pgmpy; `ParameterEstimator`,
    `StructureEstimator` and `StructureScore` derive from this class.

    Parameters
    ----------
    data: pandas DataFrame object
        object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, data=None, state_names=None):
        if data is None:
            self.data = None
            self.dtypes = None
        else:
            self.data, self.dtypes = preprocess_data(data)

        # data can be None in the case when learning structure from
        # independence conditions. Look into PC.py.
        if self.data is not None:
            self.variables = list(self.data.columns.values)
            self.state_names = build_state_names(self.data, state_names=state_names)

    def state_counts(
        self,
        variable,
        parents=[],
        weighted=False,
        reindex=True,
    ):
        """
        Return counts how often each state of 'variable' occurred in the data.
        If a list of parents is provided, counting is done conditionally
        for each state configuration of the parents.

        Parameters
        ----------
        variable: string
            Name of the variable for which the state count is to be done.

        parents: list
            Optional list of variable parents, if conditional counting is desired.
            Order of parents in list is reflected in the returned DataFrame

        weighted: bool
            If True, data must have a `_weight` column specifying the weight of the
            datapoint (row). If False, each datapoint has a weight of `1`.

        reindex: bool
            If True, returns a data frame with all possible parents state combinations
            as the columns. If False, drops the state combinations which are not
            present in the data.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.estimators import BaseEstimator
        >>> data = pd.DataFrame(
        ...     data={
        ...         "A": ["a1", "a1", "a2"],
        ...         "B": ["b1", "b2", "b1"],
        ...         "C": ["c1", "c1", "c2"],
        ...     }
        ... )
        >>> estimator = BaseEstimator(data)
        >>> estimator.state_counts(variable="A").values
        array([[2],
               [1]])
        >>> estimator.state_counts(variable="C", parents=["A", "B"]).values
        array([[1., 1., 0., 0.],
               [0., 0., 1., 0.]])
        """
        pass


class ParameterEstimator(BaseEstimator):
    """
    Base class for parameter estimators in pgmpy.

    Parameters
    ----------
    model: pgmpy.models.DiscreteBayesianNetwork or pgmpy.models.DiscreteMarkovNetwork model
        for which parameter estimation is to be done.

    data: pandas DataFrame object
        dataframe object with column names identical to the variable names of the model.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, model, data, **kwargs):
        """
        Base class for parameter estimators in pgmpy.

        Parameters
        ----------
        model: pgmpy.models.DiscreteBayesianNetwork or pgmpy.models.DiscreteMarkovNetwork model
            for which parameter estimation is to be done.

        data: pandas DataFrame object
            dataframe object with column names identical to the variable names of the model.
            (If some values in the data are missing the data cells should be set to `numpy.nan`.
            Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.nan` is used.
            This sets the behavior of the `state_count`-method.
        """
        self.model = model

        super().__init__(data, **kwargs)

    def state_counts(self, variable, weighted=False, **kwargs):
        """
        Return counts how often each state of 'variable' occurred in the data.
        If the variable has parents, counting is done conditionally
        for each state configuration of the parents.

        Parameters
        ----------
        variable: string
            Name of the variable for which the state count is to be done.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import ParameterEstimator
        >>> model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        >>> data = pd.DataFrame(
        ...     data={
        ...         "A": ["a1", "a1", "a2"],
        ...         "B": ["b1", "b2", "b1"],
        ...         "C": ["c1", "c1", "c2"],
        ...     }
        ... )
        >>> estimator = ParameterEstimator(model, data)
        >>> estimator.state_counts(variable="A").values
        array([[2],
               [1]])
        >>> estimator.state_counts(variable="C").values
        array([[1., 1., 0., 0.],
               [0., 0., 1., 0.]])
        """
        pass


class StructureEstimator(BaseEstimator):
    """
    Base class for structure estimators in pgmpy.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, data=None, independencies=None, **kwargs):
        self.independencies = independencies
        if self.independencies is not None:
            self.variables = self.independencies.get_all_variables()

        super().__init__(data=data, **kwargs)

    def estimate(self):
        pass


class MarginalEstimator(BaseEstimator):
    """
    Base class for marginal estimators in pgmpy.

    Parameters
    ----------
    model: DiscreteMarkovNetwork | FactorGraph | JunctionTree
        A model to optimize, using Belief Propagation and an estimation method.

    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.
    """

    def __init__(self, model, data, **kwargs):
        super().__init__(data, **kwargs)
        self.belief_propagation = BeliefPropagation(model=model)
        self.theta = None

    @staticmethod
    def _clique_to_marginal(marginals, clique_nodes):
        """
        Construct a minimal mapping from cliques to marginals.

        Parameters
        ----------
        marginals: FactorDict
            A mapping from cliques to factors.

        clique_nodes: List[Tuple[str, ...]]
            Cliques that exist within a different FactorDict.

        Returns
        -------
        clique_to_marginal: A mapping from clique to a list of marginals
        such that each clique is a super set of the marginals it is associated with.
        """
        pass

    def _marginal_loss(self, marginals, clique_to_marginal, metric):
        """
        Compute the loss and gradient for a given dictionary of clique beliefs.

        Parameters
        ----------
        marginals: FactorDict
            A mapping from a clique to an observed marginal represented by a `DiscreteFactor`.

        clique_to_marginal: Dict[Tuple[str, ...], List[DiscreteFactor]]
            A mapping from a Junction Tree's clique to a list of corresponding marginals
            such that a clique is a superset of the marginal with the constraint that
            each marginal only appears once across all cliques.

        metric: str
            One of either 'L1' or 'L2'.

        Returns
        -------
        Loss and gradient of the loss: Tuple[float, pgmpy.factors.FactorDict.FactorDict]
            Marginal loss and the gradients of the loss with respect to the estimated beliefs.
        """
        pass

    def estimate(self):
        pass
