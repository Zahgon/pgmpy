#!/usr/bin/env python3
from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from collections.abc import Hashable, Iterable
from functools import reduce
from operator import mul
from typing import (
    Any,
)

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import logger
from pgmpy.base import DAG
from pgmpy.factors.discrete import (
    DiscreteFactor,
    JointProbabilityDistribution,
    TabularCPD,
)
from pgmpy.models.DiscreteMarkovNetwork import DiscreteMarkovNetwork
from pgmpy.utils import compat_fns


class DiscreteBayesianNetwork(DAG):
    """
    Initializes a Discrete Bayesian Network.

    A Bayesian Network is defined using a model structure and a conditional
    probability distribution (CPDs) associated with each node (i.e., variable)
    in the network. For a discrete Bayesian Network, pgmpy offers two ways to
    define these CPDs: TabularCPD and NoisyORCPD

    Parameters
    ----------
    ebunch : input graph, optional
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    latents : set of nodes, default=None
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    exposures : set, default=None
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. Default is an empty set.

    outcomes : set, optional (default: None)
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. If None, an empty set is used.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    # Defining a Discrete Bayesian Network and adding CPDs to it.

    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.factors.discrete import TabularCPD
    >>> model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    >>> model.add_nodes_from(["A", "B", "C"])
    >>> cpd_a = TabularCPD("A", 2, [[0.6], [0.4]])
    >>> cpd_b = TabularCPD("B", 2, [[0.7], [0.3]])
    >>> cpd_c = TabularCPD(
    ...     variable="C",
    ...     variable_card=2,
    ...     values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
    ...     evidence=["A", "B"],
    ...     evidence_card=[2, 2],
    ... )
    >>> model.add_cpds(cpd_a, cpd_b, cpd_c)
    >>> model.get_cpds("C")  # doctest: +ELLIPSIS
    <TabularCPD representing P(C:2 | A:2, B:2) at 0x...>

    # Simulating data from the defined Discrete Bayesian Network.

    >>> df = model.simulate(n_samples=1000)

    # Fitting simulated data to the model.

    >>> fitted_model = model.fit(df)

    # Predicting missing values in the data.

    >>> test_data = df.copy()
    >>> test_data = test_data.drop(columns=["C"])
    >>> predicted_data = fitted_model.predict(test_data)
    >>> predicted_data.shape
    (1000, 3)
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        roles: dict[str, Iterable] | None = None,
    ) -> None:
        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_edge(self, u: Any, v: Any, w: Any | None = None, **kwargs: Any) -> None:
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
              Nodes can be any hashable python object.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> G = DiscreteBayesianNetwork()
        >>> G.add_nodes_from(["grade", "intel"])
        >>> G.add_edge("grade", "intel")
        """
        pass

    def remove_node(self, node: Any) -> None:
        """
        Remove node from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of its children.

        Parameters
        ----------
        node : node
            Node which is to be removed from the model.

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")]
        ... )
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 4)),
        ...     columns=["A", "B", "C", "D"],
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x...>
        >>> model.get_cpds()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(A:2) at 0x...>,
         <TabularCPD representing P(B:2 | A:2) at 0x...>,
         <TabularCPD representing P(C:2 | B:2, D:2) at 0x...>,
         <TabularCPD representing P(D:2 | A:2) at 0x...>]
        >>> model.remove_node("A")
        >>> model.get_cpds()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(B:2) at 0x...>,
         <TabularCPD representing P(C:2 | B:2, D:2) at 0x...>,
         <TabularCPD representing P(D:2) at 0x...>]
        """
        pass

    def remove_nodes_from(self, nodes: Iterable[Any]) -> None:
        """
        Remove multiple nodes from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of its children.

        Parameters
        ----------
        nodes : list, set (iterable)
            Nodes which are to be removed from the model.

        Returns
        -------
        None

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("B", "C"), ("A", "D"), ("D", "C")]
        ... )
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 4)),
        ...     columns=["A", "B", "C", "D"],
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x...>
        >>> model.get_cpds()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(A:2) at 0x...>,
         <TabularCPD representing P(B:2 | A:2) at 0x...>,
         <TabularCPD representing P(C:2 | B:2, D:2) at 0x...>,
         <TabularCPD representing P(D:2 | A:2) at 0x...>]
        >>> model.remove_nodes_from(["A", "B"])
        >>> model.get_cpds()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(C:2 | D:2) at 0x...>,
         <TabularCPD representing P(D:2) at 0x...>]
        """
        pass

    def add_cpds(self, *cpds: TabularCPD) -> None:
        """
        Add CPD (Conditional Probability Distribution) to the Bayesian Model.

        Parameters
        ----------
        cpds  :  list, set, tuple (array-like)
            List of CPDs which will be associated with the model

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete.CPD import TabularCPD
        >>> student = DiscreteBayesianNetwork(
        ...     [("diff", "grades"), ("aptitude", "grades")]
        ... )
        >>> grades_cpd = TabularCPD(
        ...     "grades",
        ...     3,
        ...     [
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ...     ],
        ...     evidence=["diff", "aptitude"],
        ...     evidence_card=[2, 3],
        ...     state_names={
        ...         "grades": ["gradeA", "gradeB", "gradeC"],
        ...         "diff": ["easy", "hard"],
        ...         "aptitude": ["low", "medium", "high"],
        ...     },
        ... )
        >>> student.add_cpds(grades_cpd)

        +---------+-------------------------+------------------------+
        |diff:    |          easy           |         hard           |
        +---------+------+--------+---------+------+--------+--------+
        |aptitude:| low  | medium |  high   | low  | medium |  high  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeA   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeB   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeC   | 0.8  | 0.8    |   0.8   |  0.8 |  0.8   |   0.8  |
        +---------+------+--------+---------+------+--------+--------+
        """
        pass

    def get_cpds(self, node: Any | None = None) -> TabularCPD | list[TabularCPD]:
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        cpd : TabularCPD object or list of TabularCPD objects
            If 'node' is specified, returns the 'TabularCPD' object corresponding to the node.
            If 'node' is not specified, returns a list of all 'TabularCPD' objects added to the model.

        Raises
        ------
        ValueError
            If the specified node is not present in the model.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> cpds = model.get_cpds()
        >>> cpds  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(asia:2) at 0x...>,
        <TabularCPD representing P(bronc:2 | smoke:2) at 0x...>,
        <TabularCPD representing P(dysp:2 | bronc:2, either:2) at 0x...>,
        <TabularCPD representing P(either:2 | lung:2, tub:2) at 0x...>,
        <TabularCPD representing P(lung:2 | smoke:2) at 0x...>,
        <TabularCPD representing P(smoke:2) at 0x...>,
        <TabularCPD representing P(tub:2 | asia:2) at 0x...>,
        <TabularCPD representing P(xray:2 | either:2) at 0x...>]
        >>> cpd = model.get_cpds("bronc")
        >>> cpd  # doctest: +ELLIPSIS
        <TabularCPD representing P(bronc:2 | smoke:2) at 0x...>
        """
        pass

    def remove_cpds(self, *cpds: TabularCPD | str) -> None:
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: TabularCPD object
            A CPD object on any subset of the variables of the model which
            is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        >>> cpd = TabularCPD(
        ...     "grade",
        ...     2,
        ...     [[0.1, 0.9, 0.2, 0.7], [0.9, 0.1, 0.8, 0.3]],
        ...     ["intel", "diff"],
        ...     [2, 2],
        ... )
        >>> student.add_cpds(cpd)
        >>> student.remove_cpds(cpd)
        """
        pass

    def get_cardinality(self, node: Any | None = None) -> int | dict[Any, int]:
        """
        Returns the cardinality of the node. Throws an error if the CPD for the
        queried node hasn't been added to the network.

        Parameters
        ----------
        node: Any hashable python object(optional).
              The node whose cardinality we want. If node is not specified returns a
              dictionary with the given variable as keys and their respective cardinality
              as values.

        Returns
        -------
        variable cardinalities: dict or int
            If node is specified returns the cardinality of the node else returns a dictionary
            with the cardinality of each variable in the network

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        >>> cpd_diff = TabularCPD("diff", 2, [[0.6], [0.4]])
        >>> cpd_intel = TabularCPD("intel", 2, [[0.7], [0.3]])
        >>> cpd_grade = TabularCPD(
        ...     "grade",
        ...     2,
        ...     [[0.1, 0.9, 0.2, 0.7], [0.9, 0.1, 0.8, 0.3]],
        ...     ["intel", "diff"],
        ...     [2, 2],
        ... )
        >>> student.add_cpds(cpd_diff, cpd_intel, cpd_grade)
        >>> {k: int(v) for k, v in student.get_cardinality().items()}
        {'diff': 2, 'intel': 2, 'grade': 2}

        >>> int(student.get_cardinality("intel"))
        2
        """
        pass

    @property
    def states(self) -> dict[Any, list[str]]:
        """
        Returns a dictionary mapping each node to its list of possible states.

        Returns
        -------
        state_dict: dict
            Dictionary of nodes to possible states
        """
        pass

    def check_model(self) -> bool:
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass otherwise should throw an error.
        """
        pass

    def to_markov_model(self) -> DiscreteMarkovNetwork:
        """
        Converts Bayesian Network to Markov Model. The Markov Model created would
        be the moral graph of the Bayesian Network.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> G = DiscreteBayesianNetwork(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> mm = G.to_markov_model()
        >>> mm.nodes()
        NodeView(('diff', 'grade', 'intel', 'letter', 'SAT'))
        >>> mm.edges()
        EdgeView([('diff', 'grade'), ('diff', 'intel'), ('grade', 'letter'), ('grade', 'intel'), ('intel', 'SAT')])
        """
        pass

    def to_junction_tree(self) -> Any:
        """
        Creates a junction tree (or clique tree) for a given Bayesian Network.

        For converting a Bayesian Model into a Clique tree, first it is converted
        into a Markov one.

        For a given markov model (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = DiscreteBayesianNetwork(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ...     ],
        ...     evidence=["diff", "intel"],
        ...     evidence_card=[2, 3],
        ... )
        >>> sat_cpd = TabularCPD(
        ...     "SAT",
        ...     2,
        ...     [[0.1, 0.2, 0.7], [0.9, 0.8, 0.3]],
        ...     evidence=["intel"],
        ...     evidence_card=[3],
        ... )
        >>> letter_cpd = TabularCPD(
        ...     "letter",
        ...     2,
        ...     [[0.1, 0.4, 0.8], [0.9, 0.6, 0.2]],
        ...     evidence=["grade"],
        ...     evidence_card=[3],
        ... )
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> jt = G.to_junction_tree()
        """
        pass

    def fit(self, data, estimator=None, state_names=[], n_jobs=1, **kwargs) -> DAG:
        """
        Estimates the CPD for each variable based on a given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.nan`.
            Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

        estimator: Estimator class
            One of:
            - MaximumLikelihoodEstimator (default)
            - BayesianEstimator: In this case, pass 'prior_type' and either 'pseudo_counts'
            or 'equivalent_sample_size' as additional keyword arguments.
            See `BayesianEstimator.get_parameters()` for usage.
            - ExpectationMaximization

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        n_jobs: int (default: 1)
            Number of threads/processes to use for estimation. Using n_jobs > 1
            for small models or datasets might be slower.

        Returns
        -------
        Fitted Model: DiscreteBayesianNetwork
            Returns a DiscreteBayesianNetwork object with learned CPDs.
            The DAG structure is preserved, and parameters (CPDs) are added.
            This allows the DAG to represent both the structure and the parameters of a Bayesian Network.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.base import DAG
        >>> data = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]})
        >>> model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
        >>> fitted_model = model.fit(data)
        >>> len(fitted_model.get_cpds())
        3
        >>> fitted_model.get_cpds()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(A:2) at 0x...>,
        <TabularCPD representing P(C:2 | A:2, B:2) at 0x...>,
        <TabularCPD representing P(B:2) at 0x...>]
        """
        pass

    def fit_update(self, data: pd.DataFrame, n_prev_samples: int | None = None, n_jobs: int = 1) -> None:
        """
        Method to update the parameters of the DiscreteBayesianNetwork with more data.
        Internally, uses BayesianEstimator with dirichlet prior, and uses
        the current CPDs (along with `n_prev_samples`) to compute the pseudo_counts.

        Parameters
        ----------
        data: pandas.DataFrame
            The new dataset which to use for updating the model.

        n_prev_samples: int
            The number of samples/datapoints on which the model was trained before.
            This parameter determines how much weight should the new data be given.
            If None, n_prev_samples = nrow(data).

        n_jobs: int (default: 1)
            Number of threads/processes to use for estimation. Using n_jobs > 1
            for small models or datasets might be slower.

        Returns
        -------
        Updated model: None
            Modifies the network inplace.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> model = load_model("bnlearn/alarm")
        >>> # Generate some new data.
        >>> data = BayesianModelSampling(model).forward_sample(int(1e3))
        >>> model.fit_update(data)
        """
        pass

    def predict(
        self,
        data: pd.DataFrame,
        algo: type | None = None,
        stochastic: bool = False,
        n_jobs: int = -1,
        seed: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Predicts states of all the missing variables.

        Parameters
        ----------
        data: pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.

        algo: a subclass of pgmpy.inference.Inference or pgmpy.inference.ApproxInference
            An algorithm class from pgmpy Inference algorithms. Default is Variable Elimination.

        stochastic: boolean
            If True, does prediction by sampling from the distribution of predicted variable(s).
            If False, returns the states with the highest probability value (i.e. MAP) for the
                predicted variable(s).

        n_jobs: int (default: -1)
            The number of CPU cores to use. If -1, uses all available cores.

        seed: int (default: None)
            When `stochastic=True`, the seed value to use for random number generators.

        **kwargs
            Optional keyword arguments specific to the selected algorithm.
            - Variable Elimination:
            - elimination_order: str or list (default='greedy')
                Order in which to eliminate the variables in the algorithm. If list is provided,
                should contain all variables in the model except the ones in `variables`. str options
                are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
                refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
                for details.

            - joint: boolean (should only be used with stochastic=True i.e. when not calculating MAP)
                If True, returns a Joint Distribution over `variables`.
                If False, returns a dict of distributions over each of the `variables`.

            - Belief Propagation:
                - joint: boolean (should only be used with stochastic=True i.e. when not calculating MAP)
                If True, returns a Joint Distribution over `variables`.
                If False, returns a dict of distributions over each of the `variables`.

            - Approx Inference:
                - n_samples: int
                    The number of samples to generate for computing the distributions. Higher `n_samples`
                    results in more accurate results at the cost of more computation time.

                - samples: pd.DataFrame (default: None)
                    If provided, uses these samples to compute the distribution instead
                    of generating samples. `samples` **must** conform with the
                    `evidence` and `virtual_evidence`.

                - state_names: dict (default: None)
                    A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
                    If None, inferred from the data but is possible that the final distribution misses some states.

                - seed: int (default: None)
                    Sets the seed for the random generators.

                - joint: boolean (should only be used with stochastic=True i.e. when not calculating MAP)
                    If True, returns a Joint Distribution over `variables`.
                    If False, returns a dict of distributions over each of the `variables`.

        Returns
        -------
        Inference results: Pandas DataFrame
            If `stochastic` is True, returns state(s) by sampling from the distribution of predicted variables.
            If `stochastic` is False, returns state(s) with the highest probability value.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import ApproxInference
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> train_data = values[:800]
        >>> predict_data = values[800:]
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(train_data)  # doctest: +ELLIPSIS
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x...>
        >>> predict_data = predict_data.copy()
        >>> predict_data.drop("E", axis=1, inplace=True)
        >>> approx_inf_parameters = {"n_samples": int(1e3), "seed": 42}
        >>> y_pred = model.predict(
        ...     predict_data, algo=ApproxInference, **approx_inf_parameters
        ... )
        >>> y_pred["E"].shape
        (200,)
        """
        pass

    def predict_probability(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts probabilities of all states of the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(100, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> train_data = values[:80]
        >>> predict_data = values[80:]
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x...>
        >>> predict_data = predict_data.copy()
        >>> predict_data.drop("B", axis=1, inplace=True)
        >>> y_prob = model.predict_probability(predict_data)
        >>> y_prob.shape
        (20, 2)
        """
        pass

    def get_state_probability(self, states: dict[Hashable, Hashable]) -> float:
        """
        Given a fully specified Bayesian Network, returns the probability of the given set
        of states.

        Parameters
        ----------
        state: dict
            dict of the form {variable: state}

        Returns
        -------
        float: The probability value

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> float(
        ...     model.get_state_probability(
        ...         {"either": "no", "tub": "no", "xray": "yes", "bronc": "no"}
        ...     )
        ... )
        0.02605122
        """
        pass

    def get_factorized_product(self, latex: bool = False) -> None:
        # TODO: refer to IMap class for explanation why this is not implemented.
        pass

    def is_imap(self, JPD: JointProbabilityDistribution) -> bool:
        """
        Checks whether the Bayesian Network is Imap of given JointProbabilityDistribution

        Parameters
        ----------
        JPD: An instance of JointProbabilityDistribution Class, for which you want to check the Imap

        Returns
        -------
        is IMAP: True or False
            True if Bayesian Network is Imap for given Joint Probability Distribution False otherwise

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> G = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        >>> diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ...     ],
        ...     evidence=["diff", "intel"],
        ...     evidence_card=[2, 3],
        ... )
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        >>> val = [
        ...     0.01,
        ...     0.01,
        ...     0.08,
        ...     0.006,
        ...     0.006,
        ...     0.048,
        ...     0.004,
        ...     0.004,
        ...     0.032,
        ...     0.04,
        ...     0.04,
        ...     0.32,
        ...     0.024,
        ...     0.024,
        ...     0.192,
        ...     0.016,
        ...     0.016,
        ...     0.128,
        ... ]
        >>> JPD = JointProbabilityDistribution(
        ...     ["diff", "intel", "grade"], [2, 3, 3], val
        ... )
        >>> G.is_imap(JPD)
        True
        """
        pass

    def copy(self) -> DiscreteBayesianNetwork:
        """
        Returns a copy of the model.

        Returns
        -------
        Model's copy: pgmpy.models.DiscreteBayesianNetwork
            Copy of the model on which the method was called.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> model = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
        >>> cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        >>> cpd_b = TabularCPD(
        ...     "B", 2, [[0.3, 0.7], [0.7, 0.3]], evidence=["A"], evidence_card=[2]
        ... )
        >>> cpd_c = TabularCPD(
        ...     "C", 2, [[0.1, 0.9], [0.9, 0.1]], evidence=["B"], evidence_card=[2]
        ... )
        >>> model.add_cpds(cpd_a, cpd_b, cpd_c)
        >>> copy_model = model.copy()
        >>> copy_model.nodes()
        NodeView(('A', 'B', 'C'))
        >>> copy_model.edges()
        OutEdgeView([('A', 'B'), ('B', 'C')])
        >>> len(copy_model.get_cpds())
        3
        """
        pass

    def get_markov_blanket(self, node: Hashable) -> list[Hashable]:
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        Markov Blanket: list
            List of nodes contained in Markov Blanket of `node`

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = DiscreteBayesianNetwork(
        ...     [
        ...         ("x", "y"),
        ...         ("z", "y"),
        ...         ("y", "w"),
        ...         ("y", "v"),
        ...         ("u", "w"),
        ...         ("s", "v"),
        ...         ("w", "t"),
        ...         ("w", "m"),
        ...         ("v", "n"),
        ...         ("v", "q"),
        ...     ]
        ... )
        >>> sorted(G.get_markov_blanket("y"))
        ['s', 'u', 'v', 'w', 'x', 'z']
        """
        pass

    @staticmethod
    def get_random(
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        node_names: list[Hashable] | None = None,
        n_states: int | dict[Hashable, int] | None = None,
        latents: bool = False,
        seed: int | None = None,
    ) -> DiscreteBayesianNetwork:
        """
        Returns a randomly generated Bayesian Network on `n_nodes` variables
        with edge probabiliy of `edge_prob` between variables.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        n_states: int or dict (default: None)
            The number of states of each variable in the form
            {variable: no_of_states}. If a single value is provided,
            all nodes will have the same number of states. When None
            randomly generates the number of states.

        latents: bool (default: False)
            If True, also creates latent variables.

        seed: int (default: None)
            The seed value for random number generators.

        Returns
        -------
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> model = DiscreteBayesianNetwork.get_random(n_nodes=5)
        >>> sorted(model.nodes())
        ['X_0', 'X_1', 'X_2', 'X_3', 'X_4']
        >>> sorted([cpd.variable for cpd in model.cpds])
        ['X_0', 'X_1', 'X_2', 'X_3', 'X_4']
        >>> len(model.cpds)
        5

        """
        pass

    def get_random_cpds(
        self,
        n_states: int | dict[Hashable, int] | None = None,
        inplace: bool = False,
        seed: int | None = None,
    ) -> list[TabularCPD] | DiscreteBayesianNetwork | None:
        """
        Given a `model`, generates and adds random `TabularCPD`
          for each node resulting in a fully parameterized network.

        Parameters
        ----------
        n_states: int or dict (default: None)
            The number of states of each variable in the `model`. If None, randomly
            generates the number of states.

        inplace: bool (default: False)
            If inplace=True, adds the generated TabularCPDs to `model` itself, else creates
            a copy of the model.

        seed: int (default: None)
            The seed value for random number generators.

        """
        pass

    def do(self, nodes: Hashable | list[Hashable], inplace: bool = False) -> DiscreteBayesianNetwork | None:
        """
        Applies the do operation. The do operation removes all incoming edges
        to variables in `nodes` and marginalizes their CPDs to only contain the
        variable itself.

        Parameters
        ----------
        nodes : list, array-like
            The names of the nodes to apply the do-operator for.

        inplace: boolean (default: False)
            If inplace=True, makes the changes to the current object,
            otherwise returns a new instance.

        Returns
        -------
        Modified network: pgmpy.models.DiscreteBayesianNetwork or None
            If inplace=True, modifies the object itself else returns an instance of
            DiscreteBayesianNetwork modified by the do operation.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> asia.edges()  # doctest: +NORMALIZE_WHITESPACE
        OutEdgeView([('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'),
                     ('lung', 'either'), ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')])
        >>> do_bronc = asia.do(["bronc"])
        """
        pass

    def simulate(
        self,
        n_samples: int = 10,
        do: dict[Hashable, Hashable] | None = None,
        evidence: dict[Hashable, Hashable] | None = None,
        virtual_evidence: list[TabularCPD] | None = None,
        virtual_intervention: list[TabularCPD] | None = None,
        missing_prob: TabularCPD | list[TabularCPD] | None = None,
        include_latents: bool = False,
        partial_samples: pd.DataFrame | None = None,
        seed: int | None = None,
        show_progress: bool = True,
        return_full: bool = False,
    ) -> pd.DataFrame:
        """
        Simulates data from the given model. Internally uses methods from
        pgmpy.sampling.BayesianModelSampling to generate the data.

        Parameters
        ----------
        n_samples: int
            The number of data samples to simulate from the model.

        do: dict
            The interventions to apply to the model. dict should be of the form
            {variable_name: state}

        evidence: dict
            Observed evidence to apply to the model. dict should be of the form
            {variable_name: state}

        virtual_evidence: list
            Probabilistically apply evidence to the model. `virtual_evidence` should
            be a list of `pgmpy.factors.discrete.TabularCPD` objects specifying the
            virtual probabilities.

        virtual_intervention: list
            Also known as soft intervention. `virtual_intervention` should be a list
            of `pgmpy.factors.discrete.TabularCPD` objects specifying the virtual/soft
            intervention probabilities.

        missing_prob: TabularCPD, list of TabularCPDs (default: None)
            Used to define the missingness mechanism in the simulated data. For
            each variable with missing values, provide a TabularCPD defining
            the probability of a value being missing given the variable's value
            (Missing at Random) and optionally its parents' values (Missing Not
            at Random).

            TabularCPD format: The variable name of each TabularCPD should end
              with the name of node in DiscreteBayesianNetwork with * at the end
              of the name. The state names of each TabularCPD should be the same
              as the state names of the corresponding node in
              DiscreteBayesianNetwork.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        partial_samples: pandas.DataFrame
            A pandas dataframe specifying samples on some of the variables in the model. If
            specified, the sampling procedure uses these sample values, instead of generating them.
            partial_samples.shape[0] must be equal to `n_samples`.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: bool
            If True, shows a progress bar when generating samples.


        return_full: bool (default: False)
            If True, return both full samples and samples with missing values (if performed).

        Returns
        -------
        A dataframe with the simulated data: pd.DataFrame

        Examples
        --------
        >>> from pgmpy.example_models import load_model

        Simulation without any evidence or intervention:

        >>> model = load_model("bnlearn/alarm")
        >>> model.simulate(n_samples=10).shape
        (10, 37)


        Simulation with the hard evidence: MINVOLSET = HIGH:

        >>> model.simulate(n_samples=10, evidence={"MINVOLSET": "HIGH"}).shape
        (10, 37)


        Simulation with hard intervention: CVP = LOW:

        >>> model.simulate(n_samples=10, do={"CVP": "LOW"}).shape
        (10, 37)


        Simulation with virtual/soft evidence: p(MINVOLSET=LOW) = 0.8, p(MINVOLSET=HIGH) = 0.2,
        p(MINVOLSET=NORMAL) = 0:

        >>> virt_evidence = [
        ...     TabularCPD(
        ...         "MINVOLSET",
        ...         3,
        ...         [[0.8], [0.0], [0.2]],
        ...         state_names={"MINVOLSET": ["LOW", "NORMAL", "HIGH"]},
        ...     )
        ... ]
        >>> model.simulate(n_samples=10, virtual_evidence=virt_evidence).shape
        (10, 38)


        Simulation with virtual/soft intervention: p(CVP=LOW) = 0.2, p(CVP=NORMAL)=0.5, p(CVP=HIGH)=0.3:

        >>> virt_intervention = [
        ...     TabularCPD(
        ...         "CVP",
        ...         3,
        ...         [[0.2], [0.5], [0.3]],
        ...         state_names={"CVP": ["LOW", "NORMAL", "HIGH"]},
        ...     )
        ... ]
        >>> model.simulate(n_samples=10, virtual_intervention=virt_intervention).shape
        (10, 38)


        Simulation with missing values:
        >>> from pgmpy.factors.discrete.CPD import TabularCPD
        >>> cpd = TabularCPD("HISTORY*", 2, [[0.5], [0.5]])
        >>> model.simulate(n_samples=10, missing_prob=cpd).shape
        (10, 37)
        >>> cpd = TabularCPD(
        ...     "HISTORY*",
        ...     2,
        ...     [[0.5, 0.5], [0.5, 0.5]],
        ...     ["HISTORY"],
        ...     [2],
        ...     state_names={"HISTORY*": [0, 1], "HISTORY": ["TRUE", "FALSE"]},
        ... )
        >>> model.simulate(n_samples=10, missing_prob=cpd).shape
        (10, 37)
        >>> cpd = TabularCPD(
        ...     "HISTORY*",
        ...     2,
        ...     [[0.2, 0.1, 0.6, 0.4, 0.7, 0.2], [0.8, 0.9, 0.4, 0.6, 0.3, 0.8]],
        ...     ["HYPOVOLEMIA", "LVEDVOLUME"],
        ...     [2, 3],
        ...     state_names={
        ...         "HISTORY*": [0, 1],
        ...         "HYPOVOLEMIA": ["TRUE", "FALSE"],
        ...         "LVEDVOLUME": ["LOW", "NORMAL", "HIGH"],
        ...     },
        ... )
        >>> model.simulate(n_samples=10, missing_prob=cpd).shape
        (10, 37)
        """
        pass

    def save(self, filename: str, filetype: str = "bif") -> None:
        """
        Writes the model to a file. Please avoid using any special characters or
        spaces in variable names or state names in the model.

        Parameters
        ----------
        filename: str
            The path along with the filename where to write the file.

        filetype: str (default: bif)
            The format in which to write the model to file. Can be one of
            the following: bif, uai, xmlbif, xdsl, net.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> alarm = load_model("bnlearn/alarm")
        >>> alarm.save("alarm.bif", filetype="bif")
        """
        pass

    @staticmethod
    def load(filename: str, filetype: str = "bif", **kwargs: Any) -> DiscreteBayesianNetwork:
        """
        Read the model from a file.

        Parameters
        ----------
        filename: str
            The path along with the filename where to read the file.

        filetype: str (default: bif)
            The format of the model file. Can be one of
            the following: bif, uai, xmlbif, xdsl, net.

        kwargs: kwargs
            Any additional arguments for the reader class or get_model method.
            Please refer the file format class for details.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> alarm = load_model("bnlearn/alarm")
        >>> alarm.save("alarm.bif", filetype="bif")
        >>> alarm_model = DiscreteBayesianNetwork.load("alarm.bif", filetype="bif")
        """
        pass
