from __future__ import annotations

import io
import json
import math
import os
from collections.abc import Hashable, Iterable
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression

from pgmpy import logger
from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD


class LinearGaussianBayesianNetwork(DAG):
    """
    Class to represent Linear Gaussian Bayesian Networks (LGBN).

    A LGBN is a graphical model that represents a set of continuous random variables and their conditional dependencies
    via a directed acyclic graph (DAG). In a LGBN, each variable is assumed to be conditionally normally distributed,
    and the conditional probability distribution (CPD) of each variable given its parents is modeled as a linear
    function of the parents' values plus Gaussian noise. This is equivalent to assumptions of a Linear Structural
    Equation Model (SEM) with Gaussian noise.

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

    exposures : set, default=set()
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
    # Defining a Linear Gaussian Bayesian Network.

    >>> from pgmpy.models import LinearGaussianBayesianNetwork
    >>> from pgmpy.factors.continuous import LinearGaussianCPD
    >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
    >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
    >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
    >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
    >>> model.add_cpds(cpd1, cpd2, cpd3)
    >>> for cpd in model.cpds:
    ...     print(cpd)
    pass
    P(x1) = N(1; 4)
    P(x2 | x1) = N(0.5*x1 + -5.0; 4)
    P(x3 | x2) = N(-1*x2 + 4; 3)

    # Simulating data from the model.

    >>> df = model.simulate(n_samples=100, seed=42)
    >>> print(df.columns) # doctest: +ELLIPSIS
    Index(['x1', 'x2', 'x3'], dtype='...')

    # Fitting the model to the simulated data.

    >>> model.fit(df) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    <pgmpy.models.LinearGaussianBayesianNetwork.LinearGaussianBayesianNetwork object at 0x...>

    # Predicting missing variables.

    >>> df_missing = df.drop(columns=["x3"])
    >>> pred = model.predict(df_missing)
    >>> list(pred.columns)
    ['x3']
    >>> print(pred.values)
    [[ 8.01138228]
     [13.61181367]
     [ 8.70432782]
     [ 3.71719153]
     [ 8.1509597 ]
     [ 6.24976516]
     [12.2121776 ]
     [ 6.01448446]
     [ 5.49139518]
     [ 9.23748708]
     [17.92545478]
     [ 3.24653756]
     [ 8.78452503]
     [10.3678509 ]
     [ 5.33405765]
     [ 9.09319649]
     [10.66717573]
     [10.9290793 ]
     [ 6.48827753]
     [12.7339279 ]
     [ 0.79803275]
     [ 9.69425692]
     [ 5.27994359]
     [ 8.80268511]
     [ 4.31081468]
     [10.76081874]
     [10.05810137]
     [ 5.93859429]
     [ 4.10420816]
     [ 7.74976272]
     [11.67397411]
     [ 9.63141961]
     [ 1.72775337]
     [ 2.2725024 ]
     [ 8.44578257]
     [ 7.602702  ]
     [10.53853647]
     [11.31860773]
     [ 8.00975022]
     [ 9.22702521]
     [ 3.64868722]
     [13.67114269]
     [15.01854326]
     [ 6.37691191]
     [13.14971548]
     [ 2.75588544]
     [16.93490848]
     [ 2.97009486]
     [ 5.64759205]
     [ 7.74788815]
     [ 9.86681496]
     [ 3.40585598]
     [ 9.89093876]
     [ 4.08221225]
     [15.617452  ]
     [ 4.14029637]
     [ 8.59698685]
     [11.89439088]
     [ 0.44433568]
     [ 8.42879464]
     [14.45268215]
     [10.62681186]
     [10.76349781]
     [16.0269725 ]
     [ 8.83836337]
     [ 5.30435055]
     [ 7.63843465]
     [13.18359343]
     [ 0.92282836]
     [ 3.35438779]
     [11.61943098]
     [ 4.52648267]
     [11.18074558]
     [ 4.86137485]
     [ 8.49295864]
     [ 7.07209154]
     [ 6.85461911]
     [ 3.96748462]
     [ 8.3311032 ]
     [ 8.04499479]
     [ 7.27919516]
     [ 4.77660469]
     [-0.33549712]
     [ 2.65815359]
     [15.58173105]
     [12.24334129]
     [ 7.60858529]
     [ 8.0673818 ]
     [10.30962944]
     [ 9.73931168]
     [ 5.46107107]
     [16.95243925]
     [ 2.80408287]
     [12.23910532]
     [14.03289339]
     [ 6.26117488]
     [ 7.37468791]
     [13.3850798 ]
     [ 6.83845881]
     [ 5.59547155]]
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

    @classmethod
    def load(
        cls,
        filename: str | os.PathLike | io.IOBase,
    ) -> LinearGaussianBayesianNetwork:
        """
        Read the model from a JSON file or a file-like object of a JSON file.

        Parameters
        ----------
        filename: str or file-like object
            The path along with the filename where to read the file, or a
            file-like object containing the model data.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.example_models import load_model
        >>> data = load_model("bnlearn/ecoli70")
        >>> data.save("ecoli70.json")
        >>> model = LinearGaussianBayesianNetwork.load("ecoli70.json")
        >>> print(model)
        LinearGaussianBayesianNetwork with 46 nodes and 70 edges
        """
        pass

    def save(self, filename: str) -> None:
        """
        Writes the model to a JSON file.

        Parameters
        ----------
        filename: str
            The path along with the filename where to write the file.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/ecoli70")
        >>> model.save("ecoli70.json")
        """
        pass

    def add_cpds(self, *cpds: LinearGaussianCPD) -> None:
        """
        Add Linear Gaussian CPDs (Conditional Probability Distributions)
        to the Bayesian Network.

        Parameters
        ----------
        cpds : instances of LinearGaussianCPD
            LinearGaussianCPDs which will be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.cpds:
        ...     print(cpd)
        pass
        P(x1) = N(1; 4)
        P(x2 | x1) = N(0.5*x1 + -5.0; 4)
        P(x3 | x2) = N(-1*x2 + 4; 3)
        """
        pass

    def get_cpds(self, node: Hashable | None = None) -> LinearGaussianCPD | list[LinearGaussianCPD]:
        """
        Returns the CPD of the specified node. If node is not specified, returns all CPDs
        that have been added so far to the graph.

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        list[LinearGaussianCPD] or LinearGaussianCPD
            A CPD or list of Linear Gaussian CPDs.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<LinearGaussianCPD: P(x1) = N(1; 4) at 0x...,
        <LinearGaussianCPD: P(x2 | x1) = N(0.5*x1 + -5.0; 4) at 0x...,
        <LinearGaussianCPD: P(x3 | x2) = N(-1*x2 + 4; 3) at 0x...]
        """
        pass

    def remove_cpds(self, *cpds: LinearGaussianCPD) -> None:
        """
        Removes the CPDs provided in the arguments.

        Parameters
        ----------
        *cpds: LinearGaussianCPD
            LinearGaussianCPD objects (or their variable names) to remove.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        pass
        P(x1) = N(1; 4)
        P(x2 | x1) = N(0.5*x1 + -5.0; 4)
        P(x3 | x2) = N(-1*x2 + 4; 3)

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        pass
        P(x1) = N(1; 4)


        """
        pass

    def get_random_cpds(
        self,
        loc: float = 0,
        scale: float = 1,
        inplace: bool = False,
        seed: int | None = None,
    ) -> None | list[LinearGaussianCPD]:
        """
        Generates random Linear Gaussian CPDs for the model. The coefficients
        are sampled from a normal distribution with mean `loc` and standard
        deviation `scale`.

        Parameters
        ----------
        loc: float
            Mean of the normal from which coefficients are sampled.
        scale: float
            Std dev of the normal from which coefficients are sampled.
        inplace: bool (default: False)
            If True, adds the generated LinearGaussianCPDs to the model;
            otherwise returns them.
        seed: int (optional)
            Seed for the random number generator.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> model.get_random_cpds(loc=0, scale=1, seed=42) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<LinearGaussianCPD: P(x1) = N(...; ...) at 0x...,
        <LinearGaussianCPD: P(x2 | x1) = N(...; ...) at 0x...,
        <LinearGaussianCPD: P(x3 | x2) = N(...; ...) at 0x...]
        """
        pass

    def to_joint_gaussian(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Represents the Linear Gaussian Bayesian Network as a joint
        Linear Gaussian Bayesian Networks can be represented using a joint
        Gaussian distribution over all the variables. This method gives
        the mean and covariance of this equivalent joint gaussian distribution.
        Returns
        -------
        mean, cov: np.ndarray, np.ndarray
            Mean vector and covariance matrix of the joint Gaussian.
            The mean and the covariance matrix of the joint gaussian distribution.
        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> mean, cov = model.to_joint_gaussian()
        >>> mean
        array([ 1. , -4.5,  8.5])
        >>> cov
        array([[ 16.,   8.,  -8.],
               [  8.,  20., -20.],
               [ -8., -20.,  29.]])
        """
        pass

    def log_likelihood(self, data: pd.DataFrame) -> float:
        """
        Computes the log-likelihood of the given dataset under the current
        Linear Gaussian Bayesian Network.

        Parameters
        ----------
        data : pandas.DataFrame
            Observations for all variables (columns must match model variables).

        Returns
        -------
        float
            Total log-likelihood of the data under the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> rng = np.random.default_rng(42)
        >>> df = pd.DataFrame(
        ...     rng.normal(0, 1, size=(100, 3)), columns=["x1", "x2", "x3"]
        ... )
        >>> float(round(model.log_likelihood(df), 3))
        -855.065
        """
        pass

    def copy(self):
        """
        Returns a copy of the model.

        Returns
        -------
        Model's copy: pgmpy.models.LinearGaussianBayesianNetwork
            Copy of the model on which the method was called.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("A", "B"), ("B", "C")])
        >>> cpd_a = LinearGaussianCPD(variable="A", beta=[1], std=4)
        >>> cpd_b = LinearGaussianCPD(
        ...     variable="B", beta=[-5, 0.5], std=4, evidence=["A"]
        ... )
        >>> cpd_c = LinearGaussianCPD(variable="C", beta=[4, -1], std=3, evidence=["B"])
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

    def simulate(
        self,
        n_samples: int = 1000,
        do: dict[str, float] | None = None,
        evidence: dict[str, float] | None = None,
        virtual_intervention: list[LinearGaussianCPD] | None = None,
        include_latents: bool = False,
        seed: int | None = None,
        missing_prob=None,
    ) -> pd.DataFrame:
        """
        Simulates data from the model.

        Parameters
        ----------
        n_samples: int
            Number of samples to draw.
            The number of samples to draw from the model.

        do: dict (default: None)
            The interventions to apply to the model. dict should be of the form
            {variable_name: value}

        evidence: dict (default: None)
            Observed evidence to apply to the model. dict should be of the form
            {variable_name: value}

        virtual_intervention: list
            Also known as soft intervention. `virtual_intervention` should be a list
            of `pgmpy.factors.discrete.LinearGaussianCPD` objects specifying the virtual/soft
            intervention probabilities.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            Seed for the random number generator.

        missing_prob: dict (default: None)
            A dictionary specifying the probability of missingness for each variable.
            Keys must be valid variable names in the model, and values must be floats
            between 0 and 1. Each sampled value is independently replaced with NaN
            with the specified probability (MCAR assumption). A ValueError is raised
            if a variable is not present in the sampled data or if the probability
            is outside the range [0, 1].

        Returns
        -------
        pandas.DataFrame: A pandas data frame with the generated samples.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = LinearGaussianCPD("x1", [1], 4)
        >>> cpd2 = LinearGaussianCPD("x2", [-5, 0.5], 4, ["x1"])
        >>> cpd3 = LinearGaussianCPD("x3", [4, -1], 3, ["x2"])
        >>> model.add_cpds(cpd1, cpd2, cpd3)

        Simple forward sampling
        >>> model.simulate(n_samples=3, seed=42) # doctest: +NORMALIZE_WHITESPACE
                 x1        x2        x3
        0 -3.307168 -4.270673  9.688070
        1 -7.195367 -9.833986  9.493212
        2 -0.324284 -4.959026  8.758940

        Sampling with intervention (do)
        >>> model.simulate(n_samples=3, seed=42, do={"x2": 0.0}) # doctest: +NORMALIZE_WHITESPACE
                 x1        x3   x2
        0  2.218868  0.880048  0.0
        1  4.001805  6.821694  0.0
        2 -6.804141  0.093461  0.0

        Sampling with evidence
        >>> model.simulate(n_samples=3, seed=42, evidence={"x1": 2.0}) # doctest: +NORMALIZE_WHITESPACE
            x1        x2         x3
        0  2.0 -6.753790   8.242987
        1  2.0 -5.284287  12.763190
        2  2.0  1.133549  -3.023892

        Sampling with both intervention and evidence
        >>> model.simulate(n_samples=3, seed=42, do={"x2": 1.0}, evidence={"x1": 0.0}) # doctest: +NORMALIZE_WHITESPACE
            x1        x3   x2
        0  0.0  3.914151  1.0
        1  0.0 -0.119952  1.0
        2  0.0  5.251354  1.0
        """
        pass

    def check_model(self) -> bool:
        """
        Checks the model for structural/parameter consistency.

        Currently checks:
        * Each CPD's listed parents match the graph's parents.

        Returns
        -------
        bool
            True if all checks pass; raises ValueError otherwise.
        """
        pass

    def get_cardinality(self, node: Any) -> None:
        """
        Cardinality is not defined for continuous variables.
        """
        pass

    def fit(
        self,
        data: pd.DataFrame,
        estimator: str = "mle",
        std_estimator: str = "unbiased",
    ) -> LinearGaussianBayesianNetwork:
        """
        Estimates (fits) the Linear Gaussian CPDs from data.

        Parameters
        ----------
        data : pd.DataFrame
                Continuous-valued data containing all model variables.
                A pandas DataFrame with the data to which to fit the model
                structure. All variables must be continuously valued.

        estimator : str, optional (default 'mle')
                The estimator to use for mean estimation.
                 - 'mle': Maximum Likelihood Estimation via OLS.
                Currently, MLE via OLS is the only supported method for mean estimation.

        std_estimator : str, optional (default 'unbiased')
                The estimator to use for standard deviation estimation.
                Must be one of:
                    - 'mle': Maximum Likelihood Estimation. Uses ddof=0.
                    - 'unbiased': Unbiased estimation. For root nodes, uses
                    ddof=1. For non-root nodes, uses ddof = 1 + number of parents.

        Returns
        -------
        self
        None: The estimated LinearGaussianCPDs are added to the model. They can
            be accessed using `model.cpds`.
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> rng = np.random.default_rng(42)
        >>> df = pd.DataFrame(
        ...     rng.normal(0, 1, (100, 3)), columns=["x1", "x2", "x3"]
        ... )
        >>> model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> model.fit(df, estimator="mle", std_estimator="unbiased") # doctest: +ELLIPSIS
        <pgmpy.models.LinearGaussianBayesianNetwork.LinearGaussianBayesianNetwork object at 0x...>
        >>> model.cpds # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<LinearGaussianCPD: P(x1) = N(-0.029; 0.902) at 0x...,
        <LinearGaussianCPD: P(x2 | x1) = N(0.046*x1 + -0.012; 0.981) at 0x...,
        <LinearGaussianCPD: P(x3 | x2) = N(0.172*x2 + -0.078; 0.908) at 0x...]
        """
        pass

    def predict_probability(self, data: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Predicts the conditional distribution of missing variables

        Returns the posterior mean and covariance of the missing variables
        given the observed variables in each row of data.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with a subset of model variables observed.

        Returns
        -------
        variables: list
            Missing variables (order matches returned distribution).

        mu: np.array
            Posterior mean for each row of data.

        cov: np.array
            Posterior covariance (same for all rows, depends only on structure).

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/ecoli70")
        >>> df = model.simulate(n_samples=5, seed=42)
        >>> df = df.drop(columns=["folK"])
        >>> model.predict(df) # doctest: +NORMALIZE_WHITESPACE
               folK
        0  0.903384
        1  0.576122
        2  1.331394
        3  0.027018
        4  1.731904
        """
        pass

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the MAP estimates (posterior mean) of missing variables.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with a subset of model variables observed.

        Returns
        -------
        predictions: pandas.DataFrame
            DataFrame with missing variables columns containing the posterior mean
            (MAP estimate) for each row of data.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/ecoli70")
        >>> df = model.simulate(n_samples=5, seed=42)
        >>> df = df.drop(columns=["folK"])
        >>> model.predict(df) # doctest: +NORMALIZE_WHITESPACE
               folK
        0  0.903384
        1  0.576122
        2  1.331394
        3  0.027018
        4  1.731904
        """
        pass

    def to_markov_model(self) -> None:
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        pass

    def is_imap(self, JPD: Any) -> None:
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        pass

    @staticmethod
    def get_random(
        n_nodes: int = 5,
        edge_prob: float = 0.5,
        node_names: list | None = None,
        latents: bool = False,
        loc: float = 0,
        scale: float = 1,
        seed: int | None = None,
    ) -> LinearGaussianBayesianNetwork:
        """
        Returns a randomly generated Linear Gaussian Bayesian Network on `n_nodes`
        Returns a randomly generated Linear Gaussian Bayesian Network on `n_nodes` variables
        with edge probabiliy of `edge_prob` between variables.
        Parameters
        ----------
        n_nodes: int
            Number of nodes.
            The number of nodes in the randomly generated DAG.

            Probability of an edge (consistent with a topological order).
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        latents: bool (default: False)
        loc: float

            Mean of normal for coefficients.
            The mean of the normal distribution from which the coefficients are
            sampled.

            Std dev of normal for coefficients.
            The standard deviation of the normal distribution from which the
            coefficients are sampled.

        seed: int
            The seed for the random number generator.
        Returns
        -------
        LinearGaussianBayesianNetwork
            The randomly generated model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> model = LinearGaussianBayesianNetwork.get_random(n_nodes=5, seed=42)
        >>> sorted(model.nodes())
        ['X_0', 'X_1', 'X_2', 'X_3', 'X_4']
        >>> sorted(model.edges())
        [('X_0', 'X_2'), ('X_0', 'X_3'), ('X_1', 'X_2'), ('X_2', 'X_3'), ('X_3', 'X_4')]
        >>> sorted(model.cpds, key=lambda cpd: cpd.variable) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<LinearGaussianCPD: P(X_0) = N(...; ...) at 0x...,
        <LinearGaussianCPD: P(X_1) = N(...; ...) at 0x...,
        <LinearGaussianCPD: P(X_2 | X_0, X_1) = N(...) at 0x...,
        <LinearGaussianCPD: P(X_3 | X_0, X_2) = N(...) at 0x...,
        <LinearGaussianCPD: P(X_4 | X_3) = N(...) at 0x...]
        """
        pass

    def __eq__(self, other):
        """
        Checks equality of two LinearGaussianBayesianNetwork objects. Two models are equal if they have the same
        structure and the same CPDs.

        Parameters
        ----------
        other: LinearGaussianBayesianNetwork instance
            The model to compare with.

        Returns
        -------
        bool
            True if the two LinearGaussianCPD objects are equal, False otherwise.
        """
        if not isinstance(other, LinearGaussianBayesianNetwork):
            return False

        # Test for structure equality using the DAG's __eq__ method.
        if not super().__eq__(other):
            return False

        # Test for LinearGaussianCPD equality.
        self_cpds = {cpd.variable: cpd for cpd in self.cpds}
        other_cpds = {cpd.variable: cpd for cpd in other.cpds}

        for var in self_cpds:
            if self_cpds[var] != other_cpds[var]:
                return False

        return True
