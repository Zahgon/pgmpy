import warnings
from collections.abc import Iterable
from itertools import chain, product

import networkx as nx
import numpy as np
from networkx.algorithms.dag import descendants
from tqdm.auto import tqdm

from pgmpy import config, logger
from pgmpy.base import DAG
from pgmpy.estimators.LinearModel import LinearEstimator
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import (
    DiscreteBayesianNetwork,
    FunctionalBayesianNetwork,
    LinearGaussianBayesianNetwork,
    SEMGraph,
)
from pgmpy.utils.sets import _powerset, _variable_or_iterable_to_set


class CausalInference:
    """
    This is an inference class for performing Causal Inference over Bayesian
    Networks or Structural Equation Models.

    Parameters
    ----------
    model: pgmpy.base.DAG | pgmpy.models.DiscreteBayesianNetwork | pgmpy.models.SEMGraph
        The model that we'll perform inference over.

    Examples
    --------
    Create a small Bayesian Network.

    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> game = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])

    Load the graph into the CausalInference object to make causal queries.

    >>> from pgmpy.inference.CausalInference import CausalInference
    >>> inference = CausalInference(game)
    >>> inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
    frozenset()
    >>> inference.get_all_frontdoor_adjustment_sets(X="X", Y="Y")
    frozenset({frozenset({'A'})})

    References
    ----------
    'Causality: Models, Reasoning, and Inference' - Judea Pearl (2000)
    """

    def __init__(self, model):
        if not isinstance(
            model,
            (
                DiscreteBayesianNetwork,
                LinearGaussianBayesianNetwork,
                FunctionalBayesianNetwork,
                SEMGraph,
                DAG,
            ),
        ):
            raise NotImplementedError("Causal Inference is only implemented for DAGs, BayesianNetworks, and SEMGraphs.")

        # Check if the variable names are strings. If not, raise an error.
        bad_variable = model._variable_name_contains_non_string()
        if bad_variable != False:
            raise NotImplementedError(
                f"Causal Inference is only implemented for a model with "
                "variable names with string type. "
                f"Found {bad_variable[0]} with type {bad_variable[1]}. "
                "Convert them to string to proceed."
            )

        # Initialize data structures.
        self.model = model

        if isinstance(model, SEMGraph):
            self.observed_variables = frozenset(model.observed)
            self.latent_variables = model.latents
            self.dag = DAG(
                model.full_graph_struct,
                latents=model.latents.union({var for var in model.full_graph_struct.nodes() if var.startswith(".")}),
            )

        elif isinstance(model, (DiscreteBayesianNetwork, DAG)):
            self.observed_variables = frozenset(model.nodes()).difference(model.latents)
            self.latent_variables = model.latents
            self.dag = DAG(model.to_directed(), latents=model.latents)

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return f"{self.__class__.__name__}({variables})"

    def is_valid_backdoor_adjustment_set(self, X, Y, Z=[]):
        """
        Test whether Z is a valid backdoor adjustment set for estimating the causal impact of X on Y.

        Parameters
        ----------

        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable.

        Z: list (array-like)
            List of adjustment variables.

        Returns
        -------
        Is a valid backdoor adjustment set: bool
            True if Z is a valid backdoor adjustment set else False

        Examples
        --------
        >>> game1 = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        >>> inference = CausalInference(game1)
        >>> inference.is_valid_backdoor_adjustment_set("X", "Y")
        True
        """
        pass

    def get_all_backdoor_adjustment_sets(self, X, Y):
        """
        Returns a list of all adjustment sets per the back-door criterion.

        A set of variables Z satisfies the back-door criterion relative
          to an ordered pair of variabies (Xi, Xj) in a DAG G if:
            (i) no node in Z is a descendant of Xi; and
            (ii) Z blocks every path between Xi and Xj that contains an arrow into Xi.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable.

        Returns
        -------
        frozenset: A frozenset of frozensets

        Y: str
            Target Variable

        Examples
        --------
        >>> game1 = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        >>> inference = CausalInference(game1)
        >>> inference.get_all_backdoor_adjustment_sets("X", "Y")
        frozenset()
        """
        pass

    def is_valid_frontdoor_adjustment_set(self, X, Y, Z=None):
        """
        Test whether Z is a valid frontdoor adjustment set for estimating the causal impact of X on Y via the frontdoor
        adjustment formula.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable.

        Z: list (array-like)
            List of adjustment variables.

        Returns
        -------
        Is valid frontdoor adjustment: bool
            True if Z is a valid frontdoor adjustment set.
        """
        pass

    def get_all_frontdoor_adjustment_sets(self, X, Y):
        """
        Identify possible sets of variables, Z, which satisfy the front-door criterion relative to given X and Y.

        Z satisfies the front-door criterion if:
          (i)    Z intercepts all directed paths from X to Y
          (ii)   there is no backdoor path from X to Z
          (iii)  all back-door paths from Z to Y are blocked by X

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable

        Returns
        -------
        frozenset: a frozenset of frozensets
        """
        pass

    def get_scaling_indicators(self):
        """
        Returns a scaling indicator for each of the latent variables in the model.
        The scaling indicator is chosen randomly among the observed measurement
        variables of the latent variable.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(
        ...     ebunch=[
        ...         ("xi1", "eta1"),
        ...         ("xi1", "x1"),
        ...         ("xi1", "x2"),
        ...         ("eta1", "y1"),
        ...         ("eta1", "y2"),
        ...     ],
        ...     latents=["xi1", "eta1"],
        ... )
        >>> sorted(model.get_scaling_indicators().items())
        [('eta1', 'y1'), ('xi1', 'x1')]

        Returns
        -------
        dict: Returns a dict with latent variables as the key and their value being the
                scaling indicator.
        """
        pass

    def _iv_transformations(self, X, Y, scaling_indicators={}):
        """
        Transforms the graph structure of SEM so that the d-separation criterion is
        applicable for finding IVs. The method transforms the graph for finding MIIV
        for the estimation of X \rightarrow Y given the scaling indicator for all the
        parent latent variables.

        Parameters
        ----------
        X: node
            The explantory variable.

        Y: node
            The dependent variable.

        scaling_indicators: dict
            Scaling indicator for each latent variable in the model.

        Returns
        -------
        nx.DiGraph: The transformed full graph structure.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(
        ...     ebunch=[
        ...         ("xi1", "eta1"),
        ...         ("xi1", "x1"),
        ...         ("xi1", "x2"),
        ...         ("eta1", "y1"),
        ...         ("eta1", "y2"),
        ...     ],
        ...     latents=["xi1", "eta1"],
        ... )
        >>> inference = CausalInference(model)
        >>> inference._iv_transformations(
        ...     "xi1", "eta1", scaling_indicators={"xi1": "x1", "eta1": "y1"}
        ... ) # doctest: +ELLIPSIS
        (<pgmpy.base.DAG.DAG object at 0x...>, 'y1')
        """
        pass

    def get_ivs(self, X, Y, scaling_indicators={}):
        """
        Returns the Instrumental variables(IVs) for the relation X -> Y

        Parameters
        ----------
        X: node
            The variable name (observed or latent)

        Y: node
            The variable name (observed or latent)

        scaling_indicators: dict (optional)
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.
            If not given the method automatically selects one of the measurement variables
            at random as the scaling indicator.

        Returns
        -------
        set: {str}
            The set of Instrumental Variables for X -> Y.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(
        ...     ebunch=[("I", "X"), ("X", "Y")], latents=[], err_corr=[("X", "Y")]
        ... )
        >>> inference = CausalInference(model)
        >>> inference.get_ivs("X", "Y")
        {'I'}
        """
        pass

    def get_conditional_ivs(self, X, Y, scaling_indicators={}):
        """
        Returns the conditional IVs for the relation X -> Y

        Parameters
        ----------
        X: node
            The observed variable's name

        Y: node
            The oberved variable's name

        scaling_indicators: dict (optional)
            A dict representing which observed variable to use as scaling indicator for
            the latent variables.
            If not provided, automatically finds scaling indicators by randomly selecting
            one of the measurement variables of each latent variable.

        Returns
        -------
        set: Set of 2-tuples representing tuple[0] is an IV for X -> Y given tuple[1].

        References
        ----------
        .. [1] Van Der Zander, B., Textor, J., & Liskiewicz, M. (2015, June). Efficiently finding
               conditional instruments for causal inference. In Twenty-Fourth International Joint
               Conference on Artificial Intelligence.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(
        ...     ebunch=[("I", "X"), ("X", "Y"), ("W", "I")],
        ...     latents=[],
        ...     err_corr=[("W", "Y")],
        ... )
        >>> inference = CausalInference(model)
        >>> inference.get_conditional_ivs("X", "Y")
        [('I', {'W'})]
        """
        pass

    def get_total_conditional_ivs(self, X, Y, scaling_indicators={}):
        pass

    def identification_method(self, X, Y):
        """
        Automatically identifies a valid method for estimating the causal effect from X to Y.

        Parameters
        ----------
        X: str
            The treatment/exposure variable
        Y: str
            The outcome variable

        Returns
        -------
        dict
            A dictionary containing keys as method and value as the corresponding result.
        """
        pass

    def _nearest_separator(self, G, Y, Z):
        """
        Finds the set of the nearest separators for `Y` and `Z` in `G`.

        Parameters
        ----------
        G: nx.DiGraph instance
            The graph in which to the find the nearest separation for `Y` and `Z`.

        Y: str
            The variable name for which the separators are needed.

        Z: str
            The other variable for which the separators are needed.

        Returns
        -------
        set or None: If there is a nearest separator returns the set of separators else returns None.
        """
        pass

    def _simple_decision(self, adjustment_sets=[]):
        """
        Selects the smallest set from provided adjustment sets.

        Parameters
        ----------
        adjustment_sets: iterable
            A frozenset or list of valid adjustment sets

        Returns
        -------
        frozenset
        """
        pass

    def estimate_ate(
        self,
        X,
        Y,
        data,
        estimand_strategy="smallest",
        estimator_type="linear",
        **kwargs,
    ):
        """
        Estimate the average treatment effect (ATE) of X on Y.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable

        data: pandas.DataFrame
            All observed data for this Bayesian Network.

        estimand_strategy: str or frozenset
            Either specify a specific backdoor adjustment set or a strategy.
            The available options are:
                smallest:
                    Use the smallest estimand of observed variables
                all:
                    Estimate the ATE from each identified estimand

        estimator_type: str
            The type of model to be used to estimate the ATE.
            All of the linear regression classes in statsmodels are available including:
                * GLS: generalized least squares for arbitrary covariance
                * OLS: ordinary least square of i.i.d. errors
                * WLS: weighted least squares for heteroskedastic error
            Specify them with their acronym (e.g. "OLS") or simple "linear" as an alias for OLS.

        **kwargs: dict
            Keyward arguments specific to the selected estimator.
            linear:
              missing: str
                Available options are "none", "drop", or "raise"

        Returns
        -------
        The average treatment effect: float

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> game1 = DiscreteBayesianNetwork([("X", "A"), ("A", "Y"), ("A", "B")])
        >>> data = pd.DataFrame(
        ...     rng.random(size=(1000, 4)), columns=["X", "A", "B", "Y"]
        ... )
        >>> inference = CausalInference(model=game1)
        >>> float(round(inference.estimate_ate("X", "Y", data=data, estimator_type="linear"), 15))
        0.001138244615115

        """
        pass

    def get_proper_backdoor_graph(self, X, Y, inplace=False):
        """
        Returns a proper backdoor graph for the exposure `X` and outcome `Y`.
        A proper backdoor graph is a graph which remove the first edge of every
        proper causal path from `X` to `Y`.

        Parameters
        ----------
        X: list (array-like)
            A list of exposure variables.

        Y: list (array-like)
            A list of outcome variables

        inplace: boolean
            If inplace is True, modifies the object itself. Otherwise retuns
            a modified copy of self.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import CausalInference
        >>> model = DiscreteBayesianNetwork(
        ...     [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
        ... )
        >>> c_infer = CausalInference(model)
        >>> c_infer.get_proper_backdoor_graph(X=["x1", "x2"], Y=["y1", "y2"]) # doctest: +ELLIPSIS
        <pgmpy.base.DAG.DAG object at 0x...>

        References
        ----------
        [1] Perkovic, Emilija, et al.
         "Complete graphical characterization and construction of
         adjustment sets in Markov equivalence classes of ancestral graphs."
           The Journal of Machine Learning Research 18.1 (2017): 8132-8193.
        """
        pass

    def is_valid_adjustment_set(self, X, Y, adjustment_set):
        """
        Method to test whether `adjustment_set` is a valid adjustment set for
        identifying the causal effect of `X` on `Y`.

        Parameters
        ----------
        X: list (array-like)
            The set of cause variables.

        Y: list (array-like)
            The set of predictor variables.

        adjustment_set: list (array-like)
            The set of variables for which to test whether they satisfy the
            adjustment set criteria.

        Returns
        -------
        Is valid adjustment set: bool
            Returns True if `adjustment_set` is a valid adjustment set for
            identifying the effect of `X` on `Y`. Else returns False.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import CausalInference
        >>> model = DiscreteBayesianNetwork(
        ...     [("x1", "y1"), ("x1", "z1"), ("z1", "z2"), ("z2", "x2"), ("y2", "z2")]
        ... )
        >>> c_infer = CausalInference(model)
        >>> c_infer.is_valid_adjustment_set(
        ...     X=["x1", "x2"], Y=["y1", "y2"], adjustment_set=["z1", "z2"]
        ... )
        True

        References
        ----------
        [1] Perkovic, Emilija, et al.
          "Complete graphical characterization and construction of
            adjustment sets in Markov equivalence classes of ancestral graphs."
              The Journal of Machine Learning Research 18.1 (2017): 8132-8193.
        """
        pass

    def get_minimal_adjustment_set(self, X, Y):
        """
        Returns a minimal adjustment set for
        identifying the causal effect of `X` on `Y`.

        Parameters
        ----------
        X: str (variable name)
            The cause/exposure variables.

        Y: str (variable name)
            The outcome variable

        Returns
        -------
        Minimal adjustment set: set or None
            A set of variables which are the minimal possible adjustment set. If
            None, no adjustment set is possible.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import CausalInference
        >>> dag = DiscreteBayesianNetwork([("X_1", "X_2"), ("Z", "X_1"), ("Z", "X_2")])
        >>> infer = CausalInference(dag)
        >>> infer.get_minimal_adjustment_set("X_1", "X_2")
        {'Z'}

        References
        ----------
        [1] Perkovic, Emilija, et al.
          "Complete graphical characterization and construction of
            adjustment sets in Markov equivalence classes of ancestral graphs."
              The Journal of Machine Learning Research 18.1 (2017): 8132-8193.
        """
        pass

    def query(
        self,
        variables,
        do=None,
        evidence=None,
        adjustment_set=None,
        inference_algo="ve",
        show_progress=True,
        **kwargs,
    ):
        """
        Performs a query on the model of the form :math:`P(X | do(Y), Z)` where :math:`X`
        is `variables`, :math:`Y` is `do` and `Z` is the `evidence`.

        Parameters
        ----------
        variables: list
            list of variables in the query i.e. `X` in :math:`P(X | do(Y), Z)`.

        do: dict (default: None)
            Dictionary of the form {variable_name: variable_state} representing
            the variables on which to apply the do operation i.e. `Y` in
            :math:`P(X | do(Y), Z)`.

        evidence: dict (default: None)
            Dictionary of the form {variable_name: variable_state} repesenting
            the conditional variables in the query i.e. `Z` in :math:`P(X |
            do(Y), Z)`.

        adjustment_set: str or list (default=None)
            Specifies the adjustment set to use. If None, uses the parents of the
            do variables as the adjustment set.

        inference_algo: str or pgmpy.inference.Inference instance
            The inference algorithm to use to compute the probability values.
            String options are: 1) ve: Variable Elimination 2) bp: Belief
            Propagation.

        kwargs: Any
            Additional paramters which needs to be passed to inference
            algorithms.  Please refer to the pgmpy.inference.Inference for
            details.

        Returns
        -------
        Queried distribution: pgmpy.factor.discrete.DiscreteFactor
            A factor object representing the joint distribution over the variables in `variables`.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/alarm")
        >>> infer = CausalInference(model)
        >>> infer.query(["HISTORY"], do={"CVP": "LOW"}, evidence={"HR": "LOW"}) # doctest: +ELLIPSIS
        <DiscreteFactor representing phi(HISTORY:2) at 0x...>
        """
        pass
