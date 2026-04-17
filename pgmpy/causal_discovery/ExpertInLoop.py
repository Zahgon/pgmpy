from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test
from pgmpy.global_vars import logger
from pgmpy.utils import llm_pairwise_orient


class ExpertInLoop(_BaseCausalDiscovery):
    """
    Expert-in-the-loop causal discovery algorithm.

    This class implements an iterative causal discovery algorithm that combines statistical independence testing with
    expert knowledge for edge orientation. The algorithm works by iteratively adding and removing edges between
    variables based on conditional independence tests, similar to the Greedy Equivalence Search (GES) algorithm. When
    adding edges, the algorithm queries an expert (human or automated through LLMs) for the edge orientation.

    The algorithm can use various sources for edge orientation:
    - Manual user input
    - Large Language Models (LLMs)
    - Custom orientation functions
    - Pre-specified orientations
    - Specified `expert_knowledge` argument.

    Parameters
    ----------
    pval_threshold : float, default=0.05
        The p-value threshold used in conditional independence tests. If the p-value is greater than this threshold, the
        variables are considered conditionally independent.

    effect_size_threshold : float, default=0.05
        The effect size threshold for edge suggestions.
        - If the conditional effect size between two variables is greater
          than this threshold, the algorithm suggests adding an edge.
        - If the effect size for an existing edge is less than this threshold,
          the algorithm suggests removing the edge.

    ci_test : str or callable, default=None
        The Conditional Independence test to use. When None, the algorithm
        tries to automatically detect a suitable CI test based on the variable
        types. See :mod:`pgmpy.estimators.CITests` for available tests.

    orientation_fn : callable, default=llm_pairwise_orient
        A function to determine edge orientation. The function should take at
        least two arguments (the names of the two variables) and return either:
        - A tuple (source, target) representing the directed edge from source
          to target
        - None, representing no edge between the variables

        Built-in functions that can be used:
        - `pgmpy.utils.manual_pairwise_orient`: Prompts the user to specify direction.
        - `pgmpy.utils.llm_pairwise_orient`: Uses an LLM to determine direction.

    orientations : set, default=set()
        A set of edges that will be used as the preferred orientation over
        the output of `orientation_fn`. Edges should be specified as tuples
        (source, target).

    expert_knowledge : ExpertKnowledge, default=None
        Expert knowledge about the causal structure. Can include:
        - forbidden_edges: Edges that should not be present in the final model
        - required_edges: Edges that must be present in the final model
        - temporal_order: The temporal ordering of variables

        Note: Explicit orientations in the `orientations` parameter take
        precedence over temporal ordering.

    use_cache : bool, default=True
        If True, the algorithm caches results from `orientation_fn` and reuses
        them in future calls instead of querying the orientation function again.

    show_progress : bool, default=True
        If True, prints information about the running status.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal graph as a DAG.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    orientation_cache_ : set
        Cache of edge orientations learned during fitting.

    Examples
    --------
    Basic usage with custom orientation function:

    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.causal_discovery import ExpertInLoop
    >>> model = get_example_model("cancer")
    >>> df = model.simulate(int(1e3))
    >>> def custom_orient(var1, var2, **kwargs):
    ...     return (var1, var2) if var1 < var2 else (var2, var1)
    pass
    >>> eil = ExpertInLoop(orientation_fn=custom_orient, effect_size_threshold=0.0001)
    >>> eil.fit(df)
    >>> eil.causal_graph_.edges()

    Using pre-specified orientations:

    >>> orientations = {("Pollution", "Cancer"), ("Smoker", "Cancer")}
    >>> eil = ExpertInLoop(orientations=orientations, effect_size_threshold=0.0001)
    >>> eil.fit(df)

    Using expert knowledge with temporal ordering:

    >>> from pgmpy.estimators import ExpertKnowledge
    >>> expert = ExpertKnowledge(
    ...     temporal_order=[["Pollution", "Smoker"], ["Cancer"], ["Xray", "Dyspnoea"]]
    ... )
    >>> eil = ExpertInLoop(expert_knowledge=expert, effect_size_threshold=0.0001)
    >>> eil.fit(df)

    Using LLM-based orientation (requires API key):

    >>> from functools import partial
    >>> from pgmpy.utils import llm_pairwise_orient
    >>> variable_descriptions = {
    ...     "Smoker": "Whether a person smokes",
    ...     "Cancer": "Whether a person has cancer",
    ... }
    >>> orientation_fn = partial(
    ...     llm_pairwise_orient,
    ...     variable_descriptions=variable_descriptions,
    ...     llm_model="gemini/gemini-1.5-flash",
    ... )
    >>> eil = ExpertInLoop(orientation_fn=orientation_fn)
    >>> eil.fit(df)

    References
    ----------
    The algorithm is inspired by active learning approaches to causal discovery
    and the GES algorithm.
    """

    def __init__(
        self,
        pval_threshold: float = 0.05,
        effect_size_threshold: float = 0.05,
        ci_test: str | None = None,
        orientation_fn: Callable = llm_pairwise_orient,
        orientations: set[tuple[str, str]] | None = None,
        expert_knowledge=None,
        use_cache: bool = True,
        show_progress: bool = True,
    ):
        self.pval_threshold = pval_threshold
        self.effect_size_threshold = effect_size_threshold
        self.ci_test = ci_test
        self.orientation_fn = orientation_fn
        self.orientations = orientations
        self.expert_knowledge = expert_knowledge
        self.use_cache = use_cache
        self.show_progress = show_progress

    def _test_all(self, ci_test, dag: DAG, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs CI tests on all possible combinations of variables in `dag`.

        Parameters
        ----------
        ci_test : callable
            The CI test function to use.

        dag : pgmpy.base.DAG
            The DAG on which to run the tests.

        data : pd.DataFrame
            The data to use for CI testing.

        Returns
        -------
        pd.DataFrame
            The results with p-values and effect sizes of all the tests.
        """
        pass

    def _break_cycle(self, dag, u, v, ci_test, data, effect_size_threshold, pval_threshold):
        """
        Subroutine to break any cycles that get created.

        Parameters
        ----------
        dag : pgmpy.base.DAG
            The current DAG that still doesn't have cycles.

        u, v : hashable
            The variables that create a cycle in `dag` when (u, v) edge is added.

        ci_test : callable
            The Conditional Independence test to use.

        data : pd.DataFrame
            The data for CI testing.

        effect_size_threshold : float
            Threshold for effect size.

        pval_threshold : float
            Threshold for p-value.

        Returns
        -------
        list
            List of edges to remove to break the cycle.
        """
        pass

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the ExpertInLoop algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : ExpertInLoop
            Returns the instance with the fitted attributes.
        """
        pass
