from collections.abc import Hashable, Iterable
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import PDAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.structure_score import BaseStructureScore, get_scoring_method
from pgmpy.utils.mathext import powerset


class GES(_ScoreMixin, _BaseCausalDiscovery):
    """
    Score-based causal discovery using Greedy Equivalence Search (GES).

    This class implements the GES algorithm [1]_ for causal discovery. Given a
    tabular dataset, the algorithm estimates the causal structure among the
    variables in the data as a Directed Acyclic Graph (DAG) or Partially
    Directed Acyclic Graph (PDAG).

    GES works in three phases:
        1. Forward phase: Edges are added to improve the model score.
        2. Backward phase: Edges are removed to improve the model score.
        3. Edge turning phase: Edge orientations are flipped to improve the score.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore instance, default=None
        The score to be optimized during structure estimation. Supported
        structure scores:

        - Discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'
        - Continuous data: 'll-g', 'aic-g', 'bic-g'
        - Mixed data: 'll-cg', 'aic-cg', 'bic-cg'

        If None, the appropriate scoring method is automatically selected based
        on the data type. Also accepts a custom score instance that inherits
        from `BaseStructureScore`.

    return_type : str, default='pdag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    min_improvement : float, default=1e-6
        The minimum score improvement required to perform an operation
        (edge addition, removal, or flipping). Operations with smaller
        improvements are not performed.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph at a (local) score maximum.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> import numpy as np
    >>> from pgmpy.example_models import load_model
    >>> np.random.seed(42)
    >>> model = load_model("bnlearn/alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the GES algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import GES
    >>> ges = GES(scoring_method="bic-d")
    >>> ges.fit(df)
    GES(scoring_method='bic-d')
    >>> ges.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.PDAG.PDAG object at 0x...>
    >>> ges.n_features_in_
    37

    References
    ----------
    .. [1] Chickering, David Maxwell. "Optimal structure identification with
           greedy search." Journal of machine learning research 3.Nov (2002):
           507-554.
    """

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        return_type: str = "pdag",
        min_improvement: float = 1e-6,
    ):
        self.scoring_method = scoring_method
        self.return_type = return_type
        self.min_improvement = min_improvement

    def _separates(
        self,
        S: set[Any],
        A: set[Any],
        B: set[Any],
        graph: nx.DiGraph,
    ) -> bool:
        """
        Check if S separates A and B in the graph.

        That is, every path from any node in A to any node in B
        intersects S.
        """
        pass

    def _legal_edge_deletions(
        self,
        current_model: PDAG,
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Return all edges that can be considered for deletion.
        """
        pass

    def insert(
        self,
        u: Any,
        v: Any,
        T: Iterable[Any],
        current_model: PDAG,
    ) -> PDAG:
        """
        Perform insert(u -> v) with conditioning set T.
        """
        pass

    def delete(
        self,
        u: Any,
        v: Any,
        H: set[Any],
        current_model: PDAG,
    ) -> PDAG:
        """
        Perform delete(u - v) or delete(u -> v) with conditioning set H.
        """
        pass

    def turn(
        self,
        u: Any,
        v: Any,
        C: Iterable[Any],
        current_model: PDAG,
    ) -> PDAG:
        """
        Perform turn operation (reverse or orient edge between u and v) with set C.
        """
        pass

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the GES algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.GES
            Returns the instance with the fitted attributes.
        """
        pass
