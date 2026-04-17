import warnings
from collections.abc import Callable, Hashable, Iterable
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import PDAG
from pgmpy.estimators import (
    StructureEstimator,
    StructureScore,
)
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import get_scoring_method
from pgmpy.utils.mathext import powerset


class GES(StructureEstimator):
    """
    Implementation of Greedy Equivalence Search (GES) causal discovery / structure learning algorithm.

    GES is a score-based causal discovery / structure learning algorithm that works in three phases:
        1. Forward phase: New edges are added such that the model score improves.
        2. Backward phase: Edges are removed from the model such that the model score improves.
        3. Edge turning phase: Edge orientations are turned/flipped such that model score improves.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    Chickering, David Maxwell. "Optimal structure identification with greedy search."
      Journal of machine learning research 3.Nov (2002): 507-554.
    """

    def __init__(self, data: pd.DataFrame, use_cache: bool = False, **kwargs):
        warnings.warn(
            "GES is deprecated. Please use pgmpy.causal_discovery.GES instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.use_cache = use_cache

        super().__init__(data=data, **kwargs)

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

    def _legal_edge_additions(
        self,
        current_model: PDAG,
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Return all possible directed edge additions (u -> v) between non-adjacent nodes.
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

    def _legal_edge_turns(
        self,
        current_model: PDAG,
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Return all candidate edge turns (i.e., reverse directions of existing edges).
        """
        pass

    def insert(
        self,
        u: Any,
        v: Any,
        T: Iterable[Any],
        current_model: PDAG,
    ):
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
    ):
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
    ):
        """
        Perform turn operation (reverse or orient edge between u and v) with set C.
        """
        pass

    def _score_valid_insertions(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid insert(u -> v) operations.
        """
        pass

    def _score_valid_deletions(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid delete(u - v) or delete(u -> v) operations.
        """
        pass

    def _score_valid_turns(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ):
        """
        Dispatch turn operator depending on edge type.
        """
        pass

    def _score_valid_turns_directed(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid turn(u -> v) operations.
        """
        pass

    def _score_valid_turns_undirected(
        self,
        u: Any,
        v: Any,
        current_model: PDAG,
        score_fn: Callable[[Any, list[Any]], float],
    ) -> list[tuple[float, Any, Any, set[Any]]]:
        """
        Score all valid turn(u - v) operations.
        """
        pass

    def estimate(
        self,
        scoring_method: str | StructureScore | None = None,
        min_improvement: float = 1e-6,
        debug: bool = False,
    ) -> PDAG:
        """
        Estimates the DAG from the data.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
            ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
            be an instance of `StructureScore`.

        min_improvement: float
            The operation (edge addition, removal, or turning) would only be performed if the
            model score improves by atleast `min_improvement`.

        debug: bool
            Estimate the graph in debug mode, printing the corresponding increase in score at
            each step.

        Returns
        -------
        Estimated model: pgmpy.base.PDAG
            A `PDAG` at a (local) score maximum.

        Examples
        --------
        >>> import numpy as np
        >>> # Simulate some sample data from a known model to learn the model structure from
        >>> from pgmpy.utils import get_example_model
        >>> np.random.seed(42)
        >>> model = get_example_model("alarm")
        >>> model.seed = 42
        >>> df = model.simulate(int(1e3))

        >>> # Learn the model structure using GES algorithm from `df`
        >>> from pgmpy.estimators import GES
        >>> est = GES(df)
        >>> dag = est.estimate(scoring_method="bic-d")
        >>> len(dag.nodes())
        37
        >>> len(dag.edges())
        48
        """
        pass
