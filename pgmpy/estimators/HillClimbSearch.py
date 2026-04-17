import warnings
from collections import deque
from collections.abc import Callable, Generator, Hashable
from itertools import permutations
from typing import (
    Any,
)

import networkx as nx
import pandas as pd
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.estimators import (
    StructureEstimator,
    StructureScore,
)
from pgmpy.estimators.StructureScore import get_scoring_method


class HillClimbSearch(StructureEstimator):
    """
    Class for heuristic hill climb searches for DAGs, to learn
    network structure from data. `estimate` attempts to find a model with optimal score.

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

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.4.3 (page 811ff)
    """

    def __init__(self, data: pd.DataFrame, use_cache: bool = True, **kwargs):
        warnings.warn(
            "HillClimbSearch is deprecated. Please use pgmpy.causal_discovery.HillClimbSearch instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.use_cache = use_cache

        super().__init__(data, **kwargs)

    def _legal_operations(
        self,
        model: DAG,
        score: Callable[[Any, list[Any]], float],
        structure_score: Callable[[str], float],
        tabu_list: deque[tuple[str, tuple[Hashable, Hashable]]],
        max_indegree: int,
        forbidden_edges: list[tuple[Hashable, Hashable]],
        required_edges: list[tuple[Hashable, Hashable]],
    ) -> Generator[tuple[tuple[str, tuple[Hashable, Hashable]], float]]:
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `forbidden_edges` or `required_edges` to exclude those
        edges or to force them to be present in the model, respectively.
        """
        pass

    def estimate(
        self,
        scoring_method: str | StructureScore | None = None,
        start_dag: DAG | None = None,
        tabu_length: int = 100,
        max_indegree: int | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
        epsilon: float = 1e-4,
        max_iter: int = int(1e6),
        show_progress: bool = True,
    ) -> DAG:
        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
            ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
            be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default, a completely
            disconnected network is used.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be
            reversed during the search procedure. This serves to enforce a
            wider exploration of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        expert_knowledge: pgmpy.estimators.ExpertKnowledge instance (default: None)
            Expert knowledge to be used with the algorithm. Expert knowledge
            allows specification of required and forbidden edges, as well as temporal
            order of nodes.

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less
            than `epsilon`, the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model
            when the number of iterations is greater than `max_iter`.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> # Simulate some sample data from a known model to learn the model structure from
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/alarm")
        >>> df = model.simulate(int(1e3), seed=42)

        >>> # Learn the model structure using HillClimbSearch algorithm from `df`
        >>> from pgmpy.estimators import HillClimbSearch
        >>> est = HillClimbSearch(df)
        >>> dag = est.estimate(scoring_method="bic-d")
        >>> len(dag.nodes())
        37
        >>> isinstance(dag, DAG)
        True

        """
        pass
