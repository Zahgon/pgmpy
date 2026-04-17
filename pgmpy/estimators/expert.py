import warnings
from collections.abc import Callable, Hashable
from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy import config, logger
from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import ci_registry
from pgmpy.utils import llm_pairwise_orient


class ExpertInLoop(StructureEstimator):
    def __init__(self, data: pd.DataFrame | None = None, **kwargs):
        warnings.warn(
            "ExpertInLoop is deprecated. Please use pgmpy.causal_discovery.ExpertInLoop instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(data=data, **kwargs)
        self.orientation_cache = set()

    def test_all(self, ci_test, dag: DAG) -> pd.DataFrame:
        """
        Runs CI tests on all possible combinations of variables in `dag`.

        Parameters
        ----------
        dag: pgmpy.base.DAG
            The DAG on which to run the tests.

        Returns
        -------
        pd.DataFrame: The results with p-values and effect sizes of all the tests.
        """
        pass

    def estimate(
        self,
        pval_threshold: float = 0.05,
        effect_size_threshold: float = 0.05,
        ci_test: str | None = None,
        orientation_fn: Callable[..., tuple[Hashable, Hashable] | None] = llm_pairwise_orient,
        orientations: set[tuple[str, str]] = set(),
        expert_knowledge: ExpertKnowledge | None = None,
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> DAG:
        """
        Estimates a DAG from the data by utilizing expert knowledge.

        The method iteratively adds and removes edges between variables
        (similar to Greedy Equivalence Search (GES) algorithm) based on a
        global score metric that improves the model's fit in each iteration.
        The score metric used is based on conditional independence testing.
        When adding an edge to the model, the method asks for expert knowledge
        to decide the orientation of the edge. Alternatively, an LLM can used
        to decide the orientation of the edge.

        Parameters
        ----------
        pval_threshold: float
            The p-value threshold to use for the test to determine whether
            there is a significant association between the variables or not.

        effect_size_threshold: float
            The effect size threshold to use to suggest a new edge. If the
            conditional effect size between two variables is greater than the
            threshold, the algorithm would suggest to add an edge between them.
            And if the effect size for an edge is less than the threshold,
            would suggest to remove the edge.

        ci_test: str or callable (default: None)
            The Conditional Independence test to use. When None, the algorithms
            tries to automatically detect the suitable CI test based on the variable
            types.

        orientation_fn: callable (default: pgmpy.utils.llm_pairwise_orient)
            A function to determine edge orientation. The function should at
            least take two arguments (the names of the two variables) and
            return either a tuple (source, target) representing the directed
            edge from source to target or None representing no edge between the
            variables. Any additional keyword arguments passed to estimate()
            will be forwarded to this function.

            Built-in functions that can be used:

            - `pgmpy.utils.manual_pairwise_orient`: Prompts the user to specify the direction
              between two variables by presenting options and taking input.

            - `pgmpy.utils.llm_pairwise_orient`: Uses a Large Language Model to determine direction.
              Requires additional parameters:

              * variable_descriptions: dict of {var_name: description} for context
              * llm_model: name of the LLM model (default: "gemini/gemini-1.5-flash")
              * system_prompt: optional custom system prompt

            Custom functions can be provided that implement any desired logic
            for determining edge orientation, including using local LLMs or
            domain-specific heuristics.

        orientations: set
            Users can specify a set of edges which would be used as the
            preferred orientation for edges over the output of orientation_fn.

        expert_knowledge: pgmpy.estimators.ExpertKnowledge (default: None)
            Expert knowledge about the causal structure. This can include:
            - forbidden_edges: Edges that should not be present in the final model
            - required_edges: Edges that must be present in the final model (can be removed during pruning)
            - temporal_order: The temporal ordering of variables. Note that explicit orientations
              specified in the 'orientations' parameter will override this temporal ordering.

        use_cache: bool
            If True, the method will cache the results returned by
            `orientation_fn` and reuse it in future calls of the `estimate`
            method instead of calling the `orientation_fn`.

        show_progress: bool (default: True)
            If True, prints info of the running status.

        kwargs: kwargs
            Any additional parameters to pass to the `orientation_fn`.

        Returns
        -------
        pgmpy.base.DAG: A DAG representing the learned causal structure.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.utils import (
        ...     llm_pairwise_orient,
        ...     manual_pairwise_orient,
        ... )
        >>> from pgmpy.estimators import ExpertInLoop
        >>> model = load_model("bnlearn/cancer")
        >>> df = model.simulate(int(1e3))

        >>> # Using manual orientation
        >>> dag = ExpertInLoop(df).estimate(
        ...     effect_size_threshold=0.0001, orientation_fn=manual_pairwise_orient
        ... )

        >>> # Using LLM-based orientation
        >>> variable_descriptions = {
        ...     "Smoker": "A binary variable representing whether a person smokes or not.",
        ...     "Cancer": "A binary variable representing whether a person has cancer.",
        ...     "Xray": "A binary variable representing the result of an X-ray test.",
        ...     "Pollution": "A binary variable representing whether the person is in a high-pollution area or not.",
        ...     "Dyspnoea": "A binary variable representing whether a person has shortness of breath.",
        ... }
        >>> dag = ExpertInLoop(df).estimate(
        ...     effect_size_threshold=0.0001,
        ...     orientation_fn=llm_pairwise_orient,
        ...     variable_descriptions=variable_descriptions,
        ...     llm_model="gemini/gemini-1.5-flash",
        ... )
        >>> dag.edges()
        OutEdgeView([('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea'), ('Pollution', 'Cancer')])

        >>> # Using a custom orientation function
        >>> def my_orientation_func(var1, var2, **kwargs):
        ...     # Custom logic to determine edge orientation
        ...     if var1 == "Pollution" and var2 == "Cancer":
        ...         return ("Pollution", "Cancer")  # Pollution -> Cancer
        ...     elif var1 == "Cancer" and var2 == "Pollution":
        ...         return ("Pollution", "Cancer")  # Pollution -> Cancer
        ...     elif "Smoker" in (var1, var2) and "Cancer" in (var1, var2):
        ...         return ("Smoker", "Cancer")  # Smoker -> Cancer
        ...     # For edges involving Xray, always orient from other variable to Xray
        ...     elif "Xray" in (var1, var2):
        ...         if var1 == "Xray":
        ...             return (var2, var1)
        ...         else:
        ...             return (var1, var2)
        ...     # Default: use alphabetical ordering
        ...     return (var1, var2) if var1 < var2 else (var2, var1)
        pass
        >>> dag = ExpertInLoop(df).estimate(
        ...     effect_size_threshold=0.0001, orientation_fn=my_orientation_func
        ... )
        >>> dag.edges()
        OutEdgeView([('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea'), ('Pollution', 'Cancer')])
        """
        pass

    def _break_cycle(self, dag, u, v, ci_test, effect_size_threshold, pval_threshold):
        """
        Subroutine to break any cycles that get created.

        Parameters
        ----------
        dag: pgmpy.base.DAG
            The current DAG that still doesn't have cycles.

        u, v: hashable object
            The u and v variables that create cycle in `dag` when (u, v) edge is added.

        ci_test: Callable
            The Conditional Independence test to use.
        """
        pass
