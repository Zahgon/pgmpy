import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG, PDAG
from pgmpy.metrics import _BaseSupervisedMetric


class AdjacencyConfusionMatrix(_BaseSupervisedMetric):
    """
    Computes confusion matrix based metrics for comparing causal graph skeletons.

    Treats edge presence/absence in the undirected skeleton as a binary classification
    problem and computes confusion matrix based metrics.

    Parameters
    ----------
    metrics : List[str], optional
        List of metrics to compute. If None, computes all available metrics.

            cm          : Confusion matrix for skeleton edge presence.
            precision   : Fraction of estimated skeleton edges that are correct (TP / (TP + FP)).
            recall      : Fraction of true skeleton edges that are recovered (TP / (TP + FN)).
            f1          : Harmonic mean of precision and recall.
            npv         : Fraction of absent estimated edges that are truly absent (TN / (TN + FN)).
            specificity : Fraction of truly absent edges correctly predicted absent (TN / (TN + FP)).

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics.

    Examples
    --------
    >>> from pgmpy.metrics import AdjacencyConfusionMatrix
    >>> from pgmpy.base import DAG
    >>> true_dag = DAG(
    ...     [
    ...         ("Smoking", "Lung_Cancer"),
    ...         ("Smoking", "Heart_Disease"),
    ...         ("Age", "Heart_Disease"),
    ...         ("Age", "Lung_Cancer"),
    ...     ]
    ... )
    >>> est_dag = DAG([("Smoking", "Lung_Cancer"), ("Age", "Heart_Disease")])
    >>> cm = AdjacencyConfusionMatrix()
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> result["precision"]
    1.0
    >>> result["recall"]
    0.5
    >>> result["cm"]  # doctest: +NORMALIZE_WHITESPACE
    Estimated       Est Present  Est Absent
    Actual
    Actual Present            2           2
    Actual Absent             0           2

    Compute only selected metrics:

    >>> cm = AdjacencyConfusionMatrix(metrics=["precision", "recall", "f1"])
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> "f1" in result
    True
    >>> "npv" in result
    False

    References
    ----------
    .. [1] Petersen, A. H. (2025). Are you doing better than random guessing? a call for using negative controls
           when evaluating causal discovery algorithms. Proceedings of the Forty-First Conference on Uncertainty
           in Artificial Intelligence. Rio de Janeiro, Brazil: JMLR.org. https://arxiv.org/abs/2412.10039

    """

    _tags = {
        "name": "adjacency_confusion_matrix",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG, PDAG),
    }

    def __init__(self, metrics: list[str] | None = None):
        self.metrics = metrics or [
            "cm",
            "precision",
            "recall",
            "f1",
            "npv",
            "specificity",
        ]
        super().__init__()

    def _evaluate(self, true_causal_graph, est_causal_graph):
        """Evaluate adjacency confusion matrix metrics."""
        pass
