import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.metrics import _BaseSupervisedMetric


class OrientationConfusionMatrix(_BaseSupervisedMetric):
    """
    Computes confusion matrix based metrics for comparing edge orientations in DAGs.

    Treats edge direction as a binary classification problem, conditioned on edges
    that are present in both skeletons (common edges). Only supported for DAGs.

    Parameters
    ----------
    metrics : List[str], optional
        List of metrics to compute. If None, computes all available metrics.

            cm          : Confusion matrix for edge direction among common skeleton edges.
            precision   : Fraction of estimated directed edges with correct orientation (TP / (TP + FP)).
            recall      : Fraction of true directed edges that are correctly oriented (TP / (TP + FN)).
            f1          : Harmonic mean of precision and recall.
            npv         : Fraction of absent estimated directions that are truly absent (TN / (TN + FN)).
            specificity : Fraction of truly absent directions correctly predicted absent (TN / (TN + FP)).

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics.

    Examples
    --------
    >>> from pgmpy.metrics import OrientationConfusionMatrix
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
    >>> cm = OrientationConfusionMatrix()
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> result["precision"]
    1.0
    >>> result["recall"]
    1.0
    >>> result["cm"]  # doctest: +NORMALIZE_WHITESPACE
    Estimated       Est Present  Est Absent
    Actual
    Actual Present            2           0
    Actual Absent             0           2

    Compute only selected metrics:

    >>> cm = OrientationConfusionMatrix(metrics=["precision", "recall"])
    >>> result = cm.evaluate(true_dag, est_dag)
    >>> "cm" in result
    False

    References
    ----------
    .. [1] Bryan Andrews, Joseph Ramsey, Gregory F. Cooper Proceedings of Machine Learning Research,
           PMLR 104:4-21, 2019. https://proceedings.mlr.press/v104/andrews19a.html
    """

    _tags = {
        "name": "orientation_confusion_matrix",
        "requires_true_graph": True,
        "requires_data": False,
        "lower_is_better": False,
        "is_symmetric": False,
        "supported_graph_types": (DAG,),
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
        """Evaluate orientation confusion matrix metrics."""
        pass
