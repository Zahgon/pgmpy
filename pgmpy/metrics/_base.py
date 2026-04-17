import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects


class _BaseSupervisedMetric(BaseObject):
    """
    Base class for all metric classes in pgmpy that require ground truth causal graph.
    """

    def evaluate(self, true_causal_graph, est_causal_graph, **kwargs):
        """
        Evaluate the metric by comparing the true causal graph with the estimated causal graph.

        Parameters
        ----------
        true_causal_graph: Instance of type pgmpy.base
            The ground truth causal graph.

        est_causal_graph: Instance of type pgmpy.base
            The estimated causal graph.
        """
        pass

    def __call__(self, true_causal_graph, est_causal_graph, **kwargs):
        return self.evaluate(
            true_causal_graph=true_causal_graph,
            est_causal_graph=est_causal_graph,
            **kwargs,
        )


class _BaseUnsupervisedMetric(BaseObject):
    """
    Base class for all metric classes in pgmpy that do not require ground truth causal graph.
    """

    def evaluate(self, X, causal_graph, **kwargs):
        """
        Evaluate the metric by comparing the causal graph with the data.

        Parameters
        ----------
        X: pandas.DataFrame
            The data used for evaluation.

        causal_graph: Instance of type pgmpy.base
            The causal graph to be evaluated.
        """
        pass

    def __call__(self, X, causal_graph, **kwargs):
        return self.evaluate(X=X, causal_graph=causal_graph, **kwargs)


def get_metrics(**kwargs):
    """
    Get metric classes matching the given tag filters.

    Parameters
    ----------
    **kwargs
        Keyword arguments specifying tag filters to be passed to
        :func:`skbase.lookup.all_objects` via its ``filter_tags`` parameter.

    Returns
    -------
    Type[BaseObject] or list[Type[BaseObject]]
        Metric class(es) corresponding to the given tag filters.

    Raises
    ------
    ValueError
        If no metric class matching the given tag filters is found.
    """
    pass
