import networkx as nx

from pgmpy.base import DAG
from pgmpy.identification import Adjustment, _BaseIdentification
from pgmpy.utils.sets import _powerset


class Frontdoor(_BaseIdentification):
    """
    Given a causal graph, finds the set of variables satisfying frontdoor criterion.

    Given a causal graph with `exposures` and `outcomes` roles specified, the
    `FrontdoorIdentification` class provides methods to find the set of variables
    satisfying the frontdoor criterion with respect to `exposures` and `outcomes` in
    the causal graph.

    Parameters
    ----------
    variant: all | None
        If all, returns all possible frontdoor identification causal graphs.
        If None, returns one at random.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> dag = DAG(
    ...     ebunch=[
    ...         ("X", "M"),
    ...         ("M", "Y"),
    ...         ("U", "X"),
    ...         ("U", "Y"),
    ...     ],
    ...     roles={"exposures": "X", "outcomes": "Y"},
    ... )
    >>> dag_with_adj, is_identified = Frontdoor().identify(dag)
    >>> dag_with_adj.get_role_dict()
    {'exposures': ['X'], 'outcomes': ['Y'], 'frontdoor': ['M']}
    >>> Frontdoor().validate(dag_with_adj)
    True
    """

    def __init__(self, variant=None):
        self.supported_graph_types = (DAG,)
        self.variant = variant

    def _identify(self, causal_graph):
        pass

    @staticmethod
    def _is_valid_adjustment_set(causal_graph, X, Y, Z):
        pass

    def _validate(self, causal_graph):
        """ """
        pass
