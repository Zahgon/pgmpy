import itertools

import networkx as nx

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.identification import _BaseIdentification
from pgmpy.utils.sets import _powerset


class Adjustment(_BaseIdentification):
    """
    Given a causal graph, finds the adjustment set.

    This class implements a few variants for computing adjustment sets for
    identifying the total causal effect of the variables in the `exposures`
    role on the variables in the `outcomes` role. Additionally, it provides methods to check if the
    current set of variables with role `adjustment` satisfy the backdoor
    criterion and to compute the backdoor adjustment formula.

    Parameters
    ----------
    variant: str
        The variant of backdoor identification to use. Default is 'minimal'.

        - 'all': Returns all adjustment sets that satisfy the backdoor criterion.
        - 'minimal': Returns the smallest adjustment set.
        - 'minimal_variance': Returns the adjustment set for which estimators achieve minimal variance.

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> dag = DAG(
    ...     ebunch=[
    ...         ("x1", "y1"),
    ...         ("x1", "z1"),
    ...         ("z1", "z2"),
    ...         ("z2", "x2"),
    ...         ("y2", "z2"),
    ...     ],
    ...     roles={"exposures": "x1", "outcomes": "y1"},
    ... )
    >>> dag_with_adj, success = Adjustment(variant="minimal").identify(dag)
    >>> roles = dag_with_adj.get_role_dict()
    >>> roles["exposures"]
    ['x1']
    >>> roles["outcomes"]
    ['y1']
    >>> Adjustment(variant="minimal").validate(dag_with_adj)
    True

    References
    ----------
    [1] Perkovi, Emilija, et al. "Complete graphical characterization and
        construction of adjustment sets in Markov equivalence classes of ancestral
        graphs." Journal of Machine Learning Research.
    [2] Witte, Janine, et al. "On efficient adjustment in causal graphs."
        Journal of Machine Learning Research.
    """

    def __init__(self, variant="minimal"):
        self.variant = variant
        if self.variant in ("minimal", "all"):
            self.supported_graph_types = (DAG, PDAG, ADMG, MAG)
        elif self.variant == "minimal_variance":
            self.supported_graph_types = (DAG, PDAG)

    def _get_proper_backdoor_graph(self, causal_graph, inplace=False):
        """
        Returns a proper backdoor graph of the `causal_graph`.

        For a `causal_graph` with variable roles `exposures` and `outcomes`
        defined, returns it's proper backdoor graph. A proper backdoor graph is
        a graph which removes the first edge of every proper causal path from
        `exposures` to `outcomes`.

        Parameters
        ----------
        causal_graph: pgmpy.base.DAG, pgmpy.base.PDAG, pgmpy.base.ADMG, or pgmpy.base.MAG
            The causal graph for which the proper backdoor graph is to be computed.

        inplace: boolean
            If inplace is True, modifies the object itself. Otherwise returns
            a modified copy of self.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> from pgmpy.identification import Adjustment
        >>> dag = DAG(
        ...     ebunch=[
        ...         ("x1", "y1"),
        ...         ("x1", "z1"),
        ...         ("z1", "z2"),
        ...         ("z2", "x2"),
        ...         ("y2", "z2"),
        ...     ],
        ...     roles={"exposures": "x1", "outcomes": "y1"},
        ... )
        >>> dag_proper = Adjustment()._get_proper_backdoor_graph(dag, inplace=False)
        >>> list(dag_proper.edges())
        [('x1', 'z1'), ('z1', 'z2'), ('z2', 'x2'), ('y2', 'z2')]

        References
        ----------
        [1] Perkovic, Emilija, et al. "Complete graphical characterization and
            construction of adjustment sets in Markov equivalence classes of
            ancestral graphs." The Journal of Machine Learning Research.
        """
        pass

    def _identify(self, causal_graph):
        """
        Identify adjustment sets using the backdoor criterion.

        Parameters
        ----------
        causal_graph: DAG | PDAG | ADMG | MAG | PAG
            The causal graph for which the adjustment sets are to be identified.

        Returns
        -------
        causal_graph: DAG | PDAG | ADMG | MAG | PAG
            The causal graph with the identified adjustment set added as role `adjustment`.

        success: bool
            True if the identification was successful, False otherwise.
        """
        pass

    def _validate(self, causal_graph):
        """
        Validate the causal graph for backdoor identification.

        Given a `causal_graph` with variable roles `exposures`, `outcomes`, and
        `adjustment` defined, this method checks if the given `adjustment` set
        is valid.

        Parameters
        ----------
        causal_graph: DAG | PDAG | ADMG | MAG | PAG
            The causal graph to validate.

        Returns
        -------
        bool:
            True if the `adjustment` set is valid, False otherwise.
        """
        pass
