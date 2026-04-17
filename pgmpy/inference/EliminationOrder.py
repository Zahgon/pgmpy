from abc import abstractmethod
from itertools import combinations

import numpy as np
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.models import DiscreteBayesianNetwork


class BaseEliminationOrder:
    """
    Init method for the base class of Elimination Orders.

    Parameters
    ----------
    model: DiscreteBayesianNetwork instance
        The model on which we want to compute the elimination orders.
    """

    def __init__(self, model):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise ValueError("Model should be a DiscreteBayesianNetwork instance")
        self.bayesian_model = model.copy()
        self.moralized_model = self.bayesian_model.moralize()

    @abstractmethod
    def cost(self, node):
        """
        The cost function to compute the cost of elimination of each node.
        This method is just a dummy and returns 0 for all the nodes. Actual cost functions
        are implemented in the classes inheriting BaseEliminationOrder.

        Parameters
        ----------
        node: string, any hashable python object.
            The node whose cost is to be computed.
        """
        pass

    def get_elimination_order(self, nodes=None, show_progress=True):
        """
        Returns the optimal elimination order based on the cost function.
        The node having the least cost is removed first.

        Parameters
        ----------
        nodes: list, tuple, set (array-like)
            The variables which are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference.EliminationOrder import WeightedMinFill
        >>> rng = np.random.default_rng(42)
        >>> model = DiscreteBayesianNetwork(
        ...     [
        ...         ("c", "d"),
        ...         ("d", "g"),
        ...         ("i", "g"),
        ...         ("i", "s"),
        ...         ("s", "j"),
        ...         ("g", "l"),
        ...         ("l", "j"),
        ...         ("j", "h"),
        ...         ("g", "h"),
        ...     ]
        ... )
        >>> cpd_c = TabularCPD("c", 2, rng.random((2, 1)))
        >>> cpd_d = TabularCPD("d", 2, rng.random((2, 2)), ["c"], [2])
        >>> cpd_g = TabularCPD("g", 3, rng.random((3, 4)), ["d", "i"], [2, 2])
        >>> cpd_i = TabularCPD("i", 2, rng.random((2, 1)))
        >>> cpd_s = TabularCPD("s", 2, rng.random((2, 2)), ["i"], [2])
        >>> cpd_j = TabularCPD("j", 2, rng.random((2, 4)), ["l", "s"], [2, 2])
        >>> cpd_l = TabularCPD("l", 2, rng.random((2, 3)), ["g"], [3])
        >>> cpd_h = TabularCPD("h", 2, rng.random((2, 6)), ["g", "j"], [3, 2])
        >>> model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)
        >>> WeightedMinFill(model).get_elimination_order(["c", "d", "g", "l", "s"])
        ['c', 'd', 's', 'l', 'g']
        >>> WeightedMinFill(model).get_elimination_order(["c", "d", "g", "l", "s"])
        ['c', 'd', 's', 'l', 'g']
        >>> WeightedMinFill(model).get_elimination_order(["c", "d", "g", "l", "s"])
        ['c', 'd', 's', 'l', 'g']
        """
        pass

    def fill_in_edges(self, node):
        """
        Return edges needed to be added to the graph if a node is removed.

        Parameters
        ----------
        node: string (any hashable python object)
            Node to be removed from the graph.
        """
        pass


class WeightedMinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        Cost function for WeightedMinFill.
        The cost of eliminating a node is the sum of weights of the edges that need to
        be added to the graph due to its elimination, where a weight of an edge is the
        product of the weights, domain cardinality, of its constituent vertices.
        """
        pass


class MinNeighbors(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of eliminating a node is the number of neighbors it has in the
        current graph.
        """
        pass


class MinWeight(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of eliminating a node is the product of weights, domain cardinality,
        of its neighbors.
        """
        pass


class MinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of eliminating a node is the number of edges that need to be added
        (fill in edges) to the graph due to its elimination
        """
        pass
