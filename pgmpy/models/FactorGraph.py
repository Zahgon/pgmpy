#!/usr/bin/env python3

import itertools
from collections import defaultdict

import numpy as np
from networkx.algorithms import bipartite

from pgmpy.base import UndirectedGraph
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models.DiscreteMarkovNetwork import DiscreteMarkovNetwork


class FactorGraph(UndirectedGraph):
    """
    Class for representing factor graph.

    DiscreteFactor graph is a bipartite graph representing factorization of a function.
    They allow efficient computation of marginal distributions through sum-product
    algorithm.

    A factor graph contains two types of nodes. One type corresponds to random
    variables whereas the second type corresponds to factors over these variables.
    The graph only contains edges between variables and factor nodes. Each factor
    node is associated with one factor whose scope is the set of variables that
    are its neighbors.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data is an edge list.

    Examples
    --------
    Create an empty FactorGraph with no nodes and no edges

    >>> from pgmpy.models import FactorGraph
    >>> G = FactorGraph()

    G can be grown by adding variable nodes as well as factor nodes

    **Nodes:**

    Add a node at a time or a list of nodes.

    >>> G.add_node("a")
    >>> G.add_nodes_from(["a", "b"])
    >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
    >>> G.add_factors(phi1)
    >>> G.add_nodes_from([phi1])

    **Edges:**

    G can also be grown by adding edges.

    >>> G.add_edge("a", phi1)

    or a list of edges

    >>> G.add_edges_from([("a", phi1), ("b", phi1)])
    """

    def __init__(self, ebunch=None):
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.factors = []

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between variable_node and factor_node.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> G.add_nodes_from([phi1])
        >>> G.add_edge("a", phi1)
        """
        pass

    def add_factors(self, *factors, replace=False):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values.

        Parameters
        ----------
        *factor: pgmpy.factors.DiscreteFactor object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        """
        pass

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1)
        >>> G.remove_factors(phi1)
        """
        pass

    def get_cardinality(self, node=None):
        """
        Returns the cardinality of the node

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose cardinality we want. If node is not specified returns a
            dictionary with the given variable as keys and their respective cardinality
            as values.

        Returns
        -------
        int or dict : If node is specified returns the cardinality of the node.
                      If node is not specified returns a dictionary with the given
                      variable as keys and their respective cardinality as values.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> G.add_factors(phi1, phi2)
        >>> dict(G.get_cardinality()) == {"a": 2, "b": 2, "c": 2}
        True

        >>> int(G.get_cardinality("a"))
        2
        """
        pass

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors. In the same time it also updates the cardinalities of all the
        random variables.

        * Check whether bipartite property of factor graph is still maintained
          or not.
        * Check whether factors are associated for all the random variables or not.
        * Check if factors are defined for each factor node or not.
        * Check if cardinality information for all the variables is available or not.
        * Check if cardinality of random variable remains same across all the
          factors.
        """
        pass

    def get_variable_nodes(self):
        """
        Returns variable nodes present in the graph.

        Before calling this method make sure that all the factors are added
        properly.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_factors(phi1, phi2)
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> sorted(G.get_variable_nodes())
        ['a', 'b', 'c']
        """
        pass

    def get_factor_nodes(self):
        """
        Returns factors nodes present in the graph.

        Before calling this method make sure that all the factors are added
        properly.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> rng = np.random.default_rng(42)
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], rng.random(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], rng.random(4))
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_factors(phi1, phi2)
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> sorted(
        ...     G.get_factor_nodes(), key=str
        ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<DiscreteFactor representing phi(a:2, b:2) at 0x...>,
         <DiscreteFactor representing phi(b:2, c:2) at 0x...>]
        """
        pass

    def to_markov_model(self):
        """
        Converts the factor graph into markov model.

        A markov model contains nodes as random variables and edge between
        two nodes imply interaction between them.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> mm = G.to_markov_model()
        """
        pass

    def to_junction_tree(self):
        """
        Create a junction treeo (or clique tree) for a given factor graph.

        For a given factor graph (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of
        edge to other

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> mm = G.to_markov_model()
        """
        pass

    def get_factors(self, node=None):
        """
        Returns the factors that have been added till now to the graph.

        If node is not None, it would return the factor corresponding to the
        given node.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> G.get_factors()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<DiscreteFactor representing phi(a:2, b:2) at 0x...>,
        <DiscreteFactor representing phi(b:2, c:2) at 0x...>]
        >>> G.get_factors(node=phi1)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        <DiscreteFactor representing phi(a:2, b:2) at 0x...>
        """
        pass

    def get_partition_function(self):
        r"""
        Returns the partition function for a given undirected graph.

        A partition function is defined as

        .. math:: \sum_{X}(\prod_{i=1}^{m} \phi_i)

        where m is the number of factors present in the graph
        and X are all the random variables present.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> rng = np.random.default_rng(42)
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], rng.random(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], rng.random(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> G.get_factors()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<DiscreteFactor representing phi(a:2, b:2) at 0x...>,
        <DiscreteFactor representing phi(b:2, c:2) at 0x...>]
        >>> round(float(G.get_partition_function()), 14)
        3.50451083471209
        """
        pass

    def copy(self):
        """
        Returns a copy of the model.

        Returns
        -------
        FactorGraph : Copy of FactorGraph

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(["a", "b", "c"])
        >>> phi1 = DiscreteFactor(["a", "b"], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(["b", "c"], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> G.add_nodes_from([phi1, phi2])
        >>> G.add_edges_from([("a", phi1), ("b", phi1), ("b", phi2), ("c", phi2)])
        >>> G_copy = G.copy()
        >>> G_copy.nodes()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        NodeView(('a', <DiscreteFactor representing phi(a:2, b:2) at 0x...>, 'b',
        <DiscreteFactor representing phi(b:2, c:2) at 0x...>, 'c'))

        """
        pass

    def get_point_mass_message(self, variable, observation):
        """
        Returns a point mass message for the variable given the observed state.

        Parameters
        ----------
        variable: str
            The variable for which the message needs to be computed.
        observation: int
            The observed state of the variable.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = FactorGraph()
        >>> G.add_node("a")
        >>> phi = DiscreteFactor(["a"], [4], np.random.rand(4))
        >>> G.add_factors(phi)
        >>> G.add_edges_from([("a", phi)])
        >>> G.get_point_mass_message("a", 1)
        array([0., 1., 0., 0.])
        """
        pass

    def get_uniform_message(self, variable):
        """
        Returns a uniform message for the given variable

        Parameters
        ----------
        variable: str
            The variable for which the message needs to be computed.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> G = FactorGraph()
        >>> G.add_node("a")
        >>> phi = DiscreteFactor(["a"], [4], np.random.rand(4))
        >>> G.add_factors(phi)
        >>> G.add_edges_from([("a", phi)])
        >>> G.get_uniform_message("a")
        array([0.25, 0.25, 0.25, 0.25])
        """
        pass
