#!/usr/bin/env python3
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
from networkx.algorithms.components import connected_components

from pgmpy.base import UndirectedGraph
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.utils import compat_fns


class DiscreteMarkovNetwork(UndirectedGraph):
    """
    Base class for Markov Model.

    A DiscreteMarkovNetwork stores nodes and edges with potentials

    DiscreteMarkovNetwork holds undirected edges.

    Parameters
    ----------
    data : input graph
        Data to initialize graph.  If data=None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.

    Examples
    --------
    Create an empty Markov Model with no nodes and no edges.

    >>> from pgmpy.models import DiscreteMarkovNetwork
    >>> G = DiscreteMarkovNetwork()

    G can be grown in several ways.

    **Nodes:**

    Add one node at a time:

    >>> G.add_node("a")

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(["a", "b"])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge("a", "b")

    a list of edges,

    >>> G.add_edges_from([("a", "b"), ("b", "c")])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically.  There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> "a" in G  # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3
    """

    def __init__(self, ebunch=None, latents=[]):
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.factors = []
        self.latents = latents

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
            Nodes can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> G = DiscreteMarkovNetwork()
        >>> G.add_nodes_from(["Alice", "Bob", "Charles"])
        >>> G.add_edge("Alice", "Bob")
        """
        pass

    def add_factors(self, *factors):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values

        Parameters
        ----------
        *factor: pgmpy.factors.factors object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = DiscreteMarkovNetwork(
        ...     [
        ...         ("Alice", "Bob"),
        ...         ("Bob", "Charles"),
        ...         ("Charles", "Debbie"),
        ...         ("Debbie", "Alice"),
        ...     ]
        ... )
        >>> factor = DiscreteFactor(
        ...     ["Alice", "Bob"], cardinality=[3, 2], values=np.random.rand(6)
        ... )
        >>> student.add_factors(factor)
        """
        pass

    def get_factors(self, node=None):
        """
        Returns all the factors containing the node. If node is not specified
        returns all the factors that have been added till now to the graph.

        Parameters
        ----------
        node: any hashable python object (optional)
           The node whose factor we want. If node is not specified

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = DiscreteMarkovNetwork([("Alice", "Bob"), ("Bob", "Charles")])
        >>> factor1 = DiscreteFactor(
        ...     ["Alice", "Bob"], cardinality=[2, 2], values=np.random.rand(4)
        ... )
        >>> factor2 = DiscreteFactor(
        ...     ["Bob", "Charles"], cardinality=[2, 3], values=np.ones(6)
        ... )
        >>> student.add_factors(factor1, factor2)
        >>> student.get_factors()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<DiscreteFactor representing phi(Alice:2, Bob:2) at 0x...>,
        <DiscreteFactor representing phi(Bob:2, Charles:3) at 0x...>]
        >>> student.get_factors("Alice")  # doctest: +ELLIPSIS
        [<DiscreteFactor representing phi(Alice:2, Bob:2) at 0x...>]
        """
        pass

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = DiscreteMarkovNetwork([("Alice", "Bob"), ("Bob", "Charles")])
        >>> factor = DiscreteFactor(
        ...     ["Alice", "Bob"], cardinality=[2, 2], values=np.random.rand(4)
        ... )
        >>> student.add_factors(factor)
        >>> student.remove_factors(factor)
        """
        pass

    def get_cardinality(self, node=None):
        """
        Returns the cardinality of the node. If node is not specified returns
        a dictionary with the given variable as keys and their respective cardinality
        as values.

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose cardinality we want. If node is not specified returns a
            dictionary with the given variable as keys and their respective cardinality
            as values.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = DiscreteMarkovNetwork([("Alice", "Bob"), ("Bob", "Charles")])
        >>> factor = DiscreteFactor(
        ...     ["Alice", "Bob"], cardinality=[2, 2], values=np.random.rand(4)
        ... )
        >>> student.add_factors(factor)
        >>> int(student.get_cardinality(node="Alice"))
        2
        >>> {k: int(v) for k, v in student.get_cardinality().items()}
        {'Alice': 2, 'Bob': 2}
        """
        pass

    @property
    def states(self):
        """
        Returns a dictionary mapping each node to its list of possible states.

        Returns
        -------
        state_dict: dict
            Dictionary of nodes to possible states
        """
        pass

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors -

        * Checks if the cardinalities of all the variables are consistent across all the factors.
        * Factors are defined for all the random variables.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        pass

    def to_factor_graph(self):
        """
        Converts the Markov Model into Factor Graph.

        A Factor Graph contains two types of nodes. One type corresponds to
        random variables whereas the second type corresponds to factors over
        these variables. The graph only contains edges between variables and
        factor nodes. Each factor node is associated with one factor whose
        scope is the set of variables that are its neighbors.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = DiscreteMarkovNetwork([("Alice", "Bob"), ("Bob", "Charles")])
        >>> factor1 = DiscreteFactor(["Alice", "Bob"], [3, 2], np.random.rand(6))
        >>> factor2 = DiscreteFactor(["Bob", "Charles"], [2, 2], np.random.rand(4))
        >>> student.add_factors(factor1, factor2)
        >>> factor_graph = student.to_factor_graph()
        """
        pass

    def triangulate(self, heuristic="H6", order=None, inplace=False):
        """
        Triangulate the graph.

        If order of deletion is given heuristic algorithm will not be used.

        Parameters
        ----------
        heuristic: H1 | H2 | H3 | H4 | H5 | H6
            The heuristic algorithm to use to decide the deletion order of
            the variables to compute the triangulated graph.
            Let X be the set of variables and X(i) denotes the i-th variable.

            * S(i) - The size of the clique created by deleting the variable.
            * E(i) - Cardinality of variable X(i).
            * M(i) - Maximum size of cliques given by X(i) and its adjacent nodes.
            * C(i) - Sum of size of cliques given by X(i) and its adjacent nodes.

            The heuristic algorithm decide the deletion order if this way:

            * H1 - Delete the variable with minimal S(i).
            * H2 - Delete the variable with minimal S(i)/E(i).
            * H3 - Delete the variable with minimal S(i) - M(i).
            * H4 - Delete the variable with minimal S(i) - C(i).
            * H5 - Delete the variable with minimal S(i)/M(i).
            * H6 - Delete the variable with minimal S(i)/C(i).

        order: list, tuple (array-like)
            The order of deletion of the variables to compute the triagulated
            graph. If order is given heuristic algorithm will not be used.

        inplace: True | False
            if inplace is true then adds the edges to the object from
            which it is called else returns a new object.

        References
        ----------
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.3607

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = DiscreteMarkovNetwork()
        >>> G.add_nodes_from(["x1", "x2", "x3", "x4", "x5", "x6", "x7"])
        >>> G.add_edges_from(
        ...     [
        ...         ("x1", "x3"),
        ...         ("x1", "x4"),
        ...         ("x2", "x4"),
        ...         ("x2", "x5"),
        ...         ("x3", "x6"),
        ...         ("x4", "x6"),
        ...         ("x4", "x7"),
        ...         ("x5", "x7"),
        ...     ]
        ... )
        >>> phi = [
        ...     DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in G.edges()
        ... ]
        >>> G.add_factors(*phi)
        >>> G_chordal = G.triangulate()
        """
        pass

    def to_junction_tree(self):
        """
        Creates a junction tree (or clique tree) for a given markov model.

        For a given markov model (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> mm = DiscreteMarkovNetwork()
        >>> mm.add_nodes_from(["x1", "x2", "x3", "x4", "x5", "x6", "x7"])
        >>> mm.add_edges_from(
        ...     [
        ...         ("x1", "x3"),
        ...         ("x1", "x4"),
        ...         ("x2", "x4"),
        ...         ("x2", "x5"),
        ...         ("x3", "x6"),
        ...         ("x4", "x6"),
        ...         ("x4", "x7"),
        ...         ("x5", "x7"),
        ...     ]
        ... )
        >>> phi = [
        ...     DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in mm.edges()
        ... ]
        >>> mm.add_factors(*phi)
        >>> junction_tree = mm.to_junction_tree()
        """
        pass

    def markov_blanket(self, node):
        """
        Returns a markov blanket for a random variable.

        Markov blanket is the neighboring nodes of the given node.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> mm = DiscreteMarkovNetwork()
        >>> mm.add_nodes_from(["x1", "x2", "x3", "x4", "x5", "x6", "x7"])
        >>> mm.add_edges_from(
        ...     [
        ...         ("x1", "x3"),
        ...         ("x1", "x4"),
        ...         ("x2", "x4"),
        ...         ("x2", "x5"),
        ...         ("x3", "x6"),
        ...         ("x4", "x6"),
        ...         ("x4", "x7"),
        ...         ("x5", "x7"),
        ...     ]
        ... )
        >>> mm.markov_blanket("x1")  # doctest: +ELLIPSIS
        <dict_keyiterator object at 0x...>
        """
        pass

    def get_local_independencies(self, latex=False):
        r"""
        Returns all the local independencies present in the markov model.

        Local independencies are the independence assertion in the form of
        .. math:: {X \perp W - {X} - MB(X) | MB(X)}
        where MB is the markov blanket of all the random variables in X

        Parameters
        ----------
        latex: boolean
            If latex=True then latex string of the indepedence assertion would
            be created

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> mm = DiscreteMarkovNetwork()
        >>> mm.add_nodes_from(["x1", "x2", "x3", "x4", "x5", "x6", "x7"])
        >>> mm.add_edges_from(
        ...     [
        ...         ("x1", "x3"),
        ...         ("x1", "x4"),
        ...         ("x2", "x4"),
        ...         ("x2", "x5"),
        ...         ("x3", "x6"),
        ...         ("x4", "x6"),
        ...         ("x4", "x7"),
        ...         ("x5", "x7"),
        ...     ]
        ... )
        >>> independencies = mm.get_local_independencies()
        >>> assertions = independencies.get_assertions()
        >>> len(assertions)
        7
        >>> mm.get_local_independencies()  # doctest: +SKIP
        (x1 ⟂ x7, x5, x2, x6 | x3, x4)
        (x2 ⟂ x7, x6, x1, x3 | x5, x4)
        (x3 ⟂ x7, x5, x2, x4 | x6, x1)
        (x4 ⟂ x5, x3 | x7, x6, x1, x2)
        (x5 ⟂ x6, x1, x3, x4 | x7, x2)
        (x6 ⟂ x7, x5, x1, x2 | x3, x4)
        (x7 ⟂ x6, x1, x2, x3 | x5, x4)
        """
        pass

    def to_bayesian_model(self):
        """
        Creates a Bayesian Model which is a minimum I-Map for this Markov Model.

        The ordering of parents may not remain constant. It would depend on the
        ordering of variable in the junction tree (which is not constant) all the
        time. Also, if the model is not connected, the connected components are
        treated as separate models, converted, and then joined together.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> mm = DiscreteMarkovNetwork()
        >>> mm.add_nodes_from(["x1", "x2", "x3", "x4", "x5", "x6", "x7"])
        >>> mm.add_edges_from(
        ...     [
        ...         ("x1", "x3"),
        ...         ("x1", "x4"),
        ...         ("x2", "x4"),
        ...         ("x2", "x5"),
        ...         ("x3", "x6"),
        ...         ("x4", "x6"),
        ...         ("x4", "x7"),
        ...         ("x5", "x7"),
        ...     ]
        ... )
        >>> phi = [
        ...     DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in mm.edges()
        ... ]
        >>> mm.add_factors(*phi)
        >>> bm = mm.to_bayesian_model()
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
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = DiscreteMarkovNetwork()
        >>> G.add_nodes_from(["x1", "x2", "x3", "x4", "x5", "x6", "x7"])
        >>> rng = np.random.default_rng(42)
        >>> G.add_edges_from(
        ...     [
        ...         ("x1", "x3"),
        ...         ("x1", "x4"),
        ...         ("x2", "x4"),
        ...         ("x2", "x5"),
        ...         ("x3", "x6"),
        ...         ("x4", "x6"),
        ...         ("x4", "x7"),
        ...         ("x5", "x7"),
        ...     ]
        ... )
        >>> phi = [DiscreteFactor(edge, [2, 2], rng.random(4)) for edge in G.edges()]
        >>> G.add_factors(*phi)
        >>> round(float(G.get_partition_function()), 3)
        0.82
        """
        pass

    def copy(self):
        """
        Returns a copy of this Markov Model.

        Returns
        -------
        DiscreteMarkovNetwork: Copy of this Markov model.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> G = DiscreteMarkovNetwork()
        >>> G.add_nodes_from([("a", "b"), ("b", "c")])
        >>> G.add_edge(("a", "b"), ("b", "c"))
        >>> G_copy = G.copy()
        >>> G_copy.edges()
        EdgeView([(('a', 'b'), ('b', 'c'))])
        >>> sorted(G_copy.nodes())
        [('a', 'b'), ('b', 'c')]
        >>> factor = DiscreteFactor(
        ...     [("a", "b")], cardinality=[3], values=np.random.rand(3)
        ... )
        >>> G.add_factors(factor)
        >>> G.get_factors()  # doctest: +ELLIPSIS
        [<DiscreteFactor representing phi(('a', 'b'):3) at 0x...>]
        >>> G_copy.get_factors()
        []
        """
        pass
