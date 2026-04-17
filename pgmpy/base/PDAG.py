import itertools
from collections.abc import Hashable, Iterable

import networkx as nx

from pgmpy import logger
from pgmpy.base._mixin_roles import _GraphRolesMixin


class PDAG(_GraphRolesMixin, nx.DiGraph):
    """
    Class for representing PDAGs (also known as CPDAG). PDAGs are the equivalence classes of
    DAGs and contain both directed and undirected edges.

    Note: In this class, undirected edges are represented using two edges in both direction i.e.
    an undirected edge between X - Y is represented using X -> Y and X <- Y.

    Parameters
    ----------
    directed_ebunch: list, array-like of 2-tuples
        List of directed edges in the PDAG.

    undirected_ebunch: list, array-like of 2-tuples
        List of undirected edges in the PDAG.

    latents: list, array-like
        List of nodes which are latent variables.

    exposures : set, default=set()
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. Default is an empty set.

    outcomes : set, default=set()
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. Default is an empty set.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    >>> from pgmpy.base import PDAG
    >>> pdag = PDAG(
    ...     directed_ebunch=[("A", "C"), ("D", "C")],
    ...     undirected_ebunch=[("B", "A"), ("B", "D")],
    ...     latents=["A"],
    ...     roles={"exposures": ["A"], "outcomes": ["C"]},
    ... )
    >>> sorted(pdag.directed_edges)
    [('A', 'C'), ('D', 'C')]
    >>> sorted(pdag.undirected_edges)
    [('B', 'A'), ('B', 'D')]
    >>> pdag.latents
    {'A'}
    >>> pdag.exposures
    {'A'}
    """

    def __init__(
        self,
        directed_ebunch: list[tuple[Hashable, Hashable]] = [],
        undirected_ebunch: list[tuple[Hashable, Hashable]] = [],
        latents: Iterable[Hashable] = [],
        exposures: set[Hashable] = set(),
        outcomes: set[Hashable] = set(),
        roles=None,
    ):
        self.directed_edges = set(directed_ebunch)
        self.undirected_edges = set(undirected_ebunch)

        super().__init__(
            self.directed_edges.union(self.undirected_edges).union({(Y, X) for (X, Y) in self.undirected_edges})
        )
        self.latents = set(latents)
        self.exposures = set(exposures)
        self.outcomes = set(outcomes)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def all_neighbors(self, node):
        """
        Returns a set of all neighbors of a node in the PDAG. This includes both directed and undirected edges.

        Parameters
        ----------
        node: any hashable python object
            The node for which to get the neighboring nodes.

        Returns
        -------
        set: A set of neighboring nodes.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "C"), ("D", "C")],
        ...     undirected_ebunch=[("B", "A"), ("B", "D")],
        ... )
        >>> sorted(pdag.all_neighbors("A"))
        ['B', 'C']
        """
        pass

    def directed_children(self, node):
        """
        Returns a set of children of node such that there is a directed edge from `node` to child.
        """
        pass

    def directed_parents(self, node):
        """
        Returns a set of parents of node such that there is a directed edge from the parent to `node`.
        """
        pass

    def has_directed_edge(self, u, v):
        """
        Returns True if there is a directed edge u -> v in the PDAG.
        """
        pass

    def has_undirected_edge(self, u, v):
        """
        Returns True if there is an undirected edge u - v in the PDAG.
        """
        pass

    def undirected_neighbors(self, node):
        """
        Returns a set of neighboring nodes such that all of them have an undirected edge with `node`.

        Parameters
        ----------
        node: any hashable python object
            The node for which to get the undirected neighboring nodes.

        Returns
        -------
        set: A set of neighboring nodes.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "C"), ("D", "C")],
        ...     undirected_ebunch=[("B", "A"), ("B", "D")],
        ... )
        >>> pdag.undirected_neighbors("A")
        {'B'}
        """
        pass

    def chain_component(self, node: Hashable) -> set[Hashable]:
        """
        Returns the chain component of `node`, i.e. all nodes reachable
        through undirected edges.
        """
        pass

    def has_semidirected_path(
        self,
        source: Hashable,
        target: Hashable,
        blocked_nodes: Iterable[Hashable] | None = None,
        ignore_direct_edge: bool = False,
    ) -> bool:
        """
        Returns True if there exists a semi-directed path from `source` to `target`.

        Semi-directed paths follow directed edges in their forward direction and
        can traverse undirected edges in either direction.

        Parameters
        ----------
        source, target: hashable
            The endpoints of the path.

        blocked_nodes: iterable, optional
            Nodes that are not allowed on the path.

        ignore_direct_edge: bool
            If True, ignores the direct edge `source -> target` when checking for
            a path.
        """
        pass

    def has_acyclic_extension(self) -> bool:
        """
        Returns True if the PDAG admits an acyclic DAG extension.
        """
        pass

    def is_adjacent(self, u, v):
        """
        Returns True if there is an edge between u and v. This can be either of u - v, u -> v, or u <- v.
        """
        pass

    def copy(self):
        """
        Returns a copy of the object instance.

        Returns
        -------
        Copy of PDAG: pgmpy.dag.PDAG
            Returns a copy of self.
        """
        pass

    def _directed_graph(self):
        """
        Returns a subgraph containing only directed edges.
        """
        pass

    def orient_undirected_edge(self, u, v, inplace=False):
        """
        Orients an undirected edge u - v as u -> v.

        Parameters
        ----------
        u, v: Any hashable python objects
            The node names.

        inplace: boolean (default=False)
            If True, the PDAG object is modified inplace, otherwise a new modified copy is returned.

        Returns
        -------
        None or pgmpy.base.PDAG: The modified PDAG object.
            If inplace=True, returns None and the object itself is modified.
            If inplace=False, returns a PDAG object.
        """
        pass

    def is_clique(self, nodes: Iterable) -> bool:
        """
        Checks if a set of nodes forms a clique. A clique is a subgraph
        where every pair of nodes is connected by an edge (fully connected).

        Parameters
        ----------
        nodes: Iterable
            The set of nodes to be checked for clique formation.
        """
        pass

    def _check_new_unshielded_collider(self, u, v):
        """
        Tests if orienting an undirected edge u - v as u -> v creates new unshielded V-structures in the PDAG.

        Checks whether v has any directed parents other than u that are not adjacent to u.

        Returns
        -------
        True, if the orientation u -> v would lead to creation of a new V-structure.
        False, if no new V-structures are formed.
        """
        pass

    def apply_meeks_rules(self, apply_r4=False, inplace=False, debug=False):
        """
        Applies the Meek's rules to orient the undirected edges of a PDAG to return a CPDAG.

        Parameters
        ----------
        apply_r4: boolean (default=False)
            If True, applies Rules 1 - 4 of Meek's rules.
            If False, applies only Rules 1 - 3.

        inplace: boolean (default=False)
            If True, the PDAG object is modified inplace, otherwise a new modified copy is returned.

        debug: boolean (default=False)
            If True, prints the rules being applied to the PDAG.

        Returns
        -------
        None or pgmpy.base.PDAG: The modified PDAG object.
            If inplace=True, returns None and the object itself is modified.
            If inplace=False, returns a PDAG object.

        Examples
        --------
        >>> from pgmpy.base import PDAG
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "B")], undirected_ebunch=[("B", "C"), ("C", "B")]
        ... )
        >>> pdag = pdag.apply_meeks_rules()
        >>> sorted(pdag.directed_edges)
        [('A', 'B'), ('B', 'C')]
        """
        pass

    def calibrate_directed_undirected_edges(self) -> None:
        """
        Iterates through existing edges to correctly assign directed
        and undirected edges.
        """
        pass

    def to_cpdag(self):
        """
        Returns the CPDAG corresponding to one DAG extension of the PDAG.
        """
        pass

    def to_dag(self):
        """
        Returns one possible DAG which is represented using the PDAG.

        Returns
        -------
        pgmpy.base.DAG: Returns an instance of DAG.

        Examples
        --------
        >>> pdag = PDAG(
        ...     directed_ebunch=[("A", "B"), ("C", "B")],
        ...     undirected_ebunch=[("C", "D"), ("D", "A")],
        ... )
        >>> dag = pdag.to_dag()
        >>> sorted(dag.edges())
        [('A', 'B'), ('C', 'B'), ('D', 'A'), ('D', 'C')]

        References
        ----------
        [1] Dor, Dorit, and Michael Tarsi.
          "A simple algorithm to construct a consistent extension of a partially oriented graph."
            Technicial Report R-185, Cognitive Systems Laboratory, UCLA (1992): 45.
        """
        pass

    def to_graphviz(self) -> object:
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/alarm")
        >>> model.to_graphviz()  # doctest: +ELLIPSIS
        <AGraph ...
        """
        pass

    def __eq__(self, other):
        """
        Checks if two PDAGs are equal. Two PDAGs are considered equal if they
        have the same nodes, edges, latent variables, and variable roles.

        Parameters
        ----------
        other: PDAG object
            The other PDAG to compare with.

        Returns
        -------
        bool
            True if the PDAGs are equal, False otherwise.
        """
        if not isinstance(other, PDAG):
            return False

        return (
            set(self.nodes()) == set(other.nodes())
            and set(self.directed_edges) == set(other.directed_edges)
            and set(self.undirected_edges) == set(other.undirected_edges)
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )
