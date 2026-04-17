from collections import deque
from collections.abc import Hashable, Iterable
from typing import Any

import networkx as nx

from pgmpy.base._mixin_roles import _GraphRolesMixin


class _CoreGraph(nx.MultiGraph, _GraphRolesMixin):
    """
    Base graph class for pgmpy.

    This class provides a generalized representation for all graph `edge_types` in pgmpy.
    All specific graph classes (e.g., `DAG`, `PDAG`, ...) inherit from `_CoreGraph`.

    Subclasses are expected to define their own `SUPPORTED_EDGE_TYPES` to restrict the kinds of edges they can store.

    It also provides generalized algorithms and methods that work across all inheriting graph `edge_types`.

    Parameters
    ----------
    ebunch : iterable of tuples, optional
        A list or iterable of edges to add at initialization.

    latents : set of nodes, (default=set())
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    exposures : set, (default=set())
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. Default is an empty set.

    outcomes : set, (default=set())
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. Default is an empty set.

    roles : dict, optional (default=None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).

    Examples
    --------
    Create an empty `_CoreGraph` with no nodes and no edges.

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> edges = [("A", "B", "->"), ("B", "C", "->")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.get_edges(keys=True, data=True)
    [('A', 'B', 0, '->'), ('B', 'C', 0, '->')]

    **Nodes:**

    Add one node,

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_node("A")
    >>> G.nodes
    NodeView(('A',))

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> from pgmpy.base._base import _CoreGraph
    >>> G = _CoreGraph()
    >>> G.add_edge("A", "B", "->")
    >>> G.get_edges(keys=True, data=True)
    [('A', 'B', 0, '->')]

    Remove one edge,

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G.remove_edge("A", "B", "->")
    >>> G.get_edges(keys=True, data=True)
    [('B', 'C', 0, '->'), ('C', 'D', 0, '--')]

    **Exposures, Outcomes, and Latents:**

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(ebunch=edges)

    **Roles:**

    Add node's role.

    >>> G.exposures = "A"
    >>> G.outcomes = "C"
    >>> G.latents = "D"

    Checks for the node's role.

    >>> G.exposures
    {'A'}
    >>> G.outcomes
    {'C'}
    >>> G.latents
    {'D'}

    In addition to 'exposures', 'outcomes', and 'latents', you can add custom roles.

    >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("D", "C", "-o")]
    >>> G = _CoreGraph(ebunch=edges)
    >>> G = G.with_role("Custom_role", "A", inplace=False)
    >>> G = G.with_role("latents", "D", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"], "Custom_role": ["A"]}
    True
    >>> G = G.without_role("Custom_role", "A", inplace=False)
    >>> G.get_role_dict() == {"latents": ["D"]}
    True

    """

    SUPPORTED_EDGE_TYPES = frozenset(["--", "-o", "o-", "->", "<-", "o>", "<o", "<>", "oo"])

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]] = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        latents: set[Hashable] | None = None,
        roles=None,
    ):
        super().__init__()
        if ebunch:
            self._validate_edges(ebunch=ebunch)
            for edge in ebunch:
                if len(edge) == 4:
                    u, v, key, edge_type = edge
                elif len(edge) == 3:
                    u, v, edge_type = edge
                    key = None
                self.add_edge(u, v, edge_type=edge_type, key=key)

        self.exposures = set() if exposures is None else set(exposures)
        self.outcomes = set() if outcomes is None else set(outcomes)
        self.latents = set() if latents is None else set(latents)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        # set the roles to the vertices as networkx attributes
        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    # ----------------------------------------------------------------------
    # Public API (or Public Methods)
    # ----------------------------------------------------------------------

    def add_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str = "->",
        key: Any = None,
        **kwargs,
    ) -> None:
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str (default="->")
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        key : Hashable, optional (default=None)
            Identifier for the edge. If not specified, a generic key will be used
            (usually the lowest unused integer).

        Returns
        -------
        None

        See Also
        --------
        `add_edges_from()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G = _CoreGraph()
        >>> G.add_edge("A", "B", "->")
        >>> G.get_edges(keys=True, data=True)
        [('A', 'B', 0, '->')]

        """
        pass

    def add_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable] | tuple[Hashable, Hashable, Hashable, Hashable]],
        **kwargs,
    ) -> None:
        """
        Add all the edges in ebunch.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]
            [(`u`, `v`, `key`, `edge_type`), (`u`, `v`, `key`, `edge_type`), ...]

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        `add_edge()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph()
        >>> G.add_edges_from(ebunch=edges)
        >>> G.get_edges(keys=True, data=True)
        [('A', 'B', 0, '->'), ('B', 'C', 0, '->')]

        """
        pass

    def remove_edge(
        self,
        u: Hashable,
        v: Hashable,
        edge_type: str = None,
    ) -> None:
        """
        Remove an edge between u and v.

        Parameters
        ----------
        u, v : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str (default=None)
            The type should be None or a value from SUPPORTED_EDGE_TYPES.
            If the type is `None`, remove all edges between `u` and `v`.

        kwargs : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Returns
        -------
        None

        See Also
        --------
        `remove_edges_from()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> G.remove_edge("A", "B", "->")
        >>> G.get_edges(keys=True, data=True)
        [('B', 'C', 0, '->'), ('C', 'D', 0, '--')]

        """
        pass

    def remove_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable, Hashable]],
    ) -> None:
        """
        Remove all the edges in ebunch.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]

        Returns
        -------
        None

        See Also
        --------
        `remove_edge()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->"), ("C", "D", "--")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> remove_edges = [("B", "C", "->"), ("C", "D", "--")]
        >>> G.remove_edges_from(ebunch=remove_edges)
        >>> G.get_edges(keys=True, data=True)
        [('A', 'B', 0, '->')]

        """
        pass

    def copy(self):
        """
        Returns a deep copy of the graph object.

        Parameters
        ----------
        None

        Returns
        -------
        graph: graph object
            A copy of the graph object.

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = G1.copy()
        >>> G2.__class__
        <class 'pgmpy.base._base._CoreGraph'>

        """
        pass

    def get_neighbors(self, node: Hashable, edge_type: str | None = None) -> set[Hashable]:
        """
        Returns a set of neighbors nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str (default=None)
            The type should be None or a value from SUPPORTED_EDGE_TYPES.

        Returns
        -------
        nodes : set
            Set of neighbors nodes.

        See Also
        --------
        `get_parents()`
        `get_children()`
        `get_ancestors()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_neighbors("B", "->")))
        ['C']
        >>> print(sorted(G.get_neighbors("B", "<-")))
        ['A']

        """
        pass

    def get_parents(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of parents nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of parents nodes.

        See Also
        --------
        `get_neighbors()`
        `get_children()`
        `get_ancestors()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_parents("B")))
        ['A']
        >>> print(sorted(G.get_parents("C")))
        ['B']

        """
        pass

    def get_children(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of children nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of children nodes.

        See Also
        --------
        `get_neighbors()`
        `get_parents()`
        `get_ancestors()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_children("A")))
        ['B']
        >>> print(sorted(G.get_children("B")))
        ['C']

        """
        pass

    def get_spouses(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of spouses nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of spouses nodes.

        See Also
        --------
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "<>")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_spouses("B")))
        ['C']
        >>> print(sorted(G.get_spouses("C")))
        ['B']

        """
        pass

    def get_ancestors(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of ancestors nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of ancestors nodes.

        See Also
        --------
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_descendants()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_ancestors("C")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_ancestors("B")))
        ['A', 'B']

        """
        pass

    def get_descendants(self, node: Hashable) -> set[Hashable]:
        """
        Returns a set of descendants nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        Returns
        -------
        nodes : set
            Set of descendants nodes.

        See Also
        --------
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_ancestors()`
        `get_spouses()`
        `get_reachable_nodes()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [("A", "B", "->"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_descendants("A")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_descendants("B")))
        ['B', 'C']

        """
        pass

    def get_reachable_nodes(self, node: Hashable, edge_type: str | None = None) -> set[Hashable]:
        """
        Returns a set of reachable nodes in the graph.

        Parameters
        ----------
        node : Hashable
            Nodes can be, for example, strings or numbers.
            Nodes must be hashable (and not None) Python objects.

        edge_type : str
            Type must be str (and not None) and one of the values in `SUPPORTED_EDGE_TYPES`.

        Returns
        -------
        nodes : set
            Set of reachable nodes.

        See Also
        --------
        `get_neighbors()`
        `get_parents()`
        `get_children()`
        `get_ancestors()`
        `get_spouses()`
        `get_descendants()`

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> edges = [
        ...     ("A", "B", "->"),
        ...     ("B", "C", "->"),
        ...     ("C", "D", "--"),
        ...     ("D", "F", "<>"),
        ... ]
        >>> G = _CoreGraph(ebunch=edges)
        >>> print(sorted(G.get_reachable_nodes("A", "->")))
        ['A', 'B', 'C']
        >>> print(sorted(G.get_reachable_nodes("C", "--")))
        ['C', 'D']
        >>> print(sorted(G.get_reachable_nodes("D", "<>")))
        ['D', 'F']

        """
        pass

    def get_edges(self, keys: bool = False, data: bool = False) -> list[tuple[Any, ...]]:
        """
        Retrieve edges with optional keys and API-formatted edge types.

        Parameters
        ----------
        keys : bool, optional (default=False)
            If True, returns the edge key. Default is False.
        data : bool, optional (default=False)
            If True, returns the edge type as a string (e.g., '->') instead of
            the internal dictionary representation. Default is False.

        Returns
        -------
        list
            A list of edge tuples. The format varies based on parameters:
            * (u, v, key, type) : keys=True, data=True
            * (u, v, type)      : keys=False, data=True
            * (u, v, key)       : keys=True, data=False
            * (u, v)            : keys=False, data=False

        Examples
        --------
        >>> edges = [("A", "B", "->"), ("A", "B", "<>"), ("B", "C", "->")]
        >>> G = _CoreGraph(ebunch=edges)
        >>> G.get_edges(data=True)
        [('A', 'B', '->'), ('A', 'B', '<>'), ('B', 'C', '->')]
        >>> G.get_edges(keys=True, data=True)
        [('A', 'B', 0, '->'), ('A', 'B', 1, '<>'), ('B', 'C', 0, '->')]

        """
        pass

    def get_edge_type(self) -> set:
        """
        Retrieves the list of supported edge types for the instance.

        Returns
        -------
        set

        """
        pass

    # ----------------------------------------------------------------------
    # Internal Methods (or Private Methods)
    # ----------------------------------------------------------------------

    def __eq__(self, other):
        """
        Checks if two graphs are equal. Two graphs are considered equal if they
        have the same nodes, edges, exposures, outcomes, latent variables, and variable roles.

        Parameters
        ----------
        other: graph object
            The other graph to compare with.

        Returns
        -------
        bool:
            True if the graphs are equal, False otherwise.

        Notes
        -----
        This method is expected to be usable without being implemented in a subclass of the graph class.

        Examples
        --------
        >>> from pgmpy.base._base import _CoreGraph
        >>> G1 = _CoreGraph()
        >>> G2 = _CoreGraph()
        >>> G1.__eq__(G2)
        True

        """
        if not isinstance(other, self.__class__):
            return False
        return nx.utils.graphs_equal(self, other) and self.get_role_dict() == other.get_role_dict()

    def _validate_edges(
        self,
        ebunch: (
            Iterable[tuple[Hashable, Hashable, Hashable]] | Iterable[tuple[Hashable, Hashable, Hashable, Hashable]]
        ),
    ):
        """
        Validates the value input by the user, then either raises an error.

        Parameters
        ----------
        ebunch : list of tuples
            [(`u`, `v`, `edge_type`), (`u`, `v`, `edge_type`), ...]

        Notes
        -----
        Helper method that validates the input for
            `add_edge()`,
            `add_edges_from()`,
            `remove_edge()`,
            `remove_edges_from()`.
        """
        pass

    def _from_api_edge_type(
        self,
        edge: tuple[Hashable, Hashable, str] | list[Hashable],
    ) -> dict:
        """
        The `_from_api_edge_type` method converts the user's `edge_type` input into an internal representation.
        """
        pass

    def _to_api_edge_type(
        self,
        u: Hashable,
        v: Hashable,
        markers: dict,
    ) -> str:
        """
        The `_to_api_edge_type` method converts the internal representation into the user's `edge_type` input.
        """
        pass
