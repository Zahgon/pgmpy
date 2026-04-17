from collections import deque
from collections.abc import Hashable, Iterable

import networkx as nx
import numpy as np

from pgmpy.base._mixin_roles import _GraphRolesMixin
from pgmpy.utils.parser import parse_dagitty


class AncestralBase(nx.Graph, _GraphRolesMixin):
    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] = set(),
        exposures: set[Hashable] = set(),
        outcomes: set[Hashable] = set(),
        roles=None,
    ):
        """
        Ancestral graph base class.
        Internally, each edge is stored with an attribute dictionary
        called ``marks``. The ``marks`` dict maps the two endpoint
        nodes to their respective marks, for example:

        - Directed: ("A", "B", "-", ">") is stored as
          ("A", "B", {"marks": {"A": "-", "B": ">"}})
        - Bidirected: ("A", "B", ">", ">") is stored as
          ("A", "B", {"marks": {"A": ">", "B": ">"}})
        - Undirected: ("A", "B", "-", "-") is stored as
          ("A", "B", {"marks": {"A": "-", "B": "-"}})
        - Circle endpoint: ("A", "B", "o", ">") is stored as
          ("A", "B", {"marks": {"A": "o", "B": ">"}})

        Parameters
        ----------
        ebunch : Iterable[tuple], optional
            An iterable of edges of the form (u, v, u_mark, v_mark) used to
            initialize the graph. Each mark must be one of {">", "-", "o"}.
            Default is None, which initializes an empty graph.

        latents : set, optional
            Set of latent (unobserved) variables in the graph. Default is
            an empty set.

        exposures : set, optional
            Set of exposure variables in the graph. These are the variables
            that represent the treatment or intervention being studied in a
            causal analysis. Default is an empty set.

        outcomes : set, optional
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
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> list(graph.edges(data=True))
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}})]
        >>> graph.add_edge("C", "D", "o", "o")
        >>> list(graph.edges(data=True))
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}}),
         ('C', 'D', {'marks': {'C': 'o', 'D': 'o'}})]

        Roles can be assigned to nodes in the graph at construction or using methods.

        At construction:

        >>> g = AncestralBase(
        ...     ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")],
        ...     latents={"L"},
        ...     exposures={"A"},
        ...     outcomes={"B"},
        ... )

        Roles can also be assigned after creation using ``with_role`` method.

        >>> g = g.with_role("adjustment", {"L", "C"})

        Vertices of a specific role can be retrieved using ``get_role`` method.

        >>> g.get_role("exposures")
        ["A"]
        >>> g.get_role("adjustment")
        ["L", "C"]
        """
        super().__init__()
        self.valid_marks = {">", "-", "o"}
        if ebunch:
            self.add_edges_from(ebunch)
        self.latents = set(latents)
        self.exposures = set(exposures)
        self.outcomes = set(outcomes)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as dictionary")

        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    @property
    def adjacency_matrix(self):
        """
        Return adjacency matrix with edge marks and node-to-index mapping.

        Returns
        -------
        M : np.ndarray
            A square matrix of shape (n_nodes, n_nodes) where M[i, j]
            is the mark at node j for edge (i, j).

        node_index : dict
            Mapping from node label to row/col index.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> M, node_index = graph.adjacency_matrix
        >>> print(M)
        [[0 '>' 0]
         ['-' 0 '-']
         [0 '>' 0]]
        >>> print(node_index)
        {'A': 0, 'B': 1, 'C': 2}
        """
        pass

    @adjacency_matrix.setter
    def adjacency_matrix(self, value):
        """
        Set graph edges from an adjacency matrix with edge marks.

        Parameters
        ----------
        value : np.ndarray
            A square matrix where value[i, j] is the mark at node j
            for edge (i, j). Marks must be one of {">", "-", "o
            or 0 (no edge).

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> M = np.array([[0, ">", 0], ["-", 0, ">"], [0, "-", 0]], dtype=object)
        >>> graph = AncestralBase()
        >>> graph.adjacency_matrix = M
        >>> print(graph.nodes)
        ['X_0', 'X_1', 'X_2']
        >>> print(graph.edges(data=True))
        [('X_0', 'X_1', {'marks': {'X_1': '-', 'X_0': '>'}}), ('X_1', 'X_2', {'marks': {'X_2': '-', 'X_1': '>'}})]
        """
        pass

    def add_edge(self, u, v, u_mark, v_mark):
        """
        Add an edge with specified marks.

        Parameters
        ----------
        u : Hashable
            One endpoint of the edge.

        v : Hashable
            The other endpoint of the edge.

        u_mark : str
            Mark at node u for edge (u, v). Must be one of {">", "-", "o"}.

        v_mark : str
            Mark at node v for edge (u, v). Must be one of {">",
            "-", "o"}.

        Returns
        -------
        None
            Adds the edge to the graph in-place

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> g = AncestralBase()

        # Directed edge A → B
        >>> g.add_edge("A", "B", "-", ">")
        >>> g["A"]["B"]["marks"]
        {'A': '-', 'B': '>'}

        # Bidirected edge A ↔ D
        >>> g.add_edge("A", "D", ">", ">")
        >>> g["A"]["D"]["marks"]
        {'A': '>', 'D': '>'}

        # Undirected edge C — E
        >>> g.add_edge("C", "E", "-", "-")
        >>> g["C"]["E"]["marks"]
        {'C': '-', 'E': '-'}
        """
        pass

    def add_edges_from(self, ebunch):
        """
        Add multiple edges from an iterable of (u, v, marks) tuples.

        Parameters
        ----------
        ebunch : Iterable[tuple]
            Each tuple should be of the form (u, v, u_mark, v_mark).

        Returns
        -------
        None
            Adds the edges to the graph in-place.


        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> g = AncestralBase()
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-"), ("C", "D", "o", "o")]
        >>> g.add_edges_from(edges)
        >>> list(g.edges(data=True))
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}}),
         ('C', 'D', {'marks': {'C': 'o', 'D': 'o'}})]
        """
        pass

    def get_neighbors(self, node, u_type=None, v_type=None):
        """
        Get neighbors of a node with optional edge mark constraints.

        Parameters
        ----------
        node : Hashable
            The node whose neighbors are to be found.

        u_type : Optional[str]
            Required mark at the given node for the edge.

        v_type : Optional[str]
            Required mark at the neighbor node for the edge.

        Returns
        -------
        neighbors : set
            Set of neighboring nodes satisfying the mark constraints.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-"), ("C", "D", "o", "o")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_neighbors("B"))
        {'A', 'C'}
        >>> print(graph.get_neighbors("B", u_type=">"))
        {'C', 'A'}
        >>> print(graph.get_neighbors("B", v_type="-"))
        {'A', 'C'}
        >>> print(graph.get_neighbors("B", u_type=">", v_type="-"))
        {'C', 'A'}
        """
        pass

    def get_parents(self, node):
        """
        Get nodes that point to this node with '>'

        Parameters
        ----------
        node : Hashable
            The node whose parents are to be found.

        Returns
        -------
        parents : set
            Set of parent nodes.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("C", "B", "-", ">"), ("B", "D", "-", ">")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_parents("B"))
        {'A', 'C'}
        >>> print(graph.get_parents("D"))
        {'B'}
        >>> print(graph.get_parents("A"))
        set()
        """
        pass

    def get_children(self, node):
        """
        Get nodes that this node points to with '>'

        Parameters
        ----------
        node : Hashable
            The node whose children are to be found.

        Returns
        -------
        children : set
            Set of child nodes.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("A", "C", "-", ">"), ("B", "D", "-", ">")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_children("A"))
        {'B', 'C'}
        >>> print(graph.get_children("B"))
        {'D'}
        >>> print(graph.get_children("D"))
        set()
        """
        pass

    def get_spouses(self, node):
        """
        Get nodes connected by bidirectional '>' edges (spouses).

        Parameters
        ----------
        node : Hashable
            The node whose spouses are to be found.

        Returns
        -------
        spouses : set
            Set of spouse nodes.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", ">", ">"), ("A", "C", "-", ">"), ("C", "D", ">", ">")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_spouses("A"))
        {'B'}
        >>> print(graph.get_spouses("C"))
        {'D'}
        >>> print(graph.get_spouses("B"))
        {'A'}
        """
        pass

    def get_ancestors(self, node):
        """
        Get all ancestor nodes of the given node.

        Parameters
        ----------
        node : Hashable
            The node whose ancestors are to be found.

        Returns
        -------
        ancestors : set
            Set of ancestor nodes including the starting node.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [
        ...     ("A", "B", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("C", "D", "-", ">"),
        ...     ("E", "C", "-", ">"),
        ... ]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_ancestors("D"))
        {'A', 'B', 'C', 'D', 'E'}
        >>> print(graph.get_ancestors("C"))
        {'A', 'B', 'C', 'E'}
        >>> print(graph.get_ancestors("A"))
        {'A'}
        """
        pass

    def get_descendants(self, node):
        """
        Get all descendant nodes (children, grandchildren, etc.)

        Parameters
        ----------
        node : Hashable
            The starting node.

        Returns
        -------
        descendants : set
            Set of descendant nodes including the starting node.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [
        ...     ("A", "B", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("C", "D", "-", ">"),
        ...     ("B", "E", "-", ">"),
        ... ]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_descendants("A"))
        {'A', 'B', 'C', 'D', 'E'}
        >>> print(graph.get_descendants("B"))
        {'B', 'C', 'D', 'E'}
        >>> print(graph.get_descendants("D"))
        {'D'}
        """
        pass

    def get_reachable_nodes(self, node, u_type=None, v_type=None):
        """
        Get all nodes reachable from the given node following edges
        with specified marks.

        Parameters
        ----------
        node : Hashable
            The starting node.

        u_type : Optional[str]
            Required mark at the current node for traversal.

        v_type : Optional[str]
            Required mark at the neighbor node for traversal.

        Returns
        -------
        reachable : set
            Set of reachable nodes including the starting node.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [
        ...     ("A", "B", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("A", "D", "o", "o"),
        ...     ("D", "E", "o", "o"),
        ... ]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_reachable_nodes("A", v_type=">"))
        {'A', 'B', 'C'}
        >>> print(graph.get_reachable_nodes("A", u_type="o", v_type="o"))
        {'A', 'D', 'E'}
        """
        pass

    def to_dagitty(self) -> str:
        """
        Convert the MAG to a Dagitty string representation.

        Returns
        -------
        str
            A string in Dagitty format representing the MAG.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Y", "-", ">")
        >>> mag.add_edge("Z", "Y", "-", ">")
        >>> print(mag.to_dagitty())
        mag {
        X -> Y
        Z -> Y
        }

        >>> mag2 = MAG()
        >>> mag2.add_edge("A", "B", ">", ">")
        >>> mag2.add_edge("C", "D", "-", "-")
        >>> print(mag2.to_dagitty())
        mag {
        A <-> B
        C -- D
        }

        >>> # MAG with latent variables and roles
        >>> mag3 = MAG()
        >>> mag3.add_edge("L", "X", "-", ">")
        >>> mag3.add_edge("X", "Y", "-", ">")
        >>> mag3.latents = {"L"}
        >>> mag3 = mag3.with_role("exposures", "X")
        >>> mag3 = mag3.with_role("outcomes", "Y")
        >>> print(mag3.to_dagitty())
        mag {
        L -> X
        X -> Y
        L [latents]
        Y [outcome]
        X [exposure]
        }

        References
        ----------
        dagitty syntax: https://cran.r-project.org/web/packages/dagitty/dagitty.pdf
        """
        pass

    @classmethod
    def from_dagitty(cls, string: str = None, filename: str = None):
        """
        Populate the MAG from a Dagitty string representation.

        Parameters
        ----------
        string : str
            A string in dagitty format representing the MAG.
        filename : str, optional
            Path to file containing Dagitty format string.

        Returns
        -------
        MAG
            A new MAG instance created from the Dagitty representation.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> dag_str = '''dag {
        ... L -> A
        ... B -> C
        ... L [latents]
        ... B [outcome]
        ... A [exposure]
        ... }'''
        >>> mag = MAG.from_dagitty(dag_str)
        """
        pass

    def __eq__(self, other):
        """
        Checks if two MAGs are equal. Two MAGs are equal if they have the same
        nodes, edges(including marks), latent variables, and variable roles

        Parameters
        ----------
        other: MAG object
            The other MAG to compare with

        Returns
        -------
        bool
            True if the MAGs are equal, False otherwise

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag1 = MAG(
        ...     ebunch=[("X", "Y", "-", ">"), ("Y", "Z", "-", ">")],
        ...     latents={"L"},
        ...     roles={"exposures": "X"},
        ... )
        >>> mag2 = MAG(
        ...     ebunch=[("X", "Y", "-", ">"), ("Y", "Z", "-", ">")],
        ...     latents={"L"},
        ...     roles={"exposures": "X"},
        ... )
        >>> mag1 == mag2
        True

        >>> mag3 = MAG(
        ...     ebunch=[("X", "Y", "-", ">")], latents={"L"}, roles={"exposures": "X"}
        ... )
        >>> mag1 == mag3
        False
        """
        if not isinstance(other, AncestralBase):
            return False

        self_edges = {(u, v, frozenset(data["marks"].items())) for u, v, data in self.edges(data=True)}
        other_edges = {(u, v, frozenset(data["marks"].items())) for u, v, data in other.edges(data=True)}

        return (
            set(self.nodes()) == set(other.nodes())
            and self_edges == other_edges
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )

    def copy(self):
        """
        Return a copy of the graph, preserving nodes, edges, marks, latents, and roles.

        Returns
        -------
        AncestralBase
            A new instance of the same class as self with all properties copied.
        """
        pass
