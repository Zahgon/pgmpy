from collections.abc import Hashable, Iterable

import networkx as nx

from pgmpy.base import AncestralBase


class MAG(AncestralBase):
    """
    Class for representing Maximal Ancestral Graphs (MAGs).

    A MAG is a type of graph used in causal inference to represent conditional
    independence relations when some variables are latent (unobserved). Unlike
    simple directed acyclic graphs (DAGs), MAGs allow for special edge types
    (directed and bidirected) that capture the presence of latent confounding
    and selection bias. Every pair of nodes in a MAG is connected in such a way
    that the graph is "maximal," meaning no additional edges can be added
    without changing the set of implied conditional independence relations.


    Parameters
    ----------
    ebunch : iterable of tuples, optional
        A list or iterable of edges to add at initialization.

    latents : set, default=set()
        Set of latent (unobserved) variables.

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
    >>> from pgmpy.base import MAG
    >>> mag = MAG(ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")], latents={"L"})
    >>> sorted(mag.nodes())
    ['A', 'B', 'C', 'L']

    Roles can be assigned to nodes in the graph at construction or using methods.

    At construction:

    >>> mag = MAG(
    ...     ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")],
    ...     latents={"L"},
    ...     exposures={"A"},
    ...     outcomes={"B"},
    ... )

    Roles can also be assigned after creation using ``with_role`` method.

    >>> mag = mag.with_role("adjustment", {"L", "C"})

    Vertices of a specific role can be retrieved using ``get_role`` method.

    >>> mag.get_role("exposures")
    ['A']
    >>> mag.get_role("adjustment")
    ['L', 'C']

    References
    ----------
    .. [1] Zhang, J. (2008). Causal Reasoning with Ancestral Graphs. Journal of Machine Learning Research, 9(7).
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] = set(),
        exposures: set[Hashable] = set(),
        outcomes: set[Hashable] = set(),
        roles=None,
    ):
        if ebunch:
            for _, _, u_mark, v_mark in ebunch:
                if (u_mark, v_mark) not in {
                    ("-", ">"),
                    (">", "-"),
                    (">", ">"),
                    ("-", "-"),
                }:
                    raise ValueError(
                        f"Invalid edge type ({u_mark}, {v_mark}). "
                        "MAGs only allow directed ('-', '>'), reverse directed ('>', '-'), "
                        "bidirected ('>', '>'), and undirected ('-', '-') edges."
                    )
        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )

    def _is_collider(self, u, c, v):
        """
        Check if a node is a collider in a path u - c - v.

        A collider is a node with incoming arrowheads on both sides:
        u -> c <- v.

        Parameters
        ----------
        u : Hashable
            The first endpoint in the triplet (u, c, v).

        c : Hashable
            The middle node, candidate collider.

        v : Hashable
            The second endpoint in the triplet.

        Returns
        -------
        bool
            True if `c` is a collider on the path, False otherwise.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Z", "-", ">")
        >>> mag.add_edge("Y", "Z", "-", ">")
        >>> mag._is_collider("X", "Z", "Y")
        True
        """
        pass

    def has_inducing_path(self, u, v, W):
        """
        Check if there exists an inducing path between two nodes relative to W.

        An inducing path between u and v is a path such that:
        - The path has length > 2 (at least one intermediate node),
        - Every intermediate node is a collider on the path,
        - Every intermediate node is either:
            * in W, or
            * an ancestor of u or v.

        Parameters
        ----------
        u : Hashable
            Source node.

        v : Hashable
            Target node.

        W : set
            Subset of nodes to check inducing paths through (often latents).

        Returns
        -------
        bool
            True if there exists an inducing path, False otherwise.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "L", "-", ">")
        >>> mag.add_edge("Y", "L", "-", ">")
        >>> mag.latents = {"L"}
        >>> mag.has_inducing_path("X", "Y", mag.latents)
        True
        """
        pass

    def is_visible_edge(self, u, v) -> bool:
        """
        Check if a directed edge u -> v is visible in the MAG.

        A directed edge A → B in a MAG is considered visible if there exists a vertex C
        not adjacent to B such that either:
            1. C → A exists, or
            2. There is a collider path from C to A that is into A, and every vertex
            on that path is a parent of B.

        Parameters
        ----------
        u : Hashable
            Source node (tail of the edge).

        v : Hashable
            Target node (head of the edge).

        Returns
        -------
        bool
            True if the edge u -> v is visible, False otherwise.

        Examples
        --------
        >>> edges = [
        ...     ("A", "D", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("X", "A", "-", ">"),
        ... ]
        >>> mag = MAG(ebunch=edges)
        >>> mag.is_visible_edge("A", "D")
        True
        >>> mag.is_visible_edge("B", "C")
        False
        """
        pass

    def lower_manipulation(self, X, inplace=False):
        """
        Performs lower manipulation.

        Removes all edges that are visible and originate from nodes in X.
        For edges from X that are invisible, adds bidirected edges from the other
        endpoint to its neighbors outside X to preserve conditional independencies.
        All other edges remain unchanged.

        Parameters
        ----------
        X : set
            Set of nodes to perform manipulation on.

        inplace : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        MAG
            A new MAG with outgoing edges from X removed.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("A", "B", "-", ">")
        >>> mag.add_edge("A", "C", "-", ">")
        >>> mag.add_edge("C", "B", "-", ">")
        >>> mag.add_edge("B", "C", ">", ">")
        >>> new_mag = mag.lower_manipulation({"A"})
        >>> edges = list(new_mag.edges(data=True))
        >>> len(edges)
        1
        >>> edges[0][0], edges[0][1]
        ('B', 'C')
        >>> edges[0][2]['marks']['B'], edges[0][2]['marks']['C']
        ('>', '>')
        """
        pass

    def upper_manipulation(self, X, inplace=False):
        """
        Performs upper manipulation.

        Deletes all edges (directed or bidirected) that have an arrowhead
        pointing to any variable in X. The rest of the graph remains unchanged.

        Parameters
        ----------
        X : set
            Set of nodes to perform manipulation on.

        inplace : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        MAG
            A new MAG with incoming edges to X removed.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Y", ">", "-")
        >>> mag.add_edge("Z", "X", ">", "-")
        >>> mag.add_edge("A", "X", "-", ">")
        >>> new_mag = mag.upper_manipulation({"X"})
        >>> new_mag.has_edge("Z", "X")
        True
        >>> new_mag.has_edge("A", "X")
        False
        >>> new_mag.has_edge("X", "Y")
        False
        """
        pass
