import collections

import networkx as nx
from networkx import MultiDiGraph

from pgmpy.base._mixin_roles import _GraphRolesMixin
from pgmpy.base.DAG import DAG as pgmpy_DAG


class ADMG(_GraphRolesMixin, MultiDiGraph):
    """
    A class representing an Acyclic Directed Mixed Graph (ADMG).

    An ADMG is a directed graph that allows for both directed and bidirected edges.
    This class extends the `networkx.MultiDiGraph` and provides additional functionality
    for operations involving directed and bidirected edges.

    Parameters
    ----------
    directed_ebunch : list of tuple, optional
        List of directed edges to initialize the graph, where each tuple is (u, v).
    bidirected_ebunch : list of tuple, optional
        List of bidirected edges to initialize the graph, where each tuple is (u, v).
    latents : set of str, optional
        Set of latent variables in the graph. These are not directly represented as nodes
        but are used to indicate the presence of bidirected edges.
    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    >>> from pgmpy.base.ADMG import ADMG
    >>> admg = ADMG(
    ...     directed_ebunch=[("X", "Y"), ("Z", "Y")], bidirected_ebunch=[("X", "Z")]
    ... )
    >>> sorted(admg.nodes())
    ['X', 'Y', 'Z']
    >>> sorted(admg.edges())
    [('X', 'Y'), ('X', 'Z'), ('Z', 'X'), ('Z', 'Y')]
    >>> admg.latents
    set()
    """

    def __init__(
        self,
        directed_ebunch=None,
        bidirected_ebunch=None,
        latents=None,
        roles=None,
    ):
        super().__init__()
        # Using edge attributes to distinguish bidirected edges

        if directed_ebunch:
            self.add_directed_edges(directed_ebunch)
        if bidirected_ebunch:
            self.add_bidirected_edges(bidirected_ebunch)

        self.latents = set(latents) if latents else set()

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        # set the roles to the vertices as networkx attributes
        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def add_directed_edges(self, ebunch):
        """
        Add directed edges (u -> v) to the ADMG.

        Parameters
        ----------
        ebunch : list of tuple
            List of directed edges, where each tuple is (u, v).

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG()
        >>> admg.add_directed_edges([("X", "Y"), ("Y", "Z")])
        >>> sorted(admg.nodes())
        ['X', 'Y', 'Z']
        >>> sorted(admg.edges())
        [('X', 'Y'), ('Y', 'Z')]
        """
        pass

    def add_bidirected_edges(self, ebunch):
        """
        Add bidirected edges (u <-> v) to the ADMG.

        Parameters
        ----------
        ebunch : list of tuple
            List of bidirected edges, where each tuple is (u, v).

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG()
        >>> admg.add_bidirected_edges([("X", "Z")])
        >>> sorted(admg.nodes())
        ['X', 'Z']
        >>> sorted(admg.edges())
        [('X', 'Z'), ('Z', 'X')]
        """
        pass

    def add_edge(self, u, v, **kwargs):
        """
        Raise an error if trying to add a regular edge.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG()
        >>> admg.add_edge("X", "Y")
        Traceback (most recent call last):
            pass
        NotImplementedError: Use add_directed_edge or add_bidirected_edge to add edges.
        """
        pass

    def get_directed_parents(self, nodes):
        """
        Get directed parents of given nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to query.

        Returns
        -------
        set
            Set of directed parents.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y"), ("Z", "Y")])
        >>> sorted(admg.get_directed_parents("Y"))
        ['X', 'Z']
        >>> admg.get_directed_parents("X")
        set()
        """
        pass

    def get_bidirected_parents(self, nodes):
        """
        Get bidirected parents (nodes connected via bidirected edge) of the given nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to query.

        Returns
        -------
        set
            Set of bidirected parents.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y")], bidirected_ebunch=[("X", "Z")])
        >>> sorted(admg.get_bidirected_parents("X"))
        ['Z']
        >>> admg.get_bidirected_parents("Y")
        set()
        """
        pass

    def get_children(self, nodes):
        """
        Get children of given nodes (i.e., targets of outgoing directed edges).

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of children nodes.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y"), ("X", "Z")])
        >>> sorted(admg.get_children("X"))
        ['Y', 'Z']
        >>> admg.get_children("Y")
        set()
        """
        pass

    def get_spouses(self, nodes):
        """
        Get spouses of given nodes (i.e., nodes connected via bidirected edges).

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of spouses.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y")], bidirected_ebunch=[("X", "Z")])
        >>> sorted(admg.get_spouses("X"))
        ['Z']
        >>> admg.get_spouses("Y")
        set()
        """
        pass

    def get_ancestors(self, nodes):
        """
        Get ancestors of given nodes via directed paths.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of ancestor nodes including the input nodes.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y"), ("Y", "Z")])
        >>> sorted(admg.get_ancestors("Z"))
        ['X', 'Y', 'Z']
        >>> sorted(admg.get_ancestors("X"))
        ['X']
        """
        pass

    def get_descendants(self, nodes):
        """
        Get descendants of given nodes via directed paths.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of descendant nodes including the input nodes.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y"), ("Y", "Z")])
        >>> sorted(admg.get_descendants("X"))
        ['X', 'Y', 'Z']
        >>> sorted(admg.get_descendants("Z"))
        ['Z']
        """
        pass

    def get_district(self, nodes):
        """
        Return district of a node: maximal set connected via bidirected edges.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Nodes in the same bidirected-connected component.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y")], bidirected_ebunch=[("X", "Z")])
        >>> sorted(admg.get_district("X"))
        ['X', 'Z']
        >>> admg.get_district("Y")
        {'Y'}
        """
        pass

    def get_ancestral_graph(self, nodes):
        """
        Return the ancestral graph induced by the input nodes.

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes to induce subgraph on.

        Returns
        -------
        ADMG
            Subgraph induced by ancestors of the given nodes.

        Raises
        ------
        ValueError
            If any input node is not in the graph.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(
        ...     directed_ebunch=[("X", "Y"), ("Y", "Z")], bidirected_ebunch=[("X", "Z")]
        ... )
        >>> anc = admg.get_ancestral_graph(["Y", "Z"])
        >>> sorted(anc.nodes())
        ['Y', 'Z']
        >>> anc2 = admg.get_ancestral_graph(["X", "Y", "Z"])
        >>> sorted(anc2.nodes())
        ['X', 'Y', 'Z']
        """
        pass

    def get_markov_blanket(self, nodes):
        """
        Compute the Markov blanket for the given node(s).

        Includes:
        - Parents
        - Children
        - Spouses (nodes sharing a child)
        - Parents of nodes in the district

        Parameters
        ----------
        nodes : str or iterable of str
            Node or list of nodes.

        Returns
        -------
        set
            Set of nodes in the Markov blanket.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(
        ...     directed_ebunch=[("X", "Y"), ("Z", "Y")], bidirected_ebunch=[("X", "Z")]
        ... )
        >>> sorted(admg.get_markov_blanket("Y"))
        ['X', 'Z']
        """
        pass

    def to_dag(self):
        """
        Project ADMG into a DAG by introducing latent variables for bidirected edges.

        Returns
        -------
        pgmpy.base.DAG.DAG
            DAG with latent variables replacing bidirected edges.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y")], bidirected_ebunch=[("X", "Z")])
        >>> dag = admg.to_dag()
        >>> "L_X_Z" in dag.nodes()
        True
        >>> ("X", "Y") in dag.edges()
        True
        """
        pass

    def is_mseparated(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Test m-separation between two sets of nodes given a conditioning set.

        Parameters
        ----------
        nodes_u : str or iterable of str
            First set of nodes.

        nodes_v : str or iterable of str
            Second set of nodes.

        conditional_set : set of str, optional
            Conditioning set (default is empty set).

        Returns
        -------
        bool
            True if nodes_u and nodes_v are m-separated; False otherwise.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y"), ("Z", "Y")])
        >>> admg.is_mseparated("X", "Z")
        True
        >>> admg.is_mseparated("X", "Z", conditional_set={"Y"})
        False
        """
        pass

    def is_mconnected(
        self,
        nodes_u,
        nodes_v,
        conditional_set=None,
    ):
        """
        Test m-connectedness between two node sets given a conditioning set.

        Parameters
        ----------
        nodes_u : str or iterable of str
            First set of nodes.

        nodes_v : str or iterable of str
            Second set of nodes.

        conditional_set : set of str, optional
            Conditioning set.

        Returns
        -------
        bool
            True if m-connected; False if m-separated.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y"), ("Z", "Y")])
        >>> admg.is_mconnected("X", "Z", conditional_set={"Y"})
        True
        >>> admg.is_mconnected("X", "Z")
        False
        """
        pass

    def mconnected_nodes(self, nodes_u, nodes_v=None, conditional_set=None):
        """
        Find all nodes m-connected to nodes in `nodes_u` given `conditional_set`.

        Parameters
        ----------
        nodes_u : str or iterable of str
            Set of source nodes.

        nodes_v : str or iterable of str, optional
            If provided, filters the result to this set.

        conditional_set : set of str, optional
            Conditioning set (default is empty set).

        Returns
        -------
        set
            Nodes m-connected to `nodes_u` (or their intersection with `nodes_v` if provided).

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg = ADMG(directed_ebunch=[("X", "Y"), ("Y", "Z")])
        >>> sorted(admg.mconnected_nodes("X", nodes_v=["Y", "Z"]))
        ['Y', 'Z']
        >>> sorted(admg.mconnected_nodes("X", nodes_v=["Z"]))
        ['Z']
        """
        pass

    def __eq__(self, other):
        """
        Check if two ADMGs are equal.

        Two ADMGs are considered equal if they have the same nodes, edges,
        latent variables, and variable roles.

        Parameters
        ----------
        other : ADMG
            The other ADMG to compare with.

        Returns
        -------
        bool
            True if the ADMGs are equal, False otherwise.

        Examples
        --------
        >>> from pgmpy.base.ADMG import ADMG
        >>> admg1 = ADMG(directed_ebunch=[("X", "Y")], bidirected_ebunch=[("X", "Z")])
        >>> admg2 = ADMG(directed_ebunch=[("X", "Y")], bidirected_ebunch=[("X", "Z")])
        >>> admg1 == admg2
        True
        >>> admg3 = ADMG(directed_ebunch=[("X", "Y")])
        >>> admg1 == admg3
        False
        """
        if not isinstance(other, ADMG):
            return False

        if (
            set(self.nodes()) != set(other.nodes())
            or self.latents != other.latents
            or self.get_role_dict() != other.get_role_dict()
            or set(self.edges()) != set(other.edges())
        ):
            return False

        # Check edges type more details ('directed' or 'bidirected').
        for u, v in self.edges():
            if self.get_edge_data(u, v, 0)["type"] != other.get_edge_data(u, v, 0)["type"]:
                return False
        return True
