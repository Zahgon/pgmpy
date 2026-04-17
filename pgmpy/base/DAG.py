from __future__ import annotations

import itertools
from collections.abc import Hashable, Iterable, Sequence
from os import PathLike

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy import logger
from pgmpy.base._mixin_roles import _GraphRolesMixin
from pgmpy.independencies import Independencies
from pgmpy.utils.parser import parse_dagitty, parse_lavaan


class DAG(_GraphRolesMixin, nx.DiGraph):
    """Directed Graphical Model, graph with vertex roles.

    Each node in the graph can represent either a random variable, ``Factor``,
    or a cluster of random variables. Edges in the graph represent the
    dependencies between these.

    Abstract roles can be assigned to nodes in the graph, such as
    exposures, outcomes, adjustment sets, etc. These roles are used, or created,
    by algorithms that use the graph, such as causal inference,
    causal discovery, causal prediction.

    Parameters
    ----------
    ebunch : input graph, optional
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    latents : set of nodes, default=set()
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

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
    Create an empty DAG with no nodes and no edges

    >>> from pgmpy.base import DAG
    >>> G = DAG()

    Edges and vertices can be passed to the constructor as an edge list.

    >>> G = DAG([("a", "b"), ("b", "c")])

    G can be also grown incrementally, in several ways:

    **Nodes:**

    Add one node at a time:

    >>> G.add_node("a")

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(["a", "b"])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge(u="a", v="b")

    a list of edges,

    >>> G.add_edges_from(ebunch=[("a", "b"), ("b", "c")])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically. There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> "a" in G  # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3

    Roles can be assigned to nodes in the graph at construction or using methods.

    At construction:

    >>> G = DAG(
    ...     ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")],
    ...     roles={"exposures": "X", "outcomes": "Y"},
    ... )

    Roles can also be assigned after creation using the ``with_role`` method.

    >>> G = G.with_role("adjustment", {"U", "M"})

    Vertices of a specific role can be retrieved using the ``get_role`` method.

    >>> G.get_role("exposures")
    ['X']
    >>> G.get_role("adjustment")
    ['U', 'M']

    **Latents:**
        Latent variables can be managed using the `latents` parameter at
        initialization or by assigning the "latents" role to nodes. The
        `latents` parameter is a convenient shortcut for `roles={'latents': ...}`.

    Create a graph with initial latent variables 'U' and 'V', and exposure 'X':

    >>> from pgmpy.base import DAG
    >>> G = DAG(
    ...     ebunch=[("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y"), ("V", "M")],
    ...     latents={"U", "V"},
    ...     exposures={"X"},
    ... )
    >>> sorted(G.latents)
    ['U', 'V']
    >>> G.exposures
    {'X'}

    Add a new latent variable 'Z' using the role system:

    >>> G.add_node("Z")
    >>> G.with_role(role="latents", variables="Z", inplace=True)
    >>> sorted(G.latents)
    ['U', 'V', 'Z']

    You can also check for latents using the `get_role` method:

    >>> sorted(G.get_role(role="latents"))
    ['U', 'V', 'Z']

    Remove a latent variable from the role:

    >>> G.without_role(role="latents", variables="V", inplace=True)
    >>> sorted(G.latents)
    ['U', 'Z']
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        roles: dict[str, Iterable] | None = None,
    ) -> None:
        super().__init__(ebunch)

        self._check_cycles()

        self.latents = set(latents) if latents is not None else set()
        self.exposures = set(exposures) if exposures is not None else set()
        self.outcomes = set(outcomes) if outcomes is not None else set()

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as a dictionary.")

        # set the roles to the vertices as networkx attributes
        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def _check_cycles(self):
        """Checks if the graph has cycles.

        Raises
        ------
        ValueError
            If the graph has cycles.
        """
        pass

    @classmethod
    def from_lavaan(
        cls,
        string: str | None = None,
        filename: str | PathLike | None = None,
    ) -> DAG:
        """
        Initializes a `DAG` instance using lavaan syntax.

        Parameters
        ----------
        string: str (default: None)
            A `lavaan` style multiline set of regression equation representing the model.
            Refer http://lavaan.ugent.be/tutorial/syntax1.html for details.

        filename: str (default: None)
            The filename of the file containing the model in lavaan syntax.

        Examples
        --------
        """
        pass

    @classmethod
    def from_dagitty(cls, string=None, filename=None) -> DAG:
        """
        Initializes a `DAG` instance using DAGitty syntax.

        Creates a `DAG` from the dagitty string. If parameter `beta` is specified in the DAGitty
        string, the method returns a `LinearGaussianBayesianNetwork` instead of a plain `DAG`.

        Parameters
        ----------
        string: str (default: None)
            A `DAGitty` style multiline set of regression equation representing the model.
            Refer https://www.dagitty.net/manual-3.x.pdf#page=3.58 and
            https://github.com/jtextor/dagitty/blob/7a657776dc8f5e5ba4e323edb028e2c2aaf29327/gui/js/dagitty.js#L3417

        filename: str (default: None)
            The filename of the file containing the model in DAGitty syntax.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG.from_dagitty(
        ...     "dag{'carry matches' [latent] cancer [outcome] smoking -> 'carry matches' [beta=0.2] "
        ...     "smoking -> cancer [beta=0.5] 'carry matches' -> cancer }"
        ... )

        Creating a Linear Gaussian Bayesian network from dagitty:

        >>> from pgmpy.base import DAG
        >>> from pgmpy.models import LinearGaussianBayesianNetwork as LGBN

        # Specifying beta creates a LinearGaussianBayesianNetwork instance
        >>> dag = DAG.from_dagitty("dag{X -> Y [beta=0.3] Y -> Z [beta=0.1]}")
        >>> data = dag.simulate(n_samples=int(1e4))

        >>> from pgmpy.base import DAG
        >>> from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
        """
        pass

    def add_edge(self, u: Hashable, v: Hashable, weight: int | float | None = None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(["Alice", "Bob", "Charles"])
        >>> G.add_edge(u="Alice", v="Bob")
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob')])

        When the node is not already present in the graph:

        >>> G.add_edge(u="Alice", v="Ankur")
        >>> sorted(G.nodes())
        ['Alice', 'Ankur', 'Bob', 'Charles']
        >>> sorted(G.edges())
        [('Alice', 'Ankur'), ('Alice', 'Bob')]

        Adding edges with weight:

        >>> G.add_edge("Ankur", "Maria", weight=0.1)
        >>> G.edges["Ankur", "Maria"]
        {'weight': 0.1}
        """
        pass

    def add_edges_from(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]],
        weights: list[float] | tuple[float] | None = None,
    ):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        object.

        **The behavior of adding weights is different than networkx.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(["Alice", "Bob", "Charles"])
        >>> G.add_edges_from([("Alice", "Bob"), ("Bob", "Charles")])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles')])

        When the node is not already in the model:

        >>> G.add_edges_from([("Alice", "Ankur")])
        >>> sorted(G.nodes())
        ['Alice', 'Ankur', 'Bob', 'Charles']
        >>> sorted(G.edges())
        [('Alice', 'Ankur'), ('Alice', 'Bob'), ('Bob', 'Charles')]

        Adding edges with weights:

        >>> G.add_edges_from(
        ...     [("Ankur", "Maria"), ("Maria", "Mason")], weights=[0.3, 0.5]
        ... )
        >>> G.edges["Ankur", "Maria"]
        {'weight': 0.3}
        >>> G.edges["Maria", "Mason"]
        {'weight': 0.5}

        or

        >>> G.add_edges_from([("Ankur", "Maria", 0.3), ("Maria", "Mason", 0.5)])
        """
        pass

    def get_parents(self, node: Hashable):
        """
        Returns a list of parents of node.

        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose parents would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("diff", "grade"), ("intel", "grade")])
        >>> G.get_parents(node="grade")
        ['diff', 'intel']
        """
        pass

    def moralize(self):
        """
        Removes all the immoralities in the DAG and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("diff", "grade"), ("intel", "grade")])
        >>> moral_graph = G.moralize()
        >>> sorted(list(moral_graph.edges()))
        [('diff', 'grade'), ('diff', 'intel'), ('grade', 'intel')]
        """
        pass

    def get_leaves(self):
        """
        Returns a list of leaves of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([("A", "B"), ("B", "C"), ("B", "D")])
        >>> graph.get_leaves()
        ['C', 'D']
        """
        pass

    def out_degree_iter(self, nbunch=None, weight=None):
        pass

    def in_degree_iter(self, nbunch=None, weight=None):
        pass

    def get_roots(self):
        """
        Returns a list of roots of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([("A", "B"), ("B", "C"), ("B", "D"), ("E", "B")])
        >>> graph.get_roots()
        ['A', 'E']
        """
        pass

    def get_children(self, node: Hashable):
        """
        Returns a list of children of node.
        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose children would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> g = DAG(
        ...     ebunch=[
        ...         ("A", "B"),
        ...         ("C", "B"),
        ...         ("B", "D"),
        ...         ("B", "E"),
        ...         ("B", "F"),
        ...         ("E", "G"),
        ...     ]
        ... )
        >>> g.get_children(node="B")
        ['D', 'E', 'F']
        """
        pass

    def get_independencies(self, latex=False, include_latents=False) -> Independencies | list[str]:
        """
        Computes independencies in the DAG, by checking minimal d-seperation.

        Parameters
        ----------
        latex: boolean
            If latex=True then latex string of the independence assertion
            would be created.

        include_latents: boolean
            If True, includes latent variables in the independencies. Otherwise,
            only generates independencies on observed variables.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> chain = DAG([("X", "Y"), ("Y", "Z")])
        >>> chain.get_independencies()
        (X \u27c2 Z | Y)
        """
        pass

    def local_independencies(self, variables: list[Hashable] | tuple[Hashable, ...] | str):
        """
        Returns an instance of Independencies containing the local independencies
        of each of the variables.

        Parameters
        ----------
        variables: str or array like
            variables whose local independencies are to be found.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_edges_from(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("grade", "letter"),
        ...         ("intel", "SAT"),
        ...     ]
        ... )
        >>> ind = student.local_independencies("grade")
        >>> ind
        (grade \u27c2 SAT | diff, intel)
        """
        pass

    def is_iequivalent(self, model: DAG):
        """
        Checks whether the given model is I-equivalent

        Two graphs G1 and G2 are said to be I-equivalent if they have same skeleton
        and have same set of immoralities.

        Parameters
        ----------
        model : A DAG object, for which you want to check I-equivalence

        Returns
        --------
        I-equivalence: boolean
            True if both are I-equivalent, False otherwise

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_edges_from([("V", "W"), ("W", "X"), ("X", "Y"), ("Z", "Y")])
        >>> G1 = DAG()
        >>> G1.add_edges_from([("W", "V"), ("X", "W"), ("X", "Y"), ("Z", "Y")])
        >>> G.is_iequivalent(G1)
        True

        """
        pass

    def get_immoralities(self) -> dict[Hashable, list[tuple[Hashable, Hashable]]]:
        """
        Finds all the immoralities in the model
        A v-structure X -> Z <- Y is an immorality if there is no direct edge between X and Y .

        Returns
        -------
        Immoralities: set
            A set of all the immoralities in the model

        Examples
        ---------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_edges_from(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> imm = student.get_immoralities()
        >>> imm["grade"]
        [('diff', 'intel')]
        """
        pass

    def is_dconnected(
        self,
        start: Hashable,
        end: Hashable,
        observed: Sequence[Hashable] | None = None,
        include_latents=False,
    ):
        """
        Returns True if there is an active trail (i.e. d-connection) between
        `start` and `end` node given that `observed` is observed.

        Parameters
        ----------
        start, end : int, str, any hashable python object.
            The nodes in the DAG between which to check the d-connection/active trail.

        observed : list, array-like (optional)
            If given the active trail would be computed assuming these nodes to
            be observed.

        include_latents: boolean (default: False)
            If true, latent variables are return as part of the active trail.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_nodes_from(["diff", "intel", "grades", "letter", "sat"])
        >>> student.add_edges_from(
        ...     [
        ...         ("diff", "grades"),
        ...         ("intel", "grades"),
        ...         ("grades", "letter"),
        ...         ("intel", "sat"),
        ...     ]
        ... )
        >>> student.is_dconnected("diff", "intel")
        False
        >>> student.is_dconnected("grades", "sat")
        True
        """
        pass

    def minimal_dseparator(self, start: Hashable, end: Hashable, include_latents=False) -> set[Hashable]:
        """
        Finds the minimal d-separating set for `start` and `end`.

        Parameters
        ----------
        start: node
            The first node.

        end: node
            The second node.

        include_latents: boolean (default: False)
            If true, latent variables are consider for minimal d-seperator.

        Examples
        --------
        >>> dag = DAG([("A", "B"), ("B", "C")])
        >>> dag.minimal_dseparator(start="A", end="C")
        {'B'}

        References
        ----------
        [1] Algorithm 4, Page 10: Tian, Jin, Azaria Paz, and
          Judea Pearl. Finding minimal d-separators. Computer Science Department,
            University of California, 1998.
        """
        pass

    def get_markov_blanket(self, node: Hashable) -> list[Hashable]:
        """
        Returns a markov blanket for a random variable. In the case
        of Bayesian Networks, the markov blanket is the set of
        node's parents, its children and its children's other parents.

        Returns
        -------
        Markov Blanket: list
            List of nodes in the markov blanket of `node`.

        Parameters
        ----------
        node: string, int or any hashable python object.
              The node whose markov blanket would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> G = DAG(
        ...     [
        ...         ("x", "y"),
        ...         ("z", "y"),
        ...         ("y", "w"),
        ...         ("y", "v"),
        ...         ("u", "w"),
        ...         ("s", "v"),
        ...         ("w", "t"),
        ...         ("w", "m"),
        ...         ("v", "n"),
        ...         ("v", "q"),
        ...     ]
        ... )
        >>> sorted(G.get_markov_blanket("y"))
        ['s', 'u', 'v', 'w', 'x', 'z']
        """
        pass

    def active_trail_nodes(
        self,
        variables: list[Hashable] | Hashable,
        observed: Hashable | list[Hashable] | tuple[Hashable, Hashable] | None = None,
        include_latents=False,
    ) -> dict[Hashable, set[Hashable]]:
        """
        Returns a dictionary with the given variables as keys and all the nodes reachable
        from that respective variable as values.

        Parameters
        ----------
        variables: str or array like
            variables whose active trails are to be found.

        observed : List of nodes (optional)
            If given the active trails would be computed assuming these nodes to be
            observed.

        include_latents: boolean (default: False)
            Whether to include the latent variables in the returned active trail nodes.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> student = DAG()
        >>> student.add_nodes_from(["diff", "intel", "grades"])
        >>> student.add_edges_from([("diff", "grades"), ("intel", "grades")])
        >>> {k: sorted(v) for k, v in student.active_trail_nodes("diff").items()}
        {'diff': ['diff', 'grades']}
        >>> {k: sorted(v) for k, v in student.active_trail_nodes(["diff", "intel"], observed="grades").items()}
        {'diff': ['diff', 'intel'], 'intel': ['diff', 'intel']}

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        pass

    def get_ancestors(self, nodes: str | tuple[Hashable, Hashable] | Iterable[Hashable]) -> set[Hashable]:
        """
        Returns a dictionary of all ancestors of all the observed nodes including the
        node itself.

        Parameters
        ----------
        nodes: string, list-type
            name of all the observed nodes

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> model = DAG([("D", "G"), ("I", "G"), ("G", "L"), ("I", "L")])
        >>> sorted(model.get_ancestors("G"))
        ['D', 'G', 'I']
        >>> sorted(model.get_ancestors(["G", "I"]))
        ['D', 'G', 'I']
        """
        pass

    def to_pdag(self):
        """
        Returns the CPDAG (Completed Partial DAG) of the DAG representing the equivalence class
        that the given DAG belongs to.

        Returns
        -------
        CPDAG: pgmpy.base.PDAG
            An instance of pgmpy.base.PDAG representing the CPDAG of the given DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("A", "B"), ("B", "C"), ("C", "D")])
        >>> pdag = dag.to_pdag()
        >>> pdag.directed_edges
        set()

        References
        ----------
        [1] Chickering, David Maxwell. "Learning equivalence classes of Bayesian-network structures."
          Journal of machine learning research 2.Feb (2002): 445-498. Figure 4 and 5.
        """
        pass

    def do(
        self,
        nodes: Hashable | Iterable[Hashable] | tuple[Hashable, Hashable],
        inplace=False,
    ):
        """
        Applies the do operator to the graph and returns a new DAG with the
        transformed graph.

        The do-operator, do(X = x) has the effect of removing all edges from
        the parents of X and setting X to the given value x.

        Parameters
        ----------
        nodes : list, array-like
            The names of the nodes to apply the do-operator for.

        inplace: boolean (default: False)
            If inplace=True, makes the changes to the current object,
            otherwise returns a new instance.

        Returns
        -------
        Modified DAG: pgmpy.base.DAG
            A new instance of DAG modified by the do-operator

        Examples
        --------
        Initialize a DAG

        >>> graph = DAG()
        >>> graph.add_edges_from([("X", "A"), ("A", "Y"), ("A", "B")])
        >>> # Applying the do-operator will return a new DAG with the desired structure.
        >>> graph_do_A = graph.do("A")
        >>> # Which we can verify is missing the edges we would expect.
        >>> sorted(graph_do_A.edges)
        [('A', 'B'), ('A', 'Y')]

        References
        ----------
        Causality: Models, Reasoning, and Inference, Judea Pearl (2000). p.70.
        """
        pass

    def get_ancestral_graph(self, nodes: Iterable[Hashable]):
        """
        Returns the ancestral graph of the given `nodes`. The ancestral graph only
        contains the nodes which are ancestors of at least one of the variables in
        node.

        Parameters
        ----------
        node: iterable
            List of nodes whose ancestral graph needs to be computed.

        Returns
        -------
        Ancestral Graph: pgmpy.base.DAG

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("A", "C"), ("B", "C"), ("D", "A"), ("D", "B")])
        >>> anc_dag = dag.get_ancestral_graph(nodes=["A", "B"])
        >>> anc_dag.edges()
        OutEdgeView([('D', 'A'), ('D', 'B')])
        """
        pass

    def to_daft(
        self,
        node_pos: str | dict[Hashable, tuple[int, int]] = "circular",
        latex=True,
        pgm_params={},
        edge_params={},
        node_params={},
        plot_edge_strength=False,
    ):
        """
        Returns a daft (https://docs.daft-pgm.org/en/latest/) object which can be rendered for
        publication quality plots. The returned object's render method can be called to see the plots.

        Parameters
        ----------
        node_pos: str or dict (default: circular)
            If str: Must be one of the following: circular, kamada_kawai, planar, random, shell, sprint,
                spectral, spiral. Please refer:
                  https://networkx.org/documentation/stable//reference/drawing.html#module-networkx.drawing.layout
                    for details on these layouts.

            If dict should be of the form {node: (x coordinate, y coordinate)} describing the x and y coordinate of each
            node.

            If no argument is provided uses circular layout.

        latex: boolean
            Whether to use latex for rendering the node names.

        pgm_params: dict (optional)
            Any additional parameters that need to be passed to `daft.PGM` initializer.
            Should be of the form: {param_name: param_value}

        edge_params: dict (optional)
            Any additional edge parameters that need to be passed to `daft.add_edge` method.
            Should be of the form: {(u1, v1): {param_name: param_value}, (u2, v2): {...} }

        node_params: dict (optional)
            Any additional node parameters that need to be passed to `daft.add_node` method.
            Should be of the form: {node1: {param_name: param_value}, node2: {...} }

        plot_edge_strength: bool (default: False)
            If True, displays edge strength values as labels on edges.
            Requires edge strengths to be computed first using the edge_strength() method.

        Returns
        -------
        Daft object: daft.PGM object
            Daft object for plotting the DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("a", "b"), ("b", "c"), ("d", "c")])
        >>> dag.to_daft(
        ...     node_pos={"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (1, 1)}
        ... )  # doctest: +SKIP
        <daft.PGM at ...>
        >>> dag.to_daft(node_pos="circular")  # doctest: +SKIP
        <daft.PGM at ...>
        >>> dag.to_daft(
        ...     node_pos="circular", pgm_params={"observed_style": "inner"}
        ... )  # doctest: +SKIP
        <daft.PGM at ...>
        >>> dag.to_daft(
        ...     node_pos="circular",
        ...     edge_params={("a", "b"): {"label": 2}},
        ...     node_params={"a": {"shape": "rectangle"}},
        ...     )  # doctest: +SKIP
        <daft.PGM at ...>
        """
        pass

    @staticmethod
    def get_random(
        n_nodes=5,
        edge_prob=0.5,
        node_names: list[Hashable] | None = None,
        latents=False,
        seed: int | None = None,
    ) -> DAG:
        """
        Returns a randomly generated DAG with `n_nodes` number of nodes with
        edge probability being `edge_prob`.

        Parameters
        ----------
        n_nodes: int
            The number of nodes in the randomly generated DAG.

        edge_prob: float
            The probability of edge between any two nodes in the topologically
            sorted DAG.

        node_names: list (default: None)
            A list of variables names to use in the random graph.
            If None, the node names are integer values starting from 0.

        latents: bool (default: False)
            If True, includes latent variables in the generated DAG.

        seed: int (default: None)
            The seed for the random number generator.

        Returns
        -------
        Random DAG: pgmpy.base.DAG
            The randomly generated DAG.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> random_dag = DAG.get_random(n_nodes=10, edge_prob=0.3, seed=42)
        >>> sorted(random_dag.nodes())
        ['X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9']
        >>> sorted(random_dag.edges())  # doctest: +NORMALIZE_WHITESPACE
        [('X_0', 'X_2'), ('X_0', 'X_5'), ('X_0', 'X_6'), ('X_0', 'X_7'),
         ('X_1', 'X_3'), ('X_1', 'X_8'), ('X_2', 'X_3'), ('X_2', 'X_4'),
         ('X_4', 'X_5'), ('X_7', 'X_9')]
        """
        pass

    def to_graphviz(self, plot_edge_strength=False):
        """
        Retuns a pygraphviz object for the DAG. pygraphviz is useful for
        visualizing the network structure.

        Parameters
        ----------
        plot_edge_strength: bool (default: False)
            If True, displays edge strength values as labels on edges.
            Requires edge strengths to be computed first using the edge_strength() method.

        Returns
        -------
        AGraph object: pygraphviz.AGraph
            pygraphviz object for plotting the DAG.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/alarm")
        >>> model.to_graphviz()  # doctest: +ELLIPSIS
        <AGraph b'unknown' <Swig Object of type 'Agraph_t *' at 0x...>
        """
        pass

    def to_lavaan(self) -> str:
        """
        Convert the DAG to lavaan syntax representation.

        The lavaan syntax represents structural equations where each line
        shows a dependent variable regressed on its parents using the ~ operator.
        Isolated nodes (nodes with no parents) are not included in the output.

        Returns
        -------
        str
            String representation of the DAG in lavaan syntax format.
            Each line represents a regression equation where the dependent
            variable is regressed on its parents.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("X", "Y"), ("Z", "Y")])
        >>> print(dag.to_lavaan())
        Y ~ X + Z

        >>> dag2 = DAG([("A", "B"), ("B", "C")])
        >>> print(dag2.to_lavaan())
        B ~ A
        C ~ B

        >>> # Empty DAG returns empty string
        >>> empty_dag = DAG()
        >>> empty_dag.to_lavaan()
        ''

        Notes
        -----
        - Node names are converted to string representations using str().
        - If node names contain spaces or special characters, they will be used as-is.
        - Users should ensure node names are valid in R/lavaan context if needed.

        References
        ----------
        lavaan syntax: http://lavaan.ugent.be/tutorial/syntax1.html
        """
        pass

    def to_dagitty(self) -> str:
        """
        Convert the DAG to dagitty syntax representation.

        The dagitty syntax represents directed acyclic graphs using
        the dag { statements } format with -> for directed edges.
        Isolated nodes (nodes with no edges) are included as standalone nodes.

        Returns
        -------
        str
            String representation of the DAG in dagitty syntax format.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> dag = DAG([("X", "Y"), ("Z", "Y")])
        >>> print(dag.to_dagitty())
        dag {
        X -> Y
        Z -> Y
        }

        >>> dag2 = DAG([("A", "B"), ("B", "C")])
        >>> print(dag2.to_dagitty())
        dag {
        A -> B
        B -> C
        }

        >>> # DAG with isolated node
        >>> dag3 = DAG()
        >>> dag3.add_nodes_from(["A", "B"])
        >>> dag3.add_edge("A", "B")
        >>> dag3.add_node("C")  # Isolated node
        >>> print(dag3.to_dagitty())
        dag {
        A -> B
        C
        }

        Notes
        -----
        - Node names are converted to string representations using str().
        - If node names contain spaces or special characters, they will be used as-is.
        - Users should ensure node names are valid in R/dagitty context if needed.

        References
        ----------
        dagitty syntax: https://cran.r-project.org/web/packages/dagitty/dagitty.pdf
        """
        pass

    def _variable_name_contains_non_string(self):
        """
        Checks if the variable names contain any non-string values. Used only for CausalInference class.
        """
        pass

    def copy(self):
        """Returns a copy of the DAG object."""
        pass

    def __eq__(self, other):
        """
        Checks if two DAGs are equal. Two DAGs are considered equal if they
        have the same nodes, edges, latent variables, and variable roles.

        Parameters
        ----------
        other: DAG object
            The other DAG to compare with.

        Returns
        -------
        bool
            True if the DAGs are equal, False otherwise.
        """
        if not isinstance(other, DAG):
            return False

        return (
            set(self.nodes()) == set(other.nodes())
            and set(self.edges()) == set(other.edges())
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )

    def edge_strength(self, data, edges=None):
        """
        Computes the strength of each edge in `edges`. The strength is bounded
        between 0 and 1, with 1 signifying strong effect.

        The edge strength is defined as the effect size measure of a
        Conditional Independence test using the parents as the conditional set.
        The strength quantifies the effect of edge[0] on edge[1] after
        controlling for any other influence paths. We use a residualization-based
        CI test[1] to compute the strengths.

        Interpretation:
        - The strength is the Pillai's Trace effect size of partial correlation.
        - Measures the strength of linear relationship between the residuals.
        - Works for any mixture of categorical and continuous variables.
        - The value is bounded between 0 and 1:
        - Strength close to 1 → strong dependence.
        - Strength close to 0 → conditional independence.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset to compute edge strengths on.

        edges : tuple, list, or None (default: None)
            - None: Compute for all DAG edges.
            - Tuple (X, Y): Compute for edge X → Y.
            - List of tuples: Compute for selected edges.

        Returns
        -------
        dict
            Dictionary mapping edges to their strength values.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> # Create a linear Gaussian Bayesian network
        >>> linear_model = LGBN([("X", "Y"), ("Z", "Y")])
        >>> # Create CPDs with specific beta values
        >>> x_cpd = LinearGaussianCPD(variable="X", beta=[0], std=1)
        >>> y_cpd = LinearGaussianCPD(
        ...     variable="Y", beta=[0, 0.4, 0.6], std=1, evidence=["X", "Z"]
        ... )
        >>> z_cpd = LinearGaussianCPD(variable="Z", beta=[0], std=1)
        >>> # Add CPDs to the model
        >>> linear_model.add_cpds(x_cpd, y_cpd, z_cpd)
        >>> # Simulate data from the model
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> data = linear_model.simulate(n_samples=int(1e4))
        >>> # Create DAG and compute edge strengths
        >>> dag = DAG([("X", "Y"), ("Z", "Y")])
        >>> strengths = dag.edge_strength(data)  # doctest: +SKIP
        >>> sorted(strengths.items())  # doctest: +SKIP
        [(('X', 'Y'), 0.4...), (('Z', 'Y'), 0.6...)]
        >>> strengths[("X", "Y")]  # doctest: +SKIP
        np.float64(0.1454172599124535)
        >>> strengths[("Z", "Y")]  # doctest: +SKIP
        np.float64(0.26003467856256834)

        References
        ----------
        [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional
        conditional independences for categorical and ordinal data." Proceedings of the AAAI Conference
        on Artificial Intelligence.
        """
        pass

    def __hash__(self):
        """
        Returns a hash value for the DAG object. The hash value is computed based on
        the nodes, edges, latent variables, and variable roles of the DAG.
        """
        return hash(
            (
                frozenset(self.nodes()),
                frozenset(self.edges()),
                frozenset(self.latents),
                frozenset((role, frozenset(self.get_role(role))) for role in self.get_roles()),
            )
        )

    def get_stats(self):
        """
        Returns a dictionary of summary statistics about the structure of the DAG.

        Returns
        -------
        dict
            Dictionary containing summary statistics of the DAG.

        n_nodes : int
            Number of nodes in the DAG.
        n_edges : int
            Number of edges in the DAG.
        n_root_nodes : int
            Number of nodes with no parents.
        n_leaf_nodes : int
            Number of nodes with no children.
        edge_density : float
            Ratio of edges to maximum possible edges.
        n_connected_components : int
            Number of weakly connected components.
        n_v_structures : int
            Number of v-structures (immoralities).
        avg_n_parents : float
            Average number of parents per node.
        max_n_parents : int
            Maximum number of parents of any node.
        n_latent_nodes : int
            Number of latent (unobserved) nodes in the DAG.

        Causal statistics : only computed if exposures and outcomes are provided.
        n_exposures : int
            Number of exposure nodes.
        n_outcomes : int
            Number of outcome nodes.
        n_causal_paths : int
            Number of directed paths from exposures to outcomes.
        n_direct_paths : int
            Number of causal paths with no mediator.
        n_mediated_paths : int
            Number of causal paths through at least one mediator.
        n_mediators : int
            Number of nodes mediating at least one exposure-outcome path.
        n_confounding_paths : int
            Number of backdoor paths from exposures to outcomes.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> from pprint import pprint
        >>> dag = DAG([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])
        >>> pprint(dag.get_stats())
        {'avg_n_parents': 0.8,
         'edge_density': 0.4,
         'max_n_parents': 2,
         'n_connected_components': 1,
         'n_edges': 4,
         'n_latent_nodes': 0,
         'n_leaf_nodes': 2,
         'n_nodes': 5,
         'n_root_nodes': 2,
         'n_v_structures': 1}
        >>> dag2 = DAG(
        ...     ebunch=[('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')],
        ...     roles={'exposures': 'D', 'outcomes': 'L'}
        ... )
        >>> pprint(dag2.get_stats())
        {'avg_n_parents': 0.8,
         'edge_density': 0.4,
         'max_n_parents': 2,
         'n_causal_paths': 1,
         'n_confounding_paths': 0,
         'n_connected_components': 1,
         'n_direct_paths': 0,
         'n_edges': 4,
         'n_exposures': 1,
         'n_latent_nodes': 0,
         'n_leaf_nodes': 2,
         'n_mediated_paths': 1,
         'n_mediators': 1,
         'n_nodes': 5,
         'n_outcomes': 1,
         'n_root_nodes': 2,
         'n_v_structures': 1}
        """
        pass
