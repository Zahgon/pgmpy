import typing
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, combinations

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
from pgmpy.utils import compat_fns, to_timeseries_format


@dataclass(eq=True, frozen=True)
class DynamicNode:
    """
    Class for representing the nodes of Dynamic Bayesian Networks.
    """

    node: str
    time_slice: int

    def __getitem__(self, idx: int) -> str | int:
        if idx == 0:
            return self.node
        elif idx == 1:
            return self.time_slice
        else:
            raise IndexError(f"Index {idx} out of bounds.")

    def __str__(self) -> str:
        return f"({self.node}, {self.time_slice})"

    def __repr__(self) -> str:
        return f"<DynamicNode({self.node}, {self.time_slice}) at {hex(id(self))}>"

    def __lt__(self, other) -> bool:
        if self.node < other.node:
            return True
        elif self.node > other.node:
            return False
        else:
            return self.time_slice < other.time_slice

    def __le__(self, other) -> bool:
        if self.node <= other.node:
            return True
        elif self.node > other.node:
            return False
        else:
            return self.time_slice <= other.time_slice

    def __eq__(self, other) -> bool:
        if isinstance(other, DynamicNode):
            return (self.node, self.time_slice) == (other.node, other.time_slice)
        elif isinstance(other, (list, tuple)):
            return (self.node, self.time_slice) == tuple(other)
        else:
            return False

    def to_tuple(self) -> tuple:
        """
        Returns a tuple representation as (node, time_slice) for DynamicNode object.
        """
        pass


class DynamicBayesianNetwork(DAG):
    """
    Base class for Dynamic Bayesian Network

    This is a time variant model of the static Bayesian model, where each
    time-slice has some static nodes and is then replicated over a certain
    time period.

    The nodes can be any hashable python objects.

    Parameters
    ----------
    ebunch: Data to initialize graph.  If data=None (default) an empty
          graph is created.  The data can be an edge list, or any NetworkX
          graph object

    Examples
    --------
    Create an empty Dynamic Bayesian Network with no nodes and no edges:

    >>> from pgmpy.models import DynamicBayesianNetwork as DBN
    >>> dbn = DBN()

    Adding nodes and edges inside the Dynamic Bayesian Network. A single
    node can be added using the method below. For adding edges we need to
    specify the time slice since edges can be across different time slices.

    For example for a network as [image](http://s8.postimg.org/aaybw4x2t/Blank_Flowchart_New_Page_1.png),
    we will need to add all the edges in the 2-TBN as:

    >>> dbn.add_edges_from(
    ...     [
    ...         (("D", 0), ("G", 0)),
    ...         (("I", 0), ("G", 0)),
    ...         (("G", 0), ("L", 0)),
    ...         (("D", 0), ("D", 1)),
    ...         (("I", 0), ("I", 1)),
    ...         (("G", 0), ("G", 1)),
    ...         (("G", 0), ("L", 1)),
    ...         (("L", 0), ("L", 1)),
    ...     ]
    ... )

    We can query the edges and nodes in the network as:

    >>> dbn.nodes() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    NodeView((<DynamicNode(D, 0) at 0x...>,
    <DynamicNode(G, 0) at 0x...>,
    <DynamicNode(D, 1) at 0x...>,
    <DynamicNode(G, 1) at 0x...>,
    <DynamicNode(I, 0) at 0x...>,
    <DynamicNode(I, 1) at 0x...>,
    <DynamicNode(L, 0) at 0x...>,
    <DynamicNode(L, 1) at 0x...>))
    >>> dbn.edges() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    OutEdgeView([(<DynamicNode(D, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
    (<DynamicNode(D, 0) at 0x...>, <DynamicNode(D, 1) at 0x...>),
    (<DynamicNode(G, 0) at 0x...>, <DynamicNode(L, 0) at 0x...>),
    (<DynamicNode(G, 0) at 0x...>, <DynamicNode(G, 1) at 0x...>),
    (<DynamicNode(G, 0) at 0x...>, <DynamicNode(L, 1) at 0x...>),
    (<DynamicNode(D, 1) at 0x...>, <DynamicNode(G, 1) at 0x...>),
    (<DynamicNode(G, 1) at 0x...>, <DynamicNode(L, 1) at 0x...>),
    (<DynamicNode(I, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
    (<DynamicNode(I, 0) at 0x...>, <DynamicNode(I, 1) at 0x...>),
    (<DynamicNode(I, 1) at 0x...>, <DynamicNode(G, 1) at 0x...>),
    (<DynamicNode(L, 0) at 0x...>, <DynamicNode(L, 1) at 0x...>)])

    If any variable is not present in the network while adding an edge,
    pgmpy will automatically add that variable to the network.

    But for adding nodes to the model we don't need to specify the time
    slice as it is common in all the time slices. And therefore pgmpy
    automatically replicated it all the time slices. For example, for
    adding a new variable `S` in the above network we can simply do:

    >>> dbn.add_node("S")
    >>> dbn.nodes() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    NodeView((<DynamicNode(D, 0) at 0x...>,
    <DynamicNode(G, 0) at 0x...>,
    <DynamicNode(D, 1) at 0x...>,
    <DynamicNode(G, 1) at 0x...>,
    <DynamicNode(I, 0) at 0x...>,
    <DynamicNode(I, 1) at 0x...>,
    <DynamicNode(L, 0) at 0x...>,
    <DynamicNode(L, 1) at 0x...>,
    <DynamicNode(S, 0) at 0x...>))

    Public Methods
    --------------
    add_node
    add_nodes_from
    add_edges
    add_edges_from
    add_cpds
    initialize_initial_state
    inter_slice
    intra_slice
    copy
    """

    def __init__(self, ebunch=None):
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_node(self, node, **attr):
        """
        Adds a single node to the Network

        Parameters
        ----------
        node: node
            A node can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_node("A")
        """
        pass

    def add_nodes_from(self, nodes, **attr):
        """
        Add multiple nodes to the Network.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(["A", "B", "C"])
        """
        pass

    def _nodes(self):
        """
        Returns the list of nodes present in the network

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(["A", "B", "C"])
        >>> sorted(dbn._nodes())
        ['A', 'B', 'C']
        """
        pass

    def _timeslices(self):
        pass

    def add_edge(self, start, end, **kwargs):
        """
        Add an edge between two nodes.

        The nodes will be automatically added if they are not present in the network.

        Parameters
        ----------
        start: tuple
               Both the start and end nodes should specify the time slice as
               (node_name, time_slice). Here, node_name can be any hashable
               python object while the time_slice is an integer value,
               which denotes the time slice that the node belongs to.

        end: tuple
               Both the start and end nodes should specify the time slice as
               (node_name, time_slice). Here, node_name can be any hashable
               python object while the time_slice is an integer value,
               which denotes the time slice that the node belongs to.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> model = DBN()
        >>> model.add_nodes_from(["D", "I"])
        >>> model.add_edge(("D", 0), ("I", 0))
        >>> sorted(model.edges()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [(<DynamicNode(D, 0) at 0x...>, <DynamicNode(I, 0) at 0x...>),
        (<DynamicNode(D, 1) at 0x...>, <DynamicNode(I, 1) at 0x...>)]
        """
        pass

    def add_edges_from(self, ebunch, **kwargs):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python object.

        Parameters
        ----------
        ebunch : list, array-like
                List of edges to add. Each edge must be of the form of
                ((start, time_slice), (end, time_slice)).

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(("D", 0), ("G", 0)), (("I", 0), ("G", 0))])
        >>> sorted(dbn.nodes()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<DynamicNode(D, 0) at 0x...>,
        <DynamicNode(D, 1) at 0x...>,
        <DynamicNode(G, 0) at 0x...>,
        <DynamicNode(G, 1) at 0x...>,
        <DynamicNode(I, 0) at 0x...>,
        <DynamicNode(I, 1) at 0x...>]
        >>> sorted(dbn.edges()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [(<DynamicNode(D, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
        (<DynamicNode(D, 1) at 0x...>, <DynamicNode(G, 1) at 0x...>),
        (<DynamicNode(I, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
        (<DynamicNode(I, 1) at 0x...>, <DynamicNode(G, 1) at 0x...>)]
        """
        pass

    def get_intra_edges(self, time_slice=0):
        """
        Returns the intra slice edges present in the 2-TBN.

        Parameters
        ----------
        time_slice: int (whole number)
                The time slice for which to get intra edges. The timeslice
                should be a positive value or zero.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(["D", "G", "I", "S", "L"])
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("G", 0), ("L", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...         (("G", 0), ("G", 1)),
        ...         (("G", 0), ("L", 1)),
        ...         (("L", 0), ("L", 1)),
        ...     ]
        ... )
        >>> sorted(dbn.get_intra_edges()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [(<DynamicNode(D, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
        (<DynamicNode(G, 0) at 0x...>, <DynamicNode(L, 0) at 0x...>),
        (<DynamicNode(I, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>)]
        """
        pass

    def get_inter_edges(self):
        """
        Returns the inter-slice edges present in the 2-TBN.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("G", 0), ("L", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...         (("G", 0), ("G", 1)),
        ...         (("G", 0), ("L", 1)),
        ...         (("L", 0), ("L", 1)),
        ...     ]
        ... )
        >>> sorted(dbn.get_inter_edges()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [(<DynamicNode(D, 0) at 0x...>, <DynamicNode(D, 1) at 0x...>),
        (<DynamicNode(G, 0) at 0x...>, <DynamicNode(G, 1) at 0x...>),
        (<DynamicNode(G, 0) at 0x...>, <DynamicNode(L, 1) at 0x...>),
        (<DynamicNode(I, 0) at 0x...>, <DynamicNode(I, 1) at 0x...>),
        (<DynamicNode(L, 0) at 0x...>, <DynamicNode(L, 1) at 0x...>)]
        """
        pass

    def get_interface_nodes(self, time_slice=0):
        """
        Returns the nodes in the first timeslice whose children are present in the first timeslice.

        Parameters
        ----------
        time_slice:int
                The timeslice should be a positive value greater than or equal to zero

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(["D", "G", "I", "S", "L"])
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("G", 0), ("L", 0)),
        ...         (("D", 0), ("D", 1)),
        ...     ]
        ... )
        >>> dbn.get_interface_nodes() # doctest: +ELLIPSIS
        [<DynamicNode(D, 0) at 0x...>]
        """
        pass

    def get_slice_nodes(self, time_slice=0):
        """
        Returns the nodes present in a particular timeslice

        Parameters
        ----------
        time_slice:int
                The timeslice should be a positive value greater than or equal to zero

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(["D", "G", "I", "S", "L"])
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("G", 0), ("L", 0)),
        ...         (("D", 0), ("D", 1)),
        ...     ]
        ... )
        >>> sorted(dbn.get_slice_nodes()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<DynamicNode(D, 0) at 0x...>,
        <DynamicNode(G, 0) at 0x...>,
        <DynamicNode(I, 0) at 0x...>,
        <DynamicNode(L, 0) at 0x...>,
        <DynamicNode(S, 0) at 0x...>]
        """
        pass

    def add_cpds(self, *cpds):
        """
        This method adds the cpds to the Dynamic Bayesian Network.
        Note that while adding variables and the evidence in cpd,
        they have to be of the following form
        (node_name, time_slice)
        Here, node_name is the node that is inserted
        while the time_slice is an integer value, which denotes
        the index of the time_slice that the node belongs to.

        Parameters
        ----------
        cpds : list, set, tuple (array-like)
            List of CPDs which are to be associated with the model. Each CPD
            should be an instance of `TabularCPD`.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> import numpy as np
        >>> dbn = DBN()
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...     ]
        ... )
        >>> grade_cpd = TabularCPD(
        ...     ("G", 0),
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     evidence=[("I", 0), ("D", 0)],
        ...     evidence_card=[2, 2],
        ... )
        >>> d_i_cpd = TabularCPD(
        ...     ("D", 1),
        ...     2,
        ...     [[0.6, 0.3], [0.4, 0.7]],
        ...     evidence=[("D", 0)],
        ...     evidence_card=[2],
        ... )
        >>> diff_cpd = TabularCPD(("D", 0), 2, np.array([[0.6], [0.4]]))
        >>> intel_cpd = TabularCPD(("I", 0), 2, np.array([[0.7], [0.3]]))
        >>> i_i_cpd = TabularCPD(
        ...     ("I", 1),
        ...     2,
        ...     [[0.5, 0.4], [0.5, 0.6]],
        ...     evidence=[("I", 0)],
        ...     evidence_card=[2],
        ... )
        >>> dbn.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> sorted(dbn.get_cpds(), key=lambda cpd: str(cpd.variable)) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(('D', 0):2) at 0x...>,
        <TabularCPD representing P(('D', 1):2 | ('D', 0):2) at 0x...>,
        <TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0x...>,
        <TabularCPD representing P(('I', 0):2) at 0x...>,
        <TabularCPD representing P(('I', 1):2 | ('I', 0):2) at 0x...>]
        """
        pass

    def get_cpds(self, node=None, time_slice=None):
        """
        Returns the CPDs that have been associated with the network.

        Parameters
        ----------
        node: tuple (node_name, time_slice)
            The node should be in the following form (node_name, time_slice).
            Here, node_name is the node that is inserted while the time_slice is
            an integer value, which denotes the index of the time_slice that the
            node belongs to.

        time_slice: int
            The time_slice should be a positive integer greater than or equal to zero.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...     ]
        ... )
        >>> grade_cpd = TabularCPD(
        ...     ("G", 0),
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     [("I", 0), ("D", 0)],
        ...     [2, 2],
        ... )
        >>> dbn.add_cpds(grade_cpd)
        >>> dbn.get_cpds() # doctest: +ELLIPSIS
        [<TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0x...>]
        """
        pass

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds : list, set, tuple (array-like)
            List of CPDs which are to be associated with the model. Each CPD
            should be an instance of `TabularCPD`.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...     ]
        ... )
        >>> grade_cpd = TabularCPD(
        ...     ("G", 0),
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     [("I", 0), ("D", 0)],
        ...     [2, 2],
        ... )
        >>> dbn.add_cpds(grade_cpd)
        >>> dbn.get_cpds() # doctest: +ELLIPSIS
        [<TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0x...>]
        >>> dbn.remove_cpds(grade_cpd)
        >>> dbn.get_cpds()
        []
        """
        pass

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities in each associated CPD for each
            state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        boolean: True if everything seems to be order. Otherwise raises error
            according to the problem.
        """
        pass

    def initialize_initial_state(self):
        """
        This method will automatically re-adjust the cpds and the edges added to the Bayesian Network.
        If an edge that is added as an intra time slice edge in the 0th timeslice, this method will
        automatically add it in the 1st timeslice. It will also add the cpds. However, to call this
        method, one needs to add cpds as well as the edges in the Bayesian Network of the whole
        skeleton including the 0th and the 1st timeslice,.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> import numpy as np
        >>> student = DBN()
        >>> student.add_nodes_from(["D", "G", "I", "S", "L"])
        >>> student.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...     ]
        ... )
        >>> grade_cpd = TabularCPD(
        ...     ("G", 0),
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     evidence=[("I", 0), ("D", 0)],
        ...     evidence_card=[2, 2],
        ... )
        >>> d_i_cpd = TabularCPD(
        ...     ("D", 1),
        ...     2,
        ...     [[0.6, 0.3], [0.4, 0.7]],
        ...     evidence=[("D", 0)],
        ...     evidence_card=[2],
        ... )
        >>> diff_cpd = TabularCPD(("D", 0), 2, np.array([[0.6], [0.4]]))
        >>> intel_cpd = TabularCPD(("I", 0), 2, np.array([[0.7], [0.3]]))
        >>> i_i_cpd = TabularCPD(
        ...     ("I", 1),
        ...     2,
        ...     [[0.5, 0.4], [0.5, 0.6]],
        ...     evidence=[("I", 0)],
        ...     evidence_card=[2],
        ... )
        >>> student.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> student.initialize_initial_state()
        """
        pass

    def moralize(self):
        """
        Removes all the immoralities in the Network and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN([(("D", 0), ("G", 0)), (("I", 0), ("G", 0))])
        >>> moral_graph = dbn.moralize()
        >>> sorted(moral_graph.edges()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [(<DynamicNode(D, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
        (<DynamicNode(D, 0) at 0x...>, <DynamicNode(I, 0) at 0x...>),
        (<DynamicNode(D, 1) at 0x...>, <DynamicNode(G, 1) at 0x...>),
        (<DynamicNode(D, 1) at 0x...>, <DynamicNode(I, 1) at 0x...>),
        (<DynamicNode(G, 0) at 0x...>, <DynamicNode(I, 0) at 0x...>),
        (<DynamicNode(G, 1) at 0x...>, <DynamicNode(I, 1) at 0x...>)]
        """
        pass

    def copy(self):
        """
        Returns a copy of the Dynamic Bayesian Network.

        Returns
        -------
        DynamicBayesianNetwork: copy of the Dynamic Bayesian Network

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...     ]
        ... )
        >>> grade_cpd = TabularCPD(
        ...     ("G", 0),
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     [("I", 0), ("D", 0)],
        ...     [2, 2],
        ... )
        >>> dbn.add_cpds(grade_cpd)
        >>> dbn_copy = dbn.copy()
        >>> sorted(dbn_copy.nodes()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<DynamicNode(D, 0) at 0x...>,
        <DynamicNode(D, 1) at 0x...>,
        <DynamicNode(G, 0) at 0x...>,
        <DynamicNode(G, 1) at 0x...>,
        <DynamicNode(I, 0) at 0x...>,
        <DynamicNode(I, 1) at 0x...>]
        >>> sorted(dbn_copy.edges()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [(<DynamicNode(D, 0) at 0x...>, <DynamicNode(D, 1) at 0x...>),
        (<DynamicNode(D, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
        (<DynamicNode(D, 1) at 0x...>, <DynamicNode(G, 1) at 0x...>),
        (<DynamicNode(I, 0) at 0x...>, <DynamicNode(G, 0) at 0x...>),
        (<DynamicNode(I, 0) at 0x...>, <DynamicNode(I, 1) at 0x...>),
        (<DynamicNode(I, 1) at 0x...>, <DynamicNode(G, 1) at 0x...>)]
        >>> dbn_copy.get_cpds() # doctest: +ELLIPSIS
        [<TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0x...>]
        """
        pass

    def get_markov_blanket(self, node):
        # Wrap node into DynamicNode
        pass

    def get_constant_bn(self, t_slice=0):
        """
        Returns a normal Bayesian Network object which has nodes from the first two
        time slices and all the edges in the first time slice and edges going from
        first to second time slice. The returned Bayesian Network basically represents
        the part of the DBN which remains constant.

        The node names are changed to strings in the form `{var}_{time}`.
        """
        pass

    def fit(self, data, estimator="MLE"):
        """
        Learns the CPD of the model from data.

        Since the assumption is that the 2-TBN stays constant throughtout the model,
        the algorithm iterates over every 2 consecutive time slices in the data and
        updates the CPDs based on it.

        Parameters
        ----------
        data: pandas.DataFrame instance
            The column names must be of the form (variable, time_slice). The
            time-slices must start from 0.

        estimator: str
            Currently only Maximum Likelihood Estimator is supported.

        Returns
        -------
        None: The CPDs are added to the model instance.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> model = DBN(
        ...     [
        ...         (("A", 0), ("B", 0)),
        ...         (("A", 0), ("C", 0)),
        ...         (("B", 0), ("D", 0)),
        ...         (("C", 0), ("D", 0)),
        ...         (("A", 0), ("A", 1)),
        ...         (("B", 0), ("B", 1)),
        ...         (("C", 0), ("C", 1)),
        ...         (("D", 0), ("D", 1)),
        ...     ]
        ... )
        >>> data = np.random.randint(low=0, high=2, size=(1000, 20))
        >>> colnames = []
        >>> for t in range(5):
        ...     colnames.extend([("A", t), ("B", t), ("C", t), ("D", t)])
        pass
        >>> df = pd.DataFrame(data, columns=colnames)
        >>> model.fit(df)
        """
        pass

    def active_trail_nodes(self, variables, observed=None, include_latents=False):
        pass

    @staticmethod
    def _postprocess(df):
        """
        Postprocess the generated samples before returning.
        1. Remove any of the variables created because of the virtual evidence or intervention. Variables
           starting with `__`
        2. Change the column names from str to tuples.
        """
        pass

    def simulate(
        self,
        n_samples=10,
        n_time_slices=2,
        do=None,
        evidence=None,
        virtual_evidence=None,
        virtual_intervention=None,
        include_latents=False,
        seed=None,
        show_progress=True,
        return_format="wide",
    ):
        """
        Simulates time-series data from the specified model.

        Parameters
        ----------
        n_samples: int
            The number of data samples to simulate from the model.

        n_time_slices: int
            The number of time slices for which to simulate the data.

        do: dict
            The interventions to apply to the model. dict should be of the form
            {(variable_name, time_slice): state}

        evidence: dict
            Observed evidence to apply to the model. dict should be of the form
            {(variable_name, time_slice): state}

        virtual_evidence: list
            Probabilistically apply evidence to the model. `virtual_evidence` should
            be a list of `pgmpy.factors.discrete.TabularCPD` objects specifying the
            virtual probabilities.

        virtual_intervention: list
            Also known as soft intervention. `virtual_intervention` should be a list
            of `pgmpy.factors.discrete.TabularCPD` objects specifying the virtual/soft
            intervention probabilities.

        include_latents: boolean (default: False)
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: bool
            If True, shows a progress bar when generating samples.

        return_format: {"wide", "numpy3d", "pd-multiindex", "pd-list", "sorted"}
            Controls the return representation

            - "wide" : Default option : wide format, where on rows we have samples, and on columns we have (potentially
                       unsorted)  ("variable", "timestep)

            - 'numpy3d' : returns a 3D numpy array, where first dimension represents trace, second dimension
                        represents variable, third dimension represent timestep

            - 'pd-multiindex' : returns the pandas multindex DataFrame, with indexes of ("Variable name", "timestep")

            - 'pd-list' : returns a list of pandas DataFrames. For every sample, a Dataframe is created, where rows
                        contain timestep and columns represent variables

            - 'sorted' : makes sure that the representation of [sample, ("variable", "timestep")] is sorted, which
                       makes further processing easier


        Returns
        -------
        np.ndarray or pandas.DataFrame
            Depends on `return_format` argument. `numpy3d` returns a numpy array (np.ndarray), while rest of the
            representations return a pandas DataFrame.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN(
        ...     [
        ...         (("D", 0), ("G", 0)),
        ...         (("I", 0), ("G", 0)),
        ...         (("D", 0), ("D", 1)),
        ...         (("I", 0), ("I", 1)),
        ...     ]
        ... )
        >>> diff_cpd = TabularCPD(("D", 0), 2, [[0.6], [0.4]])
        >>> grade_cpd = TabularCPD(
        ...     variable=("G", 0),
        ...     variable_card=3,
        ...     values=[
        ...         [0.3, 0.05, 0.9, 0.5],
        ...         [0.4, 0.25, 0.08, 0.3],
        ...         [0.3, 0.7, 0.02, 0.2],
        ...     ],
        ...     evidence=[("I", 0), ("D", 0)],
        ...     evidence_card=[2, 2],
        ... )
        >>> d_i_cpd = TabularCPD(
        ...     variable=("D", 1),
        ...     variable_card=2,
        ...     values=[[0.6, 0.3], [0.4, 0.7]],
        ...     evidence=[("D", 0)],
        ...     evidence_card=[2],
        ... )
        >>> intel_cpd = TabularCPD(("I", 0), 2, [[0.7], [0.3]])
        >>> i_i_cpd = TabularCPD(
        ...     variable=("I", 1),
        ...     variable_card=2,
        ...     values=[[0.5, 0.4], [0.5, 0.6]],
        ...     evidence=[("I", 0)],
        ...     evidence_card=[2],
        ... )
        >>> g_i_cpd = TabularCPD(
        ...     variable=("G", 1),
        ...     variable_card=3,
        ...     values=[
        ...         [0.3, 0.05, 0.9, 0.5],
        ...         [0.4, 0.25, 0.08, 0.3],
        ...         [0.3, 0.7, 0.02, 0.2],
        ...     ],
        ...     evidence=[("I", 1), ("D", 1)],
        ...     evidence_card=[2, 2],
        ... )
        >>> dbn.add_cpds(diff_cpd, grade_cpd, d_i_cpd, intel_cpd, i_i_cpd, g_i_cpd)

        Normal simulation from the model.

        >>> dbn.simulate(n_time_slices=4, n_samples=2, seed=42) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
          (D, 0) (G, 0) (I, 0) (D, 1)  ... (D, 3) (G, 3) (I, 2) (I, 3)
        0      0      0      1      0  ...      1      0      1      1
        1      1      1      0      0  ...      1      0      1      1
        <BLANKLINE>
        [2 rows x 12 columns]

        Simulation with evidence.

        >>> dbn.simulate(
        ...     n_time_slices=4, n_samples=2, evidence={("D", 0): 1, ("D", 2): 0}, seed=42
        ... )  # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
          (D, 0) (G, 0) (I, 0) (D, 1)  ... (D, 3) (G, 3) (I, 2) (I, 3)
        0      1      2      0      1  ...      0      0      1      1
        1      1      0      0      0  ...      1      0      0      1
        <BLANKLINE>
        [2 rows x 12 columns]

        Simulation with virtual/soft evidence.

        >>> dbn.simulate(
        ...     n_time_slices=4,
        ...     n_samples=2,
        ...     virtual_evidence=[TabularCPD(("D", 2), 2, [[0.7], [0.3]])],
        ...     seed=42
        ... ) # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
          (D, 0) (G, 0) (I, 0) (D, 1)  ... (D, 3) (G, 3) (I, 2) (I, 3)
        0      0      0      1      0  ...      1      1      1      0
        1      1      1      0      0  ...      1      1      1      0
        <BLANKLINE>
        [2 rows x 12 columns]

        Simulation with intervention.

        >>> dbn.simulate(n_time_slices=4, n_samples=2,
        ...     do={("D", 0): 1, ("D", 2): 0}, seed=42) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
          (D, 0) (G, 0) (I, 0) (D, 1)  ... (D, 3) (G, 3) (I, 2) (I, 3)
        0      1      2      0      0  ...      0      0      1      1
        1      1      0      0      0  ...      1      0      1      1
        <BLANKLINE>
        [2 rows x 12 columns]

        Simulation with virtual/soft intervention.

        >>> dbn.simulate(
        ...     n_time_slices=4,
        ...     n_samples=2,
        ...     virtual_intervention=[TabularCPD(("D", 2), 2, [[0.7], [0.3]])],
        ...     seed=42
        ... ) # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
          (D, 0) (G, 0) (I, 0) (D, 1)  ... (D, 3) (G, 3) (I, 2) (I, 3)
        0      0      0      1      0  ...      1      1      0      0
        1      1      1      0      0  ...      1      0      1      1
        <BLANKLINE>
        [2 rows x 12 columns]

        Return format selection using `return_format` argument.
        `return_format="wide"` returns the data in standard format.

        >>> dbn.simulate(n_samples=2, n_time_slices=3,
        ...         return_format="wide", seed=42) # doctest:  +ELLIPSIS +NORMALIZE_WHITESPACE
          (D, 0) (G, 0) (I, 0) (D, 1) (G, 1) (D, 2) (G, 2) (I, 1) (I, 2)
        0      0      0      1      0      0      0      0      1      1
        1      1      1      0      0      1      1      0      1      1

        `return_format="pd-multiindex"` returns pandas dataframe with indexes of ("Variable name", "timestep").

        >>> dbn.simulate(n_samples=2, n_time_slices=3,
        ...     return_format="pd-multiindex", seed=42) # doctest:  +ELLIPSIS +NORMALIZE_WHITESPACE
        variable       D  G  I
        instance time
        0        0     0  0  1
                 1     0  0  1
                 2     0  0  1
        1        0     1  1  0
                 1     0  1  1
                 2     1  0  1
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
