from collections import defaultdict
from itertools import chain, combinations, tee

from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation, Inference


class DBNInference(Inference):
    """
    Class for performing inference using Belief Propagation method
    for the input Dynamic Bayesian Network.

    For the exact inference implementation, the interface algorithm
    is used which is adapted from [1].

    Parameters
    ----------
    model: Dynamic Bayesian Network
        Model for which inference is to performed

    Examples
    --------
    >>> from pgmpy.factors.discrete import TabularCPD
    >>> from pgmpy.models import DynamicBayesianNetwork as DBN
    >>> from pgmpy.inference import DBNInference
    >>> dbnet = DBN()
    >>> dbnet.add_edges_from(
    ...     [(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))]
    ... )
    >>> z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
    >>> x_i_cpd = TabularCPD(
    ...     ("X", 0),
    ...     2,
    ...     [[0.6, 0.9], [0.4, 0.1]],
    ...     evidence=[("Z", 0)],
    ...     evidence_card=[2],
    ... )
    >>> y_i_cpd = TabularCPD(
    ...     ("Y", 0),
    ...     2,
    ...     [[0.2, 0.3], [0.8, 0.7]],
    ...     evidence=[("X", 0)],
    ...     evidence_card=[2],
    ... )
    >>> z_trans_cpd = TabularCPD(
    ...     ("Z", 1),
    ...     2,
    ...     [[0.4, 0.7], [0.6, 0.3]],
    ...     evidence=[("Z", 0)],
    ...     evidence_card=[2],
    ... )
    >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
    >>> dbnet.initialize_initial_state()
    >>> dbn_inf = DBNInference(dbnet)
    >>> sorted(dbn_inf.start_junction_tree.nodes()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [(<DynamicNode(X, 0) at 0x...>, <DynamicNode(Y, 0) at 0x...>),
    (<DynamicNode(X, 0) at 0x...>, <DynamicNode(Z, 0) at 0x...>)]
    >>> sorted(tuple(sorted(n)) for n in dbn_inf.one_and_half_junction_tree.nodes())
    ...    # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [(<DynamicNode(X, 1) at 0x...>, <DynamicNode(Y, 1) at 0x...>),
    (<DynamicNode(X, 1) at 0x...>, <DynamicNode(Z, 1) at 0x...>),
    (<DynamicNode(Z, 0) at 0x...>, <DynamicNode(Z, 1) at 0x...>)]

    References
    ----------
    [1] Dynamic Bayesian Networks: Representation, Inference and Learning
        by Kevin Patrick Murphy
        http://www.cs.ubc.ca/~murphyk/Thesis/thesis.pdf
    """

    def __init__(self, model):
        super().__init__(model)
        self._initialize_structures()

        self.interface_nodes_0 = model.get_interface_nodes(time_slice=0)
        self.interface_nodes_1 = model.get_interface_nodes(time_slice=1)

        start_markov_model = self.start_bayesian_model.to_markov_model()
        one_and_half_markov_model = self.one_and_half_model.to_markov_model()

        combinations_slice_0 = tee(combinations(set(self.interface_nodes_0), 2), 2)
        combinations_slice_1 = combinations(set(self.interface_nodes_1), 2)

        start_markov_model.add_edges_from(combinations_slice_0[0])
        one_and_half_markov_model.add_edges_from(chain(combinations_slice_0[1], combinations_slice_1))

        self.one_and_half_junction_tree = one_and_half_markov_model.to_junction_tree()
        self.start_junction_tree = start_markov_model.to_junction_tree()

        self.start_interface_clique = self._get_clique(self.start_junction_tree, self.interface_nodes_0)
        self.in_clique = self._get_clique(self.one_and_half_junction_tree, self.interface_nodes_0)
        self.out_clique = self._get_clique(self.one_and_half_junction_tree, self.interface_nodes_1)

    def _shift_nodes(self, nodes, time_slice):
        """
        Shifting the nodes to a certain required timeslice.

        Parameters
        ----------
        nodes: list, array-like
            List of node names.
            nodes that are to be shifted to some other time slice.

        time_slice: int
            time slice where to shift the nodes.
        """
        pass

    def _get_clique(self, junction_tree, nodes):
        """
        Extracting the cliques from the junction tree which are a subset of
        the given nodes.

        Parameters
        ----------
        junction_tree: Junction tree
            from which the nodes are to be extracted.

        nodes: iterable container
            A container of nodes (list, dict, set, etc.).
        """
        pass

    def _get_evidence(self, evidence_dict, time_slice, shift):
        """
        Getting the evidence belonging to a particular timeslice.

        Parameters
        ----------
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        time: int
            the evidence corresponding to the time slice

        shift: int
            shifting the evidence corresponding to the given time slice.
        """
        pass

    def _marginalize_factor(self, nodes, factor):
        """
        Marginalizing the factor selectively for a set of variables.

        Parameters
        ----------
        nodes: list, array-like
            A container of nodes (list, dict, set, etc.).

        factor: factor
            factor which is to be marginalized.
        """
        pass

    def _update_belief(self, belief_prop, clique, clique_potential, message=None):
        """
        Method for updating the belief.

        Parameters
        ----------
        belief_prop: Belief Propagation
            Belief Propagation which needs to be updated.

        in_clique: clique
            The factor which needs to be updated corresponding to the input clique.

        out_clique_potential: factor
            Multiplying factor which will be multiplied to the factor corresponding to the clique.
        """
        pass

    def _get_factor(self, belief_prop, evidence):
        """
        Extracts the required factor from the junction tree.

        Parameters
        ----------
        belief_prop: Belief Propagation
            Belief Propagation which needs to be updated.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
        """
        pass

    def _shift_factor(self, factor, shift):
        """
        Shifting the factor to a certain required time slice.

        Parameters
        ----------
        factor: DiscreteFactor
           The factor which needs to be shifted.

        shift: int
           The new timeslice to which the factor should belong to.
        """
        pass

    def forward_inference(self, variables, evidence=None, args=None):
        """
        Forward inference method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.inference import DBNInference
        >>> dbnet = DBN()
        >>> dbnet.add_edges_from(
        ...     [(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))]
        ... )
        >>> z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
        >>> x_i_cpd = TabularCPD(
        ...     ("X", 0),
        ...     2,
        ...     [[0.6, 0.9], [0.4, 0.1]],
        ...     evidence=[("Z", 0)],
        ...     evidence_card=[2],
        ... )
        >>> y_i_cpd = TabularCPD(
        ...     ("Y", 0),
        ...     2,
        ...     [[0.2, 0.3], [0.8, 0.7]],
        ...     evidence=[("X", 0)],
        ...     evidence_card=[2],
        ... )
        >>> z_trans_cpd = TabularCPD(
        ...     ("Z", 1),
        ...     2,
        ...     [[0.4, 0.7], [0.6, 0.3]],
        ...     evidence=[("Z", 0)],
        ...     evidence_card=[2],
        ... )
        >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.forward_inference(
        ...     [("X", 2)], {("Y", 0): 1, ("Y", 1): 0, ("Y", 2): 1}
        ... )[("X", 2)].values
        array([0.76738736, 0.23261264])
        """
        pass

    def backward_inference(self, variables, evidence=None):
        """
        Backward inference method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.inference import DBNInference
        >>> dbnet = DBN()
        >>> dbnet.add_edges_from(
        ...     [(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))]
        ... )
        >>> z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
        >>> x_i_cpd = TabularCPD(
        ...     ("X", 0),
        ...     2,
        ...     [[0.6, 0.9], [0.4, 0.1]],
        ...     evidence=[("Z", 0)],
        ...     evidence_card=[2],
        ... )
        >>> y_i_cpd = TabularCPD(
        ...     ("Y", 0),
        ...     2,
        ...     [[0.2, 0.3], [0.8, 0.7]],
        ...     evidence=[("X", 0)],
        ...     evidence_card=[2],
        ... )
        >>> z_trans_cpd = TabularCPD(
        ...     ("Z", 1),
        ...     2,
        ...     [[0.4, 0.7], [0.6, 0.3]],
        ...     evidence=[("Z", 0)],
        ...     evidence_card=[2],
        ... )
        >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.backward_inference(
        ...     [("X", 0)], {("Y", 0): 0, ("Y", 1): 1, ("Y", 2): 1}
        ... )[("X", 0)].values
        array([0.66594382, 0.33405618])
        """
        pass

    def query(self, variables, evidence=None, args="exact"):
        """
        Query method for Dynamic Bayesian Network using Interface Algorithm.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.inference import DBNInference
        >>> dbnet = DBN()
        >>> dbnet.add_edges_from(
        ...     [(("Z", 0), ("X", 0)), (("X", 0), ("Y", 0)), (("Z", 0), ("Z", 1))]
        ... )
        >>> z_start_cpd = TabularCPD(("Z", 0), 2, [[0.5], [0.5]])
        >>> x_i_cpd = TabularCPD(
        ...     ("X", 0),
        ...     2,
        ...     [[0.6, 0.9], [0.4, 0.1]],
        ...     evidence=[("Z", 0)],
        ...     evidence_card=[2],
        ... )
        >>> y_i_cpd = TabularCPD(
        ...     ("Y", 0),
        ...     2,
        ...     [[0.2, 0.3], [0.8, 0.7]],
        ...     evidence=[("X", 0)],
        ...     evidence_card=[2],
        ... )
        >>> z_trans_cpd = TabularCPD(
        ...     ("Z", 1),
        ...     2,
        ...     [[0.4, 0.7], [0.6, 0.3]],
        ...     evidence=[("Z", 0)],
        ...     evidence_card=[2],
        ... )
        >>> dbnet.add_cpds(z_start_cpd, z_trans_cpd, x_i_cpd, y_i_cpd)
        >>> dbnet.initialize_initial_state()
        >>> dbn_inf = DBNInference(dbnet)
        >>> dbn_inf.query([("X", 0)], {("Y", 0): 0, ("Y", 1): 1, ("Y", 2): 1})[
        ...     ("X", 0)
        ... ].values
        array([0.66594382, 0.33405618])
        """
        pass
