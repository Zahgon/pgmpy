#!/usr/bin/env python3
import copy
import itertools
from collections.abc import Hashable
from functools import reduce

import networkx as nx
import numpy as np
from opt_einsum import contract
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Inference
from pgmpy.inference.EliminationOrder import (
    MinFill,
    MinNeighbors,
    MinWeight,
    WeightedMinFill,
)
from pgmpy.models import (
    DiscreteBayesianNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    FunctionalBayesianNetwork,
    JunctionTree,
    LinearGaussianBayesianNetwork,
)
from pgmpy.utils import compat_fns


class VariableElimination(Inference):
    def _get_working_factors(self, evidence):
        """
        Uses the evidence given to the query methods to modify the factors before running
        the variable elimination algorithm.

        Parameters
        ----------
        evidence: dict
            Dict of the form {variable: state}

        Returns
        -------
        dict: Modified working factors.
        """
        pass

    def _get_elimination_order(self, variables, evidence, elimination_order, show_progress=True):
        """
        Deals with all elimination order parameters given to _variable_elimination method
        and returns a list of variables that are to be eliminated

        Parameters
        ----------
        elimination_order: str or list

        Returns
        -------
        list: A list of variables names in the order they need to be eliminated.
        """
        pass

    def _variable_elimination(
        self,
        variables,
        operation,
        evidence=None,
        elimination_order="MinFill",
        joint=True,
        show_progress=True,
    ):
        """
        Implementation of a generalized variable elimination.

        Parameters
        ----------
        variables: list, array-like
            variables that are not to be eliminated.

        operation: str ('marginalize' | 'maximize')
            The operation to do for eliminating the variable.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        elimination_order: str or list (array-like)
            If str: Heuristic to use to find the elimination order.
            If array-like: The elimination order to use.
            If None: A random elimination order is used.
        """
        pass

    def query(
        self,
        variables: list[Hashable],
        evidence: dict[Hashable, int] | None = None,
        virtual_evidence: list | None = None,
        elimination_order="greedy",
        joint=True,
        show_progress=True,
    ):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        elimination_order: str or list (default='greedy')
            Order in which to eliminate the variables in the algorithm. If list is provided,
            should contain all variables in the model except the ones in `variables`. str options
            are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
            refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
            for details.

        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models...DiscreteBayesianNetwork object at 0x...>
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.query(["A", "B"])
        """
        pass

    def max_marginal(
        self,
        variables=None,
        evidence=None,
        elimination_order="MinFill",
        show_progress=True,
    ):
        """
        Computes the max-marginal over the variables given the evidence.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models...DiscreteBayesianNetwork object at 0x...>
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.max_marginal(["A", "B"])
        """
        pass

    def map_query(
        self,
        variables=None,
        evidence=None,
        virtual_evidence=None,
        elimination_order="MinFill",
        show_progress=True,
    ):
        """
        Computes the MAP Query over the variables given the evidence. Returns the
        highest probable state in the joint distribution of `variables`.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models...DiscreteBayesianNetwork object at 0x...>
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.map_query(["A", "B"])
        """
        pass

    def induced_graph(self, elimination_order):
        """
        Returns the induced graph formed by running Variable Elimination on the network.

        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models...DiscreteBayesianNetwork object at 0x...>
        >>> inference = VariableElimination(model)
        >>> inference.induced_graph(["C", "D", "A", "B", "E"])  # doctest: +ELLIPSIS
        <networkx.classes.graph.Graph object at 0x...>
        """
        pass

    def induced_width(self, elimination_order):
        """
        Returns the width (integer) of the induced graph formed by running Variable Elimination on the network.
        The width is the defined as the number of nodes in the largest clique in the graph minus 1.

        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models...DiscreteBayesianNetwork object at 0x...>
        >>> inference = VariableElimination(model)
        >>> inference.induced_width(["C", "D", "A", "B", "E"])
        3
        """
        pass


class BeliefPropagation(Inference):
    """
    Class for performing inference using Belief Propagation method.

    Creates a Junction Tree or Clique Tree (JunctionTree class) for the input
    probabilistic graphical model and performs calibration of the junction tree
    so formed using belief propagation.

    Parameters
    ----------
    model: DiscreteBayesianNetwork, DiscreteMarkovNetwork, FactorGraph, JunctionTree
        model for which inference is to performed
    """

    def __init__(self, model):
        super().__init__(model)

        if not isinstance(model, JunctionTree):
            self.junction_tree = model.to_junction_tree()
        else:
            self.junction_tree = copy.deepcopy(model)

        self.clique_beliefs = {}
        self.sepset_beliefs = {}

    def get_cliques(self):
        """
        Returns cliques used for belief propagation.
        """
        pass

    def get_clique_beliefs(self):
        """
        Returns clique beliefs. Should be called after the clique tree (or
        junction tree) is calibrated.
        """
        pass

    def get_sepset_beliefs(self):
        """
        Returns sepset beliefs. Should be called after clique tree (or junction
        tree) is calibrated.
        """
        pass

    def _update_beliefs(self, sending_clique, receiving_clique, operation):
        """
        This is belief-update method.

        Parameters
        ----------
        sending_clique: node (as the operation is on junction tree, node should be a tuple)
            Node sending the message

        receiving_clique: node (as the operation is on junction tree, node should be a tuple)
            Node receiving the message

        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Takes belief of one clique and uses it to update the belief of the
        neighboring ones.
        """
        pass

    def _is_converged(self, operation):
        r"""
        Checks whether the calibration has converged or not. At convergence
        the sepset belief would be precisely the sepset marginal.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
            if operation == marginalize, it checks whether the junction tree is calibrated or not
            else if operation == maximize, it checks whether the junction tree is max calibrated or not

        Formally, at convergence or at calibration this condition would be satisfied for

        .. math:: \sum_{C_i - S_{i, j}} \beta_i = \sum_{C_j - S_{i, j}} \beta_j = \mu_{i, j}

        and at max calibration this condition would be satisfied

        .. math:: \max_{C_i - S_{i, j}} \beta_i = \max_{C_j - S_{i, j}} \beta_j = \mu_{i, j}
        """
        pass

    def _calibrate_junction_tree(self, operation):
        """
        Generalized calibration of junction tree or clique using belief propagation. This method can be used for both
        calibrating as well as max-calibrating.
        Uses Lauritzen-Spiegelhalter algorithm or belief-update message passing.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Reference
        ---------
        Algorithm 10.3 Calibration using belief propagation in clique tree
        Probabilistic Graphical Models: Principles and Techniques
        Daphne Koller and Nir Friedman.
        """
        pass

    def calibrate(self):
        """
        Calibration using belief propagation in junction tree or clique tree.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = DiscreteBayesianNetwork(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ...     ],
        ...     evidence=["diff", "intel"],
        ...     evidence_card=[2, 3],
        ... )
        >>> sat_cpd = TabularCPD(
        ...     "SAT",
        ...     2,
        ...     [[0.1, 0.2, 0.7], [0.9, 0.8, 0.3]],
        ...     evidence=["intel"],
        ...     evidence_card=[3],
        ... )
        >>> letter_cpd = TabularCPD(
        ...     "letter",
        ...     2,
        ...     [[0.1, 0.4, 0.8], [0.9, 0.6, 0.2]],
        ...     evidence=["grade"],
        ...     evidence_card=[3],
        ... )
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.calibrate()
        """
        pass

    def max_calibrate(self):
        """
        Max-calibration of the junction tree using belief propagation.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = DiscreteBayesianNetwork(
        ...     [
        ...         ("diff", "grade"),
        ...         ("intel", "grade"),
        ...         ("intel", "SAT"),
        ...         ("grade", "letter"),
        ...     ]
        ... )
        >>> diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD("intel", 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        ...     ],
        ...     evidence=["diff", "intel"],
        ...     evidence_card=[2, 3],
        ... )
        >>> sat_cpd = TabularCPD(
        ...     "SAT",
        ...     2,
        ...     [[0.1, 0.2, 0.7], [0.9, 0.8, 0.3]],
        ...     evidence=["intel"],
        ...     evidence_card=[3],
        ... )
        >>> letter_cpd = TabularCPD(
        ...     "letter",
        ...     2,
        ...     [[0.1, 0.4, 0.8], [0.9, 0.6, 0.2]],
        ...     evidence=["grade"],
        ...     evidence_card=[3],
        ... )
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.max_calibrate()
        """
        pass

    def _query(self, variables, operation, evidence=None, joint=True, show_progress=True):
        """
        This is a generalized query method that can be used for both query and map query.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.inference import BeliefPropagation
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> model = DiscreteBayesianNetwork(
        ...     [("A", "B"), ("C", "B"), ("C", "D"), ("B", "E")]
        ... )
        >>> model.fit(values)  # doctest: +ELLIPSIS
        <pgmpy.models...DiscreteBayesianNetwork object at 0x...>
        >>> inference = BeliefPropagation(model)
        >>> phi_query = inference.query(["A", "B"])

        References
        ----------
        Algorithm 10.4 Out-of-clique inference in clique tree
        Probabilistic Graphical Models: Principles and Techniques Daphne Koller and Nir Friedman.
        """
        pass

    def query(
        self,
        variables,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        show_progress=True,
    ):
        """
        Query method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        joint: boolean
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = DiscreteBayesianNetwork(
        ...     [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")]
        ... )
        >>> cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD(
        ...     "J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2]
        ... )
        >>> cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
        >>> cpd_l = TabularCPD(
        ...     "L",
        ...     2,
        ...     [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        ...     ["G", "J"],
        ...     [2, 2],
        ... )
        >>> cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.query(
        ...     variables=["J", "Q"], evidence={"A": 0, "R": 0, "G": 0, "L": 1}
        ... )  # doctest: +ELLIPSIS
        <DiscreteFactor representing phi(J:2, Q:2) at 0x...>
        """
        pass

    def map_query(self, variables=None, evidence=None, virtual_evidence=None, show_progress=True):
        """
        MAP Query method using belief propagation. Returns the highest probable
        state in the joint distributon of `variables`.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = DiscreteBayesianNetwork(
        ...     [("A", "J"), ("R", "J"), ("J", "Q"), ("J", "L"), ("G", "L")]
        ... )
        >>> cpd_a = TabularCPD("A", 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD("R", 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD(
        ...     "J", 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ["R", "A"], [2, 2]
        ... )
        >>> cpd_q = TabularCPD("Q", 2, [[0.9, 0.2], [0.1, 0.8]], ["J"], [2])
        >>> cpd_l = TabularCPD(
        ...     "L",
        ...     2,
        ...     [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]],
        ...     ["G", "J"],
        ...     [2, 2],
        ... )
        >>> cpd_g = TabularCPD("G", 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.map_query(
        ...     variables=["J", "Q"], evidence={"A": 0, "R": 0, "G": 0, "L": 1}
        ... )  # doctest: +SKIP
        """
        pass


class BeliefPropagationWithMessagePassing(Inference):
    """
    Class for performing efficient inference using Belief Propagation method on factor graphs with no loops.

    The message-passing algorithm recursively parses the factor graph to propagate the
    model's beliefs to infer the posterior distribution of the queried variable. The recursion
    stops when reaching an observed variable or a unobserved root/leaf variable.

    It does not work for loopy graphs.

    Parameters
    ----------
    model: FactorGraph
        Model on which to run the inference.

    References
    ----------
    Algorithm 2.1 in https://www.mbmlbook.com/LearningSkills_Testing_out_the_model.html
    by J Winn (Microsoft Research).
    """

    def __init__(self, model: FactorGraph, check_model=True):
        assert isinstance(model, FactorGraph), "Model must be an instance of FactorGraph"
        if check_model:
            model.check_model()
        self.model = model

    class _RecursiveMessageSchedulingQuery:
        """
        Private class used in `BeliefPropagationWithMessagePassing.query()` to efficiently
        manage the message scheduling across the different queried variables, in a recursive way.

        Parameters
        ----------
        Same as in the query method.
        """

        def __init__(
            self,
            belief_propagation,
            variables,
            evidence,
            virtual_evidence,
            get_messages,
        ):
            self.bp = belief_propagation
            self.variables = variables
            self.evidence = evidence
            self.virtual_evidence = virtual_evidence
            self.all_messages = {} if get_messages else None

        def run(self):
            pass

        def schedule_variable_node_messages(
            self,
            variable,
            from_factor,
        ):
            """
            Returns the message sent by the variable to the factor requesting it.
            For that, the variable requests the messages coming from its neighbouring
            factors, except the one making the request.

            Parameters
            ----------
            variable: str
                The variable node from which to compute the outgoing message
            from_factor: pgmpy.factors.discrete.DiscreteFactor or None.
                The factor requesting the message, as part of the recursion.
                None for the first time this function is called.
            """
            pass

        def schedule_factor_node_messages(self, factor, from_variable):
            """
            Returns the message sent from the factor to the variable requesting it.
            For that, the factor requests the messages coming from its neighbouring
            variables, except the one making the request.

            Parameters
            ----------
            factor: pgmpy.factors.discrete.DiscreteFactor
                The factor from which we want to compute the outgoing message.
            from_variable: str
                The variable requesting the message, as part of the recursion.
            """
            pass

    def query(self, variables, evidence=None, virtual_evidence=None, get_messages=False):
        """
        Computes the posterior distributions for each of the queried variable,
        given the `evidence`, and the `virtual_evidence`. Optionally also returns
        the computed messages.

        Parameters
        ----------
        variables: list
            List of variables for which you want to compute the posterior.
        evidence: dict or None (default: None)
            A dict key, value pair as {var: state_of_var_observed}.
            None if no evidence.
        virtual_evidence: list or None (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences. Each virtual evidence becomes a virtual message that gets added to
            the list of computed messages incoming to the variable node.
            None if no virtual evidence.

        Returns
        -------
        If `get_messages` is False, returns a dict of the variables, posterior distributions
            pairs: {variable: pgmpy.factors.discrete.DiscreteFactor}.
        If `get_messages` is True, returns:
            1. A dict of the variables, posterior distributions pairs:
            {variable: pgmpy.factors.discrete.DiscreteFactor}
            2. A dict of all messages sent from a factor to a node:
            {"{pgmpy.factors.discrete.DiscreteFactor.variables} -> variable": np.array}.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.inference import BeliefPropagation
        >>> factor_graph = FactorGraph()
        >>> factor_graph.add_nodes_from(["A", "B", "C", "D"])
        >>> phi1 = DiscreteFactor(["A"], [2], [0.4, 0.6])
        >>> phi2 = DiscreteFactor(
        ...     ["B", "A"], [3, 2], [[0.2, 0.05], [0.3, 0.15], [0.5, 0.8]]
        ... )
        >>> phi3 = DiscreteFactor(
        ...     ["C", "B"], [2, 3], [[0.4, 0.5, 0.1], [0.6, 0.5, 0.9]]
        ... )
        >>> phi4 = DiscreteFactor(
        ...     ["D", "B"], [3, 3], [[0.1, 0.1, 0.2], [0.3, 0.2, 0.1], [0.6, 0.7, 0.7]]
        ... )
        >>> factor_graph.add_factors(phi1, phi2, phi3, phi4)
        >>> factor_graph.add_edges_from(
        ...     [
        ...         (phi1, "A"),
        ...         ("A", phi2),
        ...         (phi2, "B"),
        ...         ("B", phi3),
        ...         (phi3, "C"),
        ...         ("B", phi4),
        ...         (phi4, "D"),
        ...     ]
        ... )
        >>> belief_propagation = BeliefPropagation(factor_graph)
        >>> belief_propagation.query(
        ...     variables=["B", "C"],
        ...     evidence={"D": 0},
        ...     virtual_evidence=[TabularCPD("A", 2, [[0.3], [0.7]])],
        ... )  # doctest: +ELLIPSIS
        <DiscreteFactor representing phi(B:3, C:2) at 0x...>
        """
        pass

    def calc_variable_node_message(self, variable, incoming_messages):
        """
        The outgoing message is the element wise product of all incoming messages

        If there are no incoming messages, returns a uniform message
        If there is only one incoming message, returns that message
        Otherwise, returns the product of all incoming messages

        Parameters
        ----------
        variable: str
            the variable node from which to compute the outgoing message
        incoming_messages: list
            list of messages coming to this variable node
        """
        pass

    @staticmethod
    def calc_factor_node_message(factor, incoming_messages, target_var):
        """
        Returns the outgoing message for a factor node, which is the
        multiplication of the incoming messages with the factor function (CPT).

        The variables' order in the incoming messages list must match the
        variable's order in the CPT's dimensions

        Parameters
        ----------
        factor: str
            the factor node from which to compute the outgoing message
        incoming_messages: list
            list of messages coming to this factor node
        target_var: str
            the variable node to which the outgoing message is being sent to
        """
        pass
