import copy
import itertools as it

import networkx as nx
import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Inference
from pgmpy.models import DiscreteMarkovNetwork


class Mplp(Inference):
    """
    Class for performing approximate inference using Max-Product Linear Programming method.

    We derive message passing updates that result in monotone decrease of the dual of the
    MAP LP Relaxation.

    Parameters
    ----------
    model: DiscreteMarkovNetwork for which inference is to be performed.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteMarkovNetwork
    >>> from pgmpy.inference import Mplp
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> student = DiscreteMarkovNetwork()
    >>> student.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")])
    >>> factor_a = DiscreteFactor(
    ...     ["A"], cardinality=[2], values=np.array([0.54577, 1.8323])
    ... )
    >>> factor_b = DiscreteFactor(
    ...     ["B"], cardinality=[2], values=np.array([0.93894, 1.065])
    ... )
    >>> factor_c = DiscreteFactor(
    ...     ["C"], cardinality=[2], values=np.array([0.89205, 1.121])
    ... )
    >>> factor_d = DiscreteFactor(
    ...     ["D"], cardinality=[2], values=np.array([0.56292, 1.7765])
    ... )
    >>> factor_e = DiscreteFactor(
    ...     ["E"], cardinality=[2], values=np.array([0.47117, 2.1224])
    ... )
    >>> factor_f = DiscreteFactor(
    ...     ["F"], cardinality=[2], values=np.array([1.5093, 0.66257])
    ... )
    >>> factor_a_b = DiscreteFactor(
    ...     ["A", "B"],
    ...     cardinality=[2, 2],
    ...     values=np.array([1.3207, 0.75717, 0.75717, 1.3207]),
    ... )
    >>> factor_b_c = DiscreteFactor(
    ...     ["B", "C"],
    ...     cardinality=[2, 2],
    ...     values=np.array([0.00024189, 4134.2, 4134.2, 0.00024189]),
    ... )
    >>> factor_c_d = DiscreteFactor(
    ...     ["C", "D"],
    ...     cardinality=[2, 2],
    ...     values=np.array([0.0043227, 231.34, 231.34, 0.0043227]),
    ... )
    >>> factor_d_e = DiscreteFactor(
    ...     ["E", "F"],
    ...     cardinality=[2, 2],
    ...     values=np.array([31.228, 0.032023, 0.032023, 31.228]),
    ... )
    >>> student.add_factors(
    ...     factor_a,
    ...     factor_b,
    ...     factor_c,
    ...     factor_d,
    ...     factor_e,
    ...     factor_f,
    ...     factor_a_b,
    ...     factor_b_c,
    ...     factor_c_d,
    ...     factor_d_e,
    ... )
    >>> mplp = Mplp(student)
    """

    def __init__(self, model):
        if not isinstance(model, DiscreteMarkovNetwork):
            raise TypeError("Only DiscreteMarkovNetwork is supported")

        super().__init__(model)
        self._initialize_structures()

        # S = \{c \cap c^{'} : c, c^{'} \in C, c \cap c^{'} \neq \emptyset\}
        self.intersection_set_variables = set()
        # We generate the Intersections of all the pairwise edges taken one at a time to form S
        for edge_pair in it.combinations(model.edges(), 2):
            self.intersection_set_variables.add(frozenset(edge_pair[0]) & frozenset(edge_pair[1]))

        # The corresponding optimization problem = \min_{\delta}{dual_lp(\delta)} where:
        # dual_lp(\delta) = \sum_{i \in V}{max_{x_i}(Objective[nodes])} + \sum_{f /in F}{max_{x_f}(Objective[factors])
        # Objective[nodes] = \theta_i(x_i) + \sum_{f \mid i \in f}{\delta_{fi}(x_i)}
        # Objective[factors] = \theta_f(x_f) - \sum_{i \in f}{\delta_{fi}(x_i)}
        # In a way Objective stores the corresponding optimization problem for all the nodes and the factors.

        # Form Objective and cluster_set in the form of a dictionary.
        self.objective = {}
        self.cluster_set = {}
        for factor in model.get_factors():
            scope = frozenset(factor.scope())
            self.objective[scope] = factor
            # For every factor consisting of more that a single node, we initialize a cluster.
            if len(scope) > 1:
                self.cluster_set[scope] = self.Cluster(self.intersection_set_variables, factor)

        # dual_lp(\delta) is the dual linear program
        self.dual_lp = sum([np.amax(self.objective[obj].values) for obj in self.objective])

        # Best integral value of the primal objective is stored here
        self.best_int_objective = 0

        # Assignment of the nodes that results in the "maximum" integral value of the primal objective
        self.best_assignment = {}
        # This sets the minimum width between the dual objective decrements. Default value = 0.0002. This can be
        # changed in the map_query() method.
        self.dual_threshold = 0.0002
        # This sets the threshold for the integrality gap below which we say that the solution is satisfactory.
        # Default value = 0.0002. This can be changed in the map_query() method.
        self.integrality_gap_threshold = 0.0002

    class Cluster:
        r"""
        Inner class for representing a cluster.
        A cluster is a subset of variables.

        Parameters
        ----------
        set_of_variables: tuple
            This is the set of variables that form the cluster.

        intersection_set_variables: set containing frozensets.
            collection of intersection of all pairs of cluster variables.
              For eg: \{\{C_1 \cap C_2\}, \{C_2 \cap C_3\}, \{C_3 \cap C_1\} \}
                  for clusters C_1, C_2 & C_3.

            cluster_potential: DiscreteFactor
                Each cluster has an initial probability distribution provided beforehand.
        """

        def __init__(self, intersection_set_variables, cluster_potential):
            """
            Initialization of the current cluster
            """

            # The variables with which the cluster is made of.
            self.cluster_variables = frozenset(cluster_potential.scope())

            # The cluster potentials must be specified before only.
            self.cluster_potential = copy.deepcopy(cluster_potential)

            # Generate intersection sets for this cluster; S(c)
            self.intersection_sets_for_cluster_c = [
                intersect.intersection(self.cluster_variables)
                for intersect in intersection_set_variables
                if intersect.intersection(self.cluster_variables)
            ]

            # Initialize messages from this cluster to its respective intersection sets
            # \lambda_{c \rightarrow \s} = 0
            self.message_from_cluster = {}
            for intersection in self.intersection_sets_for_cluster_c:
                # Present variable. It can be a node or an edge too. (that is ['A'] or ['A', 'C'] too)
                present_variables = list(intersection)

                # Present variables cardinality
                present_variables_card = cluster_potential.get_cardinality(present_variables)
                present_variables_card = [present_variables_card[var] for var in present_variables]

                # We need to create a new factor whose messages are blank
                self.message_from_cluster[intersection] = DiscreteFactor(
                    present_variables,
                    present_variables_card,
                    np.zeros(np.prod(present_variables_card)),
                )

    def _update_message(self, sending_cluster):
        """
        This is the message-update method.

        Parameters
        ----------
        sending_cluster: The resulting messages are lambda_{c-->s} from the given
            cluster 'c' to all of its intersection_sets 's'.
            Here 's' are the elements of intersection_sets_for_cluster_c.

        References
        ----------
        Fixing Max-Product: Convergent Message-Passing Algorithms for MAP LP Relaxations
        by Amir Globerson and Tommi Jaakkola.
        Section 6, Page: 5; Beyond pairwise potentials: Generalized MPLP
        Later Modified by Sontag in "Introduction to Dual decomposition for Inference" Pg: 7 & 17
        """
        pass

    def _local_decode(self):
        """
        Finds the index of the maximum values for all the single node dual objectives.

        Reference:
        code presented by Sontag in 2012 here: http://cs.nyu.edu/~dsontag/code/README_v2.html
        """
        pass

    def _is_converged(self, dual_threshold=None, integrality_gap_threshold=None):
        """
        This method checks the integrality gap to ensure either:
            * we have found a near to exact solution or
            * stuck on a local minima.

        Parameters
        ----------
        dual_threshold: double
                        This sets the minimum width between the dual objective decrements. If the decrement is lesser
                        than the threshold, then that means we have stuck on a local minima.

        integrality_gap_threshold: double
                                   This sets the threshold for the integrality gap below which we say that the solution
                                   is satisfactory.

        References
        ----------
        code presented by Sontag in 2012 here: http://cs.nyu.edu/~dsontag/code/README_v2.html
        """
        pass

    def find_triangles(self):
        """
        Finds all the triangles present in the given model

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.inference import Mplp
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
        >>> mplp = Mplp(mm)
        >>> mplp.find_triangles()
        []
        """
        pass

    def _update_triangles(self, triangles_list):
        """
        From a set of variables forming a triangle in the model, we form the corresponding Clusters.
        These clusters are then appended to the code.

        Parameters
        ----------
        triangle_list : list
                        The list of variables forming the triangles to be updated. It is of the form of
                        [['var_5', 'var_8', 'var_7'], ['var_4', 'var_5', 'var_7']]

        """
        pass

    def _get_triplet_scores(self, triangles_list):
        """
        Returns the score of each of the triplets found in the current model

        Parameters
        ---------
        triangles_list: list
                        The list of variables forming the triangles to be updated. It is of the form of
                        [['var_5', 'var_8', 'var_7'], ['var_4', 'var_5', 'var_7']]

        Return: {frozenset({'var_8', 'var_5', 'var_7'}): 5.024, frozenset({'var_5', 'var_4', 'var_7'}): 10.23}
        """
        pass

    def _run_mplp(self, no_iterations):
        """
        Updates messages until either Mplp converges or if it doesn't converge; halts after no_iterations.

        Parameters
        --------
        no_iterations:  integer
                        Number of maximum iterations that we want MPLP to run.
        """
        pass

    def _tighten_triplet(self, max_iterations, later_iter, max_triplets, prolong):
        """
        This method finds all the triplets that are eligible and adds them iteratively in the bunch of max_triplets

        Parameters
        ----------
        max_iterations: integer
                        Maximum number of times we tighten the relaxation

        later_iter: integer
                    Number of maximum iterations that we want MPLP to run. This is lesser than the initial number
                    of iterations.

        max_triplets: integer
                      Maximum number of triplets that can be added at most in one iteration.

        prolong: bool
                It sets the continuation of tightening after all the triplets are exhausted
        """
        pass

    def get_integrality_gap(self):
        """
        Returns the integrality gap of the current state of the Mplp algorithm. The lesser it is, the closer we are
                towards the exact solution.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.inference import Mplp
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
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
        >>> phi_single = [
        ...     DiscreteFactor([node], [2], rng.random(2))
        ...     for node in mm.nodes()
        ... ]
        >>> phi_pair = [
        ...     DiscreteFactor(edge, [2, 2], rng.random(4))
        ...     for edge in mm.edges()
        ... ]
        >>> mm.add_factors(*(phi_single + phi_pair))
        >>> mplp = Mplp(mm)
        >>> mplp.map_query() # doctest: +SKIP
        >>> {k: int(v) for k, v in mplp.map_query().items()}
        {'x1': 0, 'x2': 1, 'x3': 1, 'x4': 0, 'x5': 1, 'x6': 1, 'x7': 1}
        >>> int_gap = mplp.get_integrality_gap()
        """
        pass

    def query(self):
        pass

    def map_query(
        self,
        init_iter=1000,
        later_iter=20,
        dual_threshold=0.0002,
        integrality_gap_threshold=0.0002,
        tighten_triplet=True,
        max_triplets=5,
        max_iterations=100,
        prolong=False,
    ):
        """
        MAP query method using Max Product LP method.
        This returns the best assignment of the nodes in the form of a dictionary.

        Parameters
        ----------
        init_iter: integer
                   Number of maximum iterations that we want MPLP to run for the first time.

        later_iter: integer
                    Number of maximum iterations that we want MPLP to run for later iterations

        dual_threshold: double
                        This sets the minimum width between the dual objective decrements. If the decrement is lesser
                        than the threshold, then that means we have stuck on a local minima.

        integrality_gap_threshold: double
                                   This sets the threshold for the integrality gap below which we say that the solution
                                   is satisfactory.

        tighten_triplet: bool
                         set whether to use triplets as clusters or not.

        max_triplets: integer
                      Set the maximum number of triplets that can be added at once.

        max_iterations: integer
                        Maximum number of times we tighten the relaxation. Used only when tighten_triplet is set True.

        prolong: bool
                 If set False: The moment we exhaust of all the triplets the tightening stops.
                 If set True: The tightening will be performed max_iterations
                   number of times irrespective of the triplets.

        References
        ----------
        Section 3.3: The Dual Algorithm; Tightening LP Relaxation for MAP using Message Passing (2008)
        By Sontag Et al.

        Examples
        --------
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.inference import Mplp
        >>> import numpy as np
        >>> student = DiscreteMarkovNetwork()
        >>> student.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")])
        >>> factor_a = DiscreteFactor(
        ...     ["A"], cardinality=[2], values=np.array([0.54577, 1.8323])
        ... )
        >>> factor_b = DiscreteFactor(
        ...     ["B"], cardinality=[2], values=np.array([0.93894, 1.065])
        ... )
        >>> factor_c = DiscreteFactor(
        ...     ["C"], cardinality=[2], values=np.array([0.89205, 1.121])
        ... )
        >>> factor_d = DiscreteFactor(
        ...     ["D"], cardinality=[2], values=np.array([0.56292, 1.7765])
        ... )
        >>> factor_e = DiscreteFactor(
        ...     ["E"], cardinality=[2], values=np.array([0.47117, 2.1224])
        ... )
        >>> factor_f = DiscreteFactor(
        ...     ["F"], cardinality=[2], values=np.array([1.5093, 0.66257])
        ... )
        >>> factor_a_b = DiscreteFactor(
        ...     ["A", "B"],
        ...     cardinality=[2, 2],
        ...     values=np.array([1.3207, 0.75717, 0.75717, 1.3207]),
        ... )
        >>> factor_b_c = DiscreteFactor(
        ...     ["B", "C"],
        ...     cardinality=[2, 2],
        ...     values=np.array([0.00024189, 4134.2, 4134.2, 0.0002418]),
        ... )
        >>> factor_c_d = DiscreteFactor(
        ...     ["C", "D"],
        ...     cardinality=[2, 2],
        ...     values=np.array([0.0043227, 231.34, 231.34, 0.0043227]),
        ... )
        >>> factor_d_e = DiscreteFactor(
        ...     ["E", "F"],
        ...     cardinality=[2, 2],
        ...     values=np.array([31.228, 0.032023, 0.032023, 31.228]),
        ... )
        >>> student.add_factors(
        ...     factor_a,
        ...     factor_b,
        ...     factor_c,
        ...     factor_d,
        ...     factor_e,
        ...     factor_f,
        ...     factor_a_b,
        ...     factor_b_c,
        ...     factor_c_d,
        ...     factor_d_e,
        ... )
        >>> mplp = Mplp(student)
        >>> result = mplp.map_query()
        >>> {k: int(v) for k, v in result.items()}
        {'A': 1, 'B': 0, 'C': 1, 'D': 1, 'E': 1, 'F': 0}
        """
        pass
