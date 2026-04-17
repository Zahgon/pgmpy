#!/usr/bin/env python3

from collections import defaultdict
from itertools import chain

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import (
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    JunctionTree,
)
from pgmpy.utils import compat_fns


class Inference:
    """
    Base class for all inference algorithms.

    Converts DiscreteBayesianNetwork and DiscreteMarkovNetwork to a uniform representation so that inference
    algorithms can be applied. Also, it checks if all the associated CPDs / Factors are
    consistent with the model.

    Initialize inference for a model.

    Parameters
    ----------
    model: pgmpy.models.DiscreteBayesianNetwork or pgmpy.models.DiscreteMarkovNetwork
        model for which to initialize the inference object.

    Examples
    --------
    >>> from pgmpy.inference import Inference
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.factors.discrete import TabularCPD
    >>> student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
    >>> diff_cpd = TabularCPD("diff", 2, [[0.2], [0.8]])
    >>> intel_cpd = TabularCPD("intel", 2, [[0.3], [0.7]])
    >>> grade_cpd = TabularCPD(
    ...     "grade",
    ...     3,
    ...     [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.8, 0.8, 0.8, 0.8]],
    ...     evidence=["diff", "intel"],
    ...     evidence_card=[2, 2],
    ... )
    >>> student.add_cpds(diff_cpd, intel_cpd, grade_cpd)
    >>> model = Inference(student)

    >>> from pgmpy.models import DiscreteMarkovNetwork
    >>> from pgmpy.factors.discrete import DiscreteFactor
    >>> import numpy as np
    >>> student = DiscreteMarkovNetwork(
    ...     [
    ...         ("Alice", "Bob"),
    ...         ("Bob", "Charles"),
    ...         ("Charles", "Debbie"),
    ...         ("Debbie", "Alice"),
    ...     ]
    ... )
    >>> factor_a_b = DiscreteFactor(
    ...     ["Alice", "Bob"], cardinality=[2, 2], values=np.random.rand(4)
    ... )
    >>> factor_b_c = DiscreteFactor(
    ...     ["Bob", "Charles"], cardinality=[2, 2], values=np.random.rand(4)
    ... )
    >>> factor_c_d = DiscreteFactor(
    ...     ["Charles", "Debbie"], cardinality=[2, 2], values=np.random.rand(4)
    ... )
    >>> factor_d_a = DiscreteFactor(
    ...     ["Debbie", "Alice"], cardinality=[2, 2], values=np.random.rand(4)
    ... )
    >>> student.add_factors(factor_a_b, factor_b_c, factor_c_d, factor_d_a)
    >>> model = Inference(student)
    """

    def __init__(self, model):
        self.model = model
        model.check_model()

        if isinstance(self.model, JunctionTree):
            self.variables = set(chain(*self.model.nodes()))
        else:
            self.variables = self.model.nodes()

    def _initialize_structures(self):
        """
        Initializes all the data structures which will
        later be used by the inference algorithms.
        """
        pass

    def _prune_bayesian_model(self, variables, evidence):
        """
        Prunes unnecessary nodes from the model to optimize the computation.

        Parameters
        ----------
        variables: list
            The variables on which the query is done i.e. the variables whose
            values we are interested in.

        evidence: dict (default: None)
            The variables whose values we know. The values can be specified as
            {variable: state}.

        Returns
        -------
        Pruned model: pgmpy.models.DiscreteBayesianNetwork
            The pruned model.

        Examples
        --------
        >>>
        >>>

        References
        ----------
        [1] Baker, M., & Boult, T. E. (2013).
          Pruning Bayesian networks for efficient computation.
            arXiv preprint arXiv:1304.1112.
        """
        pass

    def _check_virtual_evidence(self, virtual_evidence):
        """
        Checks the virtual evidence's format is correct. Each evidence must:
        - Be a TabularCPD instance or a DiscreteFactor on a single variable.
        - Be targeted to a single variable
        - Be defined on a variable which is in the model
        - Have the same cardinality as its corresponding variable in the model

        Parameters
        ----------
        virtual_evidence: list
            A list of TabularCPD instances specifying the virtual evidence for each
            of the evidence variables.
        """
        pass

    def _virtual_evidence(self, virtual_evidence):
        """
        Modifies the model to incorporate virtual evidence. For each virtual evidence
        variable a binary variable is added as the child of the evidence variable to
        the model. The state 0 probabilities of the child is the evidence.

        Parameters
        ----------
        virtual_evidence: list
            A list of TabularCPD instances specifying the virtual evidence for each
            of the evidence variables.

        Returns
        -------
        None

        References
        ----------
        [1] Mrad, Ali Ben, et al. "Uncertain evidence in Bayesian networks:
          Presentation and comparison on a simple example."
            International Conference on Information Processing and Management
              of Uncertainty in Knowledge-Based Systems. Springer, Berlin, Heidelberg, 2012.
        """
        pass

    @staticmethod
    def _get_virtual_evidence_var_list(virtual_evidence):
        """
        Returns the list of variables that have a virtual evidence.

        Parameters
        ----------
        virtual_evidence: list
            A list of TabularCPD instances specifying the virtual evidence for each
            of the evidence variables.
        """
        pass
