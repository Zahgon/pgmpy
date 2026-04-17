import itertools
from functools import reduce
from operator import mul

import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.utils import compat_fns


class JointProbabilityDistribution(DiscreteFactor):
    """
    Base class for Joint Probability Distribution
    """

    def __init__(self, variables, cardinality, values):
        """
        Initialize a Joint Probability Distribution class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:

        +-----+-----+-----+-------------------------+
        |  x1 |  x2 |  x3 |    P(x1, x2, x2)        |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_0| x3_0|    P(x1_0, x2_0, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_0| x3_0|    P(x1_1, x2_0, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_1| x3_0|    P(x1_0, x2_1, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_1| x3_0|    P(x1_1, x2_1, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_0| x3_1|    P(x1_0, x2_0, x3_1)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_0| x3_1|    P(x1_1, x2_0, x3_1)  |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_1| x3_1|    P(x1_0, x2_1, x3_1)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_1| x3_1|    P(x1_1, x2_1, x3_1)  |
        +-----+-----+-----+-------------------------+

        Parameters
        ----------
        variables: list
            List of scope of Joint Probability Distribution.
        cardinality: list, array_like
            List of cardinality of each variable
        value: list, array_like
            List or array of values of factor.
            A Joint Probability Distribution's values are stored in a row
            vector in the value using an ordering such that the left-most
            variables as defined in the variable field cycle through their
            values the fastest.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(
        ...     variables=["x1", "x2", "x3"],
        ...     cardinality=[2, 2, 2],
        ...     values=np.ones(8) / 8,
        ... )
        >>> print(prob)
        +-------+-------+-------+---------------+
        | x1    | x2    | x3    |   P(x1,x2,x3) |
        +=======+=======+=======+===============+
        | x1(0) | x2(0) | x3(0) |        0.1250 |
        +-------+-------+-------+---------------+
        | x1(0) | x2(0) | x3(1) |        0.1250 |
        +-------+-------+-------+---------------+
        | x1(0) | x2(1) | x3(0) |        0.1250 |
        +-------+-------+-------+---------------+
        | x1(0) | x2(1) | x3(1) |        0.1250 |
        +-------+-------+-------+---------------+
        | x1(1) | x2(0) | x3(0) |        0.1250 |
        +-------+-------+-------+---------------+
        | x1(1) | x2(0) | x3(1) |        0.1250 |
        +-------+-------+-------+---------------+
        | x1(1) | x2(1) | x3(0) |        0.1250 |
        +-------+-------+-------+---------------+
        | x1(1) | x2(1) | x3(1) |        0.1250 |
        +-------+-------+-------+---------------+
        """
        if np.isclose(np.sum(compat_fns.to_numpy(values)), 1):
            super().__init__(variables, cardinality, values)
        else:
            raise ValueError("The probability values doesn't sum to 1.")

    def __repr__(self):
        var_card = ", ".join([f"{var}:{card}" for var, card in zip(self.variables, self.cardinality)])
        return f"<Joint Distribution representing P({var_card}) at {hex(id(self))}>"

    def __str__(self):
        return self._str(phi_or_p="P")

    def marginal_distribution(self, variables, inplace=True):
        """
        Returns the marginal distribution over variables.

        Parameters
        ----------
        variables: string, list, tuple, set, dict
                Variable or list of variables over which marginal distribution needs
                to be calculated
        inplace: Boolean (default True)
                If False return a new instance of JointProbabilityDistribution

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> values = np.random.rand(12)
        >>> prob = JointProbabilityDistribution(
        ...     ["x1", "x2", "x3"], [2, 3, 2], values / np.sum(values)
        ... )
        >>> prob.marginal_distribution(variables=["x1", "x2"])
        >>> print(prob)  # doctest: +SKIP
        +-------+-------+------------+
        | x1    | x2    |   P(x1,x2) |
        +=======+=======+============+
        | x1(0) | x2(0) |     0.1408 |
        +-------+-------+------------+
        | x1(0) | x2(1) |     0.3372 |
        +-------+-------+------------+
        | x1(0) | x2(2) |     0.1530 |
        +-------+-------+------------+
        | x1(1) | x2(0) |     0.2122 |
        +-------+-------+------------+
        | x1(1) | x2(1) |     0.0950 |
        +-------+-------+------------+
        | x1(1) | x2(2) |     0.0619 |
        +-------+-------+------------+
        """
        pass

    def check_independence(self, event1, event2, event3=None, condition_random_variable=False):
        """
        Check if the Joint Probability Distribution satisfies the given independence condition.

        Parameters
        ----------
        event1: list
            random variable whose independence is to be checked.
        event2: list
            random variable from which event1 is independent.
        values: 2D array or list like or 1D array or list like
            A 2D list of tuples of the form (variable_name, variable_state).
            A 1D list or array-like to condition over randome variables (condition_random_variable must be True)
            The values on which to condition the Joint Probability Distribution.
        condition_random_variable: Boolean (Default false)
            If true and event3 is not None than will check independence condition over random variable.

        For random variables say X, Y, Z to check if X is independent of Y given Z.
        event1 should be either X or Y.
        event2 should be either Y or X.
        event3 should Z.

        Examples
        --------
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution as JPD
        >>> prob = JPD(
        ...     variables=["I", "D", "G"],
        ...     cardinality=[2, 2, 3],
        ...     values=[
        ...         0.126,
        ...         0.168,
        ...         0.126,
        ...         0.009,
        ...         0.045,
        ...         0.126,
        ...         0.252,
        ...         0.0224,
        ...         0.0056,
        ...         0.06,
        ...         0.036,
        ...         0.024,
        ...     ],
        ... )
        >>> prob.check_independence(event1=["I"], event2=["D"])
        True
        >>> prob.check_independence(["I"], ["D"], [("G", 1)])  # Conditioning over G_1
        False
        >>> # Conditioning over random variable G
        >>> prob.check_independence(
        ...     ["I"], ["D"], ("G",), condition_random_variable=True
        ... )
        False
        """
        pass

    def get_independencies(self, condition=None):
        """
        Returns the independent variables in the joint probability distribution.
        Returns marginally independent variables if condition=None.
        Returns conditionally independent variables if condition!=None

        Parameters
        ----------
        condition: array_like
                Random Variable on which to condition the Joint Probability Distribution.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(
        ...     variables=["x1", "x2", "x3"],
        ...     cardinality=[2, 3, 2],
        ...     values=np.ones(12) / 12,
        ... )
        >>> prob.get_independencies()
        (x1 \u27c2 x2)
        (x1 \u27c2 x3)
        (x2 \u27c2 x3)
        """
        pass

    def conditional_distribution(self, values, inplace=True):
        """
        Returns Conditional Probability Distribution after setting values to 1.

        Parameters
        ----------
        values: list or array_like
            A list of tuples of the form (variable_name, variable_state).
            The values on which to condition the Joint Probability Distribution.
        inplace: Boolean (default True)
            If False returns a new instance of JointProbabilityDistribution

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(
        ...     variables=["x1", "x2", "x3"],
        ...     cardinality=[2, 2, 2],
        ...     values=np.ones(8) / 8,
        ... )
        >>> prob.conditional_distribution(values=[("x1", 1)])
        >>> print(prob)
        +-------+-------+------------+
        | x2    | x3    |   P(x2,x3) |
        +=======+=======+============+
        | x2(0) | x3(0) |     0.2500 |
        +-------+-------+------------+
        | x2(0) | x3(1) |     0.2500 |
        +-------+-------+------------+
        | x2(1) | x3(0) |     0.2500 |
        +-------+-------+------------+
        | x2(1) | x3(1) |     0.2500 |
        +-------+-------+------------+
        """
        pass

    def copy(self):
        """
        Returns A copy of JointProbabilityDistribution object

        Examples
        ---------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(
        ...     variables=["x1", "x2", "x3"],
        ...     cardinality=[2, 3, 2],
        ...     values=np.ones(12) / 12,
        ... )
        >>> prob_copy = prob.copy()
        >>> (prob_copy.values == prob.values).all()
        np.True_
        >>> prob_copy.variables == prob.variables
        True
        >>> prob_copy.variables[1] = "y"
        >>> prob_copy.variables == prob.variables
        False
        """
        pass

    def minimal_imap(self, order):
        """
        Returns a Bayesian Model which is minimal IMap of the Joint Probability Distribution
        considering the order of the variables.

        Parameters
        ----------
        order: array-like
            The order of the random variables.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(
        ...     variables=["x1", "x2", "x3"],
        ...     cardinality=[2, 3, 2],
        ...     values=np.ones(12) / 12,
        ... )
        >>> bayesian_model = prob.minimal_imap(order=["x2", "x1", "x3"])
        >>> bayesian_model # doctest: +ELLIPSIS
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x...>
        >>> bayesian_model.edges()
        OutEdgeView([('x2', 'x3'), ('x1', 'x3')])
        """
        pass

    def is_imap(self, model):
        """
        Checks whether the given DiscreteBayesianNetwork is Imap of JointProbabilityDistribution

        Parameters
        ----------
        model : An instance of DiscreteBayesianNetwork Class, for which you want to
            check the Imap

        Returns
        -------
        Is IMAP: bool
            True if given Bayesian Network is Imap for Joint Probability Distribution False otherwise

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> bm = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
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
        >>> bm.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        >>> val = [
        ...     0.01,
        ...     0.01,
        ...     0.08,
        ...     0.006,
        ...     0.006,
        ...     0.048,
        ...     0.004,
        ...     0.004,
        ...     0.032,
        ...     0.04,
        ...     0.04,
        ...     0.32,
        ...     0.024,
        ...     0.024,
        ...     0.192,
        ...     0.016,
        ...     0.016,
        ...     0.128,
        ... ]
        >>> JPD = JointProbabilityDistribution(
        ...     ["diff", "intel", "grade"], [2, 3, 3], val
        ... )
        >>> JPD.is_imap(bm)
        True
        """
        pass

    def to_factor(self):
        """
        Returns JointProbabilityDistribution as a DiscreteFactor object

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(
        ...     ["x1", "x2", "x3"], [2, 3, 2], np.ones(12) / 12
        ... )
        >>> phi = prob.to_factor()
        >>> type(phi)
        <class 'pgmpy.factors.discrete.DiscreteFactor.DiscreteFactor'>
        """
        pass

    def pmap(self):
        pass
