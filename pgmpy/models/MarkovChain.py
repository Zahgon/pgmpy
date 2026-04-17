#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
from pandas import DataFrame
from scipy.linalg import eig

from pgmpy import logger
from pgmpy.factors.discrete import State
from pgmpy.utils import sample_discrete


class MarkovChain:
    """
    Class to represent a Markov Chain with multiple kernels for factored state space,
    along with methods to simulate a run.

    Examples
    --------

    Create an empty Markov Chain:

    >>> from pgmpy.models import MarkovChain as MC
    >>> model = MC()

    And then add variables to it

    >>> model.add_variables_from(["intel", "diff"], [2, 3])

    Or directly create a Markov Chain from a list of variables and their cardinalities

    >>> model = MC(["intel", "diff"], [2, 3])

    Add transition models

    >>> intel_tm = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}
    >>> model.add_transition_model("intel", intel_tm)
    >>> diff_tm = {
    ...     0: {0: 0.1, 1: 0.5, 2: 0.4},
    ...     1: {0: 0.2, 1: 0.2, 2: 0.6},
    ...     2: {0: 0.7, 1: 0.15, 2: 0.15},
    ... }
    >>> model.add_transition_model("diff", diff_tm)

    Set a start state

    >>> from pgmpy.factors.discrete import State
    >>> model.set_start_state([State("intel", 0), State("diff", 2)])

    Sample from it

    >>> df = model.sample(size=5)
    >>> df.shape
    (5, 2)
    >>> list(df.columns)
    ['intel', 'diff']
    """

    def __init__(self, variables=None, card=None, start_state=None):
        """
        Parameters
        ----------
        variables: array-like iterable object
            A list of variables of the model.

        card: array-like iterable object
            A list of cardinalities of the variables.

        start_state: array-like iterable object
            List of tuples representing the starting states of the variables.
        """
        if variables is None:
            variables = []
        if card is None:
            card = []
        if not hasattr(variables, "__iter__") or isinstance(variables, str):
            raise ValueError("variables must be a non-string iterable.")
        if not hasattr(card, "__iter__") or isinstance(card, str):
            raise ValueError("card must be a non-string iterable.")
        self.variables = variables
        self.cardinalities = {v: c for v, c in zip(variables, card)}
        self.transition_models = {var: {} for var in variables}
        if start_state is None or self._check_state(start_state):
            self.state = start_state

    def set_start_state(self, start_state):
        """
        Set the start state of the Markov Chain. If the start_state is given as an array-like iterable, its contents
        are reordered in the internal representation.

        Parameters
        ----------
        start_state: dict or array-like iterable object
            Dict (or list) of tuples representing the starting states of the variables.

        Examples
        --------
        >>> from pgmpy.models import MarkovChain as MC
        >>> from pgmpy.factors.discrete import State
        >>> model = MC(["a", "b"], [2, 2])
        >>> model.set_start_state([State("a", 0), State("b", 1)])
        """
        pass

    def _check_state(self, state):
        """
        Checks if a list representing the state of the variables is valid.
        """
        pass

    def add_variable(self, variable, card=0):
        """
        Add a variable to the model.

        Parameters
        ----------
        variable: any hashable python object

        card: int
            Representing the cardinality of the variable to be added.

        Examples
        --------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC()
        >>> model.add_variable("x", 4)
        """
        pass

    def add_variables_from(self, variables, cards):
        """
        Add several variables to the model at once.

        Parameters
        ----------
        variables: array-like iterable object
            List of variables to be added.

        cards: array-like iterable object
            List of cardinalities of the variables to be added.

        Examples
        --------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC()
        >>> model.add_variables_from(["x", "y"], [3, 4])
        """
        pass

    def add_transition_model(self, variable, transition_model):
        """
        Adds a transition model for a particular variable.

        Parameters
        ----------
        variable: any hashable python object
            must be an existing variable of the model.

        transition_model: dict or 2d array
            dict representing valid transition probabilities defined for every possible state of the variable.
            array represent a square matrix where every row sums to 1,
            array[i,j] indicates the transition probalities from State i to State j

        Examples
        --------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC()
        >>> model.add_variable("grade", 3)
        >>> grade_tm = {
        ...     0: {0: 0.1, 1: 0.5, 2: 0.4},
        ...     1: {0: 0.2, 1: 0.2, 2: 0.6},
        ...     2: {0: 0.7, 1: 0.15, 2: 0.15},
        ... }
        >>> grade_tm_matrix = np.array(
        ...     [[0.1, 0.5, 0.4], [0.2, 0.2, 0.6], [0.7, 0.15, 0.15]]
        ... )
        >>> model.add_transition_model("grade", grade_tm)
        >>> model.add_transition_model("grade", grade_tm_matrix)
        """
        pass

    def sample(self, start_state=None, size=1, seed=None):
        """
        Sample from the Markov Chain.

        Parameters
        ----------
        start_state: dict or array-like iterable
            Representing the starting states of the variables. If None is passed, a random start_state is chosen.
        size: int
            Number of samples to be generated.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> from pgmpy.models import MarkovChain as MC
        >>> from pgmpy.factors.discrete import State
        >>> model = MC(["intel", "diff"], [2, 3])
        >>> model.set_start_state([State("intel", 0), State("diff", 2)])
        >>> intel_tm = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}
        >>> model.add_transition_model("intel", intel_tm)
        >>> diff_tm = {
        ...     0: {0: 0.1, 1: 0.5, 2: 0.4},
        ...     1: {0: 0.2, 1: 0.2, 2: 0.6},
        ...     2: {0: 0.7, 1: 0.15, 2: 0.15},
        ... }
        >>> model.add_transition_model("diff", diff_tm)
        >>> df = model.sample(size=5)
        >>> df.shape
        (5, 2)
        >>> list(df.columns)
        ['intel', 'diff']
        """
        pass

    def prob_from_sample(self, state, sample=None, window_size=None):
        """
        Given an instantiation (partial or complete) of the variables of the model,
        compute the probability of observing it over multiple windows in a given sample.

        If 'sample' is not passed as an argument, generate the statistic by sampling from the
        Markov Chain, starting with a random initial state.

        Examples
        --------
        >>> from pgmpy.models.MarkovChain import MarkovChain as MC
        >>> from pgmpy.factors.discrete import State
        >>> model = MC(["intel", "diff"], [3, 2])
        >>> intel_tm = {
        ...     0: {0: 0.2, 1: 0.4, 2: 0.4},
        ...     1: {0: 0, 1: 0.5, 2: 0.5},
        ...     2: {2: 0.5, 1: 0.5},
        ... }
        >>> model.add_transition_model("intel", intel_tm)
        >>> diff_tm = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1: 0.75}}
        >>> model.add_transition_model("diff", diff_tm)
        >>> probs = model.prob_from_sample([State("diff", 0)])
        >>> len(probs)
        100
        """
        pass

    def generate_sample(self, start_state=None, size=1, seed=None):
        """
        Generator version of self.sample

        Returns
        -------
        List of State namedtuples, representing the assignment to all variables of the model.

        Examples
        --------
        >>> from pgmpy.models.MarkovChain import MarkovChain
        >>> from pgmpy.factors.discrete import State
        >>> model = MarkovChain()
        >>> model.add_variables_from(["intel", "diff"], [3, 2])
        >>> intel_tm = {
        ...     0: {0: 0.2, 1: 0.4, 2: 0.4},
        ...     1: {0: 0, 1: 0.5, 2: 0.5},
        ...     2: {0: 0.3, 1: 0.3, 2: 0.4},
        ... }
        >>> model.add_transition_model("intel", intel_tm)
        >>> diff_tm = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1: 0.75}}
        >>> model.add_transition_model("diff", diff_tm)
        >>> gen = model.generate_sample([State("intel", 0), State("diff", 0)], 2)
        >>> [sample for sample in gen] # doctest: +SKIP
        [[State(var='intel', state=2), State(var='diff', state=1)],
         [State(var='intel', state=2), State(var='diff', state=0)]]
        """
        pass

    def is_stationarity(self, tolerance=0.2, sample=None):
        """
        Checks if the given markov chain is stationary and checks the steady state
        probability values for the state are consistent.

        Parameters
        ----------
        tolerance: float
            represents the diff between actual steady state value and the computed value
        sample: [State(i,j)]
            represents the list of state which the markov chain has sampled

        Returns
        -------
        Boolean:
            True, if the markov chain converges to steady state distribution within the tolerance
            False, if the markov chain does not converge to steady state distribution within tolerance

        Examples
        --------
        >>> from pgmpy.models.MarkovChain import MarkovChain
        >>> from pgmpy.factors.discrete import State
        >>> model = MarkovChain()
        >>> model.add_variables_from(["intel", "diff"], [3, 2])
        >>> intel_tm = {
        ...     0: {0: 0.2, 1: 0.4, 2: 0.4},
        ...     1: {0: 0, 1: 0.5, 2: 0.5},
        ...     2: {0: 0.3, 1: 0.3, 2: 0.4},
        ... }
        >>> model.add_transition_model("intel", intel_tm)
        >>> diff_tm = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1: 0.75}}
        >>> model.add_transition_model("diff", diff_tm)
        >>> model.is_stationarity()
        True
        """
        pass

    def random_state(self):
        """
        Generates a random state of the Markov Chain.

        Returns
        -------
        List of namedtuples, representing a random assignment to all variables of the model.

        Examples
        --------
        >>> from pgmpy.models import MarkovChain as MC
        >>> model = MC(["intel", "diff"], [2, 3])
        >>> model.random_state() # doctest: +SKIP
        [State(var='diff', state=2), State(var='intel', state=1)]
        """
        pass

    def copy(self):
        """
        Returns a copy of Markov Chain Model.

        Returns
        -------
        MarkovChain : Copy of MarkovChain.

        Examples
        --------
        >>> from pgmpy.models import MarkovChain
        >>> from pgmpy.factors.discrete import State
        >>> model = MarkovChain()
        >>> model.add_variables_from(["intel", "diff"], [3, 2])
        >>> intel_tm = {
        ...     0: {0: 0.2, 1: 0.4, 2: 0.4},
        ...     1: {0: 0, 1: 0.5, 2: 0.5},
        ...     2: {0: 0.3, 1: 0.3, 2: 0.4},
        ... }
        >>> model.add_transition_model("intel", intel_tm)
        >>> diff_tm = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1: 0.75}}
        >>> model.add_transition_model("diff", diff_tm)
        >>> model.set_start_state([State("intel", 0), State("diff", 1)])
        >>> model_copy = model.copy()
        >>> model_copy.transition_models == {
        ...     "intel": {
        ...         0: {0: 0.2, 1: 0.4, 2: 0.4},
        ...         1: {0: 0, 1: 0.5, 2: 0.5},
        ...         2: {0: 0.3, 1: 0.3, 2: 0.4},
        ...     },
        ...     "diff": {0: {0: 0.5, 1: 0.5}, 1: {0: 0.25, 1: 0.75}},
        ... }
        True
        """
        pass
