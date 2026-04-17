#!/usr/bin/env python3
"""Contains the different formats of CPDs used in PGM"""

import csv
import numbers
import os
from collections.abc import Hashable
from itertools import chain, product
from shutil import get_terminal_size

import numpy as np
import pandas as pd

from pgmpy import config, logger
from pgmpy.extern import tabulate
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.utils import compat_fns


class TabularCPD(DiscreteFactor):
    """
    Defines the conditional probability distribution table (CPD table)

    Parameters
    ----------
    variable: int, string (any hashable python object)
        The variable whose CPD is defined.

    variable_card: integer
        Cardinality/no. of states of `variable`

    values: 2D array, 2D list or 2D tuple
        Values for the CPD table. Please refer the example for the
        exact format needed.

    evidence: array-like
        List of variables in evidences(if any) w.r.t. which CPD is defined.

    evidence_card: array-like
        cardinality/no. of states of variables in `evidence`(if any)

    state_names: dict (default: dict())
        A dictionary of the form {variable: list of states} specifying the
        names of possible states for each variable (variable + evidence) in
        the TabularCPD. The order in which the states are specified should
        match the order in the values array. If state_names is not specified,
        auto-assigns state names starting from 0.

    Examples
    --------
    For a distribution of P(grade|diff, intel)

    +---------+-------------------------+------------------------+
    |diff     |          easy           |         hard           |
    +---------+------+--------+---------+------+--------+--------+
    |intel    | low  | medium |  high   | low  | medium |  high  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeA   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeB   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeC   | 0.8  | 0.8    |   0.8   |  0.8 |  0.8   |   0.8  |
    +---------+------+--------+---------+------+--------+--------+

    the values array should be
    [[0.1,0.1,0.1,0.1,0.1,0.1],
     [0.1,0.1,0.1,0.1,0.1,0.1],
     [0.8,0.8,0.8,0.8,0.8,0.8]]

    >>> cpd = TabularCPD(
    ...     variable="grade",
    ...     variable_card=3,
    ...     values=[
    ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ...         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ...         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    ...     ],
    ...     evidence=["diff", "intel"],
    ...     evidence_card=[2, 3],
    ...     state_names={
    ...         "diff": ["easy", "hard"],
    ...         "intel": ["low", "mid", "high"],
    ...         "grade": ["A", "B", "C"],
    ...     },
    ... )
    >>> print(cpd) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    +----------+------------+-----+------------+-------------+
    | diff     | diff(easy) | ... | diff(hard) | diff(hard)  |
    +----------+------------+-----+------------+-------------+
    | intel    | intel(low) | ... | intel(mid) | intel(high) |
    +----------+------------+-----+------------+-------------+
    | grade(A) | 0.1        | ... | 0.1        | 0.1         |
    +----------+------------+-----+------------+-------------+
    | grade(B) | 0.1        | ... | 0.1        | 0.1         |
    +----------+------------+-----+------------+-------------+
    | grade(C) | 0.8        | ... | 0.8        | 0.8         |
    +----------+------------+-----+------------+-------------+
    >>> cpd.values
    array([[[0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]],
    <BLANKLINE>
           [[0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]],
    <BLANKLINE>
           [[0.8, 0.8, 0.8],
            [0.8, 0.8, 0.8]]])
    >>> cpd.variables
    ['grade', 'diff', 'intel']
    >>> cpd.cardinality
    array([3, 2, 3])
    >>> cpd.variable
    'grade'
    >>> cpd.variable_card
    3
    """

    def __init__(
        self,
        variable: Hashable,
        variable_card: int,
        values: list | np.typing.ArrayLike,
        evidence: list | tuple | None = None,
        evidence_card: list | tuple | None = None,
        state_names={},
    ):
        self.variable = variable
        self.variable_card = None

        variables = [variable]

        if not isinstance(variable_card, numbers.Integral):
            raise TypeError("Event cardinality must be an integer")
        self.variable_card = variable_card

        cardinality = [variable_card]
        if evidence_card is not None:
            if isinstance(evidence_card, numbers.Real):
                raise TypeError("Evidence card must be a list of numbers")
            cardinality.extend(evidence_card)

        if evidence is not None:
            if isinstance(evidence, str):
                raise TypeError("Evidence must be list, tuple or array of strings.")
            if isinstance(evidence_card, type(None)):
                raise ValueError("Evidence card must be provided if Evidence is provided!")
            variables.extend(evidence)
            if not len(evidence_card) == len(evidence):
                raise ValueError("Length of evidence_card doesn't match length of evidence")

        if config.BACKEND == "numpy":
            values_casted = np.array(object=values, dtype=config.get_dtype())
        else:
            import torch

            values_casted = torch.tensor(values).type(config.get_dtype()).to(config.get_device())

        if values_casted.ndim != 2:
            raise TypeError("Values must be a 2D list/array")

        if evidence is None:
            expected_cpd_shape = (variable_card, 1)
        else:
            expected_cpd_shape = (variable_card, np.prod(evidence_card))
        if values_casted.shape != expected_cpd_shape:
            raise ValueError(f"values must be of shape {expected_cpd_shape}. Got shape: {values.shape}")

        if not isinstance(state_names, dict):
            raise ValueError(f"state_names must be of type dict. Got {type(state_names)}")

        super().__init__(variables, cardinality, values_casted.flatten(), state_names=state_names)

    def __repr__(self):
        var_str = f"<TabularCPD representing P({self.variable}:{self.variable_card}"

        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            evidence_str = " | " + ", ".join([f"{var}:{card}" for var, card in zip(evidence, evidence_card)])
        else:
            evidence_str = ""

        return var_str + evidence_str + f") at {hex(id(self))}>"

    def get_values(self):
        """
        Returns the values of the CPD as a 2-D array. The order of the
        parents is the same as provided in evidence.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd = TabularCPD(
        ...     variable="grade",
        ...     variable_card=3,
        ...     values=[[0.1, 0.1], [0.1, 0.1], [0.8, 0.8]],
        ...     evidence=["evi1"],
        ...     evidence_card=[2],
        ... )
        >>> cpd.get_values()
        array([[0.1, 0.1],
               [0.1, 0.1],
               [0.8, 0.8]])
        """
        pass

    def __str__(self):
        return self._make_table_str(tablefmt="grid")

    def _str(self, phi_or_p="p", tablefmt="fancy_grid"):
        pass

    def _make_table_str(self, tablefmt="fancy_grid", print_state_names=True, return_list=False) -> str | list[str]:
        pass

    def _truncate_strtable(self, cdf_str: str):
        pass

    def to_csv(self, filename: str | os.PathLike):
        """
        Exports the CPD to a CSV file.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/alarm")
        >>> cpd = model.get_cpds(node="SAO2")
        >>> cpd.to_csv(filename="sao2.csv")
        """
        pass

    def to_dataframe(self):
        """
        Exports the CPD as a pandas dataframe.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/insurance")
        >>> cpd = model.get_cpds(node="ThisCarCost")
        >>> df = cpd.to_dataframe()
        >>> df.query(
        ...     "CarValue=='FiftyThou' and Theft == 'True'"
        ... )  # doctest: +NORMALIZE_WHITESPACE
        ThisCarCost                 HundredThou  Million   TenThou  Thousand
        ThisCarDam CarValue  Theft
        Mild       FiftyThou True      0.950000      0.0  0.020000  0.030000
        Moderate   FiftyThou True      0.998000      0.0  0.001000  0.001000
        None       FiftyThou True      0.950000      0.0  0.010000  0.040000
        Severe     FiftyThou True      0.999998      0.0  0.000001  0.000001
        >>> # Probability sums up to zero, for every combination of evidence variables
        >>> df.sum(axis=1)
        ThisCarDam  CarValue    Theft
        Mild        FiftyThou   False    1.0
                                True     1.0
                    FiveThou    False    1.0
                                True     1.0
                    Million     False    1.0
                                True     1.0
                    TenThou     False    1.0
                                True     1.0
                    TwentyThou  False    1.0
                                True     1.0
        Moderate    FiftyThou   False    1.0
                                True     1.0
                    FiveThou    False    1.0
                                True     1.0
                    Million     False    1.0
                                True     1.0
                    TenThou     False    1.0
                                True     1.0
                    TwentyThou  False    1.0
                                True     1.0
        None        FiftyThou   False    1.0
                                True     1.0
                    FiveThou    False    1.0
                                True     1.0
                    Million     False    1.0
                                True     1.0
                    TenThou     False    1.0
                                True     1.0
                    TwentyThou  False    1.0
                                True     1.0
        Severe      FiftyThou   False    1.0
                                True     1.0
                    FiveThou    False    1.0
                                True     1.0
                    Million     False    1.0
                                True     1.0
                    TenThou     False    1.0
                                True     1.0
                    TwentyThou  False    1.0
                                True     1.0
        dtype: float64
        """
        pass

    def copy(self):
        """
        Returns a copy of the `TabularCPD` object.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd = TabularCPD(
        ...     variable="grade",
        ...     variable_card=2,
        ...     values=[[0.7, 0.6, 0.6, 0.2], [0.3, 0.4, 0.4, 0.8]],
        ...     evidence=["intel", "diff"],
        ...     evidence_card=[2, 2],
        ... )
        >>> copy = cpd.copy()
        >>> copy.variable
        'grade'
        >>> copy.variable_card
        2
        >>> copy.values
        array([[[0.7, 0.6],
                [0.6, 0.2]],
        <BLANKLINE>
               [[0.3, 0.4],
                [0.4, 0.8]]])
        """
        pass

    def normalize(self, inplace=True):
        """
        Normalizes the cpd table. The method modifies each column of values such
        that it sums to 1 without changing the proportion between states.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the CPD itself, else would return
            a new CPD

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd_table = TabularCPD(
        ...     variable="grade",
        ...     variable_card=2,
        ...     values=[[0.7, 0.2, 0.6, 0.2], [0.4, 0.4, 0.4, 0.8]],
        ...     evidence=["intel", "diff"],
        ...     evidence_card=[2, 2],
        ... )
        >>> cpd_table.normalize()
        >>> cpd_table.get_values()
        array([[0.63636364, 0.33333333, 0.6       , 0.2       ],
               [0.36363636, 0.66666667, 0.4       , 0.8       ]])
        """
        pass

    def marginalize(self, variables, inplace=True):
        """
        Modifies the CPD table with marginalized values. Marginalization refers to
        summing out variables, hence that variable would no longer appear in the
        CPD.

        Parameters
        ----------
        variables: list, array-like
            list of variable to be marginalized

        inplace: boolean
            If inplace=True it will modify the CPD itself, else would return
            a new CPD

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd_table = TabularCPD(
        ...     variable="grade",
        ...     variable_card=2,
        ...     values=[[0.7, 0.6, 0.6, 0.2], [0.3, 0.4, 0.4, 0.8]],
        ...     evidence=["intel", "diff"],
        ...     evidence_card=[2, 2],
        ... )
        >>> cpd_table.marginalize(variables=["diff"])
        >>> cpd_table.get_values()
        array([[0.65, 0.4 ],
               [0.35, 0.6 ]])
        """
        pass

    def reduce(self, values, inplace=True, show_warnings=True):
        """
        Reduces the cpd table to the context of given variable values. Reduce fixes the
        state of given variable to specified value. The reduced variables will no longer
        appear in the CPD.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_state).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd_table = TabularCPD(
        ...     variable="grade",
        ...     variable_card=2,
        ...     values=[[0.7, 0.6, 0.6, 0.2], [0.3, 0.4, 0.4, 0.8]],
        ...     evidence=["intel", "diff"],
        ...     evidence_card=[2, 2],
        ... )
        >>> cpd_table.reduce(values=[("diff", 0)])
        >>> cpd_table.get_values()
        array([[0.7, 0.6],
               [0.3, 0.4]])
        """
        pass

    def to_factor(self):
        """
        Returns an equivalent factor with the same variables, cardinality, values as that of the CPD.
        Since factor doesn't distinguish between conditional and non-conditional distributions,
        evidence information will be lost.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd = TabularCPD(
        ...     variable="grade",
        ...     variable_card=3,
        ...     values=[[0.1, 0.1], [0.1, 0.1], [0.8, 0.8]],
        ...     evidence=["evi1"],
        ...     evidence_card=[2],
        ... )
        >>> factor = cpd.to_factor()
        >>> factor # doctest: +ELLIPSIS
        <DiscreteFactor representing phi(grade:3, evi1:2) at 0x...>
        """
        pass

    def reorder_parents(self, new_order: list, inplace: bool = True):
        """
        Returns a new cpd table according to provided parent/evidence order.

        Parameters
        ----------
        new_order: list
            list of new ordering of variables

        inplace: boolean
            If inplace == True it will modify the CPD itself
            otherwise new value will be returned without affecting old values

        Examples
        --------

        Consider a CPD P(grade| diff, intel)

        >>> cpd = TabularCPD(
        ...     variable="grade",
        ...     variable_card=3,
        ...     values=[
        ...         [0.1, 0.1, 0.0, 0.4, 0.2, 0.1],
        ...         [0.3, 0.2, 0.1, 0.4, 0.3, 0.2],
        ...         [0.6, 0.7, 0.9, 0.2, 0.5, 0.7],
        ...     ],
        ...     evidence=["diff", "intel"],
        ...     evidence_card=[2, 3],
        ... )
        >>> print(cpd) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        +----------+----------+----------+----------+----------+----------+----------+
        | diff     | diff(0)  | diff(0)  | diff(0)  | diff(1)  | diff(1)  | diff(1)  |
        +----------+----------+----------+----------+----------+----------+----------+
        | intel    | intel(0) | intel(1) | intel(2) | intel(0) | intel(1) | intel(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(0) | 0.1      | 0.1      | 0.0      | 0.4      | 0.2      | 0.1      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(1) | 0.3      | 0.2      | 0.1      | 0.4      | 0.3      | 0.2      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(2) | 0.6      | 0.7      | 0.9      | 0.2      | 0.5      | 0.7      |
        +----------+----------+----------+----------+----------+----------+----------+
        >>> cpd.values
        array([[[0.1, 0.1, 0. ],
                [0.4, 0.2, 0.1]],
        <BLANKLINE>
               [[0.3, 0.2, 0.1],
                [0.4, 0.3, 0.2]],
        <BLANKLINE>
               [[0.6, 0.7, 0.9],
                [0.2, 0.5, 0.7]]])
        >>> cpd.variables
        ['grade', 'diff', 'intel']
        >>> cpd.cardinality
        array([3, 2, 3])
        >>> cpd.variable
        'grade'
        >>> cpd.variable_card
        3
        >>> cpd.reorder_parents(new_order=["intel", "diff"])
        array([[0.1, 0.4, 0.1, 0.2, 0. , 0.1],
               [0.3, 0.4, 0.2, 0.3, 0.1, 0.2],
               [0.6, 0.2, 0.7, 0.5, 0.9, 0.7]])
        >>> print(cpd) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        +----------+----------+----------+----------+----------+----------+----------+
        | intel    | intel(0) | intel(0) | intel(1) | intel(1) | intel(2) | intel(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        | diff     | diff(0)  | diff(1)  | diff(0)  | diff(1)  | diff(0)  | diff(1)  |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(0) | 0.1      | 0.4      | 0.1      | 0.2      | 0.0      | 0.1      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(1) | 0.3      | 0.4      | 0.2      | 0.3      | 0.1      | 0.2      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(2) | 0.6      | 0.2      | 0.7      | 0.5      | 0.9      | 0.7      |
        +----------+----------+----------+----------+----------+----------+----------+
        >>> cpd.values
        array([[[0.1, 0.4],
                [0.1, 0.2],
                [0. , 0.1]],
        <BLANKLINE>
               [[0.3, 0.4],
                [0.2, 0.3],
                [0.1, 0.2]],
        <BLANKLINE>
               [[0.6, 0.2],
                [0.7, 0.5],
                [0.9, 0.7]]])
        >>> cpd.variables
        ['grade', 'intel', 'diff']
        >>> cpd.cardinality
        array([3, 3, 2])
        >>> cpd.variable
        'grade'
        >>> cpd.variable_card
        3
        """
        pass

    def get_evidence(self):
        """
        Returns the evidence variables of the CPD.
        """
        pass

    @staticmethod
    def get_random(variable, evidence=None, cardinality=None, state_names={}, seed=None):
        """
        Generates a TabularCPD instance with random values on `variable` with
        parents/evidence `evidence` with cardinality/number of states as given
        in `cardinality`.

        Parameters
        ----------
        variable: str, int or any hashable python object.
            The variable on which to define the TabularCPD.

        evidence: list, array-like
            A list of variable names which are the parents/evidence of `variable`.

        cardinality: dict (default: None)
            A dict of the form {var_name: card} specifying the number of states/
            cardinality of each of the variables. If None, assigns each variable
            2 states.

        state_names: dict (default: {})
            A dict of the form {var_name: list of states} to specify the state names
            for the variables in the CPD. If state_names=None, integral state names
            starting from 0 is assigned.

        Returns
        -------
        Random CPD: pgmpy.factors.discrete.TabularCPD
            A TabularCPD object on `variable` with `evidence` as evidence with random values.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> TabularCPD.get_random(
        ...     variable="A", evidence=["B", "C"], cardinality={"A": 3, "B": 2, "C": 4}
        ... ) # doctest: +ELLIPSIS
        <TabularCPD representing P(A:3 | ...) at 0x...>
        >>> TabularCPD.get_random(
        ...     variable="A",
        ...     evidence=["B", "C"],
        ...     cardinality={"A": 2, "B": 2, "C": 2},
        ...     state_names={"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]},
        ... ) # doctest: +ELLIPSIS
        <TabularCPD representing P(A:2 | B:2, C:2) at 0x...>
        """
        pass

    @staticmethod
    def get_uniform(variable, evidence=None, cardinality=None, state_names={}, seed=None):
        """
        Generates a TabularCPD instance with uniform values (i.e., all
        probabilities are 0.5) on `variable` with parents/evidence `evidence`
        with cardinality/number of states as given in `cardinality`.

        Parameters
        ----------
        variable: str, int or any hashable python object.
            The variable on which to define the TabularCPD.

        evidence: list, array-like
            A list of variable names which are the parents/evidence of `variable`.

        cardinality: dict (default: None)
            A dict of the form {var_name: card} specifying the number of states/
            cardinality of each of the variables. If None, assigns each variable
            2 states.

        state_names: dict (default: {})
            A dict of the form {var_name: list of states} to specify the state names
            for the variables in the CPD. If state_names=None, integral state names
            starting from 0 is assigned.

        Returns
        -------
        Uniform CPD: pgmpy.factors.discrete.TabularCPD
            A TabularCPD object on `variable` with `evidence` as evidence with
            all probabilities set to 0.5.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> TabularCPD.get_uniform(
        ...     variable="A", evidence=["B", "C"], cardinality={"A": 3, "B": 2, "C": 4}
        ... ) # doctest: +ELLIPSIS
        <TabularCPD representing P(A:3 | ...) at 0x...>
        >>> TabularCPD.get_uniform(
        ...     variable="A",
        ...     evidence=["B", "C"],
        ...     cardinality={"A": 2, "B": 2, "C": 2},
        ...     state_names={"A": ["a1", "a2"], "B": ["b1", "b2"], "C": ["c1", "c2"]},
        ... ) # doctest: +ELLIPSIS
        <TabularCPD representing P(A:2 | B:2, C:2) at 0x...>
        """
        pass
