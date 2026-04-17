#!/usr/bin/env python3
from __future__ import annotations

from numbers import Number

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from pgmpy.factors.base import factor_product
from pgmpy.factors.discrete import DiscreteFactor


class FactorDict(dict):
    @classmethod
    def from_dataframe(cls, df, marginals):
        """Create a `FactorDict` from a given set of marginals.

        Parameters
        ----------
        df: pandas DataFrame object

        marginals: List[Tuple[str]]
            List of Tuples containing the names of the marginals.

        Returns
        -------
        Factor dictionary: FactorDict
            FactorDict with each marginal's Factor representing the empirical
                frequency of the marginal from the dataset.
        """
        pass

    def get_factors(self):
        pass

    def __mul__(self, const):
        return FactorDict({clique: const * self[clique] for clique in self})

    def __rmul__(self, const):
        return self.__mul__(const)

    def __add__(self, other):
        return FactorDict(
            {clique: self[clique] + other for clique in self}
            if isinstance(other, Number)
            else {clique: self[clique] + other[clique] for clique in self}
        )

    def __sub__(self, other):
        return self + -1 * other

    def dot(self, other):
        pass

    def product(self):
        pass
