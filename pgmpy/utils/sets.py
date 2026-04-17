from collections.abc import Iterable
from itertools import chain, combinations


def _variable_or_iterable_to_set(x):
    """
    Convert variable, set, or iterable x to a frozenset.

    If x is None, returns the empty set.

    Parameters
    ---------
    x : None, str or Iterable[str]

    Returns
    -------
    frozenset : frozenset representation of string or iterable input
    """
    pass


def _powerset(iterable):
    """
    https://docs.python.org/3/library/itertools.html#recipes
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Parameters
    ----------
    iterable: any iterable

    Returns
    -------
    chain: a generator of the powerset of the input
    """
    pass
