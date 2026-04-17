"""Common API for torch and numpy backends."""

from copy import deepcopy

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config

torch = _safe_import("torch")


def _is_torch_tensor(obj):
    pass


def size(arr):
    pass


def copy(arr):
    pass


def tobytes(arr):
    pass


def max(arr, axis=None):
    pass


def einsum(*args):
    pass


def argmax(arr):
    pass


def stack(arr_iter):
    pass


def to_numpy(arr, decimals=None):
    pass


def ravel_f(arr):
    pass


def ones(n):
    pass


def get_compute_backend():
    pass


def unique(arr, axis=0, return_counts=False, return_inverse=False):
    pass


def flip(arr, axis=0):
    pass


def transpose(arr, axis):
    pass


def exp(arr):
    pass


def sum(arr):
    pass


def allclose(arr1, arr2, atol):
    pass
