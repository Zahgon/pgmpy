from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from skbase.base import BaseObject
from skbase.lookup import all_objects

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.utils.hf_hub import read_hf_file


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    expert_knowledge: ExpertKnowledge | None = None
    ground_truth: DAG | None = None

    tags: dict[str, Any] = None

    def __str__(self) -> str:
        return (
            f"Dataset(name={self.name}, \n data=DataFrame of size: {self.data.shape}, \n "
            f"expert_knowledge={self.expert_knowledge}, \n ground_truth={self.ground_truth}, \n tags={self.tags})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class _BaseDataset(BaseObject):
    """
    Base class for all datasets in pgmpy.
    Inherits from skbase.base.BaseObject to utilize its tag and lookup functionality.
    """

    # define tags
    _tags = {
        "name": None,
        "n_variables": None,
        "n_samples": None,
        "has_ground_truth": False,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": False,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": False,
        "is_mixed": False,
        "is_ordinal": False,
    }

    base_url = ""
    repo_id = "pgmpy/example_datasets"
    repo_type = "dataset"
    revision = "main"

    @staticmethod
    def _parse_expert_knowledge(raw_expert_knowledge: bytes) -> ExpertKnowledge:
        """
        Helper method to parse expert knowledge from raw bytes.
        """
        pass

    @classmethod
    def _get_raw_data(cls, filename) -> bytes:
        """
        Fetches a dataset file from the Hugging Face Hub cache.
        """
        pass

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """
        Fetches/reads from cache the data associated with the dataset.
        """
        pass

    @classmethod
    def load_expert_knowledge(cls) -> ExpertKnowledge:
        """Fetches/reads from cache the expert knowledge associated with the dataset."""
        pass

    @classmethod
    def load_ground_truth(cls) -> DAG:
        """Fetches/reads from cache the ground truth DAG associated with the dataset."""
        pass


class _CovarianceMixin:
    """
    This mixin class provides functionality to load datasets defined by a covariance matrix. Mainly the `load_dataframe`
    method is overridden to generate data from the covariance matrix instead of loading a static data file as is the
    case with `_BaseDataset`.
    """

    @classmethod
    def _load_covariance_matrix(cls) -> pd.DataFrame:
        """
        Fetches the data and creates a covariance matrix DataFrame.
        """
        pass

    @classmethod
    def load_dataframe(cls) -> pd.DataFrame:
        """Method to create data from covariance matrix. When the `_CovarDatasetMixin is
        used this method is supposed to override the _BaseDataset.load_dataframe method.

        ** Hence, when using this mixin, _CovarDatasetMixin should be the first parent class. **
        """
        pass


class _TubingenBenchmarkMixin:
    """
    Mixin for Tubingen datasets that consist of multiple independent pairs/files.
    URL: https://webdav.tuebingen.mpg.de/cause-effect/
    """

    @classmethod
    def load_dataframe(cls, pair_id: int) -> pd.DataFrame:
        pass

    @classmethod
    def load_ground_truth(cls, pair_id: int) -> DAG:
        pass


def load_dataset(name: str) -> Dataset:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> dataset = load_dataset("sachs_mixed")
    >>> df = dataset.data
    >>> ground_truth = dataset.ground_truth
    """
    pass


def list_datasets(**filter_tags) -> list[str]:
    """
    Returns a list of all available datasets, optionally filtered by a query string.

    Parameters
    ----------
    **filter_tags : optional arguments
        If specified, returns only datasets matching the provided tag filters. Any dataset tag can be used as a filter.
        Available tags:
            - n_variables
            - n_samples
            - has_ground_truth
            - has_expert_knowledge
            - has_missing_data
            - is_simulated
            - is_interventional
            - is_discrete
            - is_continuous
            - is_mixed
            - is_ordinal

    Returns
    -------
    list of str
        A sorted list of available dataset names.

    Examples
    --------
    >>> from pgmpy.datasets import list_datasets
    >>> list_datasets()
    ['abalone_continuous', 'abalone_mixed', ..., 'sachs_continuous', ...]

    >>> list_datasets(is_discrete=True, has_ground_truth=True)
    ['sachs_discrete']
    """
    pass
