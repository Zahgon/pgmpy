#!/usr/bin/env python

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator


class TreeSearch(StructureEstimator):
    """
    Search class for learning tree related graph structure. The algorithms
    supported are Chow-Liu and Tree-augmented naive bayes (TAN).

    Chow-Liu constructs the maximum-weight spanning tree with mutual information
    score as edge weights.

    TAN is an extension of Naive Bayes classifier to allow a tree structure over
    the independent variables to account for interaction.

    Parameters
    ----------
    data: pandas.DataFrame object
        dataframe object where each column represents one variable.

    root_node: str, int, or any hashable python object, default is None.
        The root node of the tree structure. If None then root node is auto-picked
        as the node with the highest sum of edge weights.

    n_jobs: int (default: -1)
        Number of jobs to run in parallel. `-1` means use all processors.

    References
    ----------
    [1] Chow, C. K.; Liu, C.N. (1968), "Approximating discrete probability
        distributions with dependence trees", IEEE Transactions on Information
        Theory, IT-14 (3): 462–467

    [2] Friedman N, Geiger D and Goldszmidt M (1997). Bayesian network classifiers.
        Machine Learning 29: 131–163
    """

    def __init__(self, data, root_node=None, n_jobs=-1, **kwargs):
        if root_node is not None and root_node not in data.columns:
            raise ValueError(f"Root node: {root_node} not found in data columns.")

        self.data = data
        self.root_node = root_node
        self.n_jobs = n_jobs

        super().__init__(data, **kwargs)

    def estimate(
        self,
        estimator_type="chow-liu",
        class_node=None,
        edge_weights_fn="mutual_info",
        show_progress=True,
    ):
        """
        Estimate the `DAG` structure that fits best to the given data set without
        parametrization.

        Parameters
        ----------
        estimator_type: str (chow-liu | tan)
            The algorithm to use for estimating the DAG.

        class_node: string, int or any hashable python object. (optional)
            Needed only if estimator_type = 'tan'. In the estimated DAG, there would be
            edges from class_node to each of the feature variables.

        edge_weights_fn: str or function (default: mutual info)
            Method to use for computing edge weights. By default, Mutual Info Score is
            used.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        Estimated Model: pgmpy.base.DAG
            The estimated model structure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> est = TreeSearch(values, root_node="B")
        >>> model = est.estimate(estimator_type="chow-liu")
        >>> nx.draw_circular(
        ...     model, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
        ... )
        >>> plt.show()
        >>> est = TreeSearch(values)
        >>> model = est.estimate(estimator_type="chow-liu")
        >>> nx.draw_circular(
        ...     model, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
        ... )
        >>> plt.show()
        >>> est = TreeSearch(values, root_node="B")
        >>> model = est.estimate(estimator_type="tan", class_node="A")
        >>> nx.draw_circular(
        ...     model, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
        ... )
        >>> plt.show()
        >>> est = TreeSearch(values)
        >>> model = est.estimate(estimator_type="tan")
        >>> nx.draw_circular(
        ...     model, with_labels=True, arrowsize=20, arrowstyle="fancy", alpha=0.3
        ... )
        >>> plt.show()
        """
        pass

    @staticmethod
    def _get_weights(data, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True):
        """
        Helper function to Chow-Liu algorithm for estimating tree structure from given data. Refer to
        pgmpy.estimators.TreeSearch for more details. This function returns the edge weights matrix.

        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.

        edge_weights_fn: str or function (default: mutual_info)
            Method to use for computing edge weights. Options are:
                1. 'mutual_info': Mutual Information Score.
                2. 'adjusted_mutual_info': Adjusted Mutual Information Score.
                3. 'normalized_mutual_info': Normalized Mutual Information Score.
                4. function(array[n_samples,], array[n_samples,]): Custom function.

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        weights: numpy 2D array, shape = (n_columns, n_columns)
            symmetric matrix where each element represents an edge weight.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> est = TreeSearch(values, root_node="B")
        >>> model = est.estimate(estimator_type="chow-liu")
        """
        pass

    @staticmethod
    def _get_conditional_weights(data, class_node, edge_weights_fn="mutual_info", n_jobs=-1, show_progress=True):
        """
        Helper function to TAN (Tree Augmented Naive Bayes) algorithm for
        estimating tree structure from given data. Refer to
        pgmpy.estimators.TreeSearch for more details. This function returns the
        edge weights matrix.

        Parameters
        ----------
        data: pandas.DataFrame object
            dataframe object where each column represents one variable.

        class_node: str
            The class node for TAN. The edge weight is computed as I(X, Y | class_node).

        edge_weights_fn: str or function (default: mutual_info)
            Method to use for computing edge weights. Options are:
                1. 'mutual_info': Mutual Information Score.
                2. 'adjusted_mutual_info': Adjusted Mutual Information Score.
                3. 'normalized_mutual_info': Normalized Mutual Information Score.
                4. function(array[n_samples,], array[n_samples,]): Custom function.

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. `-1` means use all processors.

        show_progress: boolean
            If True, shows a progress bar for the running algorithm.

        Returns
        -------
        weights: numpy 2D array, shape = (n_columns, n_columns)
            symmetric matrix where each element represents an edge weight.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> est = TreeSearch(values, root_node="B")
        >>> model = est.estimate(estimator_type="tan")
        """
        pass

    @staticmethod
    def _create_tree_and_dag(weights, columns, root_node):
        """
        Helper function to Chow-Liu algorithm for estimating tree structure from given data. Refer to
        pgmpy.estimators.TreeSearch for more details. This function returns the DAG based on the edge weights matrix.

        Parameters
        ----------
        weights: numpy 2D array, shape = (n_columns, n_columns)
            symmetric matrix where each element represents an edge weight.

        columns: list or array
            Names of the columns (& rows) of the weights matrix.

        root_node: str, int, or any hashable python object.
            The root node of the tree structure.

        Returns
        -------
        model: pgmpy.base.DAG
            The estimated model structure.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.estimators import TreeSearch
        >>> values = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(1000, 5)),
        ...     columns=["A", "B", "C", "D", "E"],
        ... )
        >>> est = TreeSearch(values, root_node="B")
        >>> model = est.estimate(estimator_type="chow-liu")
        """
        pass
