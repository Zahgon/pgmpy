import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.factors import factor_product
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork, MarkovChain
from pgmpy.sampling import BayesianModelInference, _return_samples
from pgmpy.utils.mathext import sample_discrete, sample_discrete_maps

State = namedtuple("State", ["var", "state"])


class BayesianModelSampling(BayesianModelInference):
    """
    Class for sampling methods specific to Bayesian Models

    Parameters
    ----------
    model: instance of DiscreteBayesianNetwork
        model on which inference queries will be computed
    """

    def __init__(self, model):
        super().__init__(model)

    def forward_sample(
        self,
        size=1,
        include_latents=False,
        seed=None,
        show_progress=True,
        partial_samples=None,
        n_jobs=-1,
    ):
        """
        Generates sample(s) from joint distribution of the Bayesian Network.

        Parameters
        ----------
        size: int
            size of sample to be generated

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: boolean
            Whether to show a progress bar of samples getting generated.

        partial_samples: pandas.DataFrame
            A pandas dataframe specifying samples on some of the variables in the model. If
            specified, the sampling procedure uses these sample values, instead of generating them.

        n_jobs: int (default: -1)
            The number of CPU cores to use. Default uses all cores.

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        >>> cpd_d = TabularCPD("diff", 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD("intel", 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     ["intel", "diff"],
        ...     [2, 2],
        ... )
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> inference.forward_sample(size=2)
        rec.array([(0, 0, 1), (1, 0, 2)], dtype=
                  [('diff', '<i8'), ('intel', '<i8'), ('grade', '<i8')])
        """
        pass

    def rejection_sample(
        self,
        evidence=[],
        size=1,
        include_latents=False,
        seed=None,
        show_progress=True,
        partial_samples=None,
    ):
        """
        Generates sample(s) from joint distribution of the Bayesian Network,
        given the evidence.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence

        size: int
            size of sample to be generated

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: boolean
            Whether to show a progress bar of samples getting generated.

        partial_samples: pandas.DataFrame
            A pandas dataframe specifying samples on some of the variables in the model. If
            specified, the sampling procedure uses these sample values, instead of generating them.

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.factors.discrete import State
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        >>> cpd_d = TabularCPD("diff", 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD("intel", 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     ["intel", "diff"],
        ...     [2, 2],
        ... )
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> evidence = [State(var="diff", state=0)]
        >>> inference.rejection_sample(
        ...     evidence=evidence, size=2, return_type="dataframe"
        ... )
                intel       diff       grade
        0         0          0          1
        1         0          0          1
        """
        pass

    def likelihood_weighted_sample(
        self,
        evidence=[],
        size=1,
        include_latents=False,
        seed=None,
        show_progress=True,
        n_jobs=-1,
    ):
        """
        Generates weighted sample(s) from joint distribution of the Bayesian
        Network, that comply with the given evidence.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Algorithm 12.2 pp 493.

        Parameters
        ----------
        evidence: list of `pgmpy.factor.State` namedtuples
            None if no evidence

        size: int
            size of sample to be generated

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: boolean
            Whether to show a progress bar of samples getting generated.

        n_jobs: int (default: -1)
            The number of CPU cores to use. Default uses all cores.

        Returns
        -------
        sampled: A pandas.DataFrame
            The generated samples with corresponding weights

        Examples
        --------
        >>> from pgmpy.factors.discrete import State
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.sampling import BayesianModelSampling
        >>> student = DiscreteBayesianNetwork([("diff", "grade"), ("intel", "grade")])
        >>> cpd_d = TabularCPD("diff", 2, [[0.6], [0.4]])
        >>> cpd_i = TabularCPD("intel", 2, [[0.7], [0.3]])
        >>> cpd_g = TabularCPD(
        ...     "grade",
        ...     3,
        ...     [[0.3, 0.05, 0.9, 0.5], [0.4, 0.25, 0.08, 0.3], [0.3, 0.7, 0.02, 0.2]],
        ...     ["intel", "diff"],
        ...     [2, 2],
        ... )
        >>> student.add_cpds(cpd_d, cpd_i, cpd_g)
        >>> inference = BayesianModelSampling(student)
        >>> evidence = [State("diff", 0)]
        >>> inference.likelihood_weighted_sample(
        ...     evidence=evidence, size=2, return_type="recarray"
        ... )
        rec.array([(0, 0, 1, 0.6), (0, 0, 2, 0.6)], dtype=
                  [('diff', '<i8'), ('intel', '<i8'), ('grade', '<i8'), ('_weight', '<f8')])
        """
        pass


class GibbsSampling(MarkovChain):
    """
    Class for performing Gibbs sampling.

    Parameters
    ----------
    model: DiscreteBayesianNetwork or DiscreteMarkovNetwork
        Model from which variables are inherited and transition probabilities computed.

    Examples
    --------
    Initialization from a DiscreteBayesianNetwork object:

    >>> from pgmpy.factors.discrete import TabularCPD
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> intel_cpd = TabularCPD("intel", 2, [[0.7], [0.3]])
    >>> sat_cpd = TabularCPD(
    ...     "sat", 2, [[0.95, 0.2], [0.05, 0.8]], evidence=["intel"], evidence_card=[2]
    ... )
    >>> student = DiscreteBayesianNetwork()
    >>> student.add_nodes_from(["intel", "sat"])
    >>> student.add_edge("intel", "sat")
    >>> student.add_cpds(intel_cpd, sat_cpd)
    >>> from pgmpy.sampling import GibbsSampling
    >>> gibbs_chain = GibbsSampling(student)
    >>> gibbs_chain.sample(size=3)
       intel  sat
    0      0    0
    1      0    0
    2      1    1
    """

    def __init__(self, model=None):
        super().__init__()
        if isinstance(model, DiscreteBayesianNetwork):
            self._get_kernel_from_bayesian_model(model)
        elif isinstance(model, DiscreteMarkovNetwork):
            self._get_kernel_from_markov_model(model)

    def _get_kernel_from_bayesian_model(self, model):
        """
        Computes the Gibbs transition models from a Bayesian Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters
        ----------
        model: DiscreteBayesianNetwork
            The model from which probabilities will be computed.
        """
        pass

    def _get_kernel_from_markov_model(self, model):
        """
        Computes the Gibbs transition models from a Markov Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters
        ----------
        model: DiscreteMarkovNetwork
            The model from which probabilities will be computed.
        """
        pass

    def sample(self, start_state=None, size=1, seed=None, include_latents=False):
        """
        Sample from the Markov Chain.

        Parameters
        ----------
        start_state: dict or array-like iterable
            Representing the starting states of the variables. If None is passed, a random start_state is chosen.

        size: int
            Number of samples to be generated.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        Returns
        -------
        sampled: pandas.DataFrame
            The generated samples

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.sampling import GibbsSampling
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> model = DiscreteMarkovNetwork([("A", "B"), ("C", "B")])
        >>> factor_ab = DiscreteFactor(["A", "B"], [2, 2], [1, 2, 3, 4])
        >>> factor_cb = DiscreteFactor(["C", "B"], [2, 2], [5, 6, 7, 8])
        >>> model.add_factors(factor_ab, factor_cb)
        >>> gibbs = GibbsSampling(model)
        >>> gibbs.sample(size=4, return_tupe="dataframe")
           A  B  C
        0  0  1  1
        1  1  0  0
        2  1  1  0
        3  1  1  1
        """
        pass

    def generate_sample(self, start_state=None, size=1, include_latents=False, seed=None):
        """
        Generator version of self.sample

        Returns
        -------
        List of State namedtuples, representing the assignment to all variables of the model.

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.sampling import GibbsSampling
        >>> from pgmpy.models import DiscreteMarkovNetwork
        >>> model = DiscreteMarkovNetwork([("A", "B"), ("C", "B")])
        >>> factor_ab = DiscreteFactor(["A", "B"], [2, 2], [1, 2, 3, 4])
        >>> factor_cb = DiscreteFactor(["C", "B"], [2, 2], [5, 6, 7, 8])
        >>> model.add_factors(factor_ab, factor_cb)
        >>> gibbs = GibbsSampling(model)
        >>> gen = gibbs.generate_sample(size=2)
        >>> [sample for sample in gen]
        [[State(var='C', state=1), State(var='B', state=1), State(var='A', state=0)],
         [State(var='C', state=0), State(var='B', state=1), State(var='A', state=1)]]
        """
        pass
