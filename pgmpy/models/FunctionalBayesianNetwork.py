from collections.abc import Callable, Hashable, Iterable
from typing import (
    Any,
)

import networkx as nx
import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config, logger
from pgmpy.factors.hybrid import FunctionalCPD
from pgmpy.models import DiscreteBayesianNetwork

pyro = _safe_import("pyro", pkg_name="pyro-ppl")


class FunctionalBayesianNetwork(DiscreteBayesianNetwork):
    """
    Class for representing Functional Bayesian Network.

    Functional Bayesian Networks allow for flexible representation of probability distribution using CPDs in functional
    form (FunctionalCPD). As these CPDs are defined using functions that return pyro distributions, they can represent
    any distribution supported by Pyro.

    Parameters
    ----------
    ebunch : input graph, optional
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    latents : set of nodes, default=None
        A set of latent variables in the graph. These are not observed
        variables but are used to represent unobserved confounding or
        other latent structures.

    exposures : set, default=None
        Set of exposure variables in the graph. These are the variables
        that represent the treatment or intervention being studied in a
        causal analysis. If None, exposures will be treated as an empty set.

    outcomes : set, optional (default: None)
        Set of outcome variables in the graph. These are the variables
        that represent the response or dependent variables being studied
        in a causal analysis. If None, defaults to an empty set.

    roles : dict, optional (default: None)
        A dictionary mapping roles to node names.
        The keys are roles, and the values are role names (strings or iterables of str).
        If provided, this will automatically assign roles to the nodes in the graph.
        Passing a key-value pair via ``roles`` is equivalent to calling
        ``with_role(role, variables)`` for each key-value pair in the dictionary.

    Examples
    --------
    # Defining a Functional Bayesian Network

    >>> from pgmpy.models import FunctionalBayesianNetwork
    >>> from pgmpy.factors.hybrid import FunctionalCPD
    >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
    >>> model.add_cpds(
    ...     FunctionalCPD("x1", lambda _: dist.Normal(0, 1)),
    ...     FunctionalCPD(
    ...         "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
    ...     ),
    ...     FunctionalCPD(
    ...         "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
    ...     ),
    ... )
    >>> model.check_model()
    True

    # Simulating data from the Functional Bayesian Network

    >>> samples = model.simulate(n_samples=1000)

    # Fitting the Functional Bayesian Network to the simulated data

    >>> fitted_params = model.fit(samples, estimator="SVI", num_steps=1000)
    """

    def __init__(
        self,
        ebunch: Iterable[tuple[Hashable, Hashable]] | None = None,
        latents: set[Hashable] | None = None,
        exposures: set[Hashable] | None = None,
        outcomes: set[Hashable] | None = None,
        roles: dict[str, Iterable] | None = None,
    ) -> None:
        if config.get_backend() == "numpy":
            msg = (
                f"{type(self)} requires pytorch backend, currently it is "
                "set to numpy."
                "Call pgmpy.config.set_backend('torch') to switch the backend globally."
            )
            logger.info(msg)
            raise ValueError(msg)

        _check_soft_dependencies("pyro-ppl", obj=self)

        super().__init__(
            ebunch=ebunch,
            latents=latents,
            exposures=exposures,
            outcomes=outcomes,
            roles=roles,
        )

    def add_cpds(self, *cpds: FunctionalCPD) -> None:
        """
        Adds FunctionalCPDs to the Bayesian Network.

        Parameters
        ----------
        cpds: instances of FunctionalCPD
            List of FunctionalCPDs which will be associated with the model

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import pyro.distributions as dist
        >>> import numpy as np

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)

        """
        pass

    def get_cpds(self, node: Any | None = None) -> list[FunctionalCPD] | FunctionalCPD:
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameter
        ---------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of Functional CPDs.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds()
        """
        pass

    def remove_cpds(self, *cpds: FunctionalCPD) -> None:
        """
        Removes the given `cpds` from the model.

        Parameters
        ----------
        *cpds: FunctionalCPD objects
            A list of FunctionalCPD objects that need to be removed from the model.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        pass

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)
        pass
        """
        pass

    def check_model(self) -> bool:
        """
        Checks the model for various errors. This method checks for the following
        error -

        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass.

        """
        pass

    def simulate(
        self,
        n_samples: int = 1000,
        do: dict[Hashable, Any] | None = None,
        virtual_intervention: list[FunctionalCPD] | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Simulate samples from the model.

        Parameters
        ----------
        n_samples : int, optional (default: 1000)
            Number of samples to generate

        seed : int, optional
            The seed value for the random number generator.

        do : dict, optional
            Specifies hard interventions to the model. The dict should be of
            the form {variable: value}. Incoming edges into each intervened
            variable are severed and the variable is set to the given constant
            for all rows.

        virtual_intervention : list[FunctionalCPD], optional
            A list of unconditional FunctionalCPD objects (no parents) that
            replace the corresponding node’s CPD during simulation (i.e.,
            stochastic interventions like do(X ~ Normal(...))).

        Returns
        -------
        pandas.DataFrame
            Simulated samples with columns corresponding to network variables

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2"), ("x2", "x3")])
        >>> cpd1 = FunctionalCPD("x1", lambda _: dist.Normal(0, 1))
        >>> cpd2 = FunctionalCPD(
        ...     "x2", lambda parent: dist.Normal(parent["x1"] + 2.0, 1), parents=["x1"]
        ... )
        >>> cpd3 = FunctionalCPD(
        ...     "x3", lambda parent: dist.Normal(parent["x2"] + 0.3, 2), parents=["x2"]
        ... )
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.simulate(n_samples=1000)
        """
        pass

    def fit(
        self,
        data: pd.DataFrame,
        estimator: str = "SVI",
        optimizer: pyro.optim.PyroOptim = pyro.optim.Adam({"lr": 1e-2}),
        prior_fn: Callable | None = None,
        num_steps: int = 1000,
        seed: int | None = None,
        nuts_kwargs: dict | None = None,
        mcmc_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """
        Fit the Bayesian network to data using Pyro's stochastic variational inference.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with observations of variables.

        estimator: str (default: "SVI")
            Fitting method to use. Currently supports "SVI" and "MCMC".

        optimizer: Instance of pyro optimizer (default: pyro.optim.Adam({"lr": 1e-2}))
            Only used if `estimator` is "SVI". The optimizer to use for optimization.

        prior_fn: function
            Only used if `estimator` is "MCMC". A function that returns a dictionary of
            pyro distributions for each parameter in the model.

        num_steps: int (default: 100)
            Number of optimization steps. For SVI it is the `num_steps`
            argument for pyro.infer.SVI. For MCMC, it is the `num_samples`
            argument for pyro.infer.MCMC.

        seed: int (default: None)
            Seed value for random number generator.

        nuts_kwargs: dict (default: None)
            Only used if `estimator` is "MCMC". Additional arguments to pass to
            pyro.infer.NUTS.

        mcmc_kwargs: dict (default: None)
            Only used if `estimator` is "MCMC". Additional arguments to pass to
            pyro.infer.MCMC.

        Returns
        -------
        dict: If `estimator` is "SVI", returns a dictionary of parameter values.
              If `estimator` is "MCMC", returns a dictionary of posterior samples for each parameter.

        Examples
        --------
        >>> from pgmpy.factors.hybrid import FunctionalCPD
        >>> from pgmpy.models import FunctionalBayesianNetwork
        >>> import numpy as np
        >>> import pyro.distributions as dist

        >>> model = FunctionalBayesianNetwork([("x1", "x2")])
        >>> x1 = np.random.normal(0.2, 0.8, size=10000)
        >>> x2 = np.random.normal(0.6 + x1, 1)
        >>> data = pd.DataFrame({"x1": x1, "x2": x2})

        >>> def x1_fn(parents):
        ...     mu = pyro.param("x1_mu", torch.tensor(1.0))
        ...     sigma = pyro.param(
        ...         "x1_sigma", torch.tensor(1.0), constraint=constraints.positive
        ...     )
        ...     return dist.Normal(mu, sigma)
        pass

        >>> def x2_fn(parents):
        ...     intercept = pyro.param("x2_inter", torch.tensor(1.0))
        ...     sigma = pyro.param(
        ...         "x2_sigma", torch.tensor(1.0), constraint=constraints.positive
        ...     )
        ...     return dist.Normal(intercept + parents["x1"], sigma)
        pass

        >>> cpd1 = FunctionalCPD("x1", fn=x1_prior)
        >>> cpd2 = FunctionalCPD("x2", fn=x2_prior, parents=["x1"])
        >>> model.add_cpds(cpd1, cpd2)
        >>> params = model.fit(data, estimator="SVI", num_steps=100)
        >>> print(params)

        >>> def prior_fn():
        ...     return {
        ...         "x1_mu": dist.Uniform(0, 1),
        ...         "x1_sigma": dist.HalfNormal(5),
        ...         "x2_inter": dist.Normal(1.0),
        ...         "x2_sigma": dist.HalfNormal(1),
        ...     }
        pass

        >>> def x1_fn(priors, parents):
        ...     return dist.Normal(priors["x1_mu"], priors["x1_sigma"])
        pass

        >>> def x2_fn(priors, parents):
        ...     return dist.Normal(
        ...         priors["x2_inter"] + parent["x1"], priors["x2_sigma"]
        ...     )
        pass

        >>> cpd1 = FunctionalCPD("x1", fn=x1_fn)
        >>> cpd2 = FunctionalCPD("x2", fn=x2_fn, parents=["x1"])
        >>> model.add_cpds(cpd1, cpd2)

        >>> params = model.fit(data, estimator="MCMC", prior_fn=prior_fn, num_steps=100)
        >>> print(params["x1_mu"].mean(), params["x1_std"].mean())
        """
        pass
