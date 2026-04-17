import itertools

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import DiscreteBayesianNetwork, DynamicBayesianNetwork
from pgmpy.utils import compat_fns


class ApproxInference:
    """
    Initializes the Approximate Inference class.

    Parameters
    ----------
    model: Instance of pgmpy.models.DiscreteBayesianNetwork or pgmpy.models.DynamicBayesianNetwork

    Examples
    --------
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/alarm")
    >>> infer = ApproxInference(model)
    """

    def __init__(self, model):
        if not isinstance(model, (DiscreteBayesianNetwork, DynamicBayesianNetwork)):
            raise ValueError(
                f"model should either be a Bayesian Network or Dynamic Bayesian Network. Got {type(model)}."
            )
        model.check_model()
        self.model = model

    @staticmethod
    def _get_factor_from_df(df, state_names):
        """
        Takes a groupby dataframe and converts it into a pgmpy.factors.discrete.DiscreteFactor object.
        """
        pass

    def get_distribution(self, samples, variables, state_names=None, joint=True):
        """
        Computes distribution of `variables` from given data `samples`.

        Parameters
        ----------
        samples: pandas.DataFrame
            A dataframe of samples generated from the model.

        variables: list (array-like)
            A list of variables whose distribution needs to be computed.

        state_names: dict (default: None)
            A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
            If None, inferred from the data but is possible that the final distribution misses some states.

        joint: boolean
            If joint=True, computes the joint distribution over `variables`.
            Else, returns a dict with marginal distribution of each variable in
            `variables`.
        """
        pass

    def query(
        self,
        variables,
        n_samples=int(1e4),
        samples=None,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        state_names=None,
        show_progress=True,
        seed=None,
    ):
        """
        Method for doing approximate inference based on sampling in Bayesian
        Networks and Dynamic Bayesian Networks.

        Parameters
        ----------
        variables: list
            List of variables for which the probability distribution needs to be calculated.

        n_samples: int
            The number of samples to generate for computing the distributions. Higher `n_samples`
            results in more accurate results at the cost of more computation time.

        samples: pd.DataFrame (default: None)
            If provided, uses these samples to compute the distribution instead
            of generating samples. `samples` **must** conform with the provided
            `evidence` and `virtual_evidence`.

        evidence: dict (default: None)
            The observed values. A dict key, value pair of the form {var: state_name}.

        virtual_evidence: list (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual/soft
            evidence.

        state_names: dict (default: None)
            A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
            If None, inferred from the data but is possible that the final distribution misses some states.

        show_progress: boolean (default: True)
            If True, shows a progress bar when generating samples.

        seed: int (default: None)
            Sets the seed for the random generators.

        Returns
        -------
        Probability distribution: pgmpy.factors.discrete.TabularCPD
            The queried probability distribution.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.inference import ApproxInference
        >>> model = load_model("bnlearn/alarm")
        >>> infer = ApproxInference(model)
        >>> infer.query(variables=["HISTORY"])  # doctest: +ELLIPSIS
        <DiscreteFactor representing phi(HISTORY:2) at 0x...>
        >>> infer.query(variables=["HISTORY", "CVP"], joint=True)  # doctest: +ELLIPSIS
        <DiscreteFactor representing phi(HISTORY:2, CVP:3) at 0x...>
        >>> infer.query(
        ...     variables=["HISTORY", "CVP"], joint=False
        ... )  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        {'HISTORY': <DiscreteFactor representing phi(HISTORY:2) at 0x...>,
         'CVP': <DiscreteFactor representing phi(CVP:3) at 0x...>}
        """
        pass

    def map_query(
        self,
        variables,
        n_samples=int(1e4),
        samples=None,
        evidence=None,
        virtual_evidence=None,
        state_names=None,
        show_progress=True,
        seed=None,
    ):
        """
        Finds the most probable state in the joint distribution of variables. Calculates the
        result by generating samples and calculating most probable states based on the probabilities.

        Parameters
        ----------
        variables: list
            List of variables for which the probability distribution needs to be calculated.

        n_samples: int
            The number of samples to generate for computing the distributions. Higher `n_samples`
            results in more accurate results at the cost of more computation time.

        samples: pd.DataFrame (default: None)
            If provided, uses these samples to compute the distribution instead
            of generating samples. `samples` **must** conform with the provided
            `evidence` and `virtual_evidence`.

        evidence: dict (default: None)
            The observed values. A dict key, value pair of the form {var: state_name}.

        virtual_evidence: list (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual/soft
            evidence.

        state_names: dict (default: None)
            A dict of state names for each variable in `variables` in the form {variable_name: list of states}.
            If None, inferred from the data but is possible that the final distribution misses some states.

        show_progress: boolean (default: True)
            If True, shows a progress bar when generating samples.

        seed: int (default: None)
            Sets the seed for the random generators.

        Returns
        -------
        MAP values: dict
            The most probable state of provided `variables` given the evidence.

        Examples
        --------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.inference import ApproxInference
        >>> from pgmpy.factors.discrete import State, TabularCPD
        >>> model = load_model("bnlearn/alarm")
        >>> infer = ApproxInference(model)
        >>> print(infer.map_query(variables=["HISTORY", "CVP"]))
        {'HISTORY': 'FALSE', 'CVP': 'NORMAL'}
        >>> virtual_evidence_history = TabularCPD(
        ...     variable="HISTORY",
        ...     variable_card=2,
        ...     values=[[0.99], [0.01]],
        ...     state_names={"HISTORY": ["TRUE", "FALSE"]},
        ... )
        >>> evidence = {"CVP": "NORMAL"}
        >>> print(
        ...     infer.map_query(
        ...         variables=["HISTORY"],
        ...         evidence=evidence,
        ...         virtual_evidence=[virtual_evidence_history],
        ...     )
        ... )
        {'HISTORY': 'TRUE'}
        """
        pass
