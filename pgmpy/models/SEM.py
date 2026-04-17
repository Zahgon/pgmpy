import itertools

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.utils.parser import parse_lavaan


class SEMGraph:
    """
    Base class for graphical representation of Structural Equation Models(SEMs).

    All variables are by default assumed to have an associated error latent variable, therefore
    doesn't need to be specified.

    Parameters
    ----------
    ebunch: list/array-like
        List of edges in form of tuples. Each tuple can be of two possible shape:
            1. (u, v): This would add an edge from u to v without setting any parameter
                       for the edge.
            2. (u, v, parameter): This would add an edge from u to v and set the edge's
                        parameter to `parameter`.

    latents: list/array-like
        List of nodes which are latent. All other variables are considered observed.

    err_corr: list/array-like
        List of tuples representing edges between error terms. It can be of the following forms:
            1. (u, v): Add correlation between error terms of `u` and `v`. Doesn't set any variance or
                       covariance values.
            2. (u, v, covar): Adds correlation between the error terms of `u` and `v` and sets the
                              parameter to `covar`.

    err_var: dict (variable: variance)
        Sets variance for the error terms in the model.

    Examples
    --------
    Defining a model (Union sentiment model[1]) without setting any paramaters:

    >>> from pgmpy.models import SEMGraph
    >>> sem = SEMGraph(
    ...     ebunch=[
    ...         ("deferenc", "unionsen"),
    ...         ("laboract", "unionsen"),
    ...         ("yrsmill", "unionsen"),
    ...         ("age", "deferenc"),
    ...         ("age", "laboract"),
    ...         ("deferenc", "laboract"),
    ...     ],
    ...     latents=[],
    ...     err_corr=[("yrsmill", "age")],
    ...     err_var={},
    ... )

    Defining a model (Education [2]) with all the parameters set. For not setting any
    parameter `np.nan` can be explicitly passed.

    >>> sem_edu = SEMGraph(
    ...     ebunch=[
    ...         ("intelligence", "academic", 0.8),
    ...         ("intelligence", "scale_1", 0.7),
    ...         ("intelligence", "scale_2", 0.64),
    ...         ("intelligence", "scale_3", 0.73),
    ...         ("intelligence", "scale_4", 0.82),
    ...         ("academic", "SAT_score", 0.98),
    ...         ("academic", "High_school_gpa", 0.75),
    ...         ("academic", "ACT_score", 0.87),
    ...     ],
    ...     latents=["intelligence", "academic"],
    ...     err_corr=[],
    ...     err_var={"intelligence": 1},
    ... )

    References
    ----------
    [1] McDonald, A, J., & Clelland, D. A. (1984). Textile Workers and Union Sentiment.
        Social Forces, 63(2), 502–521
    [2] https://en.wikipedia.org/wiki/Structural_equation_modeling#/
        media/File:Example_Structural_equation_model.svg

    Attributes
    ----------
    latents: list
        List of all the latent variables in the model except the error terms.

    observed: list
        List of all the observed variables in the model.

    graph: nx.DirectedGraph
        The graphical structure of the latent and observed variables except the error terms.
        The parameters are stored in the `weight` attribute of each edge.

    err_graph: nx.Graph
        An undirected graph representing the relations between the error terms of the model.
        The node of the graph has the same name as the variable but represents the error terms.
        The variance is stored in the `weight` attribute of the node and the covariance are stored
        in the `weight` attribute of the edge.

    full_graph_struct: nx.DiGraph
        Represents the full graph structure. The names of error terms start with `.` and
        new nodes are added for each correlation which starts with `..`.

    """

    def __init__(self, ebunch=[], latents=[], err_corr=[], err_var={}):
        super().__init__()

        # Construct the graph and set the parameters.
        self.graph = nx.DiGraph()
        for t in ebunch:
            if len(t) == 3:
                self.graph.add_edge(t[0], t[1], weight=t[2])
            elif len(t) == 2:
                self.graph.add_edge(t[0], t[1], weight=np.nan)
            else:
                raise ValueError(f"Expected tuple length: 2 or 3. Got {t} of len {len(t)}")

        self.latents = set(latents)
        self.observed = set(self.graph.nodes()) - self.latents

        # Construct the error graph and set the parameters.
        self.err_graph = nx.Graph()
        self.err_graph.add_nodes_from(self.graph.nodes())
        for t in err_corr:
            if len(t) == 2:
                self.err_graph.add_edge(t[0], t[1], weight=np.nan)
            elif len(t) == 3:
                self.err_graph.add_edge(t[0], t[1], weight=t[2])
            else:
                raise ValueError(f"Expected tuple length: 2 or 3. Got {t} of len {len(t)}")

        # Set the error variances
        for var in self.err_graph.nodes():
            self.err_graph.nodes[var]["weight"] = err_var[var] if var in err_var.keys() else np.nan

        self.full_graph_struct = self._get_full_graph_struct()

    def _variable_name_contains_non_string(self):
        """
        Checks if the variable names contain any non-string values. Used only for CausalInference class.
        """
        pass

    def _get_full_graph_struct(self):
        """
        Creates a directed graph by joining `self.graph` and `self.err_graph`.
        Adds new nodes to replace undirected edges (u <--> v) with two directed
        edges (u <-- ..uv) and (..uv --> v).

        Returns
        -------
        nx.DiGraph: A full directed graph strucuture with error nodes starting
                    with `.` and bidirected edges replaced with common cause
                    nodes starting with `..`.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> sem = SEMGraph(
        ...     ebunch=[
        ...         ("deferenc", "unionsen"),
        ...         ("laboract", "unionsen"),
        ...         ("yrsmill", "unionsen"),
        ...         ("age", "deferenc"),
        ...         ("age", "laboract"),
        ...         ("deferenc", "laboract"),
        ...     ],
        ...     latents=[],
        ...     err_corr=[("yrsmill", "age")],
        ... )
        >>> sem._get_full_graph_struct()
        """
        pass

    def get_scaling_indicators(self):
        """
        Returns a scaling indicator for each of the latent variables in the model.
        The scaling indicator is chosen randomly among the observed measurement
        variables of the latent variable.

        Examples
        --------
        >>> from pgmpy.models import SEMGraph
        >>> model = SEMGraph(
        ...     ebunch=[
        ...         ("xi1", "eta1"),
        ...         ("xi1", "x1"),
        ...         ("xi1", "x2"),
        ...         ("eta1", "y1"),
        ...         ("eta1", "y2"),
        ...     ],
        ...     latents=["xi1", "eta1"],
        ... )
        >>> model.get_scaling_indicators()
        {'xi1': 'x1', 'eta1': 'y1'}

        Returns
        -------
        dict: Returns a dict with latent variables as the key and their value being the
                scaling indicator.
        """
        pass

    def active_trail_nodes(self, variables, observed=[], avoid_nodes=[], struct="full"):
        """
        Finds all the observed variables which are d-connected to `variables` in the `graph_struct`
        when `observed` variables are observed.

        Parameters
        ----------
        variables: str or array like
            Observed variables whose d-connected variables are to be found.

        observed : list/array-like
            If given the active trails would be computed assuming these nodes to be observed.

        avoid_nodes: list/array-like
            If specificed, the algorithm doesn't account for paths that have influence flowing
            through the avoid node.

        struct: str or nx.DiGraph instance
            If "full", considers correlation between error terms for computing d-connection.
            If "non_error", doesn't condised error correlations for computing d-connection.
            If instance of nx.DiGraph, finds d-connected variables on the given graph.

        Examples
        --------
        >>> from pgmpy.models import SEM
        >>> model = SEMGraph(
        ...     ebunch=[
        ...         ("yrsmill", "unionsen"),
        ...         ("age", "laboract"),
        ...         ("age", "deferenc"),
        ...         ("deferenc", "laboract"),
        ...         ("deferenc", "unionsen"),
        ...         ("laboract", "unionsen"),
        ...     ],
        ...     latents=[],
        ...     err_corr=[("yrsmill", "age")],
        ... )
        >>> model.active_trail_nodes("age")

        Returns
        -------
        dict: {str: list}
            Returns a dict with `variables` as the key and a list of d-connected variables as the
            value.

        References
        ----------
        Details of the algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        """
        pass

    def moralize(self, graph="full"):
        """
        TODO: This needs to go to a parent class.
        Removes all the immoralities in the DirectedGraph and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Parameters
        ----------
        graph:

        Examples
        --------
        """
        pass

    def _nearest_separator(self, G, Y, Z):
        """
        Finds the set of the nearest separators for `Y` and `Z` in `G`.

        Parameters
        ----------
        G: nx.DiGraph instance
            The graph in which to the find the nearest separation for `Y` and `Z`.

        Y: str
            The variable name for which the separators are needed.

        Z: str
            The other variable for which the separators are needed.

        Returns
        -------
        set or None: If there is a nearest separator returns the set of separators else returns None.
        """
        pass

    def to_lisrel(self):
        r"""
        Converts the model from a graphical representation to an equivalent algebraic
        representation. This converts the model into a Reticular Action Model (RAM) model
        representation which is implemented by `pgmpy.models.SEMAlg` class.

        Returns
        -------
        SEMAlg instance: Instance of `SEMAlg` representing the model.

        Examples
        --------
        >>> from pgmpy.models import SEM
        >>> sem = SEM.from_graph(
        ...     ebunch=[
        ...         ("deferenc", "unionsen"),
        ...         ("laboract", "unionsen"),
        ...         ("yrsmill", "unionsen"),
        ...         ("age", "deferenc"),
        ...         ("age", "laboract"),
        ...         ("deferenc", "laboract"),
        ...     ],
        ...     latents=[],
        ...     err_corr=[("yrsmill", "age")],
        ...     err_var={},
        ... )
        >>> sem.to_lisrel()
        # TODO: Complete this.

        See Also
        --------
        to_standard_lisrel: Converts to the standard lisrel format and returns the parameters.
        """
        pass

    @staticmethod
    def __standard_lisrel_masks(graph, err_graph, weight, var):
        r"""
        This method is called by `get_fixed_masks` and `get_masks` methods.

        Parameters
        ----------
        weight: None | 'weight'
            If None: Returns a 1.0 for an edge in the graph else 0.0
            If 'weight': Returns the weight if a weight is assigned to an edge
                    else 0.0

        var: dict
            Dict with keys eta, xi, y, and x representing the variables in them.

        Returns
        -------
        np.ndarray: Adjacency matrix of model's graph structure.

        Notes
        -----
        B: Effect matrix of eta on eta
        \gamma: Effect matrix of xi on eta
        \wedge_y: Effect matrix of eta on y
        \wedge_x: Effect matrix of xi on x
        \phi: Covariance matrix of xi
        \psi: Covariance matrix of eta errors
        \theta_e: Covariance matrix of y errors
        \theta_del: Covariance matrix of x errors

        Examples
        --------
        """
        pass

    def to_standard_lisrel(self):
        r"""
        Transforms the model to the standard LISREL representation of latent and measurement
        equations. The standard LISREL representation is given as:

        ..math::
            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\Gamma \xi} + \mathbf{\zeta} \\
            \mathbf{y} = \mathbf{\wedge_y \eta} + \mathbf{\epsilon} \\
            \mathbf{x} = \mathbf{\wedge_x \xi} + \mathbf{\delta} \\
            \mathbf{\Theta_e} = COV(\mathbf{\epsilon}) \\
            \mathbf{\Theta_\delta} = COV(\mathbf{\delta}) \\
            \mathbf{\Psi} = COV(\mathbf{\eta}) \\
            \mathbf{\Phi} = COV(\mathbf{\xi}) \\

        Since the standard LISREL representation has restrictions on the types of model,
        this method adds extra latent variables with fixed loadings of `1` to make the model
        consistent with the restrictions.

        Returns
        -------
        var_names: dict (keys: eta, xi, y, x)
            Returns the variable names in :math:`\mathbf{\eta}`, :math:`\mathbf{\xi}`,
            :math:`\mathbf{y}`, :math:`\mathbf{x}`.

        params: dict (keys: B, gamma, wedge_y, wedge_x, theta_e, theta_del, phi, psi)
            Returns a boolean matrix for each of the parameters. A 1 in the matrix
            represents that there is an edge in the model, 0 represents there is no edge.

        fixed_values: dict (keys: B, gamma, wedge_y, wedge_x, theta_e, theta_del, phi, psi)
            Returns a matrix for each of the parameters. A value in the matrix represents the
            set value for the parameter in the model else it is 0.

        See Also
        --------
        to_lisrel: Converts the model to `pgmpy.models.SEMAlg` instance.

        Examples
        --------
        TODO: Finish this.
        """
        pass


class SEMAlg:
    """
    Base class for algebraic representation of Structural Equation Models(SEMs). The model is
    represented using the Reticular Action Model (RAM).
    """

    def __init__(self, eta=None, B=None, zeta=None, wedge_y=None, fixed_values=None):
        r"""
        Initializes SEMAlg model. The model is represented using the Reticular Action Model(RAM)
        which is given as:
        ..math::
            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\zeta}
            \mathbf{y} = \mathbf{\wedge_y \eta}

        where :math:`\mathbf{\eta}` is the set of all the observed and latent variables in the
        model, :math:`\mathbf{y}` are the set of observed variables, :math:`\mathbf{\zeta}` is
        the error terms for :math:`\mathbf{\eta}`, and \mathbf{\wedge_y} is a boolean array to
        select the observed variables from :math:`\mathbf{\eta}`.

        Parameters
        ----------
        The following set of parameters are used to set the learnable parameters in the model.
        To specify the values of the parameter use the `fixed_values` parameter. Either `eta`,
        `B`, `zeta`, and `wedge_y`, or `fixed_values` need to be specified.

        eta: list/array-like
            The name of the variables in the model.

        B: 2-D array (boolean)
            The learnable parameters in the `B` matrix.

        zeta: 2-D array (boolean)
            The learnable parameters in the covariance matrix of the error terms.

        wedge_y: 2-D array
            The `wedge_y` matrix.

        fixed_params: dict (default: None)
            A dict of fixed values for parameters.

            If None all the parameters specified by `B`, and `zeta` are learnable.

        Returns
        -------
        pgmpy.models.SEMAlg instance: An instance of the object with initalized values.

        Examples
        --------
        >>> from pgmpy.models import SEMAlg
        # TODO: Finish this example
        """
        self.eta = eta
        self.B = np.array(B)
        self.zeta = np.array(zeta)
        self.wedge_y = wedge_y

        # Get the observed variables
        self.y = []
        for row_i in range(self.wedge_y.shape[0]):
            for index, val in enumerate(self.wedge_y[row_i]):
                if val:
                    self.y.append(self.eta[index])

        if fixed_values:
            self.B_fixed_mask = fixed_values["B"]
            self.zeta_fixed_mask = fixed_values["zeta"]
        else:
            self.B_fixed_mask = np.zeros(self.B.shape)
            self.zeta_fixed_mask = np.zeros(self.zeta.shape)

        # Masks represent the parameters which need to be learnt while training.
        self.B_mask = np.multiply(np.where(self.B_fixed_mask != 0, 0.0, 1.0), self.B)
        self.zeta_mask = np.multiply(np.where(self.zeta_fixed_mask != 0, 0.0, 1.0), self.zeta)

    def to_SEMGraph(self):
        """
        Creates a graph structure from the LISREL representation.

        Returns
        -------
        pgmpy.models.SEMGraph instance: A path model of the model.

        Examples
        --------
        >>> from pgmpy.models import SEMAlg
        >>> model = SEMAlg()
        # TODO: Finish this example
        """
        pass

    def set_params(self, B, zeta):
        """
        Sets the fixed parameters of the model.

        Parameters
        ----------
        B: 2D array
            The B matrix.

        zeta: 2D array
            The covariance matrix.
        """
        pass

    def generate_samples(self, n_samples=100):
        """
        Generates random samples from the model.

        Parameters
        ----------
        n_samples: int
            The number of samples to generate.

        Returns
        -------
        pd.DataFrame: The generated samples.
        """
        pass


class SEM(SEMGraph):
    """
    Class for representing Structural Equation Models. This class is a wrapper over
    `SEMGraph` and `SEMAlg` to provide a consistent API over the different representations.

    Attributes
    ----------
    model: SEMGraph instance
        A graphical representation of the model.
    """

    def __init__(self, syntax, **kwargs):
        """
        Initialize a `SEM` object. Preferred way to initialize the object is to use one of
        the `from_lavaan`, `from_graph`, `from_lisrel`, or `from_RAM` methods.

        There are three possible ways to initialize the model:
            1. Lavaan syntax: `lavaan_str` needs to be specified.
            2. Graph structure: `ebunch`, `latents`, `err_corr`, and `err_var` need to be specified.
            3. LISREL syntax: `var_names`, `params`, and `fixed_masks` need to be specified.
            4. Reticular Action Model (RAM/all-y) syntax: `var_names`, `B`, `zeta`, and `wedge_y`
                                                            need to be specified.

        Parameters
        ----------
        syntax: str (lavaan|graph|lisrel|ram)
            The syntax used to initialize the model.

        kwargs:
            For parameter details, check docstrings for `from_lavaan`, `from_graph`, `from_lisrel`,
            and `from_RAM` methods.

        See Also
        --------
        from_lavaan: Initialize a model using lavaan syntax.
        from_graph: Initialize a model using graph structure.
        from_lisrel: Initialize a model using LISREL syntax.
        from_RAM: Initialize a model using Reticular Action Model(RAM/all-y) syntax.
        """
        if syntax.lower() == "lavaan":
            # Create a SEMGraph model using the lavaan str.
            ebunch, latents, err_corr, err_var = parse_lavaan(kwargs["lavaan_str"])

            # Call the parent __init__ with the arguments
            super().__init__(ebunch=ebunch, latents=latents, err_corr=err_corr)

        elif syntax.lower() == "graph":
            super().__init__(
                ebunch=kwargs["ebunch"],
                latents=kwargs["latents"],
                err_corr=kwargs["err_corr"],
                err_var=kwargs["err_var"],
            )

        elif syntax.lower() == "lisrel":
            model = SEMAlg(
                var_names=kwargs["var_names"],
                params=kwargs["params"],
                fixed_masks=kwargs.get("fixed_masks"),
            ).to_SEMGraph()
            # Initialize an empty SEMGraph instance and set the properties.
            # TODO: Boilerplate code, find a better way to do this.
            super().__init__(ebunch=[], latents=[], err_corr=[], err_var={})
            self.graph = model.graph
            self.latents = model.latents
            self.obseved = model.observed
            self.err_graph = model.err_graph
            self.full_graph_struct = model.full_graph_struct

        elif syntax.lower() == "ram":
            model = SEMAlg(
                eta=kwargs["var_names"],
                B=kwargs["B"],
                zeta=kwargs["zeta"],
                wedge_y=kwargs["wedge_y"],
                fixed_values=kwargs.get("fixed_masks"),
            )

    @classmethod
    def from_lavaan(cls, string=None, filename=None):
        """
        Initializes a `SEM` instance using lavaan syntax.

        Parameters
        ----------
        string: str (default: None)
            A `lavaan` style multiline set of regression equation representing the model.
            Refer http://lavaan.ugent.be/tutorial/syntax1.html for details.

        filename: str (default: None)
            The filename of the file containing the model in lavaan syntax.

        Examples
        --------
        """
        pass

    @classmethod
    def from_graph(cls, ebunch, latents=[], err_corr=[], err_var={}):
        """
        Initializes a `SEM` instance using graphical structure.

        Parameters
        ----------
        ebunch: list/array-like
            List of edges in form of tuples. Each tuple can be of two possible shape:
                1. (u, v): This would add an edge from u to v without setting any parameter
                           for the edge.
                2. (u, v, parameter): This would add an edge from u to v and set the edge's
                            parameter to `parameter`.

        latents: list/array-like
            List of nodes which are latent. All other variables are considered observed.

        err_corr: list/array-like
            List of tuples representing edges between error terms. It can be of the following forms:
                1. (u, v): Add correlation between error terms of `u` and `v`. Doesn't set any variance or
                           covariance values.
                2. (u, v, covar): Adds correlation between the error terms of `u` and `v` and sets the
                                  parameter to `covar`.

        err_var: dict
            Dict of the form (var: variance).

        Examples
        --------
        Defining a model (Union sentiment model[1]) without setting any paramaters.

        >>> from pgmpy.models import SEM
        >>> sem = SEM.from_graph(
        ...     ebunch=[
        ...         ("deferenc", "unionsen"),
        ...         ("laboract", "unionsen"),
        ...         ("yrsmill", "unionsen"),
        ...         ("age", "deferenc"),
        ...         ("age", "laboract"),
        ...         ("deferenc", "laboract"),
        ...     ],
        ...     latents=[],
        ...     err_corr=[("yrsmill", "age")],
        ...     err_var={},
        ... )

        Defining a model (Education [2]) with all the parameters set. For not setting any
        parameter `np.nan` can be explicitly passed.

        >>> sem_edu = SEM.from_graph(
        ...     ebunch=[
        ...         ("intelligence", "academic", 0.8),
        ...         ("intelligence", "scale_1", 0.7),
        ...         ("intelligence", "scale_2", 0.64),
        ...         ("intelligence", "scale_3", 0.73),
        ...         ("intelligence", "scale_4", 0.82),
        ...         ("academic", "SAT_score", 0.98),
        ...         ("academic", "High_school_gpa", 0.75),
        ...         ("academic", "ACT_score", 0.87),
        ...     ],
        ...     latents=["intelligence", "academic"],
        ...     err_corr=[],
        ...     err_var={},
        ... )

        References
        ----------
        [1] McDonald, A, J., & Clelland, D. A. (1984). Textile Workers and Union Sentiment.
            Social Forces, 63(2), 502–521
        [2] https://en.wikipedia.org/wiki/Structural_equation_modeling#/
            media/File:Example_Structural_equation_model.svg
        """
        pass

    @classmethod
    def from_lisrel(cls, var_names, params, fixed_masks=None):
        r"""
        Initializes a `SEM` instance using LISREL notation. The LISREL notation is defined as:
        ..math::

            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\Gamma \xi} + mathbf{\zeta} \\
            \mathbf{y} = \mathbf{\wedge_y \eta} + \mathbf{\epsilon} \\
            \mathbf{x} = \mathbf{\wedge_x \xi} + \mathbf{\delta}

        where :math:`\mathbf{\eta}` is the set of endogenous variables, :math:`\mathbf{\xi}`
        is the set of exogeneous variables, :math:`\mathbf{y}` and :math:`\mathbf{x}` are the
        set of measurement variables for :math:`\mathbf{\eta}` and :math:`\mathbf{\xi}`
        respectively. :math:`\mathbf{\zeta}`, :math:`\mathbf{\epsilon}`, and :math:`\mathbf{\delta}`
        are the error terms for :math:`\mathbf{\eta}`, :math:`\mathbf{y}`, and :math:`\mathbf{x}`
        respectively.

        Parameters
        ----------
        str_model: str (default: None)
            A `lavaan` style multiline set of regression equation representing the model.
            Refer http://lavaan.ugent.be/tutorial/syntax1.html for details.

            If None requires `var_names` and `params` to be specified.

        var_names: dict (default: None)
            A dict with the keys: eta, xi, y, and x. Each keys should have a list as the value
            with the name of variables.

        params: dict (default: None)
            A dict of LISREL representation non-zero parameters. Must contain the following
            keys: B, gamma, wedge_y, wedge_x, phi, theta_e, theta_del, and psi.

            If None `str_model` must be specified.

        fixed_params: dict (default: None)
            A dict of fixed values for parameters. The shape of the parameters should be same
            as params.

            If None all the parameters are learnable.

        Returns
        -------
        pgmpy.models.SEM instance: An instance of the object with initalized values.

        Examples
        --------
        >>> from pgmpy.models import SEMAlg
        # TODO: Finish this example
        """
        pass

    @classmethod
    def from_RAM(cls, variables, B, zeta, observed=None, wedge_y=None, fixed_values=None):
        r"""
        Initializes a `SEM` instance using Reticular Action Model(RAM) notation. The model
        is defined as:

        ..math::

            \mathbf{\eta} = \mathbf{B \eta} + \mathbf{\epsilon} \\
            \mathbf{\y} = \wedge_y \mathbf{\eta}
            \zeta = COV(\mathbf{\epsilon})

        where :math:`\mathbf{\eta}` is the set of variables (both latent and observed),
        :math:`\mathbf{\epsilon}` are the error terms, :math:`\mathbf{y}` is the set
        of observed variables, :math:`\wedge_y` is a boolean array of the shape (no of
        observed variables, no of total variables).

        Parameters
        ----------
        variables: list, array-like
            List of variables (both latent and observed) in the model.

        B: 2-D boolean array (shape: `len(variables)` x `len(variables)`)
            The non-zero parameters in :math:`B` matrix. Refer model definition in docstring for details.

        zeta: 2-D boolean array (shape: `len(variables)` x `len(variables)`)
            The non-zero parameters in :math:`\zeta` (error covariance) matrix. Refer model definition
            in docstring for details.

        observed: list, array-like (optional: Either `observed` or `wedge_y` needs to be specified)
            List of observed variables in the model.

        wedge_y: 2-D array (shape: no. observed x total vars) (optional: Either `observed` or `wedge_y`)
            The :math:`\wedge_y` matrix. Refer model definition in docstring for details.

        fixed_values: dict (optional)
            If specified, fixes the parameter values and are not changed during estimation.
            A dict with the keys B, zeta.

        Returns
        -------
        pgmpy.models.SEM instance: An instance of the object with initialized values.

        Examples
        --------
        >>> from pgmpy.models import SEM
        >>> SEM.from_RAM  # TODO: Finish this
        """
        pass

    def fit(self):
        pass
