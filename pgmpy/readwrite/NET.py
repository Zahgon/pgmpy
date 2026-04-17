import collections
import warnings
from math import prod
from string import Template

import numpy as np

from pgmpy import logger

try:
    from pyparsing import (
        CharsNotIn,
        Group,
        OneOrMore,
        Optional,
        Suppress,
        Word,
        ZeroOrMore,
        alphanums,
        alphas,
        cppStyleComment,
        nums,
        printables,
    )
except ImportError as e:
    raise ImportError(
        f"{e}. pyparsing is required for using read/write methods. Please install using: pip install pyparsing."
    ) from None

from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns


class NETWriter:
    """
    Base class for writing network file in net format

    Parameters
    ----------
    model: DiscreteBayesianNetwork Instance

    Examples
    ----------
    >>> from pgmpy.readwrite import NETWriter
    >>> from pgmpy.example_models import load_model
    >>> asia = load_model("bnlearn/asia")
    >>> writer = NETWriter(asia)
    >>> writer # doctest: +ELLIPSIS
    <pgmpy.readwrite.NET.NETWriter object at 0x...>
    >>> writer.write("asia.net")

    Reference
    ---------
    [1] HUGIN EXPERT A/S . The HUGIN file format. http://www.hugin.com, 2011.
    """

    def __init__(self, model):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("model must be an instance of DiscreteBayesianNetwork")

        self.model = model

        if not self.model.name:
            self.network_name = "unknown"
        else:
            self.network_name = self.model.name

        self.variables = self.get_variables()
        self.variable_states = self.get_states()
        self.property_tag = self.get_properties()
        self.variable_parents = self.get_parents()
        self.tables = self.get_cpds()

    def NET_templates(self):
        """
        Create template for writing in NET format
        """
        pass

    def __str__(self):
        """Return the NET"""
        (
            network_template,
            node_template,
            potential_template,
            property_template,
        ) = self.NET_templates()

        network = ""
        network += network_template.substitute()
        variables = self.variables

        for var in sorted(variables):
            quoted_states = ['"' + state + '"' for state in self.variable_states[var]]
            states = "  ".join(quoted_states)

            if not self.property_tag[var]:
                properties = ""
            else:
                properties = ""
                for prop_val in self.property_tag[var]:
                    properties += property_template.substitute(prop=prop_val)

            network += node_template.substitute(name=var, states=states, properties=properties)

        for var in sorted(variables):
            if not self.variable_parents[var]:
                parents = ""
                separator = " |"
            else:
                parents = " ".join(self.variable_parents[var])
                separator = " | "
            potentials = self.net_cpd(var)
            network += potential_template.substitute(
                variable_=var,
                separator_=separator,
                parents=parents,
                values=potentials,
            )

        return network

    def net_cpd(self, var_name):
        """
        Util function for turning pgmpy CPT values into CPT format of .net files
        Inputs
        -------
        var_name: string, name of the variable

        Returns
        -------
        string: CPT format of .net files
        """
        pass

    def get_variables(self):
        """
        Add variables to NET

        Returns
        -------
        list: a list containing names of variable

        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.get_variables()
        ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
        """
        pass

    def get_cpds(self):
        """
        Adds tables to NET

        Returns
        -------
        dict: dict of type {variable: array}

        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.get_cpds() # doctest: +NORMALIZE_WHITESPACE
        {'asia': array([0.01, 0.99]), 'bronc': array([[0.6, 0.3],
           [0.4, 0.7]]), 'dysp': array([[[0.9, 0.8],
            [0.7, 0.1]],
           [[0.1, 0.2],
            [0.3, 0.9]]]), 'either': array([[[1., 1.],
            [1., 0.]],
           [[0., 0.],
            [0., 1.]]]), 'lung': array([[0.1 , 0.01],
           [0.9 , 0.99]]), 'smoke': array([0.5, 0.5]), 'tub': array([[0.05, 0.01],
           [0.95, 0.99]]), 'xray': array([[0.98, 0.05],
           [0.02, 0.95]])}
        """
        pass

    def get_properties(self):
        """
        Add property to variables in NET

        Returns
        -------
        dict: dict of type {variable: list of properties }

        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.get_properties()
        {'asia': [], 'bronc': [], 'dysp': [], 'either': [], 'lung': [], 'smoke': [], 'tub': [], 'xray': []}
        """
        pass

    def get_states(self):
        """
        Add states to variable of NET

        Returns
        -------
        dict: dict of type {variable: a list of states}


        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.get_states() # doctest: +NORMALIZE_WHITESPACE
        {'asia': ['yes', 'no'], 'bronc': ['yes', 'no'], 'dysp': ['yes', 'no'],
        'either': ['yes', 'no'], 'lung': ['yes', 'no'], 'smoke': ['yes', 'no'],
        'tub': ['yes', 'no'], 'xray': ['yes', 'no']}
        """
        pass

    def get_parents(self):
        """
        Add the parents to NET

        Returns
        -------
        dict: dict of type {variable: a list of parents}

        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.get_parents() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [], 'bronc': ['smoke'], 'dysp': ['bronc', 'either'],
        'either': ['lung', 'tub'], 'lung': ['smoke'], 'smoke': [],
        'tub': ['asia'], 'xray': ['either']}
        """
        pass

    def write(self, filename):
        """
        Writes the NET data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import NETWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write(filename="asia.net")
        """
        pass

    def write_net(self, filename):
        pass


class NETReader:
    """
    Initializes a NETReader object.

    Parameters
    ----------
    path : file or str
        File of net data

    string : str
        String of net data

    include_properties: boolean
        If True, gets the properties tag from the file and stores in graph properties.

    defaultname: int (default: "bn_model")
        Default name for the network if a network name is not available in the net file.

    Examples
    --------
    # asia.net file is present at
    # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
    >>> from pgmpy.readwrite import NETReader
    >>> from pgmpy.example_models import load_model
    >>> asia = load_model("bnlearn/asia")
    >>> writer = NETWriter(asia)
    >>> writer.write("asia.net")
    >>> reader = NETReader("asia.net")
    >>> reader # doctest: +ELLIPSIS
    <pgmpy.readwrite.NET.NETReader object at 0x...>
    >>> model = reader.get_model()
    """

    def __init__(self, path=None, string=None, include_properties=False, defaultName="bn_model"):
        if path:
            with open(path) as network:
                self.network = network.read()

        elif string:
            self.network = string

        else:
            raise ValueError("Must specify either path or string")

        self.include_properties = include_properties

        if "/*" in self.network or "//" in self.network:
            self.network = cppStyleComment.suppress().transform_string(self.network)  # removing comments from the file

        (
            self.name_expr,
            self.state_expr,
            self.property_expr,
        ) = self.get_variable_grammar()

        self.potential_expr, self.cpd_expr = self.get_probability_grammar()

        if not self.get_network_name():
            self.network_name = defaultName
        else:
            self.network_name = self.get_network_name()

        self.variable_names = self.get_variables()
        self.variable_states = self.get_states()
        if self.include_properties:
            self.variable_properties = self.get_property()
        self.variable_parents = self.get_parents()
        self.variable_cpds = self.get_values()
        self.edges = self.get_edges()

    def get_variable_grammar(self):
        """
        A method that returns variable grammar
        """
        pass

    def get_probability_grammar(self):
        """
        A method that returns probability grammar
        """
        pass

    def get_network_name(self):
        """
        Returns the name of the network. Returns false if no network name is available

        Example
        ---------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> reader.get_network_name()
        False
        """
        pass

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        ---------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> sorted(reader.get_variables())
        ['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray']
        """
        pass

    def get_states(self):
        """
        Returns the states of each variable in the network

        Example
        ---------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> reader.get_states() # doctest: +NORMALIZE_WHITESPACE
        {'asia': ['yes', 'no'], 'bronc': ['yes', 'no'], 'dysp': ['yes', 'no'],
        'either': ['yes', 'no'], 'lung': ['yes', 'no'], 'smoke': ['yes', 'no'],
        'tub': ['yes', 'no'], 'xray': ['yes', 'no']}
        """
        pass

    def get_property(self):
        """
        Returns the property of the variable

        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> sorted(reader.get_property()) # doctest: +NORMALIZE_WHITESPACE
        ['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray']
        """
        pass

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> reader.get_parents() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [], 'bronc': ['smoke'], 'dysp': ['bronc', 'either'],
        'either': ['lung', 'tub'], 'lung': ['smoke'], 'smoke': [],
        'tub': ['asia'], 'xray': ['either']}
        """
        pass

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> reader.get_values() # doctest: +NORMALIZE_WHITESPACE
        {'asia': array([[0.01],
           [0.99]]), 'bronc': array([[0.6, 0.3],
           [0.4, 0.7]]), 'dysp': array([[0.9, 0.8, 0.7, 0.1],
           [0.1, 0.2, 0.3, 0.9]]), 'either': array([[1., 1., 1., 0.],
           [0., 0., 0., 1.]]), 'lung': array([[0.1 , 0.01],
           [0.9 , 0.99]]), 'smoke': array([[0.5],
           [0.5]]), 'tub': array([[0.05, 0.01],
           [0.95, 0.99]]), 'xray': array([[0.98, 0.05],
           [0.02, 0.95]])}
        """
        pass

    def get_edges(self):
        """
        Returns the edges of the network



        Example
        -------------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> sorted(reader.get_edges()) # doctest: +NORMALIZE_WHITESPACE
        [['asia', 'tub'], ['bronc', 'dysp'], ['either', 'dysp'], ['either', 'xray'],
        ['lung', 'either'], ['smoke', 'bronc'], ['smoke', 'lung'], ['tub', 'either']]

        """
        pass

    def get_model(self, state_name_type=str):
        """
        Returns the Bayesian Model read from the file/str.

        Parameters
        ----------
        state_name_type: int, str or bool (default: str)
            The data type to which to convert the state names of the variables.

        Example
        ----------
        # asia.net file is present at
        # https://www.bnlearn.com/bnrepository/discrete-small.html#asia
        >>> from pgmpy.readwrite import NETReader
        >>> from pgmpy.example_models import load_model
        >>> asia = load_model("bnlearn/asia")
        >>> writer = NETWriter(asia)
        >>> writer.write("asia.net")
        >>> reader = NETReader("asia.net")
        >>> reader.get_model() # doctest: +ELLIPSIS
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x...>
        """
        pass
