import warnings
from itertools import combinations

import numpy as np

try:
    from pyparsing import Combine, Literal, Optional, Regex, Word, alphas, nums
except ImportError as e:
    raise ImportError(
        f"{e}. pyparsing is required for using read/write methods. Please install using: pip install pyparsing."
    ) from None

from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork
from pgmpy.utils import compat_fns


class UAIReader:
    """
    Initialize an instance of UAI reader class

    Parameters
    ----------
    path : file or str
        Path of the file containing UAI information.

    string : str
        String containing UAI information.

    Examples
    --------
    >>> from pgmpy.readwrite import UAIReader, UAIWriter
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/asia")
    >>> writer = UAIWriter(model)
    >>> writer.write("asia.uai")
    >>> reader = UAIReader("asia.uai")
    >>> model = reader.get_model()

    Reference
    ---------
    [1] https://uaicompetition.github.io/uci-2022/file-formats/model-format/
    [2] https://forgemia.inra.fr/thomas.schiex/toulbar2/-/blob/master/doc/UAI08Format.txt
    """

    def __init__(self, path=None, string=None):
        if path:
            with open(path) as f:
                self.network = f.read()
        elif string:
            self.network = string
        else:
            raise ValueError("Must specify either path or string.")

        if "#" in self.network:
            self.network = Regex("#.*").suppress().transform_string(self.network)  # removing comments from the file

        self.grammar = self.get_grammar()
        self.network_type = self.get_network_type()
        self.variables = self.get_variables()
        self.domain = self.get_domain()
        self.edges = self.get_edges()
        self.tables = self.get_tables()

    def get_grammar(self):
        """
        Returns the grammar of the UAI file.
        """
        pass

    def get_network_type(self):
        """
        Returns the type of network defined by the file.

        Returns
        -------
        string : str
            String containing network type.

        Examples
        --------
        >>> from pgmpy.readwrite import UAIReader, UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.write("asia.uai")
        >>> reader = UAIReader("asia.uai")
        >>> reader.get_network_type()
        'BAYES'
        """
        pass

    def get_variables(self):
        """
        Returns a list of variables.
        Each variable is represented by an index of list.
        For example if the no of variables are 4 then the list will be
        [var_0, var_1, var_2, var_3]

        Returns
        -------
        list: list of variables

        Examples
        --------
        >>> from pgmpy.readwrite import UAIReader, UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.write("asia.uai")
        >>> reader = UAIReader("asia.uai")
        >>> reader.get_variables()
        ['var_0', 'var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7']
        """
        pass

    def get_domain(self):
        """
        Returns the dictionary of variables with keys as variable name
        and values as domain of the variables.

        Returns
        -------
        dict: dictionary containing variables and their domains

        Examples
        --------
        >>> from pgmpy.readwrite import UAIReader, UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.write("asia.uai")
        >>> reader = UAIReader("asia.uai")
        >>> reader.get_domain() # doctest: +NORMALIZE_WHITESPACE
        {'var_0': '2', 'var_1': '2', 'var_2': '2', 'var_3': '2',
        'var_4': '2', 'var_5': '2', 'var_6': '2', 'var_7': '2'}
        """
        pass

    def get_edges(self):
        """
        Returns the edges of the network.

        Returns
        -------
        set: set containing the edges of the network

        Examples
        --------
        >>> from pgmpy.readwrite import UAIReader, UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.write("asia.uai")
        >>> reader = UAIReader("asia.uai")
        >>> sorted(reader.get_edges()) # doctest: +NORMALIZE_WHITESPACE
        [('var_0', 'var_6'), ('var_1', 'var_2'), ('var_3', 'var_2'), ('var_3', 'var_7'),
        ('var_4', 'var_3'), ('var_5', 'var_1'), ('var_5', 'var_4'), ('var_6', 'var_3')]
        """
        pass

    def get_tables(self):
        """
        Returns list of tuple of child variable and CPD in case of Bayesian
        and list of tuple of scope of variables and values in case of Markov.

        Returns
        -------
        list : list of tuples of child variable and values in Bayesian
            list of tuples of scope of variables and values in case of Markov.

        Examples
        --------
        >>> from pgmpy.readwrite import UAIReader, UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.write("asia.uai")
        >>> reader = UAIReader("asia.uai")
        >>> reader.get_tables() # doctest: +NORMALIZE_WHITESPACE
        [('var_0', ['0.01', '0.99']), ('var_1', ['0.6', '0.3', '0.4', '0.7']),
        ('var_2', ['0.9', '0.8', '0.7', '0.1', '0.1', '0.2', '0.3', '0.9']),
        ('var_3', ['1.0', '1.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0']),
        ('var_4', ['0.1', '0.01', '0.9', '0.99']), ('var_5', ['0.5', '0.5']),
        ('var_6', ['0.05', '0.01', '0.95', '0.99']),
        ('var_7', ['0.98', '0.05', '0.02', '0.95'])]
        """
        pass

    def get_model(self):
        """
        Returns an instance of Bayesian Model or Markov Model.
        Variables are in the pattern var_0, var_1, var_2 where var_0 is
        0th index variable, var_1 is 1st index variable.

        Return
        ------
        model: an instance of Bayesian or Markov Model.

        Examples
        --------
        >>> from pgmpy.readwrite import UAIReader, UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.write("asia.uai")
        >>> reader = UAIReader("asia.uai")
        >>> reader.get_model() # doctest: +ELLIPSIS
        <pgmpy.models.DiscreteBayesianNetwork.DiscreteBayesianNetwork object at 0x...>
        """
        pass


class UAIWriter:
    """
    Initialize an instance of UAI writer class

    Parameters
    ----------
    model: A Bayesian or Markov model
        The model to write

    round_values: int (default: None)
        The number to decimals to which to round the probability values. If None, keeps all decimals points.

    Examples
    --------
    >>> from pgmpy.readwrite import UAIWriter
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/asia")
    >>> writer = UAIWriter(model)
    >>> writer.write("asia.uai")
    """

    def __init__(self, model, round_values=None):
        if isinstance(model, DiscreteBayesianNetwork):
            self.network = "BAYES\n"
        elif isinstance(model, DiscreteMarkovNetwork):
            self.network = "MARKOV\n"
        else:
            raise TypeError("Model must be an instance of Bayesian or Markov model.")

        self.model = model
        self.round_values = round_values
        self.no_nodes = self.get_nodes()
        self.domain = self.get_domain()
        self.functions = self.get_functions()
        self.tables = self.get_tables()

    def __str__(self):
        """
        Returns the UAI file as a string.
        """
        self.network += self.no_nodes + "\n"
        domain = sorted(self.domain.items(), key=lambda x: (x[1], x[0]))
        self.network += " ".join([var[1] for var in domain]) + "\n"
        self.network += str(len(self.functions)) + "\n"
        for fun in self.functions:
            self.network += str(len(fun)) + " "
            self.network += " ".join(fun) + "\n"
        self.network += "\n"
        for table in self.tables:
            self.network += str(len(table)) + "\n"
            self.network += " ".join(table) + "\n"
        return self.network[:-1]

    def get_nodes(self):
        """
        Adds variables to the network.

        Examples
        --------
        >>> from pgmpy.readwrite import UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.get_nodes()
        '8'
        """
        pass

    def get_domain(self):
        """
        Adds domain of each variable to the network.

        Examples
        --------
        >>> from pgmpy.readwrite import UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.get_domain()
        {'asia': '2', 'bronc': '2', 'dysp': '2', 'either': '2', 'lung': '2', 'smoke': '2', 'tub': '2', 'xray': '2'}
        """
        pass

    def get_functions(self):
        """
        Adds functions to the network.

        Examples
        -------_
        >>> from pgmpy.readwrite import UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.get_functions() # doctest: +NORMALIZE_WHITESPACE
        [['0'], ['5', '1'], ['3', '1', '2'], ['6', '4', '3'],
        ['5', '4'], ['5'], ['0', '6'], ['3', '7']]
        """
        pass

    def get_tables(self):
        """
        Adds tables to the network.

        Examples
        --------
        >>> from pgmpy.readwrite import UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.get_tables() # doctest: +NORMALIZE_WHITESPACE
        [['0.01', '0.99'], ['0.6', '0.3', '0.4', '0.7'],
        ['0.9', '0.8', '0.7', '0.1', '0.1', '0.2', '0.3', '0.9'],
        ['1.0', '1.0', '1.0', '0.0', '0.0', '0.0', '0.0', '1.0'],
        ['0.1', '0.01', '0.9', '0.99'], ['0.5', '0.5'],
        ['0.05', '0.01', '0.95', '0.99'], ['0.98', '0.05', '0.02', '0.95']]
        """
        pass

    def write(self, filename):
        """
        Write the xml data into the file.

        Parameters
        ----------
        filename: Name of the file.

        Examples
        --------
        >>> from pgmpy.readwrite import UAIWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = UAIWriter(model)
        >>> writer.write("asia.uai")
        """
        pass

    def write_uai(self, filename):
        pass
