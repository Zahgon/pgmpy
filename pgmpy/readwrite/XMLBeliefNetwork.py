import itertools
import warnings
import xml.etree.ElementTree as etree

import numpy as np

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


class XBNReader:
    """
    Initializer for XBNReader class.

    Parameters
    ----------
    path: str or file
        Path of the file containing XBN data.

    string: str
        String of XBN data

    Examples
    --------
    >>> reader = XBNReader("test_XBN.xml")

    Reference
    ---------
    [1] Microsoft Research. XML belief network file format.
        http://xml.coverpages.org/xbn-MSdefault19990414.html, 1999.
    """

    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.parse(path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")

        self.bnmodel = self.network.find("BNMODEL")
        self.analysisnotebook = self.get_analysisnotebook_values()
        self.model_name = self.get_bnmodel_name()
        self.static_properties = self.get_static_properties()
        self.variables = self.get_variables()
        self.edges = self.get_edges()
        self.variable_CPD = self.get_distributions()

    def get_analysisnotebook_values(self):
        """
        Returns a dictionary of the attributes of ANALYSISNOTEBOOK tag

        Examples
        --------
        >>> reader = XBNReader("xbn_test.xml")
        >>> reader.get_analysisnotebook_values()
        {'NAME': "Notebook.Cancer Example From Neapolitan",
         'ROOT': "Cancer"}
        """
        pass

    def get_bnmodel_name(self):
        """
        Returns the name of the BNMODEL.

        Examples
        --------
        >>> reader = XBNReader("xbn_test.xml")
        >>> reader.get_bnmodel_name()
        'Cancer'
        """
        pass

    def get_static_properties(self):
        """
        Returns a dictionary of STATICPROPERTIES

        Examples
        --------
        >>> reader = XBNReader("xbn_test.xml")
        >>> reader.get_static_properties()
        {'FORMAT': 'MSR DTAS XML', 'VERSION': '0.2', 'CREATOR': 'Microsoft Research DTAS'}
        """
        pass

    def get_variables(self):
        """
        Returns a list of variables.

        Examples
        --------
        >>> reader = XBNReader("xbn_test.xml")
        >>> reader.get_variables()
        {'a': {'TYPE': 'discrete', 'XPOS': '13495',
               'YPOS': '10465', 'DESCRIPTION': '(a) Metastatic Cancer',
               'STATES': ['Present', 'Absent']}
        'b': {'TYPE': 'discrete', 'XPOS': '11290',
               'YPOS': '11965', 'DESCRIPTION': '(b) Serum Calcium Increase',
               'STATES': ['Present', 'Absent']},
        'c': {....},
        'd': {....},
        'e': {....}
        }
        """
        pass

    def get_edges(self):
        """
        Returns a list of tuples. Each tuple contains two elements (parent, child) for each edge.

        Examples
        --------
        >>> reader = XBNReader("xbn_test.xml")
        >>> reader.get_edges()
        [('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd'), ('c', 'e')]
        """
        pass

    def get_distributions(self):
        """
        Returns a dictionary of name and its distribution. Distribution is a ndarray.

        The ndarray is stored in the standard way such that the rightmost variable
        changes most often. Consider a CPD of variable 'd' which has parents 'b' and
        'c' (distribution['CONDSET'] = ['b', 'c'])

                  |  d_0     d_1
        ---------------------------
        b_0, c_0  |  0.8     0.2
        b_0, c_1  |  0.9     0.1
        b_1, c_0  |  0.7     0.3
        b_1, c_1  |  0.05    0.95

        The value of distribution['d']['DPIS'] for the above example will be:
        array([[ 0.8 ,  0.2 ], [ 0.9 ,  0.1 ], [ 0.7 ,  0.3 ], [ 0.05,  0.95]])

        Examples
        --------
        >>> reader = XBNReader("xbn_test.xml")
        >>> reader.get_distributions()
        {'a': {'TYPE': 'discrete', 'DPIS': array([[ 0.2,  0.8]])},
         'e': {'TYPE': 'discrete', 'DPIS': array([[ 0.8,  0.2],
                 [ 0.6,  0.4]]), 'CONDSET': ['c'], 'CARDINALITY': [2]},
         'b': {'TYPE': 'discrete', 'DPIS': array([[ 0.8,  0.2],
                 [ 0.2,  0.8]]), 'CONDSET': ['a'], 'CARDINALITY': [2]},
         'c': {'TYPE': 'discrete', 'DPIS': array([[ 0.2 ,  0.8 ],
                 [ 0.05,  0.95]]), 'CONDSET': ['a'], 'CARDINALITY': [2]},
         'd': {'TYPE': 'discrete', 'DPIS': array([[ 0.8 ,  0.2 ],
                 [ 0.9 ,  0.1 ],
                 [ 0.7 ,  0.3 ],
                 [ 0.05,  0.95]]), 'CONDSET': ['b', 'c']}, 'CARDINALITY': [2, 2]}
        """
        pass

    def get_model(self):
        """
        Returns an instance of Bayesian Model.
        """
        pass


class XBNWriter:
    """
    Initializer for XBNWriter class

    Parameters
    ----------
    model: DiscreteBayesianNetwork Instance
        Model to write
    encoding: str(optional)
        Encoding for test data
    prettyprint: Bool(optional)
        Indentation in output XML if true

    Reference
    ---------
    http://xml.coverpages.org/xbn-MSdefault19990414.html

    Examples
    --------
    >>> writer = XBNWriter(model)
    """

    def __init__(self, model, encoding="utf-8", prettyprint=True):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("Model must be an instance of Bayesian Model.")
        self.model = model

        self.encoding = encoding
        self.prettyprint = prettyprint

        self.network = etree.Element("ANALYSISNOTEBOOK")
        self.bnmodel = etree.SubElement(self.network, "BNMODEL")
        if self.model.name:
            etree.SubElement(self.bnmodel, "NAME").text = self.model.name

        self.variables = self.set_variables(self.model.nodes)
        self.structure = self.set_edges(sorted(self.model.edges()))
        self.distribution = self.set_distributions()

    def __str__(self):
        """
        Return the XML as string.
        """
        if self.prettyprint:
            self.indent(self.network)
        return etree.tostring(self.network, encoding=self.encoding)

    def indent(self, elem, level=0):
        """
        Inplace prettyprint formatter.
        """
        pass

    def set_analysisnotebook(self, **data):
        """
        Set attributes for ANALYSISNOTEBOOK tag

        Parameters
        ----------
        **data: dict
            {name: value} for the attributes to be set.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_analysisnotebook(
        ...     NAME="Notebook.Cancer Example From Neapolitan", ROOT="Cancer"
        ... )
        """
        pass

    def set_bnmodel_name(self, name):
        """
        Set the name of the BNMODEL.

        Parameters
        ----------
        name: str
            Name of the BNModel.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_bnmodel_name("Cancer")
        """
        pass

    def set_static_properties(self, **data):
        """
        Set STATICPROPERTIES tag for the network

        Parameters
        ----------
        **data: dict
            {name: value} for name and value of the property.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_static_properties(
        ...     FORMAT="MSR DTAS XML", VERSION="0.2", CREATOR="Microsoft Research DTAS"
        ... )
        """
        pass

    def set_variables(self, data):
        """
        Set variables for the network.

        Parameters
        ----------
        data: dict
            dict for variable in the form of example as shown.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_variables(
        ...     {
        ...         "a": {
        ...             "TYPE": "discrete",
        ...             "XPOS": "13495",
        ...             "YPOS": "10465",
        ...             "DESCRIPTION": "(a) Metastatic Cancer",
        ...             "STATES": ["Present", "Absent"],
        ...         },
        ...         "b": {
        ...             "TYPE": "discrete",
        ...             "XPOS": "11290",
        ...             "YPOS": "11965",
        ...             "DESCRIPTION": "(b) Serum Calcium Increase",
        ...             "STATES": ["Present", "Absent"],
        ...         },
        ...     }
        ... )
        """
        pass

    def set_edges(self, edge_list):
        """
        Set edges/arc in the network.

        Parameters
        ----------
        edge_list: array_like
            list, tuple, dict or set whose each element has two values (parent, child).

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_edges(
        ...     [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("c", "e")]
        ... )
        """
        pass

    def set_distributions(self):
        """
        Set distributions in the network.

        Examples
        --------
        >>> from pgmpy.readwrite.XMLBeliefNetwork import XBNWriter
        >>> writer = XBNWriter()
        >>> writer.set_distributions()
        """
        pass

    def write(self, filename):
        """
        Writes the BIF data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.example_models import load_model
        >>> from pgmpy.readwrite import XBNReader, XBNWriter
        >>> asia = load_model("bnlearn/asia")
        >>> writer = XBNWriter(asia)
        >>> writer.write(filename="asia.xbn")
        """
        pass

    def write_xbn(self, filename):
        pass
