#!/usr/bin/env python

import warnings
import xml.etree.ElementTree as etree
from io import BytesIO
from itertools import chain

import numpy as np

from pgmpy import logger
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns

try:
    import pyparsing as pp
except ImportError as e:
    raise ImportError(
        f"{e} . pyparsing is required for using read/write methods. Please install using: pip install pyparsing."
    ) from None


class XMLBIFReader:
    """
    Initialisation of XMLBIFReader object.

    Parameters
    ----------
    path : file or str
        File of XMLBIF data
        File of XMLBIF data

    string : str
        String of XMLBIF data

    Examples
    --------
    >>> # xmlbif_test.xml is the file present in
    >>> # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
    >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/asia")
    >>> writer = XMLBIFWriter(model)
    >>> writer.write("xmlbif_test.xml")
    >>> reader = XMLBIFReader("xmlbif_test.xml")
    >>> model = reader.get_model()

    Reference
    ---------
    [1] https://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/
    """

    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.ElementTree(file=path).getroot().find("NETWORK")
        elif string:
            self.network = etree.fromstring(string.encode("utf-8")).find("NETWORK")
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = self.network.find("NAME").text
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_values()
        self.variable_property = self.get_property()
        self.state_names = self.get_states()

    def get_variables(self):
        """
        Returns list of variables of the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> sorted(reader.get_variables())
        ['asia', 'bronc', 'dysp', 'either', 'lung', 'smoke', 'tub', 'xray']
        """
        pass

    def get_edges(self):
        """
        Returns the edges of the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_edges() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [['smoke', 'bronc'], ['bronc', 'dysp'],
        ['either', 'dysp'], ['lung', 'either'],
        ['tub', 'either'], ['smoke', 'lung'],
        ['asia', 'tub'], ['either', 'xray']]
        """
        pass

    def get_states(self):
        """
        Returns the states of variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_states() # doctest: +NORMALIZE_WHITESPACE
        {'asia': ['yes', 'no'],
        'bronc': ['yes', 'no'],
        'dysp': ['yes', 'no'],
        'either': ['yes', 'no'],
        'lung': ['yes', 'no'],
        'smoke': ['yes', 'no'],
        'tub': ['yes', 'no'],
        'xray': ['yes', 'no']}
        """
        pass

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_parents() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [],
        'bronc': ['smoke'],
        'dysp': ['bronc', 'either'],
        'either': ['lung', 'tub'],
        'lung': ['smoke'],
        'smoke': [],
        'tub': ['asia'],
        'xray': ['either']}
        """
        pass

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
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

    def get_property(self):
        """
        Returns the property of the variable

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> reader.get_property() # doctest: +NORMALIZE_WHITESPACE
        {'asia': [None], 'bronc': [None], 'dysp': [None],
        'either': [None], 'lung': [None], 'smoke': [None],
        'tub': [None], 'xray': [None]}
        """
        pass

    def get_model(self, state_name_type=str):
        """
        Returns a Bayesian Network instance from the file/string.

        Parameters
        ----------
        state_name_type: int, str, or bool (default: str)
            The data type to which to convert the state names of the variables.

        Returns
        -------
        DiscreteBayesianNetwork instance: The read model.

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter, XMLBIFReader
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("xmlbif_test.xml")
        >>> reader = XMLBIFReader("xmlbif_test.xml")
        >>> model = reader.get_model()
        """
        pass


class XMLBIFWriter:
    """
    Initialise a XMLBIFWriter object.

    Parameters
    ----------
    model: DiscreteBayesianNetwork Instance
        Model to write

    encoding: str (optional)
        Encoding for text data

    prettyprint: Bool(optional)
        Indentation in output XML if true

    Examples
    --------
    >>> from pgmpy.readwrite import XMLBIFWriter
    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/asia")
    >>> writer = XMLBIFWriter(model)
    >>> writer.write("asia.xml")

    Reference
    ---------
    [1] https://www.cs.cmu.edu/afs/cs/user/fgcozman/www/Research/InterchangeFormat/
    """

    def __init__(self, model, encoding="utf-8", prettyprint=True):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("model must an instance of DiscreteBayesianNetwork")
        self.model = model

        self.encoding = encoding
        self.prettyprint = prettyprint

        self.xml = etree.Element("BIF", attrib={"VERSION": "0.3"})
        self.network = etree.SubElement(self.xml, "NETWORK")
        if self.model.name:
            etree.SubElement(self.network, "NAME").text = self.model.name
        else:
            etree.SubElement(self.network, "NAME").text = "UNTITLED"

        self.variables = self.get_variables()
        self.states = self.get_states()
        self.properties = self.get_properties()
        self.definition = self.get_definition()
        self.tables = self.get_values()

    def __str__(self):
        """
        Return the XML as string.
        """
        if self.prettyprint:
            self.indent(self.xml)
        f = BytesIO()
        et = etree.ElementTree(self.xml)
        et.write(f, encoding=self.encoding, xml_declaration=True)
        return f.getvalue().decode(self.encoding)

    def indent(self, elem, level=0):
        """
        Inplace prettyprint formatter.
        """
        pass

    def get_variables(self):
        """
        Add variables to XMLBIF

        Return
        ------
        dict: dict of type {variable: variable tags}

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_variables() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'VARIABLE' at 0x...>,
        'bronc': <Element 'VARIABLE' at 0x...>,
        'dysp': <Element 'VARIABLE' at 0x...>,
        'either': <Element 'VARIABLE' at 0x...>,
        'lung': <Element 'VARIABLE' at 0x...>,
        'smoke': <Element 'VARIABLE' at 0x...>,
        'tub': <Element 'VARIABLE' at 0x...>,
        'xray': <Element 'VARIABLE' at 0x...>}
        """
        pass

    def get_states(self):
        """
        Add outcome to variables of XMLBIF

        Return
        ------
        dict: dict of type {variable: outcome tags}

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_states() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'bronc': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'dysp': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'either': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'lung': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'smoke': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'tub': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>],
        'xray': [<Element 'OUTCOME' at 0x...>, <Element 'OUTCOME' at 0x...>]}
        """
        pass

    def _make_valid_state_name(self, state_name):
        """Transform the input state_name into a valid state in XMLBIF.
        XMLBIF states must start with a letter and only contain letters,
        numbers and underscores.
        """
        pass

    def get_properties(self):
        """
        Add property to variables in XMLBIF

        Return
        ------
        dict: dict of type {variable: property tag}

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_properties() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'PROPERTY' at 0x...>,
        'bronc': <Element 'PROPERTY' at 0x...>,
        'dysp': <Element 'PROPERTY' at 0x...>,
        'either': <Element 'PROPERTY' at 0x...>,
        'lung': <Element 'PROPERTY' at 0x...>,
        'smoke': <Element 'PROPERTY' at 0x...>,
        'tub': <Element 'PROPERTY' at 0x...>,
        'xray': <Element 'PROPERTY' at 0x...>}
        """
        pass

    def get_definition(self):
        """
        Add Definition to XMLBIF

        Return
        ------
        dict: dict of type {variable: definition tag}

        Examples
        --------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_definition() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'DEFINITION' at 0x...>,
        'bronc': <Element 'DEFINITION' at 0x...>,
        'dysp': <Element 'DEFINITION' at 0x...>,
        'either': <Element 'DEFINITION' at 0x...>,
        'lung': <Element 'DEFINITION' at 0x...>,
        'smoke': <Element 'DEFINITION' at 0x...>,
        'tub': <Element 'DEFINITION' at 0x...>,
        'xray': <Element 'DEFINITION' at 0x...>}
        """
        pass

    def get_values(self):
        """
        Add Table to XMLBIF.

        Return
        ---------------
        dict: dict of type {variable: table tag}

        Examples
        -------
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.get_values() # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'asia': <Element 'TABLE' at 0x...>,
        'bronc': <Element 'TABLE' at 0x...>,
        'dysp': <Element 'TABLE' at 0x...>,
        'either': <Element 'TABLE' at 0x...>,
        'lung': <Element 'TABLE' at 0x...>,
        'smoke': <Element 'TABLE' at 0x...>,
        'tub': <Element 'TABLE' at 0x...>,
        'xray': <Element 'TABLE' at 0x...>}
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
        >>> from pgmpy.readwrite import XMLBIFWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XMLBIFWriter(model)
        >>> writer.write("asia.xml")
        """
        pass

    def write_xmlbif(self, filename):
        pass
