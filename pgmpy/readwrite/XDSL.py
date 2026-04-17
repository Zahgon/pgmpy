import random
import warnings
import xml.dom.minidom as md
import xml.etree.ElementTree as etree
from itertools import chain

import networkx as nx

from pgmpy import logger
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import compat_fns


class XDSLReader:
    """
    Initializes the reader object for XDSL file formats[1] created through GeNIe[2].
    Note that XDSLReader only supports cpt blocks from the XDSL file format; elements like
    'deterministic' need to be aapropriately converted into 'cpt' elements before usage.

    Parameters
    ----------
    path : file or str
        Path to the XDSL file.

    string : str
        A string containing the XDSL file content.

    Examples
    --------
    >>> # AsiaDiagnosis.xdsl is an example file downloadable from
    >>> # https://repo.bayesfusion.com/bayesbox.html
    >>> # The file has been modified slightly to adhere to XDSLReader requirements
    >>> from pgmpy.readwrite import XDSLReader
    >>> reader = XDSLReader("AsiaDiagnosis.xdsl")
    >>> model = reader.get_model()

    Reference
    ---------
    [1] https://support.bayesfusion.com/docs/GeNIe/saving_xdslfileformat.html
    [2] https://www.bayesfusion.com/genie/
    """

    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.ElementTree(file=path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = self.network.attrib["id"]
        self.cpt_elements = self.network.find("nodes").findall("cpt")
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_values()

    def get_variables(self):
        """
        Returns list of variables of the network

        Examples
        --------
        >>> reader = XDSLReader("AsiaDiagnosis.xdsl")
        >>> reader.get_variables()
        ['asia', 'tub', 'smoke', 'lung', 'either', 'xray', 'bronc', 'dysp']
        """
        pass

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Examples
        --------
        >>> reader = XDSLReader("AsiaDiagnosis.xdsl")
        >>> reader.get_parents()
        {'asia': [],
        'tub': ['asia'],
        'smoke': [],
        'lung': ['smoke'],
        'either': ['tub', 'lung'],
        'xray': ['either'],
        'bronc': ['smoke'],
        'dysp': ['either', 'bronc']
        }
        """
        pass

    def get_edges(self):
        """
        Returns the edges of the network

        Examples
        --------
        >>> reader = XDSLReader("AsiaDiagnosis.xdsl")
        >>> reader.get_edges()
        [['asia', 'tub'],
        ['smoke', 'lung'],
        ['tub', 'either'],
        ['lung', 'either'],
        ['either', 'xray'],
        ['smoke', 'bronc'],
        ['either', 'dysp'],
        ['bronc', 'dysp']]
        """
        pass

    def get_states(self):
        """
        Returns the states of variables present in the network

        Examples
        --------
        >>> reader = XDSLReader("AsiaDiagnosis.xdsl")
        >>> reader.get_states()
        {'asia': ['no', 'yes'],
        'tub': ['no', 'yes'],
        'smoke': ['no', 'yes'],
        'lung': ['no', 'yes'],
        'either': ['Nothing',
        'CancerORTuberculosis'],
        'xray': ['Normal', 'Abnormal'],
        'bronc': ['Absent', 'Present'], '
        dysp': ['Absent', 'Present']
        }
        """
        pass

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> reader = XDSLReader("AsiaDiagnosis.xdsl")
        >>> reader.get_values()
        {'asia': [[0.99], [0.01]],
        'tub': [[0.99, 0.95], [0.01, 0.05]],
        'smoke': [[0.5], [0.5]],
        'lung': [[0.99, 0.9], [0.01, 0.1]],
        'either': [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        'xray': [[0.95, 0.02], [0.05, 0.98]],
        'bronc': [[0.7, 0.4], [0.3, 0.6]],
        'dysp': [[0.9, 0.2, 0.3, 0.1], [0.1, 0.8, 0.7, 0.9]]
        }
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
        >>> from pgmpy.readwrite import XDSLReader
        >>> reader = XDSLReader("AsiaDiagnosis.xdsl")
        >>> model = reader.get_model()
        """
        pass


class XDSLWriter:
    """
    Initialise a XDSL writer object to export pgmpy models to XDSL file format[1] used by GeNIe[2].

    Parameters
    ----------
    model: pgmpy.models.DiscreteBayesianNetwork instance.
        The model to write to the file.

    network_id: str (default: "MyNetwork")
        Name/id of the network

    num_samples: int (default: 0)
        Number of samples used for continuous variables

    disc_samples: int (default: 0)
        Number of samples used for discrete variables

    encoding: str (optional, default='utf-8')
        Encoding for text data

    Examples
    ---------
    >>> from pgmpy.readwrite import XDSLWriter
    >>> from pgmpy.example_models import load_model
    >>> asia = load_model("bnlearn/asia")
    >>> writer = XDSLWriter(asia)
    >>> writer.write("asia.xdsl")

    Reference
    ---------
    [1] https://support.bayesfusion.com/docs/GeNIe/saving_xdslfileformat.html
    [2] https://www.bayesfusion.com/genie/
    """

    def __init__(
        self,
        model,
        network_id="MyNetwork",
        num_samples="0",
        disc_samples="0",
        encoding="utf-8",
    ):
        if not isinstance(model, DiscreteBayesianNetwork):
            raise TypeError("model must an instance of DiscreteBayesianNetwork")
        self.model = model
        self.encoding = encoding
        self.network_id = network_id
        self.root = etree.Element(
            "smile",
            {
                "version": "1.0",
                "id": network_id,
                "numsamples": num_samples,
                "discsamples": disc_samples,
            },
        )

        self.variables = self.get_variables()
        self.cpds = self.get_cpds()
        self._create_extensions()

    def get_variables(self):
        """
        Add variables and their XML elements/representation to XDSL

        Return
        ------
        dict: dict of type {variable: variable tags}

        Examples
        --------
        >>> writer = XDSLWriter(model)
        >>> writer.get_variables()
        {'asia': <Element 'cpt' at 0x000001DC6BFA1350>,
        'tub': <Element 'cpt' at 0x000001DC6BFA35B0>,
        'smoke': <Element 'cpt' at 0x000001DC6BFA3560>,
        'lung': <Element 'cpt' at 0x000001DC6BFA12B0>,
        'bronc': <Element 'cpt' at 0x000001DC6BFA1260>,
        'either': <Element 'cpt' at 0x000001DC6BFA3510>,
        'xray': <Element 'cpt' at 0x000001DC6BFA34C0>,
        'dysp': <Element 'cpt' at 0x000001DC6BFA1210>}
        """
        pass

    def get_cpds(self):
        """
        Add the complete CPT element (with states and probabilities) to XDSL.

        Return
        ---------------
        dict: dict of type {variable: table tag}

        Examples
        -------
        >>> writer = XDSLWriter(model)
        >>> writer.get_values()
        {'asia': <TabularCPD representing P(asia:2) at 0x1885817c830>,
        'tub': <TabularCPD representing P(tub:2 | asia:2) at 0x1885a7e57c0>,
        'smoke': <TabularCPD representing P(smoke:2) at 0x18858327950>,
        'lung': <TabularCPD representing P(lung:2 | smoke:2) at 0x188583278f0>,
        'bronc': <TabularCPD representing P(bronc:2 | smoke:2) at 0x18855e05610>,
        'either': <TabularCPD representing P(either:2 | lung:2, tub:2) at 0x188582792e0>,
        'xray': <TabularCPD representing P(xray:2 | either:2) at 0x1885a7e5910>,
        'dysp': <TabularCPD representing P(dysp:2 | bronc:2, either:2) at 0x18858278b90>}
        """
        pass

    def _create_extensions(self):
        """
        Create the <extensions> block with a minimal <genie> element for layout information.

        Parameters
        ----------

        """
        pass

    def write(self, filename=None):
        """
        Write the xdsl data into the file.

        Parameters
        ----------
        filename: Name (path) of the file.

        Examples
        --------
        >>> from pgmpy.readwrite import XDSLWriter
        >>> from pgmpy.example_models import load_model
        >>> model = load_model("bnlearn/asia")
        >>> writer = XDSLWriter(model)
        >>> writer.write("asia.xdsl")
        """
        pass

    def write_xdsl(self, filename):
        pass
