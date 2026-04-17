#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import xml.etree.ElementTree as etree
from collections import defaultdict


class PomdpXReader:
    """
    Initialize an instance of PomdpX reader class

    Parameters
    ----------
    path : file or str
        Path of the file containing PomdpX information.

    string : str
        String containing PomdpX information.

    Example
    -------
    reader = PomdpXReader('TestPomdpX.xml')

    Reference
    ---------
    http://bigbird.comp.nus.edu.sg/pmwiki/farm/appl/index.php?n=Main.PomdpXDocumentation
    """

    def __init__(self, path=None, string=None):
        if path:
            self.network = etree.ElementTree(file=path).getroot()
        elif string:
            self.network = etree.fromstring(string)
        else:
            raise ValueError("Must specify either path or string")

    def get_description(self):
        """
        Return the problem description

        Examples
        --------
        >>> reader = PomdpXReader("Test_Pomdpx.xml")
        >>> reader.get_description()
        'RockSample problem for map size 1 x 3.
        Rock is at 0, Rover’s initial position is at 1.
        Exit is at 2.'
        >>> reader = PomdpXReader("Test_PomdpX.xml")
        >>> reader.get_description()
        'RockSample problem for map size 1 x 3.
         Rock is at 0, Rover’s initial position is at 1.
         Exit is at 2.'
        """
        pass

    def get_discount(self):
        """
        Returns the discount factor for the problem

        Example
        --------
        >>> reader = PomdpXReader("Test_PomdpX.xml")
        >>> reader.get_discount()
        0.95
        """
        pass

    def get_variables(self):
        """
        Returns list of variables of the network

        Example
        -------
        >>> reader = PomdpXReader("pomdpx.xml")
        >>> reader.get_variables()
        {'StateVar': [
                        {'vnamePrev': 'rover_0',
                         'vnameCurr': 'rover_1',
                         'ValueEnum': ['s0', 's1', 's2'],
                         'fullyObs': True},
                        {'vnamePrev': 'rock_0',
                         'vnameCurr': 'rock_1',
                         'fullyObs': False,
                         'ValueEnum': ['good', 'bad']}],
                        'ObsVar': [{'vname': 'obs_sensor',
                                    'ValueEnum': ['ogood', 'obad']}],
                        'RewardVar': [{'vname': 'reward_rover'}],
                        'ActionVar': [{'vname': 'action_rover',
                                       'ValueEnum': ['amw', 'ame',
                                                     'ac', 'as']}]
                        }
        """
        pass

    def get_initial_beliefs(self):
        """
        Returns the state, action and observation variables as a dictionary
        in the case of table type parameter and a nested structure in case of
        decision diagram parameter

        Examples
        --------
        >>> reader = PomdpXReader("Test_PomdpX.xml")
        >>> reader.get_initial_beliefs()
        [{'Var': 'rover_0',
          'Parent': ['null'],
          'Type': 'TBL',
          'Parameter': [{'Instance': ['-'],
          'ProbTable': ['0.0', '1.0', '0.0']}]
         },
         {'Var': '',
          '...': ...,'
          '...': '...',
          }]
        """
        pass

    def get_state_transition_function(self):
        """
        Returns the transition of the state variables as nested dict in the
        case of table type parameter and a nested structure in case of
        decision diagram parameter

        Example
        --------
        >>> reader = PomdpXReader("Test_PomdpX.xml")
        >>> reader.get_state_transition_function()
        [{'Var': 'rover_1',
          'Parent': ['action_rover', 'rover_0'],
          'Type': 'TBL',
          'Parameter': [{'Instance': ['amw', 's0', 's2'],
                         'ProbTable': ['1.0']},
                         {'Instance': ['amw', 's1', 's0'],
                         'ProbTable': ['1.0']},
                         pass
                        ]
        }]
        """
        pass

    def get_obs_function(self):
        """
        Returns the observation function as nested dict in the case of table-
        type parameter and a nested structure in case of
        decision diagram parameter

        Example
        --------
        >>> reader = PomdpXReader("Test_PomdpX.xml")
        >>> reader.get_obs_function()
        [{'Var': 'obs_sensor',
              'Parent': ['action_rover', 'rover_1', 'rock_1'],
              'Type': 'TBL',
              'Parameter': [{'Instance': ['amw', '*', '*', '-'],
                             'ProbTable': ['1.0', '0.0']},
                         pass
                        ]
        }]
        """
        pass

    def get_reward_function(self):
        """
        Returns the reward function as nested dict in the case of table-
        type parameter and a nested structure in case of
        decision diagram parameter

        Example
        --------
        >>> reader = PomdpXReader("Test_PomdpX.xml")
        >>> reader.get_reward_function()
        [{'Var': 'reward_rover',
              'Parent': ['action_rover', 'rover_0', 'rock_0'],
              'Type': 'TBL',
              'Parameter': [{'Instance': ['ame', 's1', '*'],
                             'ValueTable': ['10']},
                         pass
                        ]
        }]
        """
        pass

    def get_parameter(self, var):
        """
        This method supports the functional tags by providing the actual
        values in the function as list of dict in case of table type parameter or as
        nested dict in case of decision diagram
        """
        pass

    def get_parameter_tbl(self, parameter):
        """
        This method returns parameters as list of dict in case of table type
        parameter
        """
        pass

    def get_parameter_dd(self, parameter):
        """
        This method returns parameters as nested dicts in case of decision
        diagram parameter.
        """
        pass


class PomdpXWriter:
    """
    Initialise a PomdpXWriter Object

    Parameters
    ----------
        model: A Bayesian of Markov Model
            The model to write

    encoding: String(optional)
        Encoding for text data

    prettyprint: Bool(optional)
        Indentation in output XML if true
    """

    def __init__(self, model_data, encoding="utf-8", prettyprint=True):
        self.model = model_data

        self.encoding = encoding
        self.prettyprint = prettyprint

        self.xml = etree.Element("pomdpx", attrib={"version": "1.0"})
        self.description = etree.SubElement(self.xml, "Description")
        self.discount = etree.SubElement(self.xml, "Discount")
        self.variable = etree.SubElement(self.xml, "Variable")
        self.initial_belief = etree.SubElement(self.xml, "InitialStateBelief")
        self.transition_function = etree.SubElement(self.xml, "StateTransitionFunction")
        self.observation_function = etree.SubElement(self.xml, "ObsFunction")
        self.reward_function = etree.SubElement(self.xml, "RewardFunction")

    def __str__(self, xml):
        """
        Return the XML as string.
        """
        if self.prettyprint:
            self.indent(xml)
        return etree.tostring(xml, encoding=self.encoding)

    def indent(self, elem, level=0):
        """
        Inplace prettyprint formatter.
        """
        pass

    def _add_value_enum(self, var, tag):
        """
        supports adding variables to the xml

        Parameters
        ---------------
        var: The SubElement variable
        tag: The SubElement tag to which enum value is to be added

        Return
        ---------------
        None
        """
        pass

    def get_variables(self):
        """
        Add variables to PomdpX

        Return
        ---------------
        xml containing variables tag
        """
        pass

    def add_parameter_dd(self, dag_tag, node_dict):
        """
        helper function for adding parameters in condition

        Parameters
        ---------------
        dag_tag: etree SubElement
                 the DAG tag is contained in this subelement
        node_dict: dictionary
                   the decision diagram dictionary

        Return
        ---------------
        None
        """
        pass

    def add_conditions(self, condition, condprob):
        """
        helper function for adding probability conditions for model\

        Parameters
        ---------------

        condition:  dictionary
                    contains and element of conditions list
        condprob:   etree SubElement
                    the tag to which condition is added

        Return
        ---------------
        None
        """
        pass

    def add_initial_belief(self):
        """
        add initial belief tag to pomdpx model

        Return
        ---------------
        string containing the xml for initial belief tag
        """
        pass

    def add_state_transition_function(self):
        """
        add state transition function tag to pomdpx model

        Return
        ---------------
        string containing the xml for state transition tag
        """
        pass

    def add_obs_function(self):
        """
        add observation function tag to pomdpx model

        Return
        ---------------
        string containing the xml for observation function tag
        """
        pass

    def add_reward_function(self):
        """
        add reward function tag to pomdpx model

        Return
        ---------------
        string containing the xml for reward function tag
        """
        pass
