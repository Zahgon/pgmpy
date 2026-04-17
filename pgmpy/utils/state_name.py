class StateNameMixin:
    """
    This class is inherited by classes which deal with state names of variables.
    The state names are stored in instances of `StateNameMixin`. The conversion between
    state number and names are also handled by methods in this class.
    """

    def store_state_names(self, variables, cardinality, state_names):
        """
        Initialize an instance of StateNameMixin.

        Example
        -------
        >>> import numpy as np
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> sn = {
        ...     "speed": ["low", "medium", "high"],
        ...     "switch": ["on", "off"],
        ...     "time": ["day", "night"],
        ... }
        >>> phi = DiscreteFactor(
        ...     variables=["speed", "switch", "time"],
        ...     cardinality=[3, 2, 2],
        ...     values=np.ones(12),
        ...     state_names=sn,
        ... )
        >>> print(phi.state_names)
        {'speed': ['low', 'medium', 'high'], 'switch': ['on', 'off'], 'time': ['day', 'night']}
        """
        pass

    def get_state_names(self, var, state_no):
        """
        Given `var` and `state_no` returns the state name.
        """
        pass

    def get_state_no(self, var, state_name):
        """
        Given `var` and `state_name` return the state number.
        """
        pass

    def add_state_names(self, phi1):
        """
        Updates the attributes of this class with another factor `phi1`.
        Ensures state name consistency.

        Parameters
        ----------
        phi1: Instance of pgmpy.factors.DiscreteFactor
            The factor whose states and variables need to be added.
        """
        pass

    def del_state_names(self, var_list):
        """
        Deletes the state names for variables in var_list
        """
        pass
