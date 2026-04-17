#!/usr/bin/env python3
__all__ = ["_GraphRolesMixin"]


class _GraphRolesMixin:
    """Mixin class for handling roles in a causal graph."""

    def get_role(self, role: str):
        """Return list of nodes in graph G with a specific role.

        Parameters
        ----------
        role : str
            The role to match.

        Returns
        -------
        List of nodes with the specified role.
        """
        pass

    def get_roles(self):
        """Get list of all roles present in the graph.

        Returns
        -------
        List of str
            list of all roles defined in the graph.
        """
        pass

    def get_role_dict(self):
        """Get dict of lists of roles preset in the graph.

        Returns
        -------
        Dict with str keys and values being list of nodes
            keys are roles present in the graph, and lists are nodes with that role
        """
        pass

    def has_role(self, role: str) -> bool:
        """Check if a role is defined and non-empty.

        Parameters
        ----------
        role : str
            The name of the role to check.

        Returns
        -------
        bool
            True if the role exists and has variables assigned, False otherwise.
        """
        pass

    def with_role(self, role: str, variables, inplace=False):
        """Return a new graph with the specified role assignment.

        Parameters
        ----------
        role : str
            The name of the role to assign, e.g., "exposures", "outcomes".
        variables : str, set, list, or any iterable
            The variables to assign to the role.
        inplace=False : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        graph of same type as self
            A new instance with the specified role assigned, to the variables provided.
        """
        pass

    def without_role(self, role: str, variables=None, inplace=False):
        """Return a new graph with the specified role removed.

        Parameters
        ----------
        role : str
            The name of the role to remove, e.g., "exposures", "outcomes".
        variables : str, set, list, or iterable, default = all variables with the role
            The variables to remove the role from. If not provided,
            all variables with the specified role will have it removed.
        inplace=False : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        graph of same type as self
            A new instance with the specified role removed from all nodes that had it.
        """
        pass

    def is_valid_causal_structure(self) -> bool:
        """Validate that the causal structure makes sense."""
        pass

    @property
    def latents(self):
        """
        Returns the set of latent variables in the causal model.

        Property
        --------
        latents : set of nodes (default: empty set)
            A set of latent variables in the graph. These are not observed
            variables but are used to represent unobserved confounding or
            other latent structures.

        Examples
        --------
        Create a DAG with latents and check the latents value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], latents="a")
        >>> G.latents
        {'a'}
        """
        pass

    @latents.setter
    def latents(self, variables):
        """
        Sets the latent variables in the model. If latents already exist, they will be replaced.

        Parameters
        ----------
        variables: set of nodes (default: empty set)
            A set of latent variables in the graph. These are not observed
            variables but are used to represent unobserved confounding or
            other latent structures.
        """
        pass

    @property
    def observed(self):
        """
        Returns the set of observed variables in the causal model.

        Property
        --------
        observed: set of nodes (default: empty set)
            A set of observed variables in the graph. These are the variables
            that can be measured directed and have data available for them.

        Examples
        --------
        Create a DAG with latents and check the observed value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], latents="a")
        >>> G.observed
        {'b'}
        """
        pass

    @property
    def exposures(self):
        """
        Returns the set of exposure variables in the causal model.

        Property
        --------
        exposures : set of nodes (default: empty set)
            A set of exposure variables in the graph. These are the variables
            that represent the treatment or intervention being studied in a
            causal analysis.

        Examples
        --------
        Create a DAG with exposures and check the exposures value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], exposures="a")
        >>> G.exposures
        {'a'}
        """
        pass

    @exposures.setter
    def exposures(self, variables):
        """
        Sets the exposure variables in the model. If exposure variables are already defined, they will be replaced.

        Parameters
        ----------
        variables: set of nodes (default: empty set)
            A set of exposure variables in the graph. These are the variables that represent the treatment or
            intervention being studied in a causal analysis.
        """
        pass

    @property
    def outcomes(self):
        """
        Returns the set of outcome variables in the causal model.

        Property
        --------
        outcomes : set of nodes (default: empty set)
            A set of outcome variables in the graph. These are the variables
            that represent the response or dependent variables being studied
            in a causal analysis.

        Examples
        --------
        Create a DAG with outcomes and check the outcomes value.

        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[("a", "b")], outcomes="b")
        >>> G.outcomes
        {'b'}
        """
        pass

    @outcomes.setter
    def outcomes(self, variables):
        """
        Sets the outcome variables in the model. If outcome variables are already defined, they will be replaced.

        Parameters
        ----------
        variables: set of nodes (default: empty set)
            A set of outcome variables in the graph. These are the variables
            that represent the response or dependent variables being studied
            in a causal analysis.
        """
        pass
