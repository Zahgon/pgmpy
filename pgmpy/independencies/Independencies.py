import itertools


class Independencies:
    """
    Base class for independencies.
    independencies class represents a set of Conditional Independence
    assertions (eg: "X is independent of Y given Z" where X, Y and Z
    are random variables) or Independence assertions (eg: "X is
    independent of Y" where X and Y are random variables).
    Initialize the independencies Class with Conditional Independence
    assertions or Independence assertions.

    Parameters
    ----------
    assertions: Lists or tuples
            Each assertion is a list or tuple of the form: [event1,
            event2 and event3]
            eg: assertion ['X', 'Y', 'Z'] would be X is independent
            of Y given Z.

    Examples
    --------
    Creating an independencies object with one independence assertion:
    Random Variable X is independent of Y

    >>> from pgmpy.independencies import Independencies
    >>>
    >>> independencies = Independencies(["X", "Y"])

    Creating an independencies object with three conditional
    independence assertions:
    First assertion is Random Variable X is independent of Y given Z.

    >>> independencies = Independencies(
    ...     ["X", "Y", "Z"], ["a", ["b", "c"], "d"], ["l", ["m", "n"], "o"]
    ... )

    Public Methods
    --------------
    add_assertions
    get_assertions
    get_factorized_product
    closure
    entails
    is_equivalent
    """

    def __init__(self, *assertions):
        self.independencies = []
        self.add_assertions(*assertions)

    def __str__(self):
        string = "\n".join([str(assertion) for assertion in self.independencies])
        return string

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, Independencies):
            return False
        return all(independency in other.get_assertions() for independency in self.get_assertions()) and all(
            independency in self.get_assertions() for independency in other.get_assertions()
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def contains(self, assertion):
        """
        Returns `True` if `assertion` is contained in this `Independencies`-object,
        otherwise `False`.

        Parameters
        ----------
        assertion: IndependenceAssertion()-object

        Examples
        --------
        >>> from pgmpy.independencies import Independencies, IndependenceAssertion
        >>> ind = Independencies(["A", "B", ["C", "D"]])
        >>> IndependenceAssertion("A", "B", ["C", "D"]) in ind
        True
        >>> # does not depend on variable order:
        >>> IndependenceAssertion("B", "A", ["D", "C"]) in ind
        True
        >>> # but does not check entailment:
        >>> IndependenceAssertion("X", "Y", "Z") in Independencies(["X", "Y"])
        False
        """
        pass

    __contains__ = contains

    def get_all_variables(self):
        """
        Returns a set of all the variables in all the independence assertions.
        """
        pass

    def get_assertions(self):
        """
        Returns the independencies object which is a set of IndependenceAssertion objects.

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> independencies = Independencies(["X", "Y", "Z"])
        >>> independencies.get_assertions()
        [(X ⟂ Y | Z)]
        """
        pass

    def add_assertions(self, *assertions):
        """
        Adds assertions to independencies.

        Parameters
        ----------
        assertions: Lists or tuples
                Each assertion is a list or tuple of variable, independent_of and given.

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> independencies = Independencies()
        >>> independencies.add_assertions(["X", "Y", "Z"])
        >>> independencies.add_assertions(["a", ["b", "c"], "d"])
        """
        pass

    def closure(self):
        """
        Returns a new `Independencies()`-object that additionally contains those `IndependenceAssertions`
        that are implied by the current independencies (using with the `semi-graphoid axioms
        <https://en.wikipedia.org/w/index.php?title=Conditional_independence&oldid=708760689#Rules_of_conditional_independence>`_;
        see (Pearl, 1989, `Conditional Independence and its representations
        <http://www.cs.technion.ac.il/~dang/journal_papers/pearl1989conditional.pdf>`_)).

        Might be very slow if more than six variables are involved.

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> ind1 = Independencies(("A", ["B", "C"], "D"))
        >>> result1 = ind1.closure()
        >>> len(result1.get_assertions())
        5
        >>> all("A" in str(a) for a in result1.get_assertions())
        True

        >>> ind2 = Independencies(("W", ["X", "Y", "Z"]))
        >>> result2 = ind2.closure()
        >>> len(result2.get_assertions()) >= 9
        True
        """
        pass

    def entails(self, entailed_independencies):
        """
        Returns `True` if the `entailed_independencies` are implied by this `Independencies`-object, otherwise `False`.
        Entailment is checked using the semi-graphoid axioms.

        Might be very slow if more than six variables are involved.

        Parameters
        ----------
        entailed_independencies: Independencies()-object

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> ind1 = Independencies([["A", "B"], ["C", "D"], "E"])
        >>> ind2 = Independencies(["A", "C", "E"])
        >>> ind1.entails(ind2)
        True
        >>> ind2.entails(ind1)
        False
        """
        pass

    def is_equivalent(self, other):
        """
        Returns True if the two Independencies-objects are equivalent, otherwise False.
        (i.e. any Bayesian Network that satisfies the one set
        of conditional independencies also satisfies the other).

        Might be very slow if more than six variables are involved.

        Parameters
        ----------
        other: Independencies()-object

        Examples
        --------
        >>> from pgmpy.independencies import Independencies
        >>> ind1 = Independencies(["X", ["Y", "W"], "Z"])
        >>> ind2 = Independencies(["X", "Y", "Z"], ["X", "W", "Z"])
        >>> ind3 = Independencies(
        ...     ["X", "Y", "Z"], ["X", "W", "Z"], ["X", "Y", ["W", "Z"]]
        ... )
        >>> ind1.is_equivalent(ind2)
        False
        >>> ind1.is_equivalent(ind3)
        True
        """
        pass

    def reduce(self, inplace=False):
        """
        Return list of Independence Assertions without any duplicate or redundant Independence assertions.

        Might be very slow due to bidirectional entailment being checked.

        Assumption:
            If an assertion A entails assertion B and assertion B doesn't entails assertion A then
            assertion A is consider more informative and assertion B is removed.

        Parameters
        ----------

        inplace: bool (default: False)
            If True, the Independencies object will permanently removes duplicate or redundant Independence Assertions.

        """
        pass

    def latex_string(self) -> list[str]:
        """
        Returns a list of string.
        Each string represents the IndependenceAssertion in latex.
        """
        pass

    def get_factorized_product(self, random_variables=None, latex=False):
        # TODO: Write this whole function
        #
        # The problem right now is that the factorized product for all
        # P(A, B, C), P(B, A, C) etc should be same but on solving normally
        # we get different results which have to be simplified to a simpler
        # form. How to do that ??? and also how to decide which is the most
        # simplified form???
        #
        pass


class IndependenceAssertion:
    r"""
    Represents Conditional Independence or Independence assertion.

    Each assertion has 3 attributes: event1, event2, event3.
    The attributes for

    .. math:: U \perp X, Y | Z

    is read as: Random Variable U is independent of X and Y given Z would be:

    event1 = {U}

    event2 = {X, Y}

    event3 = {Z}

    Parameters
    ----------
    event1: String or List of strings
            Random Variable which is independent.

    event2: String or list of strings.
            Random Variables from which event1 is independent

    event3: String or list of strings.
            Random Variables given which event1 is independent of event2.

    Examples
    --------
    >>> from pgmpy.independencies import IndependenceAssertion
    >>> assertion = IndependenceAssertion("U", "X")
    >>> assertion = IndependenceAssertion("U", ["X", "Y"])
    >>> assertion = IndependenceAssertion("U", ["X", "Y"], "Z")
    >>> assertion = IndependenceAssertion(["U", "V"], ["X", "Y"], ["Z", "A"])


    Public Methods
    --------------
    get_assertion
    """

    def __init__(self, event1=[], event2=[], event3=[]):
        r"""
        Initialize an IndependenceAssertion object with event1, event2 and event3 attributes.

                    event2
                    ^
        event1     /   event3
           ^      /     ^
           |     /      |
          (U || X, Y | Z) read as Random variable U is independent of X and Y given Z.
            ---
        """
        if event1 and not event2:
            raise ValueError("event2 needs to be specified")
        if any([event2, event3]) and not event1:
            raise ValueError("event1 needs to be specified")
        if event3 and not all([event1, event2]):
            raise ValueError("event1" if not event1 else "event2" + " needs to be specified")

        self.event1 = frozenset(self._return_list_if_not_collection(event1))
        self.event2 = frozenset(self._return_list_if_not_collection(event2))
        self.event3 = frozenset(self._return_list_if_not_collection(event3))
        self.all_vars = frozenset().union(self.event1, self.event2, self.event3)

    def __str__(self):
        if self.event3:
            return "({event1} \u27c2 {event2} | {event3})".format(
                event1=", ".join(sorted([str(e) for e in self.event1])),
                event2=", ".join(sorted([str(e) for e in self.event2])),
                event3=", ".join(sorted([str(e) for e in self.event3])),
            )
        else:
            return "({event1} \u27c2 {event2})".format(
                event1=", ".join(sorted([str(e) for e in self.event1])),
                event2=", ".join(sorted([str(e) for e in self.event2])),
            )

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, IndependenceAssertion):
            return False
        return (self.event1, self.event2, self.event3) == other.get_assertion() or (
            self.event2,
            self.event1,
            self.event3,
        ) == other.get_assertion()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((frozenset((self.event1, self.event2)), self.event3))

    @staticmethod
    def _return_list_if_not_collection(event):
        """
        If variable is a string returns a list containing variable.
        Else returns variable itself.
        """
        pass

    def get_assertion(self):
        """
        Returns a tuple of the attributes: variable, independent_of, given.

        Examples
        --------
        >>> from pgmpy.independencies import IndependenceAssertion
        >>> asser = IndependenceAssertion("X", "Y", "Z")
        >>> asser.get_assertion()
        (frozenset({'X'}), frozenset({'Y'}), frozenset({'Z'}))
        """
        pass

    def latex_string(self):
        pass
