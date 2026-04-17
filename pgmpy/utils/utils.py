import gzip
import warnings

import pandas as pd

try:
    from importlib.resources import files
except ImportError:
    # For python 3.8 and lower
    from importlib_resources import files

from pgmpy import logger


def get_example_model(model: str):
    """
    Fetches the specified model from bnlearn repository and returns a
    pgmpy.model instance.

    Parameter
    ---------
    model: str
        Any model from bnlearn repository (http://www.bnlearn.com/bnrepository)
          and dagitty (https://www.dagitty.net/)
        Discrete Bayesian Network Options:
            Small Networks: asia, cancer, earthquake, sachs, survey
            Medium Networks: alarm, barley, child, insurance, mildew, water
            Large Networks: hailfinder, hepar2, win95pts
            Very Large Networks: andes, diabetes, link, munin1, munin2, munin3,
            munin4, pathfinder, pigs, munin
        Gaussian Bayesian Network Options: ecoli70,
        magic-niab, magic-irri, arth150
        Conditional Linear Gaussian Bayesian Network Options: sangiovese, mehra
        DAG Options: M-bias, confounding, mediator, paths,
          Sebastiani_2005, Polzer_2012,
          Schipf_2010, Shrier_2008, Acid_1996,
            Thoemmes_2013, Kampen_2014, Didelez_2010

    Example
    -------
    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model(model="asia")
    >>> model

    Returns
    -------
    pgmpy.models instance: An instance of
      one of the model classes in pgmpy.models
                           depending on the type of dataset.
    """
    pass


def discretize(data, cardinality, labels=dict(), method="rounding"):
    """
    Discretizes a given continuous dataset.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset to discretize. All columns must have continuous values.

    cardinality: dict
        A dictionary of the form (str: int) representing the number of bins
        to create for each of the variables.

    labels: dict (default: None)
        A dictionary of the form (str: list) representing the label names for
        each variable in the discretized dataframe.

    method: rounding or quantile
        If rounding, equal width bins are created and
          data is discretized into these bins.
          Refer pandas.cut for more details.
        If quantile, creates bins such that each
          bin has an equal number of datapoints.
            Refer pandas.qcut for more details.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.utils import discretize
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal(1000)
    >>> Y = 0.2 * X + rng.standard_normal(1000)
    >>> Z = 0.4 * X + 0.5 * Y + rng.standard_normal(1000)
    >>> df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>> df_disc = discretize(
    ...     df,
    ...     cardinality={"X": 3, "Y": 3, "Z": 3},
    ...     labels={
    ...         "X": ["low", "mid", "high"],
    ...         "Y": ["low", "mid", "high"],
    ...         "Z": ["low", "mid", "high"],
    ...     },
    ... )
    >>> df_disc.head()
        X    Y    Z
    0   mid  mid  mid
    1   mid  mid  low
    2   mid  mid  mid
    3  high  mid  mid
    4   low  mid  low

    Returns
    -------
    pandas.DataFrame: A discretized dataframe.
    """
    pass


def llm_pairwise_orient(
    x,
    y,
    descriptions,
    system_prompt=None,
    llm_model="gemini/gemini-1.5-flash",
    **kwargs,
):
    """
    Asks a Large Language Model (LLM) for the
     orientation of an edge between `x` and `y`.

    Parameters
    ----------
    x: str
        The first variable's name

    y: str
        The second variable's name

    descriptions: dict
        A dict of the form {variable: description}
          containing text description of the variables.

    system_prompt: str
        A system prompt to give the LLM.

    llm_model: str (default: gemini/gemini-pro)
        The LLM model to use. Please refer to litellm
          documentation (https://docs.litellm.ai/docs/providers)
        for available model options. Default is gemini-pro.

    kwargs: kwargs
        Any additional parameters to pass to litellm.completion method.

    Returns
    -------
    tuple:
        Returns a tuple (source, target) representing the edge direction.
    """
    pass


def manual_pairwise_orient(x, y):
    """
    Generates a prompt for the user to
      input the direction between the variables.

    Parameters
    ----------
    x: str
        The first variable's name

    y: str
        The second variable's name

    Returns
    -------
    tuple:
        Returns a tuple (source, target) representing the edge direction.
    """
    pass


def preprocess_data(df):
    """
    Tries to figure out the data type of each variable `df`.

    Assigns one of (numerical, categorical unordered, categorical ordered) datatypes to each column in `df`. Also
    changes any object datatypes to categorical.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe.

    Returns
    -------
    (pd.DataFrame, dtypes): tuple of transformed dataframe and a dictionary with inferred datatype of each column.
    """
    pass


def _heuristic_categorical_detection(df, dtypes):
    """
    Creates a warning if numerical values are detected for a categorical variable.
    """
    pass


def get_dataset_type(data: pd.DataFrame) -> str:
    """
    Returns continuous, discrete or mixed depending on the type of variable
    data in the given dataset.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame to analyze

    Returns
    -------
    str
        `continuous`, `discrete` or `mixed`.
    """
    pass


def to_timeseries_format(df: pd.DataFrame, return_format: str = "pd-multiindex"):
    """
    Converts given wide format dataframe to different time series formats.

    Takes a pandas dataframe with columns taken as ("Variable name", timestep) and rows represented as
    traces ( "wide" format) and converts it to different format as specified in `return_format` argument.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe represented in the wide format (on rows we have samples, on columns, unsorted pairs of
        ("Variable", "timestep")

    return_format : {'pd-multiindex', 'numpy3d', 'pd-list', 'sorted'}
        Controls the return representation. The options are:

        "numpy3d" : returns a numpy 3D tensor, where first dimension represents trace, second dimension
                    represents variable, third dimension represent timestep

        "pd-multiindex" : returns the pandas multiindex DataFrame, with indexes of ("Variable name", "timestep")

        "pd-list" : returns a list of pandas DataFrames. For every sample, a Dataframe is created, where rows
                    contain timestep and columns represent variables

        "sorted" : makes sure that the representation of [sample, ("variable", "timestep")] is sorted, which
                   makes further processing easier

    Returns
    -------
    np.ndarray or pd.DataFrame or list of pd.DataFrame:
        Depends on `return_format` variable. `numpy3d` returns a numpy array (`np.ndarray`), while rest of the
        representations return a pandas DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     [
    ...         [1, 1, 0, 0, 0, 0, 0, 1, 0],
    ...         [0, 2, 0, 1, 1, 1, 1, 1, 1],
    ...     ],
    ...     columns=[
    ...         ("D", 0), ("G" , 0), ("I" , 0),
    ...         ("D", 1), ("G", 1),
    ...         ("D", 2), ("G", 2),
    ...         ("I", 1), ("I", 2)
    ...     ],
    ... )

    For input dataframe `df`, represented in the wide format

      (D, 0) (G, 0) (I, 0) (D, 1) (G, 1) (D, 2) (G, 2) (I, 1) (I, 2)
    0      1      1      0      0      0      0      0      1      0
    1      0      2      0      1      1      1      1      1      1

    >>> to_timeseries_format(df, return_format="numpy3d")
    array([[[1, 0, 0],
            [1, 0, 0],
            [0, 1, 0]],
            [[0, 1, 1],
            [2, 1, 1],
            [0, 1, 1]]])

    >>> to_timeseries_format(df, return_format="pd-multiindex")
    variable       D  G  I
    instance time
    0        0     1  1  0
             1     0  0  1
             2     0  0  0
    1        0     0  2  0
             1     1  1  1
             2     1  1  1

    >>> to_timeseries_format(df, return_format="pd-list")
    [variable  D  G  I
     time
     0         1  1  0
     1         0  0  1
     2         0  0  0,
     variable  D  G  I
     time
     0         0  2  0
     1         1  1  1
     2         1  1  1]

    >>> to_timeseries_format(df, return_format="sorted")
    variable D     G     I
    time     0 1 2 0 1 2 0 1 2
    0        1 0 0 1 0 0 0 1 0
    1        0 1 1 2 1 1 0 1 1
    """
    pass
