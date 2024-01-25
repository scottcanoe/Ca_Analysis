from numbers import Number
from typing import (
    Any, Callable,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.random
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import xarray as xr
from .common import ArrayLike, is_int



__all__ = [

    # Random Sampling
    "sample",
    "shuffled",

    # Bootstrapping, density estimation, etc.
    "bootstrap",
    "block_bootstrap",
    "gaussian_kde",

    # Model fitting
    "LinearSVC",
    "one_hot",
    "train_test_split",

    # Hypothesis testing
    "ttest",

    # etc.
    "autocorr",
    "normalize_confusion_matrix",
    "normalize_confusion_matrices",    
]


#------------------------------------------------------------------------------#
# Random sampling




def sample(
    arr: ArrayLike,
    size: int,
    replace: bool = True,
    axis: Optional[Union[int, str]] = None,
    dim: Optional[str] = None,
    ) -> Union[np.ndarray, xr.DataArray]:

    """
    Generates a random sample from a collection.

    By default, samples are drawn from the flattened input data, but the ``axis``
    parameters allows subarrays (such as rows or columns) to be sampled when the
    input is multidimensional. Output types match input types when called with a
    ``numpy.ndarray``, ``pandas.Series``, or an ``xarray.DataArray``.

    When sampling from a ``xarray.DataArray``, ``axis`` may be specified as an integer
    (axis number) or a string (dimension name)). Otherwise,
    the returned data array should contain all relevant dimension and coordinate
    information from the input data.

    Sampling from all other container types (e.g., ``set``, ``list``,
    ``tuple``, etc.)
    will result in a list.


    Parameters
    ----------

    arr : container-like
        Container from which samples will be drawn.
    size : int
        Number of samples to draw.
    replace : bool (optional)
        Whether to sample with replacement.
    axis : int (optional)
        Optionally sample subarrays (such a rows or columns) when input is
        multidimensional.
    dim : int (optional)
        Optionally sample subarrays (such a rows or columns) when input is
        multidimensional.



    """
    
    # Handle dataarray
    if isinstance(arr, xr.DataArray):
        if dim is None:
            data = arr.data.ravel()
            inds = numpy.random.choice(data.size, size, replace=replace)
            return data[inds]
        
        inds = numpy.random.choice(arr.sizes[dim], size, replace=replace)
        return arr.isel(dim=inds)
        
                            
    # Handle all others
    arr = np.asanyarray(arr)
    if axis is None:
        data = arr.ravel()
        inds = numpy.random.choice(data.size, size, replace=replace)
        return data[inds]
    
    inds = numpy.random.choice(arr.shape[axis], size, replace=replace)
    slc = [None] * arr.ndim
    slc[axis] = inds
    
    out = arr[tuple(slc)]
    return out


def shuffled(arr: ArrayLike) -> ArrayLike:
    """
    Shuffle the elements of a 1-d array.
    """
    arr = np.asarray(arr)
    N = len(arr)
    ixs = np.random.choice(N, N, replace=False)
    return arr[ixs]


#------------------------------------------------------------------------------#
# Bootstrapping, density estimation, etc.


def bootstrap(
    data: Union[ArrayLike, Set],
    stat: Callable,
    n_iter: int = 1000,
    q: Sequence[Number] = (2.5, 50.0, 97.5),
    ) -> Tuple[np.ndarray, float]:

    """
    Returns confidence interval/percentile information describing the distribution
    of bootstrap-resampled data. Multidimensional input will be treated as if 1-d.


    Parameters
    ----------

    data : array-like, set-like
        Container from which samples will be drawn.
    stat : callable
        Statistic to being bootstrapped.
        If the ``axis`` argument is provided, then ``stat`` must be a callable capable
        of taking the argument ``axis=0``.
    n_iter : int (optional)
        Number of sampling iterations.


    Returns
    -------

    ptiles : np.ndarray
       Percentiles corresponding to ``q``.

    mu : np.ndarray
        Mean of bootstrapped distribution.
    """

    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (pd.Series, xr.DataArray)):
        arr = data.values
    elif isinstance(data, Set):
        arr = np.array(list(data))
    else:
        arr = np.array(data)

    # Perform resampling, computing statistics along the way.
    arr = arr.reshape(-1) if arr.ndim > 1 else arr
    scores = np.array([stat(_sample_dim_0(arr, len(arr))) \
                       for _ in range(n_iter)])

    # Produce summary.
    ptiles = np.percentile(scores, q)
    mu = np.mean(scores)

    return ptiles, mu



def block_bootstrap(
    data: Union[np.ndarray, xr.DataArray],
    stat: Callable,
    axis: Union[int, str] = 0,
    n_iter: int = 1000,
    q: Sequence[Number] = (2.5, 50.0, 97.5),
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:


    assert isinstance(data, (np.ndarray, xr.DataArray))
    assert data.ndim == 2
    arr = data.values if isinstance(data, xr.DataArray) else arr

    sample_axis, sample_dim = _handle_axis_arg(data, axis)
    if sample_axis < 0:
        sample_axis += 2
    if sample_axis not in (0, 1):
        raise ValueError(f"Invalid axis '{axis}'")
    if sample_axis == 1:
        arr = arr.T

    # Perform resampling.
    n_rows, n_cols = arr.shape[0], arr.shape[1]

    scores = np.zeros([n_iter, n_cols])

    for i in range(n_iter): 

        # Sample from blocks, yielding a matrix the same shape as `arr`.
        resamples = _sample_dim_0(arr, n_rows)

        # Compute statistic down the first axis.
        sc = stat(resamples, axis=0)

        # Store the score vector.
        scores[i] = sc

    # Compute summaries.
    ptiles = np.percentile(scores, q, axis=0)
    mu = np.mean(scores, axis=0)

    # If data array, return data arrays with correct dimensions.
    if isinstance(data, xr.DataArray):
        stat_axis = (sample_axis + 1) % 2
        stat_dim = data.dims[stat_axis]

        ptiles = xr.DataArray(ptiles, dims=('q', stat_dim))
        ptiles.coords['q'] = np.array(q)
        if stat_dim in data.coords:
            ptiles.coords[stat_dim] = data.coords[stat_dim]

        mu = xr.DataArray(mu, dims=(stat_dim, ))
        if stat_dim in data.coords:
            mu.coords[stat_dim] = data.coords[stat_dim]

    return ptiles, mu



def _handle_axis_arg(
    data: Union[Sequence, np.ndarray, pd.Series, xr.DataArray],
    axis: Optional[Union[int, str]],
    ) -> Tuple[Optional[int], Optional[str]]:


    if axis is None:
        return None, None

    if isinstance(data, np.ndarray):
        if is_int(axis):
            return axis, None
        raise ValueError(f"Invalid axis argument '{axis}' for ndarray")

    if isinstance(data, xr.DataArray):
        if is_int(axis):
            return axis, data.dims[axis]
        elif isinstance(axis, str):
            return data.get_axis_num(axis), axis
        else:
            raise ValueError(f"Invalid axis argument '{axis}' for DataArray")

    if is_int(axis):
        return axis, None

    raise ValueError(f"Invalid axis argument '{axis}'for object of type {type(data)}")


def gaussian_kde(
    data: ArrayLike,
    X: Optional[ArrayLike] = None,
    N: int = 1000,
    bw_method: Optional[Union[str, Number, Callable]] = None,
    plot: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:

    """
    Generate a kernel-density estimate from a dataset. Useful for quickly
    creating x- and y- arrays used for plotting on top (or in place of)
    a histogram. This function utilizes scipy.stats.gaussian_kde.


    Parameters
    ----------

    data : array-like
        Univariate datapoints to estimate from.
    X : array-like, optional
        The domain of the kernel estimate's pdf. By default, an `X` will
        be generated that evenly spans the entire range of `arr` in `N`-1
        segments. For more control, `X` can be passed in directly.
    N : int, optional
        If generating `X` (see above), then it will be an `N`-length array
        that spans the entire range of `arr`.
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See scipy.stats.gaussian_kde
        for details.
    plot : bool, optional
        Optionally show a plot of the kernel-density estimate. This is a
        convenience.


    Returns
    -------

    X, Y : 2-tuple of ndarrays
        Support and pdf of the kernel density estimate.


    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> a = np.random.randn(100)
    >>> X, Y = KDE(a)
    >>> fig, ax = plt.subplots()
    >>> ax.hist(a, density=True)
    >>> ax.plot(X, Y)
    >>> plt.show()
    >>> # Or just plot quickly for a one-off visualization.
    >>> KDE(a, plot=True)
    """



    arr = np.asarray(data)
    kde = scipy.stats.gaussian_kde(arr, bw_method=bw_method)

    if X is None:
        X = np.linspace(np.min(arr), np.max(arr), N)
    else:
        X = np.asarray(X)

    Y = kde.pdf(X)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(X, Y)
        plt.show()

    return X, Y


#------------------------------------------------------------------------------#
# Model fitting


def one_hot(
    df: pd.DataFrame,
    feature: str,
    values: Optional[ArrayLike] = None,
    ) -> pd.DataFrame:
    """
    Return a DataFrame with a categorical column replaced by one-hot encoded
    columns. The new columns will be inserted where the old column was.


    Parameters
    ----------

    df : pandas.DataFrame
        Data containing a categorical column.
    feature : str
        Name of column to be replaced.
    values : array-like (optional)
        By default, the unique values in the categorical column will be used
        to construct new columns. If these values are insufficient, they
        can be determined manually through this parameter. See the examples
        for clarification.


    Returns
    -------

    new_df : pandas.DataFrame
        A copy of the original data frame but with the categorical column
        replaced by one-hoted encoded columns.


    Examples
    --------

    >>> df = pd.DataFrame({'sex' : ['male', 'male', 'female'],
    ...                    'age' : [24, 58, 50]})
    >>> df
    ...       sex  age
    ... 0    male   24
    ... 1    male   58
    ... 2  female   50
    ...
    >>> df = one_hot(df, 'sex')
    >>> df
    ...    female  male  age
    ... 0       0     1   24
    ... 1       0     1   58
    ... 2       1     0   50
    >>> # Let's say the data happens to not include females, but we
    >>> # still want a female column.
    >>> df = pd.DataFrame({'sex' : ['male', 'male'],
    ...                    'age' : [24, 58]})
    >>> df = one_hot(df, 'sex', values=('female', 'male'))
    >>> df
    ...    female  male  age
    ... 0       0     1   24
    ... 1       0     1   58

    """

    # Get categorical sequence, and infer its possible values
    # if necessary. Also make sure values are unique, and sort
    # them for convenience.ensure uniqueness and sort
    feature_arr = df[feature].values
    if values is None:
        values = np.sort(np.unique(feature_arr))
    else:
        unique_vals = np.unique(values)
        if len(values) > len(unique_vals):
            values = np.sort(unique_vals)

    # Prepare return data frame.
    out = df.copy(deep=False)
    del out[feature]

    # Append one-hot columns to the data frame.
    for val in values:
        out[val] = (feature_arr == val).astype(int)
    return out


#------------------------------------------------------------------------------#
# Hypothesis testing


def ttest(
    a: ArrayLike,
    b: Any,
    *args,
    paired: bool = False,
    **kw,
    ) -> Tuple[float, float]:
    """
    Perform a one-sample t-test or paired or unpaired two-sample t-tested.


    Returns
    -------

    t, p : tuple
        The t-static and corresponding p-value.

    """

    a = np.asanyarray(a)

    # 1-sample t-test
    if np.isscalar(b):
        return scipy.stats.ttest_1samp(a, b, *args, **kw)

    # paired 2-sample t-test
    if paired:
        return scipy.stats.ttest_rel(a, b, *args, **kw)

    # unpaired 2-sample t-test
    return scipy.stats.ttest_ind(a, b, *args, **kw)



#------------------------------------------------------------------------------#
# etc.


def autocorr(arr: ArrayLike, lag: int = 1) -> Number:
    """
    Compute pearson autocorrelation coefficient
    """
    
    return scipy.stats.pearsonr(arr[:-lag], arr[lag:])[0]


def normalize_confusion_matrix(cm: np.ndarray, axis: int = 1) -> np.ndarray:
    cm = np.asarray(cm, float)
    return cm / cm.sum(axis=axis)[:, np.newaxis]


def normalize_confusion_matrices(mat: np.ndarray, axis: int = 1) -> np.ndarray:
    
    """
    Batch normalization of confusion matrices.
    """
    
    lst = [normalize_confusion_matrix(mat[i], axis) for i in range(len(mat))]
    return np.stack(lst)
