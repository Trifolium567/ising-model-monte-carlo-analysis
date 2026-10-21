# %%
import numpy as np
import polars as pl
import scipy as sp
from matplotlib import pyplot as plt


# %%
def normaldist(x, variance, mu):
    """Function that reuturns the normal distribution with syntax for `polars.Series` objects
    
    Returns the result of the pdf of the normal distribution
    with a mean `mu` and variance `variance`. This function uses
    syntax for polars arrays. It returns the probability density
    of a normal distribution.
    
    Parameters
    ----------
    x : float | array_like
        The indepenedent variable that is normally distributed
    variance : float
        The variance of the normal distribution
    mu : float
        The mean of the normal distribution
    
    Returns
    -------
    The probability density of `x`
    """
    return 1 / np.sqrt(2*np.pi*variance) * np.exp(-np.power((x-mu), 2)/(2*variance))


def histogram_ish(data, density=True):
    """Function that returns data points which can be fitted the same way a histogram can.
    
    Returns the 'derivative' of the cumulative sum or the
    cumulative distribution function. The function is essentially
    taking bins of unequal length in such a way that avoids
    binning artefacts.

    This method is really only viable for data which is of a low
    resolution and with a high number of data points,
    where binning artefacts will have the greatest effect.

    Parameters
    ----------
    data : array_like
        Data, in an array-like object
    density : bool
        if `density=True`, the function will 'differentiate' the cumulative distribution function
        (instead of the cumulative sum function). `density` is set to True by default
    
    Returns
    -------
    hist_x : polars.Series
        the unique values in the `data` array
    hist_y : polars.Series
        equivalent to the frequency or density in a histogram
    """
    data = pl.Series(data)
    values, cumulative_sum = data.value_counts().sort(by="")
    
    cumulative_sum = cumulative_sum.cum_sum()
    
    if density: cumulative_sum /= data.describe()["value"][0]  # divide cum_sum_y by number of data points
                                                               # if density = True
    dx = values.diff()[1:]
    dy = cumulative_sum.diff()[1:]

    hist_y = dy/dx
    hist_x = (values[1:] + values[:-1])/2

    return hist_x, hist_y

def fit_normal(x_data, y_data, start_var, start_mu, plot=False):
    """Fit `data` to a normal probability density function.

    This fits given probability density `data` to the normal
    distribution using `scipy.optimise.curve_fit`.
    
    Parameters
    ----------
    x_data : array_like
        The independent varible that is normally distributed.
        This should be a 1-d array.
    y_data : array_like
        The calculated densities of each value.
    start_var : float
        The starting variance for the fit
    start_mu : float
        The starting mean value for the normal distribution fit
    plot : bool
        `False` by default. If set to `True`, the function plots
        the fit using matplotlib.pyplot and returns the
        `matplotlib.figure.Figure` and `matplotlib.axes._axes.Axes`
        in a list.

    Returns
    -------
    If `plot` is set to `False`, return:
        `variance, mu, covariance`
        
        Where variance and mu are the final fit parameters
        and covariance is a 2-d array of the covariances
        of the parameters.
    
    If `plot` is set to `True`, return:
        `variance, mu, covariance, ndarray[fig, ax]`
        
        where `ndarray[fig, ax]` is a numpy array of the 
        `matplotlib.figure.Figure` and `matplotlib.axes._axes.Axes`
        objects that make the plot.
    """
    popt, pcov = sp.optimize.curve_fit(normaldist,
                                       x_data, y_data,
                                       p0=[start_var, start_mu])
    variance, mu = popt
    x_data.sort()
    x = np.linspace(x_data[0], x_data[-1], 100)
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x_data, y_data, label="data", marker=".", color="blue")
        ax.plot(x, normaldist(x, variance=variance, mu=mu), label="fit", color="red")
        ax.legend()
        return variance, mu, pcov, np.array([fig, ax])
    # else:
    return variance, mu, pcov
# %%
