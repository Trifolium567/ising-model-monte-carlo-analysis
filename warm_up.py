# %%
# import modules and set up random number generator
import numpy as np
import polars as pl
import scipy as sp
from matplotlib import pyplot as plt

rng = np.random.default_rng(seed=1925347)
# generator is seeded to make results reproducible
# rng = np.random.default_rng()
# %%
# generate random data and count how many are inside
n = 1000
pi_estimates = pl.Series("pi_estimates", dtype=pl.Float64)

for i in range(50):  # chunking makes it run faster
    _ = np.array([])
    for j in range(20000):
        # make arrays of N of random numbers
        points_x, points_y = rng.uniform(0, 1, size=(2, n))

        # check how many are within the circle
        in_circle = (points_x**2 + points_y**2) <= 1
        pi_estimate = 4*np.count_nonzero(in_circle)/n
        _ = np.append(_, pi_estimate)
    pi_estimates.append(pl.Series(_))

pi_estimates = pl.concat([pi_estimates], rechunk=True)
print(f"chunks = {pi_estimates.n_chunks()}")

N = int(pi_estimates.describe()["value"][0])
pi_estimates.describe()
# %%
# sort the data into equal size bins (histogram)
bins_n = 100
# pi_estimates = np.array(pi_estimates)
# counts, edges = np.histogram(pi_estimates, bins=bins_n, density=False)
hist_ = pi_estimates.hist(bin_count=bins_n, include_breakpoint=True)
# find start edge
start_edge = 2*hist_[0, "breakpoint"] - hist_[1, "breakpoint"]
breakpoints, _, counts = hist_
breakpoints = pl.Series([start_edge]).append(breakpoints)

# %%
# plot the data (histogram)
print(bins_n)
plt.stairs(counts, breakpoints, fill=True)
plt.axvline(np.pi, color="red", label="$\\pi$")
plt.legend()
plt.show()

# %%
# make cumulative sum plot

# histograms introduce binning artefacts so try cumulative
# distribution function
# pi_estimates = pi_estimates.sort()
# counts = np.arange(1, pi_estimates.shape[0] + 1)
# edges_cdf = pl.Series(pi_estimates)  # copy the Series

values, frequencies = pi_estimates.value_counts(parallel=True).sort(by="pi_estimates")
counts_cdf = frequencies.cum_sum()

edges_cdf = pl.Series(values)  # copy the Series
edges_cdf.append(pl.Series([edges_cdf[-1]]))
plt.stairs(counts_cdf, edges_cdf, fill=False)
plt.show()
# %%
# turn cumulative sum into cdf and 'differentiate'
dx = values.diff()[1:]
dy = (frequencies.cum_sum()/N).diff()[1:]
print(dx)
print(dy)
pdf_y = dy/dx

pdf_x = (values[1:] + values[:-1])/2

plt.scatter(pdf_x, pdf_y, marker=".")
plt.show()

 # %%
# np.savetxt("placeholder.txt", pi_estimates)
# %%
plt.scatter(values, frequencies, marker=".")
plt.show()
# %%
# Explanation:
# After talking to a supervisor (Rich Mason), found that this isn't really 'differentiation'.
# It is essentially instead of creating equal width bins, (which was causing artefacts)
# creating bins of unequal size tailored to where the data points actually appear.
# this avoids the binning artefacts.

# %%
# Fit a normal distribution to the curve
def normaldist(x, variance, mu):  # defining function to optimise for Polars Series (not numpy arrays)
    return 1 / np.sqrt(2*np.pi*variance) * ( -(x-mu).pow(2)/(2*variance) ).exp()

pdf_y[0]

popt, pcov = sp.optimize.curve_fit(normaldist, pdf_x, pdf_y, p0=[0.025, np.pi])

sigma_squared_pdf, mean_pdf = popt
plt.scatter(pdf_x, pdf_y, marker=".", color="red", label="data")
plt.plot(pdf_x, normaldist(pdf_x, sigma_squared_pdf, mean_pdf), label="fit")
plt.legend()
plt.show()

print("------------- fit to pdf -------------")
print(f"variance = {sigma_squared_pdf}")
sigma_pdf = np.sqrt(sigma_squared_pdf)
print(f"standard deviation = {sigma_pdf}")
print(f"mean = {mean_pdf}")

# %%
# fit a normal cdf to the cumulative distribution function
cdf_x, cdf_y = pi_estimates.value_counts(parallel=True).sort(by="pi_estimates")
cdf_y = cdf_y.cum_sum() / N
# turn what was a frequency into a cumulative sum and then a cumulative distribution (by dividing by N)

def normalcdf(x, variance, mu):
    transformed_x = (x - mu) / np.sqrt(variance)
    return sp.stats.norm.cdf(transformed_x)

popt, pcov = sp.optimize.curve_fit(normalcdf, cdf_x, cdf_y, p0=[0.025, np.pi])
sigma_squared_cdf, mean_cdf = popt
sigma_cdf = np.sqrt(sigma_squared_cdf)

print("---------- fit to cdf -----------")
print(f"variance = {sigma_squared_cdf}")
print(f"standard deviation = {sigma_cdf}")
print(f"mean = {mean_cdf}")

plt.scatter(cdf_x, cdf_y, marker=".", color="red", label="data")
plt.plot(cdf_x, sp.stats.norm.cdf((cdf_x - mean_cdf)/sigma_cdf), label="fit")
plt.legend()
plt.show()

# fit to the 'derivative' is more accurate
# This *might* be because it does not use an approximation
# and uses the analytical function for the pdf. (Instead of
# an approximation to the standard normal cdf)
# %%