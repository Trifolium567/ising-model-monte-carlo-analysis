# %%
# import modules and set up random number generator
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

rng = np.random.default_rng(seed=1925347)
# generator is seeded to make results reproducible
# rng = np.random.default_rng()
# %%
# generate random data and count how many are inside
N = 1000
pi_estimates = pl.Series("pi_estimates", dtype=pl.Float64)

for i in range(10):
    _ = np.array([])
    for j in range(100000):
        # make arrays of N of random numbers
        points_x, points_y = rng.uniform(0, 1, size=(2, N))

        # check how many are within the circle
        in_circle = (points_x**2 + points_y**2) <= 1
        pi_estimate = 4*np.count_nonzero(in_circle)/N
        _ = np.append(_, pi_estimate)
    pi_estimates.append(pl.Series(_))

pi_estimates = pl.concat([pi_estimates], rechunk=True)
print(f"chunks = {pi_estimates.n_chunks()}")


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
# make cumulative distribution plot

# histograms introduce binning artefacts so try cumulative
# distribution function
# pi_estimates = pi_estimates.sort()
# counts = np.arange(1, pi_estimates.shape[0] + 1)
# edges_cdf = pl.Series(pi_estimates)  # copy the Series

values, frequencies = pi_estimates.value_counts().sort(by="pi_estimates")
counts_cdf = frequencies.cum_sum()

edges_cdf = pl.Series(values)
edges_cdf.append(pl.Series([edges_cdf[-1]]))
plt.stairs(counts_cdf, edges_cdf, fill=False)
plt.show()
# %%
# try to differentiate
plt.scatter(values, frequencies)

# %%
# np.savetxt("hundred-thousand-x-million.txt", pi_estimates)
# %%
plt.plot(values, frequencies)
plt.show()
# %%
