# %%
# import modules and set up random number generator
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

# rng = np.random.default_rng(seed=1925347)
# generator is seeded to make results reproducible
rng = np.random.default_rng()

# %%
help(np.reciprocal)
# %%
# generate random data and count how many are inside
N = 10000
pi_estimates = []

for i in range(100000):
    # make arrays of N of random numbers
    points_x, points_y = rng.uniform(0, 1, size=(2, N))

    # check how many are within the circle
    in_circle = (points_x**2 + points_y**2) <= 1
    pi_estimate = 4*np.count_nonzero(in_circle)/N
    pi_estimates.append(pi_estimate)

pi_estimates = np.array(pi_estimates)

# %%
# sort the data into equal size bins
bins_n = 50
counts, edges = np.histogram(pi_estimates, bins=bins_n, density=False)
# %%
# plot the data
print(bins_n)
plt.stairs(counts, edges, fill=True)
plt.axvline(np.pi, color="red", label="$\\pi$")
plt.legend()
plt.show()
# %%
# make cumulative distribution plot

# histograms introduce binning artefacts so try cumulative
# distribution function
pi_estimates = np.sort(pi_estimates)
counts = np.arange(1, pi_estimates.shape[0]+1)
edges = np.append([pi_estimates[0]], pi_estimates)
plt.stairs(counts, edges, fill=True)
plt.show()
# %%
# try to differentiate
counts_histish = np.reciprocal(np.diff(edges))
plt.plot(edges[:-1], counts_histish)

# %%
