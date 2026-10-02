# %%
# import modules and set up random number generator
import numpy as np
from matplotlib import pyplot as plt

# rng = np.random.default_rng(seed=1925347)
# generator is seeded to make results reproducible
rng = np.random.default_rng()

# %%
help(np.cumsum)
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
# introduces binning artefacts
# %%
