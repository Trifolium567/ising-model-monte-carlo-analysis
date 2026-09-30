# %%
# import modules and set up random number generator
import numpy as np
from matplotlib import pyplot as plt
rng = np.random.default_rng()

# %%
# help(plt.vlines)
# %%
N = 1000000
pi_estimates = []

for i in range(1000):
    # make arrays of N pairs of random numbers
    points_x, points_y = rng.uniform(0, 1, size=(2, N))

    # check how many are within the circle
    in_circle = (points_x**2 + points_y**2) <= 1
    pi_estimate = 4*np.count_nonzero(in_circle)/N
    pi_estimates.append(pi_estimate)

pi_estimates = np.array(pi_estimates)

# %%
plt.hist(pi_estimates)
plt.vlines(np.pi)
plt.show()
# %%
