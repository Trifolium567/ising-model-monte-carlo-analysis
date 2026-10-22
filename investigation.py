# %%
import numpy as np
from matplotlib import pyplot as plt

import ising_model_setup

rng = np.random.default_rng(1925347)

# %%
def compute_n_times(system, temperature, rng, algorithm, n, step=None):
    """Compute an `algorithm` over a `system` at a `temperature` `n` times.
    
    Computes the algorithm that is passed to the function over a system.
    The conditions of the system should be specified in the function.
    It returns either a `numpy.ndarray` of the energy and the magnetisation
    after each iteration or the energy and magnetisation after x number of
    iterations depending on what is passed to `step`.
    
    Parameters
    ----------
    n : int | numpy.ndarray[points, step]
        If nothing is passed to `step`:
            the function computes the `algorithm` `n` times.
        If something is passed to `step`:
            the function computes the algorithm `n * step` times,
            records the energy after `step` iterations and
            returns a `numpy.ndarray` containing all recorded 
            energy values.
    algorithm : function
        The algorithm function should be of signature:
        `f(system, temperature, rng) -> (Delta_E, Delta_M)`
        where `system` is the system, `temperature` is the temperature
        that the system is at and `rng` is a `numpy.random.Generator`
        object. `Delta_E` and `Delta_M` are the change in energy
        and magnetism of the system, respectively.
        It should perform only one iteration of the algorithm.
    system : numpy.ndarray
        The system of spins in a `numpy.ndarray` object. With `1`
        represpenting spin-up  and `-1` representing spin-down.
    temperature : float
        The temperature that the system is at, in terms of the bond strength, J
    rng : numpy.random.Generator
        The `numpy.random.Generator` object that will be passed to the algorithm.

    Returns
    -------
    return E_array, M_array where:
    If nothing passed to `step`:
        `E_array` and `M_array` are `numpy.ndarray` objects containing the
        starting energy and magnetism values and the values after each iteration.
        `len(E_array) = len(M_array) = n + 1`
    If an integer is passed to `step`:
        `E_array` and `M_array` are `numpy.ndarray` objects containing `n`
        energy and magnetism values with each data point being collected after
        `step` iterations.
        `len(compute_n_times(..., n, step)) = n`
    """
    E = ising_model_setup.calculate_energy(system)
    M = np.sum(system)

    if step:
        num_iterations = n * step
        E_list = []
        M_list = []
        for i in range(1, num_iterations+1):
            Delta_E, Delta_M = algorithm(system, temperature, rng)
            E += Delta_E
            M += Delta_M
            if i % step == 0:
                E_list.append(E)
                M_list.append(M)
        return np.array(E_list), np.array(M_list)
    
    # else:
    E_list = []
    M_list = []
    E_list.append(E)
    M_list.append(M)
    for i in range(n):
        Delta_E, Delta_M =  algorithm(system, temperature, rng)
        E += Delta_E
        M += Delta_M
        E_list.append(E)
        M_list.append(M)
    return np.array(E_list), np.array(M_list)

# %%
# Try to get a feel of the system:
edge_length = 100  # the grid of spins will be a 'hypercube' of side length `size`
dimension = 1
size = edge_length ** dimension
# initialise 1-d system:
store_initial = ising_model_setup.initialise_system(dimension, edge_length, rng)

# compute Metropolis algorithm at T = 1J and see the change in energy over time
iteration_number = size * 10

# %%
# use metropolis function
system_1d = store_initial.copy()
E_array, M_array = compute_n_times(system_1d, 1, rng,
                                   ising_model_setup.metropolis,
                                   iteration_number)

e_array = E_array / size
x = np.linspace(0, iteration_number, iteration_number+1)

fig, ax = plt.subplots()
ax.plot(x, e_array)
ax.set_title("Energy of system over the number of iterations\n(metropolis function)")
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Internal Energy per spin / $J$")
plt.show()

fig, ax = plt.subplots()
ax.plot(x, M_array)
ax.set_title("Magnetisation of system over the number of iterations\n(metropolis function)")
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Sum of spins")
plt.show()

# %%
# compare with metropolis_checkall function
system_1d = store_initial.copy()
E_array, M_array = compute_n_times(system_1d, 1, rng,
                                   ising_model_setup.metropolis_checkall,
                                   iteration_number)

e_array = E_array / size

fig, ax = plt.subplots()
ax.plot(x, e_array)
ax.set_title("Energy of system over the number of iterations\n(metropolis_checkall function)")
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Internal Energy per spin / $J$")
plt.show()

fig, ax = plt.subplots()
ax.plot(x, M_array)
ax.set_title("Magnetisation of system over the number of iterations\n(metropolis_checkall function)")
ax.set_xlabel("Number of Iterations")
ax.set_ylabel("Sum of spins")
plt.show()

# %%
# compute over T=0.01 J to T = 4 to see how it changes over time
num_T = 100  # the resolution over which to step over T
upper_T = 4
lower_T = 0.01

T_array = np.linspace(upper_T, lower_T, num_T)
# go backward because the system is disordered and does not
# have enough temperature to go to equilibrium if start at
# very low temperature.

e_over_T = []
magnetisation_over_T = []
c_V_over_T = []

for T in T_array:
    points = 100
    # allow the system to go to equilibrium
    compute_n_times(system_1d, T, rng,
                    ising_model_setup.metropolis_checkall,
                    3, size)
    # compute `size` times between recordings so that
    # one recording is not correlated to the next
    E_array, M_array = compute_n_times(system_1d, T, rng,
                               ising_model_setup.metropolis_checkall,
                               points, size)
    
    # Calculate the values that I need:
    E_bar = np.sum(E_array) / points  # estimate population mean from a sample
    E_var = np.sum((E_array - E_bar)**2) / (points - 1)  # estimate population variance from a sample
    c_V  = E_var / (size * T**2)  # specific heat capacity
    M = np.mean(np.abs(M_array))  # just to see what's going on

    # plt.show()
    e_over_T.append(E_bar)
    magnetisation_over_T.append(M)
    c_V_over_T.append(c_V)
    
    print(T)

e_over_T = np.array(e_over_T) / size
magnetisation_over_T = np.array(magnetisation_over_T) / size
c_V_over_T = np.array(c_V_over_T)

# %%
# Transfer Matrix solution for 1-d chain
T_array_tmat = np.linspace(upper_T, lower_T, 10000)
Z_tmat = []
for T in T_array_tmat:
    tmat = np.matrix([[np.exp(np.reciprocal(T)), np.exp(-np.reciprocal(T))],
                      [np.exp(-np.reciprocal(T)), np.exp(np.reciprocal(T))]])
    Z_tmat.append((tmat ** size).trace()[0,0])
print(Z_tmat)

Beta_array_tmat = np.reciprocal(T_array_tmat)
Z_tmat = np.array(Z_tmat)
e_tmat = - np.diff(np.log(Z_tmat)) / np.diff(Beta_array_tmat) / size
c_V_tmat = np.diff(e_tmat) / np.diff(T_array_tmat)[1:]


# %%
plt.scatter(T_array, e_over_T, marker=".", label="Monte Carlo")
plt.plot(T_array_tmat[1:], e_tmat, color="red", label="Analytical Solution")
plt.title("Internal Energy per Spin over Temperature")
plt.xlabel(r"Temperature / J ${k_B}^{-1}$")
plt.ylabel("Internal Energy per Spin / J")
plt.legend()
plt.show()
plt.scatter(T_array, magnetisation_over_T, marker=".", label="Monte Carlo")
plt.title("|M| over Temperature")
plt.xlabel(r"Temperature / J ${k_B}^{-1}$")
plt.ylabel("|Sum over the system| / size")
plt.legend()
plt.show()
plt.scatter(T_array, c_V_over_T, marker=".", label="Monte Carlo")
plt.plot(T_array_tmat[2:], c_V_tmat, color="red", label="Analytical Solution")
plt.title("Heat Capacity over Temperature")
plt.xlabel(r"Temperature / J ${k_B}^{-1}$")
plt.ylabel(r"Heat Capacity / $k_B$")
plt.legend()
plt.show()

# %%
# investigate 2-d system similarly:
edge_length = 10  # the grid of spins will be a 'hypercube' of side length `size`
dimension = 2
size = edge_length ** dimension
# initialise 1-d system:
system_2d = ising_model_setup.initialise_system(dimension, edge_length, rng)
E = ising_model_setup.calculate_energy(system_2d)
M = np.sum(system_2d)

# compute over T=0.01 J to T = 4 to see how it changes over time
num_T = 50  # the resolution over which to step over T

T_array = np.linspace(4,0.01, num_T)
# go backward because the system is disordered and does not
# have enough temperature to go to equilibrium if start at
# very low temperature.

e_over_T_2d = []
M_over_T_2d= []
c_V_over_T_2d = []

for T in T_array:
    points = 1000
    # allow the system to go to equilibrium
    compute_n_times(system_2d, T, rng,
                    ising_model_setup.metropolis_checkall,
                    3, size)
    # compute `size` times between recordings so that
    # one recording is not correlated to the next
    E_array, M_array = compute_n_times(system_2d, T, rng,
                               ising_model_setup.metropolis_checkall,
                               points, size)
    
    # Calculate the values that I need:
    E_bar = np.sum(E_array) / points  # estimate population mean from a sample
    E_var = np.sum((E_array - E_bar)**2) / (points - 1)  # estimate population variance from a sample
    c_V  = E_var / (size * T**2)  # specific heat capacity
    M = np.mean(np.abs(M_array))

    # plt.show()
    e_over_T_2d.append(E_bar)
    M_over_T_2d.append(M)
    c_V_over_T_2d.append(c_V)
    
    print(T)
# %%
# T_array = T_array[:60]
e_over_T_2d = np.array(e_over_T_2d) / size
M_over_T_2d = np.array(M_over_T_2d) / size
c_V_over_T_2d = np.array(c_V_over_T_2d)
plt.plot(T_array, e_over_T_2d)
plt.title("Internal Energy per Spin over Temperature")
plt.xlabel(r"Temperature / J ${k_B}^{-1}$")
plt.ylabel("Internal Energy per Spin / J")
plt.show()
plt.plot(T_array, M_over_T_2d)
plt.title("|M| over Temperature")
plt.xlabel(r"Temperature / J ${k_B}^{-1}$")
plt.ylabel("|Sum over the system| / size")
plt.show()
plt.plot(T_array, c_V_over_T_2d)
plt.title("Heat Capacity over Temperature")
plt.xlabel(r"Temperature / J ${k_B}^-1$")
plt.ylabel(r"Heat Capacity / $k_B$")
plt.show()
# %%
