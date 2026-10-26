# %%
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import ising_model_setup

rng = np.random.default_rng(1925347)

# %%
def compute_n_times(system, temperature, rng,
                    algorithm, n, step=None,
                    correlation_function=False):
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
    step :
            TODO: COMPLETE THIS
    correlation_function: 
            TODO: COMPLETE THIS
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
    if correlation_function:
        r_values, G_values = ising_model_setup.correlation_function(system)
        G_values_list = [G_values]
    
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
                if correlation_function:
                    _, G_values = ising_model_setup.correlation_function(system)
                    G_values_list.append(G_values)
        if correlation_function:
            G_values_list = np.array(G_values_list)
            G_values = np.mean(G_values_list, axis=0)
            if np.count_nonzero(G_values > 1):
                print(G_values)
            return np.array(E_list), np.array(M_list), r_values, G_values
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
        if correlation_function:
            _, new_G_values = ising_model_setup.correlation_function(system)
            G_values = G_values + new_G_values
    if correlation_function:
        G_values /= n
        if np.count_nonzero(G_values > 1):
            print(G_values)
        return np.array(E_list), np.array(M_list)
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

system_1d = ising_model_setup.initialise_system(dimension, edge_length, rng,
                                                ground_state=True)
T_array = np.linspace(lower_T, upper_T, num_T)
# go from 0.01 from a ground state to make sure that the ground state
# is represented

e_over_T = []
magnetisation_over_T = []
c_V_over_T = []

for T in T_array:
    points = 100
    # allow the system to go to equilibrium
    compute_n_times(system_1d, T, rng,
                    ising_model_setup.metropolis_checkall,
                    100, size)
    # compute `size` times between recordings so that
    # one recording is not correlated to the next
    E_array, M_array = compute_n_times(
        system_1d, T, rng,
        ising_model_setup.metropolis_checkall, points, size
    )
    
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

# ------------- Numerical Transfer Matrix Solution -------------
# Z_tmat = []
# for T in T_array_tmat:
#     tmat = np.matrix([[np.exp(np.reciprocal(T)), np.exp(-np.reciprocal(T))],
#                       [np.exp(-np.reciprocal(T)), np.exp(np.reciprocal(T))]])
#     Z_tmat.append((tmat ** size).trace()[0,0])

# The code above does not deal with large values very well.
# Try to use natural logarithms to multiply more accurately
Z_log_tmat = []
for T in T_array_tmat:
    tmat_log = np.array([[np.reciprocal(T), -np.reciprocal(T)],
                         [-np.reciprocal(T), np.reciprocal(T)]])
    end_mat = tmat_log.copy()
    tmat_log = tmat_log.T  # transpose the matrix

    for k in range(size-1): # matrix multiplication `size` times
        _ = end_mat.copy()
        for i, j in np.ndindex((2,2)):  # do matrix multiplication but addition of natural logarithms
            end_mat[i, j] = np.sum(np.exp(_[i] + tmat_log[j]))
        end_mat = np.log(end_mat)
    Z_log_tmat.append(np.trace(np.exp(end_mat)))


Beta_array_tmat = np.reciprocal(T_array_tmat)

# Z_tmat = np.array(Z_tmat)
# e_tmat = - np.diff(np.log(Z_tmat)) / np.diff(Beta_array_tmat) / size
# c_V_tmat = np.diff(e_tmat) / np.diff(T_array_tmat)[1:]

Z_log_tmat = np.array(Z_log_tmat)
e_log_tmat = - np.diff(np.log(Z_log_tmat)) / np.diff(Beta_array_tmat) / size
c_V_log_tmat = np.diff(e_log_tmat) / np.diff(T_array_tmat)[1:]

# ------------- Analytical Transfer Matrix Solution -------------
e_analytical = - np.tanh(Beta_array_tmat)
c_V_analytical = Beta_array_tmat ** 2 * np.cosh(Beta_array_tmat) ** -2

# %%
plt.scatter(T_array, e_over_T, marker=".", label="Monte Carlo")
plt.plot(T_array_tmat, e_analytical, color="green", label="Analytical Transfer Matrix Solution")
plt.plot(T_array_tmat[1:], e_log_tmat, color="red", label="Numerical Transfer Matrix Solution")
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
plt.plot(T_array_tmat, c_V_analytical, color="green", label="Analytical Transfer Matrix Solution")
plt.plot(T_array_tmat[2:], c_V_log_tmat, color="red", label="Numerical Transfer Matrix Solution")
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
num_T = 100  # the resolution over which to step over T

T_array = np.linspace(4,0.01, num_T)
# go backward because the system is disordered and does not
# have enough temperature to go to equilibrium if start at
# very low temperature.

e_over_T_2d = []
M_over_T_2d= []
c_V_over_T_2d = []
G_values_over_T_2d = []

for T in T_array:
    points = 100
    # allow the system to go to equilibrium
    compute_n_times(system_2d, T, rng,
                    ising_model_setup.metropolis_checkall,
                    3, size)
    # compute `size` times between recordings so that
    # one recording is not correlated to the next
    E_array, M_array, r_array, G_values = compute_n_times(
        system_2d, T, rng,
        ising_model_setup.metropolis_checkall,
        points, size, correlation_function=True
    )
    
    # Calculate the values that I need:
    E_bar = np.sum(E_array) / points  # estimate population mean from a sample
    E_var = np.sum((E_array - E_bar)**2) / (points - 1)  # estimate population variance from a sample
    c_V  = E_var / (size * T**2)  # specific heat capacity
    M = np.mean(np.abs(M_array))

    # plt.show()
    e_over_T_2d.append(E_bar)
    M_over_T_2d.append(M)
    c_V_over_T_2d.append(c_V)
    G_values_over_T_2d.append(G_values)
    
    print(T)
# %%
# T_array = T_array[:60]
e_over_T_2d = np.array(e_over_T_2d) / size
M_over_T_2d = np.array(M_over_T_2d) / size
c_V_over_T_2d = np.array(c_V_over_T_2d)
G_values_over_T_2d = np.array(G_values_over_T_2d)

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

print(G_values_over_T_2d)

plt.pcolormesh(r_array, T_array, G_values_over_T_2d)
plt.colorbar()
plt.title("G(r, T)")
plt.ylabel("Temperature, T / J $k_B ^{-1}$")
plt.xlabel("Distance, r")
plt.show()

plt.scatter(r_array, G_values_over_T_2d[T_array == 3.0327272727272727][0])
plt.title("G(r, 3 J $k_B^{-1}$)")
plt.ylabel("Correlation")
plt.xlabel("Distance, r")
plt.show()
# %%
# At high temperatures, the correlation functions falls off
# exponentially:
# G(r, T) ~ exp(- r / l(T)), where l is the correlation length
# this does not hold near the critical temperature, so only
# fit for the higher temperatures, 3-20
# use scipy to fit the temperatures to an exponential function.
# TODO: do another simulation with higher resolution at higher temperatures and use that.

edge_length = 10  # the grid of spins will be a 'hypercube' of side length `size`
dimension = 2
size = edge_length ** dimension
# initialise 1-d system:
system_2d = ising_model_setup.initialise_system(dimension, edge_length, rng)
E = ising_model_setup.calculate_energy(system_2d)
M = np.sum(system_2d)

# compute over T=0.01 J to T = 4 to see how it changes over time
num_T = 100  # the resolution over which to step over T

T_array = np.linspace(20, 3, num_T)
# go backward because the system is disordered and does not
# have enough temperature to go to equilibrium if start at
# very low temperature.

e_over_T_2d = []
M_over_T_2d= []
c_V_over_T_2d = []
G_values_over_T_2d = []

for T in T_array:
    points = 50
    # allow the system to go to equilibrium
    compute_n_times(system_2d, T, rng,
                    ising_model_setup.metropolis_checkall,
                    30, size)
    # compute `size` times between recordings so that
    # one recording is not correlated to the next
    E_array, M_array, r_array, G_values = compute_n_times(
        system_2d, T, rng,
        ising_model_setup.metropolis_checkall,
        points, size, correlation_function=True
    )
    
    # Calculate the values that I need:
    E_bar = np.sum(E_array) / points  # estimate population mean from a sample
    E_var = np.sum((E_array - E_bar)**2) / (points - 1)  # estimate population variance from a sample
    c_V  = E_var / (size * T**2)  # specific heat capacity
    M = np.mean(np.abs(M_array))

    # plt.show()
    e_over_T_2d.append(E_bar)
    M_over_T_2d.append(M)
    c_V_over_T_2d.append(c_V)
    G_values_over_T_2d.append(G_values)
    
    print(T)


# %%

e_over_T_2d = np.array(e_over_T_2d) / size
M_over_T_2d = np.array(M_over_T_2d) / size
c_V_over_T_2d = np.array(c_V_over_T_2d)
G_values_over_T_2d = np.array(G_values_over_T_2d)


fit_temperatures = T_array
# fit_temperatures = fit_temperatures[fit_temperatures <= 3.8]
def expected_correlation(r, l, a):
    return np.exp(- r / l)

lengths = []
for T in fit_temperatures:
    popt, pcov = sp.optimize.curve_fit(expected_correlation, r_array, G_values_over_T_2d[T_array == T][0])
    print(T, popt[0])
    # plt.scatter(r_array, G_values_over_T_2d[T_array==T][0])
    # plt.plot(r_array, expected_correlation(r_array, popt[0], popt[1]))
    # plt.show()
    lengths.append(popt[0])
lengths = np.array(lengths)
# plt.plot(r_array, expected_correlation(r_array, popt[0]))


# plt.plot(r_array, G_values_over_T_2d[T_array == 4][0])
# plt.show()
# %%
# fit lengths and temperatures to a power law
def power_law(x, a, b):
    return np.abs(x - a) ** b

popt, pcov = sp.optimize.curve_fit(power_law, fit_temperatures, lengths, p0=[2.25, -0.8])
x = np.linspace(3, 20, 1000)

plt.scatter(fit_temperatures, lengths, marker=".", label="data")
plt.plot(x, power_law(x, popt[0], popt[1]), label=f"fit, T_c = {popt[0]}, \n$\\alpha$ = {popt[1]}", color="red")
plt.title("Correlation Length over temperature fit to |T - T$_c$|$^\\alpha$")
plt.legend()
print(popt)
plt.show()
plt.title("Correlation Length over temperature fit to |T - T$_c$|$^\\alpha$")
plt.loglog(fit_temperatures, lengths, marker=".", linestyle="", label="data")
plt.loglog(x, power_law(x, popt[0], popt[1]), label=f"fit, T_c = {popt[0]}, \n$\\alpha$ = {popt[1]}", color="red")
plt.legend()
plt.show()
# %%
