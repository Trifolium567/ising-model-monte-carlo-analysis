import numpy as np

# rng = np.random.default_rng(1925347)
# create array to represent the system
# -1 to represent spin down, 1 to represent spin-up

def calculate_energy(system):
    """Return the energy of an n-d `system`
    
    Return the energy of an n-d system using the Ising Model,
    which assumes that the energy is the sum of the product
    of the spins of the nearest neighbours of each particle. 
    (multiplied by J, the bond strength, but that is not
    done in this function. This is assumed to be positive and
    constant f)

    Functionally, it returns the sum of the products of the
    unique combinations of the spins of the nearest neighbours
    of each particle and itself.

    Parameters
    ----------
    system : numpy.ndarray
        The 1-d system whose energy is calculated.
        It should represent spin-up as `1` and spin-down as `-1`

    Returns
    -------
    energy : int
        The energy of the system as a multiple of J, the
        bond strength

    """
    system_shape = system.shape
    ndim = system.ndim
    energy = 0
    for i in range(ndim):
        cond_array_1 = np.full(system_shape[i], True)
        cond_array_1[0] = False
        cond_array_2 = cond_array_1[::-1]
        energy += np.tensordot(system.compress(cond_array_1, axis=i), system.compress(cond_array_2, axis=i), axes=ndim)
    return energy

def metropolis(system, temperature, rng):
    """Perform one iteration of the Metropolis Algorithm.
    
    The metropolis algorithm:
        1.  Makes an alteration to one randomly chosen
            particle
        2.  Calculates the change in energy that
            the alteration would have caused.
        3. Then if the energy:
            a)  decreases, always
                accept the change
            b)  does not change, always
                reject the change
            c)  increases, accept the
                change with a probability
                exp(- change_in_energy / (k_B * T ))
    
    This function will:
        -   will not make an alteration that keeps the energy the same
        -   flip one 'spin' in `system`
        -   return the change in energy due to that change
    
    Parameters
    ----------
    system : numpy.ndarray
        The system over which the algorithm is performed.
        It should be represented in an n-d array with
        `1` representing spin-up and `-1` representing
        spin-down
    temperature : float
        The temperature at which the system is in terms
        of the bond strength, J
    rng: numpy.random.Generator
        The Generator object to use to implement the
        probability aspect of the Metropolis Algorithm

    Returns
    -------
    Delta_E: numpy.ndarray
        The change in energies of the alteration in terms of
        the bond strengh, J.
    """
    # generalise for an n-d system
    system_shape = system.shape
    ndim = system.ndim
    
    while True:
        index = rng.integers(system_shape, dtype=np.int_)
        Delta_E = 0  # This variable will not be used as Delta_E first,
                     # it will be used to calculate the energies that
                     # the chosen particle will influence
        _ = np.zeros(ndim, dtype=np.int8)
        for i in range(ndim):
            _[i] = 1
            try: Delta_E += system[(*(index + _),)]
            except IndexError: pass
            if index[i] != 0:
                Delta_E += system[(*(index - _),)]
            _[i] = 0

        # calculate the sum of the energies that the chosen particle will influence
        Delta_E *= system[(*index,)]
        if Delta_E > 0:  # if Delta_E greater than zero, flip will decrease energy,
                         # so accept
            Delta_E *= -2
            system[*index] *= -1
            return Delta_E
        elif Delta_E < 0 and rng.random() >= np.exp(2*Delta_E/temperature):
            # if Delta_E is less than zero, flip will increase energy, so
            # accept with a certain probability
            Delta_E *=-2
            system[*index] *= -1
            return Delta_E
        # else, (if Delta_E = 0) then loop again
