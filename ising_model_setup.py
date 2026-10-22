import numpy as np


# rng = np.random.default_rng(1925347)
# create array to represent the system
# -1 to represent spin down, 1 to represent spin-up
def initialise_system(dim, edge_length, rng):
    """Create a 'hypercube' of spins of dimension `dim`
    and of edge length `edge_length` and return it as a `numpy.ndarray`
    of dtype `np.int_`. It uses the provided
    `numpy.random.Generator` object `rng` to generate the
    system. The system will be randomised to be of roughly
    equal spin-up (1) and spin-down (-1).
    """
    shape = np.full(dim, edge_length, dtype=np.int_)
    return rng.choice([-1, 1], size=shape)
    # return np.full(shape, 1)

def pbc(i, shape):
    """Returns the index, `i`, modulo the shape of the system, `shape`.
    This applies periodic boundary conditions."""
    return np.mod(i, shape)

def calculate_energy(system):
    """Return the energy of an n-d `system`
    
    Return the energy of an n-d `system` using the Ising Model,
    which assumes that the energy is the sum of the product
    of the spins of the nearest neighbours of each particle. 
    (multiplied by J, the bond strength, but that is not
    done in this function. This is assumed to be positive and
    constant f)

    Functionally, it returns the sum of the products of the
    unique combinations of the spins of the nearest neighbours
    of each particle and itself.

    Periodic boundary conditions are applied.

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
    ndim = system.ndim
    energy = 0
    if ndim == 1:
        energy = np.dot(np.roll(system, 1), system)
    else:
        for i in range(ndim):
            energy += np.tensordot(np.roll(system, 1, axis=i), system)
    return energy * -1

def check_energy(system, index):
    """Checks the energies that are influenced by the spin
    state that is at `index` in `system` using the Ising Model
    Hamiltonian. `index` must be a `numpy.ndarray` object"""
    ndim = system.ndim
    system_shape = system.shape
    Delta_E = 0  # This variable will not be used as Delta_E first,
                 # it will be used to calculate the energies that
                 # the chosen particle will influence
    _ = np.zeros(ndim, dtype=np.int8)
    for i in range(ndim):
        _[i] = 1
        Delta_E += system[*pbc(index + _, system_shape)]
        Delta_E += system[*pbc(index - _, system_shape)]
        _[i] = 0

    # calculate the sum of the energies that the chosen particle will influence
    Delta_E *= system[*index]
    Delta_E *= -1
    return Delta_E


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
        -   flip one 'spin' in `system` in place
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
    Delta_E, change_in_spin

    Delta_E: int
        The change in energies of the alteration in terms of
        the bond strengh, J.
    change_in_spin: int
        The change in spin of the system.
    """
    # generalise for an n-d system
    system_shape = system.shape

    # n_loops=1
    while True:
        index = rng.integers(system_shape, dtype=np.int_)

        Delta_E = check_energy(system, index)
        # if Delta_E greater than zero, flip will decrease energy,
        # so accept
        # if Delta_E is less than zero, flip will increase energy, so
        # accept with a certain probability
        prob_accept =  np.exp(2*Delta_E/temperature)
        if Delta_E > 0 \
            or (Delta_E < 0 and rng.random() <= prob_accept):  # if it gets to looping 100000 times, just accept the alteration
            Delta_E *= -2
            system[*index] *= -1
            # change_in_spin = system[*index] * 2 
            return Delta_E, system[*index] * 2
        # n_loops+=1
        # if Delta_E = 0 then loop agai

def metropolis_checkall(system, temperature, rng, level_of_patience=2):
    """Perform one iteration of a modified Metropolis Algorithm.
    

    This algorithm is designed to increase the acceptance rate of
    the Metropolis algorithm by getting forcing acceptance past a
    point. Instead of choosing a random index via the normal
    distribution, it shuffles an array containing the indexes of
    all the values in the system and iterates through that
    `level_of_patience` times before then forcing an alteration,
    including one that doesn't change the energy. This way, it
    gets to equilibrium faster.

    This function will:
        -   flip one 'spin' in `system` in place
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
    level_of_patience : int
        The number of times the function goes through
        the whole system before forcing an alteration.
        
    Returns
    -------
    Delta_E, change_in_spin

    Delta_E: int
        The change in energies of the alteration in terms of
        the bond strengh, J.
    change_in_spin: int
        The change in spin of the system.
    
    See also
    --------
    metropolis : the original algorithm that was modified
    """
    system_shape = system.shape
    index_array = np.array([i for i in np.ndindex(system_shape)])
    rng.shuffle(index_array)

    # for looping through the whole grid randomly
    # allows knowing that the whole system has been checked
    isupindex_not_saved = True
    for i in range(level_of_patience):
        for index in index_array:
            Delta_E = check_energy(system, index)
            prob_accept =  np.exp(2*Delta_E/temperature)
            
            if Delta_E > 0 \
                or (Delta_E < 0 and rng.random() <= prob_accept):
                Delta_E *= -2
                system[*index] *= -1
                # change_in_spin = system[*index] * 2 
                return Delta_E, system[*index] * 2
            elif isupindex_not_saved and Delta_E <= 0:
                # save index of first one that would go up
                # or stay the same
                upindex = index.copy()
                isupindex_not_saved = False
    # if nothing happened, system is stuck in local minimum.
    # The only thing that can be done is flip a state up
    # such that it increases in energy or does not change it
    # to try and get a configuration which can decrease
    # further in energy
        
    # flip the first one that would increase energy
    # (or keep it the same)
    Delta_E = check_energy(system, upindex)
    Delta_E *= -2
    system[*upindex] *= -1
    # change_in_spin = system[*index] * 2 
    return Delta_E, system[*upindex] * 2
