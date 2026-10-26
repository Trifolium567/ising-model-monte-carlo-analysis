import numpy as np

# rng = np.random.default_rng(1925347)
# create array to represent the system
# -1 to represent spin down, 1 to represent spin-up

# Initialising and checking the state of the system
def pbc(i, shape):
    """Returns the index, `i`, modulo the shape of the system, `shape`.
    This applies periodic boundary conditions."""
    return np.mod(i, shape)

def initialise_system(dim, edge_length, rng, ground_state=False):
    """Create a 'hypercube' of spins of dimension `dim`
    and of edge length `edge_length` and return it as a `numpy.ndarray`
    of dtype `np.int_`. It uses the provided
    `numpy.random.Generator` object `rng` to generate the
    system. The system will be randomised to be of roughly
    equal spin-up (1) and spin-down (-1).
    If `ground_state` is set to True, it returns a system full of
    spin-down (-1) states.
    """
    # TODO: implement non-square system
    shape = np.full(dim, edge_length, dtype=np.int_)
    if ground_state: return np.full(shape, -1, dtype=np.int_)
    return rng.choice([-1, 1], size=shape)
    # return np.full(shape, 1)

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
    
    for i in range(ndim):
        energy += np.tensordot(np.roll(system, 1, axis=i),  system, axes=ndim)
    
    return energy * -1  # multiplied by -1 since -J in Hamiltonian

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
    Delta_E *= -1  # because it's -J in the Hamiltonian
    return Delta_E

def correlation_function(system):
    """Find the correlation function, G(r), of the `system`, which is an
    n-d `numpy.ndarray`, and return the result as two `numpy.ndarrays`
    in this format:
    ```
        [r values],
        [G(r) values]
    ```
    """
    shape = np.array(system.shape)
    ndim = system.ndim
    size = np.prod(shape)

    # the code below finds vectors that are in the top left
    # 'hypercube' section, including the halfway row.
    # e.g. for a 10 by 10 system (with the top left quadrant being the 'hypercube' section in 2d)
    # [[1 1 1 1 1 1 0 0 0 0]
    #  [1 1 1 1 1 1 0 0 0 0]
    #  [1 1 1 1 1 1 0 0 0 0]
    #  [1 1 1 1 1 1 0 0 0 0]
    #  [1 1 1 1 1 1 0 0 0 0]
    #  [1 1 1 1 1 1 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 0]
    #  [0 0 0 0 0 0 0 0 0 0]]
    # the 1's above represent the vectors chosen, with the index
    # of each 1 being the vector.
    # then it translates the system by these vectors and finds
    # the correlation for each spin to the corresponding spin
    # in the translated system and records their mean along with the radius
    # of the vector
    # This method is highly vectorised and should run fast
    translation_array = np.array([i for i in np.ndindex(*shape)])
    translation_array = translation_array[np.count_nonzero(((translation_array / (shape/2)) <= 1) == False, axis=1) == 0]
    translation_array = translation_array[1:]  # remove zero vector
    # translation_array = translation_array[np.sum((translation_array / (shape/2))**2, axis=1) <= 1]
    # _ = np.full(ndim, 1)
    # for i in range(ndim):
    #     _[i] = -1
    #     translation_array = np.append(translation_array, translation_array * _, axis=0)
    #     _[i] = 1

    G_function_raw = []
    for vector in translation_array:
        translated_system = system.copy()
        for i in range(ndim):  # this for loop shifts the system
            translated_system = np.roll(translated_system, shift=vector[i], axis=i)
        # calculate r, G(r) and then record them
        r = np.sqrt(np.sum(vector ** 2))
        G = np.tensordot(translated_system, system, axes=ndim) / size
        # If any combinations appear multiple times, it does not
        # affect the result as the mean is taken and they are condensed
        # into one data point in the calculation directly above.
        G_function_raw.append((r, G))

    # find unique r's and take the mean of their corresponding
    # G-values
    G_function_raw = np.array(G_function_raw)
    r_values_raw, G_values_raw = G_function_raw.T

    r_values = np.unique(r_values_raw)
    r_values.sort()
    # the r_values are sorted to make the result comparable to other results.
    G_values = []
    for r in r_values:
        _ = np.mean(G_values_raw[r_values_raw == r])
        G_values.append(_)        

    return np.array(r_values), np.array(G_values)



# Algorithms to compute the system
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
        # if Delta_E = 0 then loop again

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
