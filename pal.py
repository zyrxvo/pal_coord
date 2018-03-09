import rebound
import numpy as np
import numpy.polynomial.chebyshev as CB
import numpy.polynomial.polynomial as poly
from ctypes import c_double, pointer
import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

#####################################################################################################################
############################################# PAL FUNCITONS #########################################################
#####################################################################################################################

# Checks whether or not a simulation is currently in the centre of mass frame by 
#     comparing the values to a specified tolerance.
def sim_is_in_com(_sim):
    com = _sim.calculate_com()
    tolerance = 1.0e-15
    x = com.x < tolerance
    y = com.y < tolerance
    z = com.z < tolerance
    vx = com.vx < tolerance
    vy = com.vy < tolerance
    vz = com.vz < tolerance
    return x and y and z and vx and vy and vz


# Calculates x, where x = m (mod n), offset by an amount d such that d <= x < d + n.
def mod(m, n, d):
    assert not(n==0), 'Dividing by zero is undefined.'
    return m - n*((m - d)//n)


#  Converts a simulation of particles to Pal coordinates where the first list of parameters are the cartesian 
#     coordinates for the centre of mass of the simulation.
def convert_sim_to_pal(_sim):
#     Asserts that the object passed in is the correct type.
    assert type(_sim) == type(rebound.Simulation()), 'The given object is not of type: rebound.Simulation'

    particlesList = _sim.particles
#     Asserts that there is at least one particle in the system.
    assert len(particlesList) > 0, 'The list of particles in the given simulation was empty.'
    
    Nparams = 7 # The mass and six Pal coordinates (a, l, k, h, ix, iy).
    pal_label = ['m','a','l','k','h','ix','iy']
    
#     List with Cartesian coordinats of the centre of mass followed by the particles' Pal coordinates.
    pal = np.zeros((len(particlesList), Nparams))
    
    if not sim_is_in_com(_sim): _sim.move_to_com()
    
#     Extract the centre of mass position and velocity as a particle, but use the star's mass.
    com_p = _sim.calculate_com()
    star = particlesList[0]
    com_star_particle = np.array([star.m, com_p.x, com_p.y, com_p.z, com_p.vx, com_p.vy, com_p.vz])
    pal[0] = com_star_particle

#     Create a simulation to convert coordinates using Jacobi coordinates.
    temp_sim = rebound.Simulation()
    G = c_double(temp_sim.G)
#     Add the star to the simulation for Jacobi coordinates.
    temp_sim.add(star)

#     Convert remaining particles to Pal coordinates (skipping the star).
    for i,p in enumerate(particlesList[1:]):
#         Initialize the Pal coordinates for next particle.
        a = c_double(0)
        lamda = c_double(0)
        k = c_double(0)
        h = c_double(0)
        ix = c_double(0)
        iy = c_double(0)
#         Initialize pointers to Pal coordinates (denoted here by the underscore).
        a_ = pointer(a)
        lamda_ = pointer(lamda)
        k_ = pointer(k)
        h_ = pointer(h)
        ix_ = pointer(ix)
        iy_ = pointer(iy)
        
#         Calculate the centre of mass (for Jacobi coordinates) to get Pal orbital elements.
        primary = temp_sim.calculate_com()
        rebound.clibrebound.reb_tools_particle_to_pal(G, p, primary, a_, lamda_, k_, h_, ix_, iy_)
        
#         Extract values from ctype objects for use in python.
        pal_particle = np.array([p.m, a.value, mod(lamda.value, 2*np.pi, -np.pi), k.value, h.value, ix.value, iy.value])
    
#         Assert that the returned values are valid numbers.
        for j in range(Nparams): assert not np.isnan(pal_particle[j]), 'Fatal Error. Returned value for Object '\
            + str(i + 1) + ', parameter \'' + pal_label[j] + '\'' ' is not a number.'
    
#         Add Pal coordinates of particle to list and then add particle to simulation for COM calculation.
        pal[i + 1] = pal_particle
        temp_sim.add(p)

#     Returns a list of lists with the particles in the order given, now in Pal coordinates.
#      The attributes are listed as [mass, a, l, k, h, ix, iy] for each particle (except the star/COM).
#      The attributes of the star/COM (the first element in the list) is listed as [m, x, y, z, vx, vy, vz].
    return pal


# Given a list of Pal coordinates (where the first list of parameters is the centre of mass in cartesian coordinates),
#     this returns a simulation of particles in the centre of mass frame.
def convert_pal_to_sim(listOfPalCoordinates):
#     Must have a non-empty list.
    assert len(listOfPalCoordinates) > 0, 'The given list was empty.'
#     Create a temporty simulation to extract the particles from.
    temp_sim = rebound.Simulation()
#     Extract the center of mass from list and add to simulation.
    com = listOfPalCoordinates[0]
    com_p = rebound.Particle(m=com[0], x=com[1], y=com[2], z=com[3], vx=com[4], vy=com[5], vz=com[6])
    
#     Add the first particle at a random position (zero is as good as any).
    temp_sim.add(com_p)
    
#     Convert remaining particles into cartesian coordinates.
    for p in listOfPalCoordinates[1:]:
        temp_sim.add(primary=temp_sim.calculate_com(), m=p[0], a=p[1], l=p[2], k=p[3], h=p[4], ix=p[5], iy=p[6])

#     Reposition the system according the the centre of mass information.
    temp_sim.move_to_com()
    for p in temp_sim.particles:
        p.x = p.x - com_p.x
        p.y = p.y - com_p.y
        p.z = p.z - com_p.z
        p.vx = p.vx - com_p.vx
        p.vy = p.vy - com_p.vy
        p.vz = p.vz - com_p.vz
    return temp_sim


# Given a simulation, this returns an np.array() of the mass and cartesian coordinates of the centre of mass
#     and each of the particles in the simulation.
def extract_cartesian(_sim):
    assert type(_sim) == type(rebound.Simulation()), 'The given object is not of type: rebound.Simulation'
    particles = _sim.particles
    assert len(particles) > 0, 'There are no particles in the simulation to get coordinates from.'
    Nparams = 7 # The mass and six Cartesian coordinates (x, y, z, vx, vy, vz).
    cart = np.zeros((len(particles), Nparams))
    
#     Checks if the simulation is in the centre of mass frame.
    if not sim_is_in_com(_sim): _sim.move_to_com()
    
    com_p = _sim.calculate_com()
    star_mass = particles[0].m
    com_star_particle = np.array([star_mass, com_p.x, com_p.y, com_p.z, com_p.vx, com_p.vy, com_p.vz])
    cart[0] = com_star_particle
    
    for i,p in enumerate(particles[1:]):
        cart_coord = np.array([p.m, p.x, p.y, p.z, p.vx, p.vy, p.vz])
        cart[i + 1] = cart_coord
    
    return cart


# Defines a metric with which to check the overall error build up.
def calc_err(measured, exact):
    return np.abs( np.linalg.norm(measured - exact) / np.linalg.norm(exact))

def calc_single_err(measured, exact):
    return np.abs(measured - exact)/np.abs(exact)

def instant_metric(_pal_coord, _cart_coord):
    Nbodies = len(_pal_coord)
    summation = 0
    new_cart = extract_cartesian(convert_pal_to_sim(_pal_coord[:,:]))
    for i in range(1, Nbodies - 1):
        err_r = calc_err(new_cart[i,1:4], _cart_coord[i,1:4])
        err_v = calc_err(new_cart[i,4:], _cart_coord[i,4:])
        summation = summation + err_r + err_v
    return summation

def monty_metric(_pal_coord, _cart_coord):
    assert len(_pal_coord) > 0, 'The given list of Pal coordinates is empty.'
    assert len(_cart_coord) > 0, 'The given list of cartesian coordinates is empty.'
    assert _pal_coord.shape == _cart_coord.shape, 'The given lists of coordinates have differing shapes.'
    n_samples = 1000
    rnd_ind = np.random.choice(np.arange(len(_pal_coord)), n_samples, replace=False)
    rnd_ind = np.sort(rnd_ind)
    _pal_sample = _pal_coord[rnd_ind]
    _cart_sample = _cart_coord[rnd_ind]
    
    order_parameter = np.zeros(n_samples)
    for t in range(n_samples):
        order_parameter[t] = instant_metric(_pal_sample[t], _cart_sample[t])
    
    return np.mean(order_parameter)
    

def metric(_pal_coord, _cart_coord):
    assert len(_pal_coord) > 0, 'The given list of Pal coordinates is empty.'
    assert len(_cart_coord) > 0, 'The given list of cartesian coordinates is empty.'
    assert len(_pal_coord) == len(_cart_coord), 'The given lists of coordinates have differing lengths.'
    Ntime = len(_pal_coord)
    Nbodies = len(_pal_coord[0])
    
    order_parameter = np.zeros(Ntime)
    for t in range(Ntime):
        summation = 0
        new_cart = extract_cartesian(convert_pal_to_sim(_pal_coord[t,:,:]))
        for i in range(1, Nbodies - 1):
            err_r = calc_err(new_cart[i,1:4], _cart_coord[t,i,1:4])
            err_v = calc_err(new_cart[i,4:], _cart_coord[t,i,4:])
            summation = summation + err_r + err_v
        order_parameter[t] = summation
    return order_parameter


# Calculates the difference between the Pal coordinates and cartesian coordinates.
def coord_diff(_pal_coord, _cart_coord):
    assert len(_pal_coord) > 0, 'The given list of Pal coordinates is empty.'
    assert len(_cart_coord) > 0, 'The given list of cartesian coordinates is empty.'
    assert _pal_coord.shape == _cart_coord.shape, 'The given lists have differing shapes.'
    
    # Extract the shape of the array and pre-allocate a new numpy array.
    (Ntime, Nbodies, Nparam) = _pal_coord.shape
    differ = np.zeros((Ntime, Nbodies, Nparam))
    
    # For each time step, convert the Pal coordinates to cartesian coordinates.
    for t in range(Ntime):
        new_cart = extract_cartesian(convert_pal_to_sim(_pal_coord[t]))
        diff_parts = np.zeros((Nbodies, Nparam))
        # For each particle in the time step, compare the conversion differences.
        for j in range(Nbodies):
            diff_parts[j] = new_cart[j] - _cart_coord[t][j]
        differ[t] = diff_parts

    # Return an numpy array of the comparision between Pal and cartesian coordinates.
    #    The shape of the array is the same as the shape of the two given numpy arrays.
    return differ

# Given a set of Pal coordinates, this function removes the restricted domain of the mean longitude
#     so that it is a monotonically increasing linear function. It does this for all particles in the system.
def unmod_mean_longitude(_pal_coord):
    assert len(_pal_coord) > 0, 'The given list of coordinates is empty.'
    (Ntime, Nbodies, Nparam) = _pal_coord.shape
    assert Ntime > 1, 'More than one time step is needed.'
    
    # Copies the old coordinates instead of changing the given ones.
    # If the old set of coordinates are no longer required, then this code can easily be 
    #    refactored to save space. Simply remove the declaration of the new_pal variable 
    #    and replace all other instances of new_pal with _pal_coord.
    new_pal = np.copy(_pal_coord)
    
    # Index for mean longitude parameter.
    _l = 2
    
    # Iterate over each particle for all time steps and add 2*Pi*n to values.
    for i in range(1, Nbodies):
        n = 0
        prev = new_pal[0, i, _l]
        curr = new_pal[1, i, _l]
        for t in range(1,Ntime):
            if curr < prev: n = n + 1
            prev = curr
            new_pal[t, i, _l] = curr + 2 * np.pi * n
            if t < Ntime - 1: curr = new_pal[t + 1, i, _l]
    
    return new_pal


# Given a set of Pal coordinates, this function reapplies the restricted domain of the mean longitude
#     so that it is a sawtooth function. It does this for all particles in the system.
def remod_mean_longitude(_pal_coord):
    assert len(_pal_coord) > 0, 'The given list of coordinates is empty.'
    (Ntime, Nbodies, Nparam) = _pal_coord.shape
    assert Ntime > 1, 'More than one time step is needed.'
    
    # Copies the old coordinates instead of changing the given ones.
    # If the old set of coordinates are no longer required, then this code can easily be 
    #    refactored to save space. Simply remove the declaration of the new_pal variable 
    #    and replace all other instances of new_pal with _pal_coord.
    new_pal = np.copy(_pal_coord)
    
    # Index for mean longitude parameter.
    _l = 2
    
    # Iterate over each particle for all time steps and add 2*Pi*n to values.
    for i in range(1, Nbodies):
        for t in range(1,Ntime):
            new_pal[t, i, _l] = mod(new_pal[t, i, _l], 2 * np.pi, -np.pi)
    
    return new_pal


#####################################################################################################################
############################################# PLOTTING ##############################################################
#####################################################################################################################

# Used for printing coordinates out nicely.
def print_particle_nicely(p):
    print '[{0:13e}, {1:13e}, {2:13e}, {3:13e}, {4:13e}, {5:13e}, {6:13e}]'.format(p.m,p.x,p.y,p.z,p.vx,p.vy,p.z)
def print_coord_nicely(p):
    print '[{0:13e}, {1:13e}, {2:13e}, {3:13e}, {4:13e}, {5:13e}, {6:13e}]'.format(p[0],p[1],p[2],p[3],p[4],p[5],p[6])
def print_labels_nicely(p):
    print '[{0:13}, {1:13}, {2:13}, {3:13}, {4:13}, {5:13}, {6:13}]'.format(p[0],p[1],p[2],p[3],p[4],p[5],p[6])
def print_time_nicely(t):
    r,s = divmod(t,60)
    r,m = divmod(r,60)
    r,h = divmod(r,24)
    if h == 0 and m == 0: return '{0:.6g}s'.format(s)
    if h == 0 and m > 0: return '{0:2}m {1:.6g}s'.format(int(m),s)
    print '{0:2}h {1:2}m {2:.6g}s'.format(int(h),int(m),s)


# Plots the differences in converting to Pal coordinates and back again.
def plot_and_save_diff(time_domain, _diff, isSS=True, isPal=False, isEarthOrbits=True):
    assert len(time_domain) > 0, 'The given list of the time domain is empty.'
    assert len(_diff) > 0, 'The given list of differences is empty.'
    assert len(time_domain) == len(_diff), 'The two given lists must be the same length.'
 
    if isEarthOrbits: xlabel = 'Earth Orbits'
    else: xlabel = 'Time'
    _ss_names = ['The Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    _pal_params = ['m', 'Semi-major Axis', 'Mean Longitude', 'k', 'h', 'ix', 'iy']
    _cart_params = ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    _x, _y, _z, _vx, _vy, _vz = [1, 2, 3, 4, 5, 6]
    (Ntime, Nbodies, Nparams) = _diff.shape
    dt_n = time_domain
    
    for i in range(Nbodies):
        if isSS: _planet_name = _ss_names[i]
        else: _planet_name = 'Object_' + str(i)
            
        if i == 0:
            _params = _cart_params
            if not isSS: _planet_name = 'Star'
        else:
            if isPal: _params = _pal_params
            else: _params = _cart_params
            
        dx = np.zeros(Ntime)
        dy = np.zeros(Ntime)
        dz = np.zeros(Ntime)
        dvx = np.zeros(Ntime)
        dvy = np.zeros(Ntime)
        dvz = np.zeros(Ntime)
        filename = 'Plots/' + _planet_name + '.pdf'

    #     The first element is acting strangely, so it is currently ommitted.
        for j,d in enumerate(_diff):
            dx[j] = d[i][_x]
            dy[j] = d[i][_y]
            dz[j] = d[i][_z]
            dvx[j] = d[i][_vx]
            dvy[j] = d[i][_vy]
            dvz[j] = d[i][_vz]
        fig, ((ax0,ax1,ax2),(ax3,ax4,ax5)) = plt.subplots(2,3)
        fig.set_figheight(10)
        fig.set_figwidth(20)

        ax0.plot(dt_n, dx)
        ax0.set_title('Difference in ' + _params[_x] + ' of ' + _planet_name)
        ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax0.set_xlabel(xlabel)

        ax1.plot(dt_n, dy)
        ax1.set_title('Difference in ' + _params[_y] + ' of ' + _planet_name)
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax1.set_xlabel(xlabel)

        ax2.plot(dt_n, dz)
        ax2.set_title('Difference in ' + _params[_z] + ' of ' + _planet_name)
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax2.set_xlabel(xlabel)

        ax3.plot(dt_n, dvx)
        ax3.set_title('Difference in ' + _params[_vx] + ' of ' + _planet_name)
        ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax3.set_xlabel(xlabel)

        ax4.plot(dt_n, dvy)
        ax4.set_title('Difference in ' + _params[_vy] + ' of ' + _planet_name)
        ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax4.set_xlabel(xlabel)

        ax5.plot(dt_n, dvz)
        ax5.set_title('Difference in ' + _params[_vz] + ' of ' + _planet_name)
        ax5.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax5.set_xlabel(xlabel)

        fig.savefig(filename, bbox_inches='tight')
        plt.show()
        print '\n'
        
# Plots the Pal coordinates of the specified planet (either calculated or cheby-fit).
def plot_and_save_pal_coord(time_domain, _pal_coord, _planet_index, isSS=True, isEarthOrbits=True):
    assert len(time_domain) > 0, 'The given list of the time domain is empty.'
    assert len(_pal_coord) > 0, 'The given list of coordinates is empty.'
    assert len(time_domain) == len(_pal_coord), 'The two given lists must be the same length.'
    assert type(_planet_index) == type(0), 'The given planet index is not of type: int.'
    assert _planet_index >= 0, 'The given planet index must be non-negative.'
    
    Ntime = len(time_domain)
    Nparam = 7
    
    if isEarthOrbits: xlabel = 'Earth Orbits'
    else: xlabel = 'Time'
    _ss_names = ['The Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    _params = ['Semi-major Axis', 'Mean Longitude', 'Pal: k', 'Pal: h', 'Pal: ix', 'Pal: iy']

    if isSS: _planet_name = _ss_names[_planet_index]
    else: _planet_name = 'Object_' + str(_planet_index)
    
    if _planet_index == 0:
        _params = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
        _planet_name = _ss_names[_planet_index] if isSS else 'Star'
    dt_n = time_domain
    _planet_data = np.zeros((Ntime, Nparam))
    
    _planet_data[:] = _pal_coord[:,_planet_index]
    
    _mass = _planet_data[:,0]
    _a = _planet_data[:,1]
    _l = _planet_data[:,2]
    _k = _planet_data[:,3]
    _h = _planet_data[:,4]
    _ix = _planet_data[:,5]
    _iy = _planet_data[:,6]

    # Plots the mass of the planet
    print 'Mass of {0}: {1:5e}'.format(_planet_name, _mass[0])

    pi = np.pi*np.ones(Ntime)
    # Plots the semi-major axis and mean longitude of the data
    filename = 'Plots/' + _planet_name + '_a_l' + '.pdf'
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    ax0.plot(dt_n, _a)
    ax0.set_title(_params[0] + ', ' + _planet_name)
    if _planet_index == 0: ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
    ax0.set_xlabel(xlabel)
    ax1.plot(dt_n, _l)
    ax1.set_title(_params[1] + ', ' + _planet_name)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
    ax1.set_xlabel(xlabel)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()

    # Plots the Pal Orbital Elements h and k of Earth
    filename = 'Plots/' + _planet_name + '_k_h' + '.pdf'
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    ax0.plot(dt_n, _k)
    ax0.set_title(_params[2] + ', ' + _planet_name)
    ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
    ax0.set_xlabel(xlabel)
    ax1.plot(dt_n, _h)
    ax1.set_title(_params[3] + ', ' + _planet_name)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
    ax1.set_xlabel(xlabel)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()

    # Plots the Pal Orbital Elements ix and iy of the data
    filename = 'Plots/' + _planet_name + '_ix_iy' + '.pdf'
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    ax0.plot(dt_n, _ix)
    ax0.set_title(_params[4] + ', ' + _planet_name)
    ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
    ax0.set_xlabel(xlabel)
    ax1.plot(dt_n, _iy)
    ax1.set_title(_params[5] + ', ' + _planet_name)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
    ax1.set_xlabel(xlabel)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    


#####################################################################################################################
####################################### INTERPOLATION ###############################################################    
#####################################################################################################################

    # TODO: NOT WORKING CORRECTY: NEED TO ALTER HOW l IS STORED SO THAT IT IS NOT SAWTOOTH.

# *********************** Interpolates the Pal coordinates with chebyshev polynomials ****************************


def random_sample(nparray, amount):
    rand_elements = random.sample(nparray, amount-2)
    rand_elements.append(nparray[0])
    rand_elements.append(nparray[len(nparray)-1])
    rand_elements = np.sort(rand_elements)
    idx = []
    for i in range(amount):
        idx.append(np.where(nparray == rand_elements[i])[0][0])
    return [rand_elements, idx]

def cheby_fit_this(time, time_obs, data, order):
    chebycoeff_data = CB.chebfit(time_obs, data, order)
    return CB.chebval(time, chebycoeff_data)

def cheby_extract_and_interpolate(pal_coord, timedomain, pts, printout=False, isSS=True, isAlwaysOne=False):
    assert len(pal_coord) > 0, 'The given list of Pal coordinates is empty.'
    assert len(timedomain) > 0, 'The given list of timesteps is empty.'
    
    pal_coord = unmod_mean_longitude(pal_coord)

    _ss_names = ['The Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    param_names = ['m','a','l','k','h','ix','iy']
    _m,_a,_l,_k,_h,_ix,_iy = [0,1,2,3,4,5,6]

    Ntime = len(timedomain)
    Nbodies = len(pal_coord[0])
    Nparam = 7
    
    seed = 998
    random.seed(seed)
    if printout: print 'Random seed:',seed

    # Number of 'observed' data points.
    if pts > Ntime: pts = np.round(Ntime*0.25)
    if pts < 2: pts = 2

    # Extracts the coordinates of the 'observed' points
    dt_s,dt_rand = random_sample(timedomain, pts)

    if printout: print 'Number of points for resolution:',Ntime
    if printout: print 'Number of \'observation\' points:',pts

    # Fits the 'observed' data with Chebyshev Polynomials up to O(order).
    # Finds the fit with the smallest maximum error and extracts it.
    order = pts # Sets how high of an polynomial order to test up to; helps find the best fit.

    cheby_pal = np.zeros((Nbodies, Nparam, Ntime))
    for pn in range(Nbodies):
        if isSS: _object_name = _ss_names[pn]
        else: 
            if pn == 0: _object_name = 'Star'
            else: _object_name = 'Object_' + str(pn)
        if printout: print "\nFor",_object_name
            
        # Set up fitting for each celestial body in the simulation.
        cheby_fits = np.zeros(Ntime)
        cheby_parameters = np.zeros((Nparam,Ntime))
        maxerr = np.zeros(order)

        # Extract the original data for a given planet.
        m = pal_coord[:,pn,_m]
        a = pal_coord[:,pn,_a]
        l = pal_coord[:,pn,_l]
        k = pal_coord[:,pn,_k]
        h = pal_coord[:,pn,_h]
        ix = pal_coord[:,pn,_ix]
        iy = pal_coord[:,pn,_iy]

        # Extract the 'observed' data for a given planet.
        m_s = pal_coord[dt_rand][:,pn,_m]
        a_s = pal_coord[dt_rand][:,pn,_a]
        l_s = pal_coord[dt_rand][:,pn,_l]
        k_s = pal_coord[dt_rand][:,pn,_k]
        h_s = pal_coord[dt_rand][:,pn,_h]
        ix_s = pal_coord[dt_rand][:,pn,_ix]
        iy_s = pal_coord[dt_rand][:,pn,_iy]

        # Combine the data to loop over for fittin purposes.
        parameters = [m, a, l, k, h, ix, iy]
        parameters_s = [m_s, a_s, l_s, k_s, h_s, ix_s, iy_s]

        # Mass data is not interpolated fit.
        cheby_parameters[0] = m
        # Iterate through each Pal coordinate and fit it with chebyshev polynomials.
        for i in range(1,Nparam):
            # Try various polynomial orders to find which order has the best fit.
            if not isAlwaysOne:
                for j in range(order):
                    cheby_fits = cheby_fit_this(timedomain, dt_s, parameters_s[i], j)
                    maxerr[j] = np.amax(np.abs(parameters[i]-cheby_fits))
            # Using the polynomial order that fit best, fit the Pal coordinate one more time.
            if i==_l or isAlwaysOne: cheby_parameters[i] = cheby_fit_this(timedomain, dt_s, parameters_s[i], 1)
            else: cheby_parameters[i] = cheby_fit_this(timedomain, dt_s, parameters_s[i], np.argmin(maxerr))
            
            if printout: print 'Order of best ',param_names[i],'fit:', 1 if i==_l or isAlwaysOne else np.argmin(maxerr)

        cheby_pal[pn] = cheby_parameters
        
    # Rearrange the newly fit data back into the original shape.
    cheby_out = np.zeros((Ntime, Nbodies, Nparam))
    for i in range(Ntime):
        for j in range(Nbodies):
            for k in range(Nparam):
                cheby_out[i,j,k] = cheby_pal[j,k,i]
                
    cheby_out = remod_mean_longitude(cheby_out)
    
    return cheby_out


# Recieves data that is already a subset of the 'actual' coordinates.
# Returns the coefficients of the polynomials used to fit the data.
def poly_coefs(timedomain, data, fit_order=2):
    assert len(timedomain) > 0, 'The given list for the domain is empty.'
    assert len(data) > 0, 'The given list of data is empty.'
    (Ntime, Nbodies, Nparam) = data.shape
    
    poly_fit_coefs = np.zeros((Nbodies, Nparam, fit_order + 1))
    
    for i in range(Nbodies):
        for j in range(Nparam):
            poly_fit_coefs[i,j] = poly.polyfit(timedomain, data[:,i,j], fit_order)
    
    return poly_fit_coefs

# Given the polynomial coefficients from the fit on the data, this returns a set
#    of polynomials that can be evaluated along any domain.
def poly_from_coef(poly_coefs):
    assert len(poly_coefs) > 0, 'The given list of coefficients is empty.'
    (Nbodies, Nparam, fit_order) = poly_coefs.shape
    
    poly_fits = []
    for i in range(Nbodies):
        params = []
        for j in range(Nparam):
            params.append(poly.Polynomial(poly_coefs[i,j]))
        poly_fits.append(params)
        
    return poly_fits

def numpy_from_poly(timedomain, poly):
    assert len(poly) > 0, 'The given list of polynomials is empty.'
    Ntime = len(timedomain)
    Nbodies = len(poly)
    Nparam = len(poly[0])
    
    np_fit = np.zeros((Ntime, Nbodies, Nparam))
    for i in range(Nbodies):
        for j in range(Nparam):
            np_fit[:,i,j] = poly[i][j](timedomain)
    np_fit[0,:,:] = np_fit[1,:,:]
    return np_fit

def ift(w, fw, t):
    N = len(w)
    ft = np.zeros(t.shape, dtype=complex)
    for i in range(N):
        ft = ft + fw[i]*np.exp(1j*2*np.pi*w[i]*t)
    return ft.real

def param_fourier_coefs(domain, data, errgoal=1e-12, iterlim=500, takelargest=4):
    assert len(domain) > 0, 'The given domain is empty.'
    assert len(data) > 0, 'The given list of data is empty.'
    N = len(data)
    L = domain[-1]
    n = takelargest
    total_sln = 0
    total_err = np.zeros(N, dtype=complex)
    freqs = np.zeros((iterlim, n), dtype=complex)
    amps = np.zeros((iterlim, n), dtype=complex)
    y_temp = data.copy()
    err = 1
    maxerr = 1e2
    its = -1
    broke = False
    w = blackman(N)
    while err > errgoal and its < iterlim:
        sp = np.fft.fft(y_temp*w)
        freq = np.fft.fftfreq(N, d=(L/N))
        max0 = np.abs(sp).argsort()[-n:][::-1]
        
        sp2 = np.zeros(N, dtype=complex)
        sp2[max0] = sp[max0]
        sln = np.fft.ifft(sp2)
        
        freqs[its] = freq[max0]
        amps[its] = sp[max0]/N
        sln = np.fft.ifft(sp2)
#        sln = ift(freq[max0], sp[max0]/N, domain)
        y_temp = y_temp - sln
        total_sln = total_sln + sln
        err = np.mean(np.abs(data - total_sln.real))
#        if its < 1: maxerr = err
        total_err[its] = err
        its = its + 1
        if err > maxerr:
            broke = True
            break
    freqs = freqs.flatten()
    amps = amps.flatten()
    m = its
#    if broke:
#        m = np.argmin(total_err)
    return freqs[:m],amps[:m],total_err[:m].real,m









