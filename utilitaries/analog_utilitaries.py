import numpy as np

########################
####### KERNELS ########
########################

# Commonly used kernels

def gaussian_kernel(x):
    return np.exp(-.5*x**2)

def exp_kernel(x):
    return np.exp(-np.abs(x))

def pexp_kernel(x, p):
    return np.exp(-np.abs(x)**p)

def inverse_kernel(x, p):
    return 1./(1+np.abs(x)**p)

def quadratic_kernel(x):
    if np.abs(x)<1:
        return 1-x**2
    else:
        return 0

def uniform_kernel(x):
    if np.abs(x)<1:
        return 1
    else:
        return 0    
    
def triangular_kernel(x):
    if np.abs(x)<1:
        return 1-np.abs(x)
    else:
        return 0    
    
def tricube_kernel(x):
    return np.clip( (1-(.5*np.abs(x))**3)**3, 0, None)

######################################
#### COMPUTE WEIGHTS FROM KERNELS ####
######################################

def weights_from_kernel(dist, std = 0, kernel = gaussian_kernel, lbda = False):
    # dist: array, analog-to-target distances (if the analogs have non-zero variances, analog_mean-to-target distances), size format (nquery, nneighbours)
    # std: array, standard deviation of the analogs (we use only the vector std and neglect covariance effects for simplicity), this can represent the observation error if we use a catalog of observations, for instance
    # kernel: function, kernel function
    # lbda: a scalar value to rescale distances
    # if lbda = False, then the median of distances is used
    
    if not(lbda):
        lbda = np.median(np.sqrt(dist**2+std**2))
        
    weights = kernel(np.sqrt(dist**2+std**2)/lbda)
    weights = weights.T*(1./np.sum(weights, axis=1))
    weights = weights.T
    
    return weights

########################################
######## SEPARATE TRAJECTORIES #########
########################################

# Sub-routine to separate analogs that belong to the same trajectory, i.e. analogs that are neighbours in time.
# It assumes that the catalog is evenly time-sampled, i.e. following indices correspond to time-neighbours in the catalog.
# For every group of analogs that are time-neighbours, this sub-routine keeps only one analog.
# Setting closest=True will return the analog with the lowest dist from each group.
# Setting closest=False will return one analog from each group, picked randomly in the group.
# /!\ This routine only works for (dist, ind) corresponding to one query, not multiple queries.

def sep_traj(dist, ind, closest=True):
    
    ind_sametraj = []; ind_sametraj.append([np.argsort(ind)[0]])
    i = 0; l = 0; K = len(ind)
    
    while True:
        while np.sort(ind)[i+1]-np.sort(ind)[i] <= 3:
            ind_sametraj[l].append(np.argsort(ind)[i+1])
            i += 1
            if i >= K-1:
                break
        if i >= K-1:
            break
        ind_sametraj.append([np.argsort(ind)[i+1]])
        l += 1; i += 1
        if i >= K-1:
            break
    
    ind_difftraj = []
    dist_difftraj = []
    
    for l in range(len(ind_sametraj)):
        if closest: # keep only the closest analogs inside a given trajectory
            k = np.argmin(dist[ind_sametraj[l]])
        else: # keep a random element inside a given trajectory
            k = np.random.choice( np.arange(len(ind_sametraj[l])) )
        ind_difftraj.append(ind[ind_sametraj[l]][k])
        dist_difftraj.append(dist[ind_sametraj[l]][k])
    
    ind_separated = np.array(ind_difftraj)[np.argsort(dist_difftraj)]
    dist_separated = np.sort(dist_difftraj)
    
    return dist_separated, ind_separated

