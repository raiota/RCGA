
"""
Module for crossover
"""

from enum import Enum

import numpy


#######################
# CrossType Selection #
#######################

class crossoverType(Enum):
    UNDX_N = 0
    E_UNDX = 1
    GENERAL_REX = 2


def cross(crossover_type, *args, **kwargs):

    if crossover_type == crossover_type.UNDX_N:
        return REX(crossover_type, *args, **kwargs)

    elif crossover_type == crossover_type.E_UNDX:
        return REX(crossover_type, *args, **kwargs)
    
    elif crossover_type == crossover_type.GENERAL_REX:
        return REX(crossover_type, *args, **kwargs)
    
    else:
        raise ValueError("Argument `crossover_type` is invalid.")


########################
# RandomType Selection #
########################

class randomType(Enum):
    NORMAL = 0
    UNIFORM = 1
    V_SHAPE = 2


class randomGenerator(object):

    def __init__(self, randomtype, seed=None):
        numpy.random.seed(seed=seed)
        self.randomtype = randomtype
    
    def normal(self, mean, var):
        return numpy.random.normal(mean, var)
    
    def uniform(self, min_, max_):
        return (max_ - min_) * numpy.random.rand() + min_
    
    def v_shape(self, min_, max_):
        # building now...
        return 
    
    def __call__(self, dim, k):
        if self.randomtype == randomType.NORMAL:
            return self.normal(mean=0, var=1/(dim+k))

        elif self.randomtype == randomType.UNIFORM:
            val = numpy.sqrt(3/(dim+k))
            return self.uniform(min_=-val, max_=val)

        elif self.randomtype == randomType.V_SHAPE:
            val = numpy.sqrt(2/(dim+k))
            return self.v_shape(min_=-val, max_=val)

        else:
            raise ValueError("Argument `randomtype` is invalid.")


##############
# Crossovers #
##############

class REX(object):
    """
    Executes Real-coded Ensemble Crossover (REX) on the input parents.

    Parameters
    ----------
    crosstype : enum
        Crossover types which are choosed from :class:`crossoverType`.
    dim : int
        Dimension of optimization, which indicates the length of chromosome.
    k : int
        Number of additional parents which is used in crossovers,
        actually the used number of parents is `dim`+`k`.
    seed : 
        the seed value of randomness.
    """

    def __init__(self, crossover_type, random_type, dim, k=None, seed=None):
        """
        Notes
        ----------

        """
        numpy.random.seed(seed=seed)

        self.dim = dim
        self.rand = randomGenerator(random_type, seed=seed)

        if crossover_type == crossoverType.UNDX_N:
            self.k = 0
            self.crossover_num = self.dim
            self.var = 1 / self.crossover_num
        
        elif crossover_type == crossoverType.E_UNDX:
            self.k = 1
            self.crossover_num = self.dim + 1
            self.var = 1 / self.crossover_num
        
        elif crossover_type == crossoverType.GENERAL_REX:
            self.k = k
            self.crossover_num = self.dim + self.k
            self.var = 1 / self.crossover_num
        
        else:
            raise ValueError("Argument `crosstype` is invalid.")
    

    def __call__(self, *parents):
        """
        Executes crossover.

        Parameters
        ----------
        *parents : tuple(dtype=:numpy.ndarray:)
            chromosome vector of all parents.
        
        Returns
        ----------
        child_vector : numpy.ndarray
            chromosome vector of a generated child.
        """
        gravity_vector = numpy.average(parents, axis=0)

        child_vector = gravity_vector
        for i in range(self.crossover_num):
            child_vector += self.rand(dim=self.dim, k=self.k) * (parents[i] - gravity_vector)
        
        return child_vector
