
"""
Module for new generation alternation model.
"""

import numpy

from .individual import *
from .crossover import *


#####################
# Alternation Model #
#####################

class JGG(object):
    """
    Executes Just Generation Gap (JGG) alternation on the populations.

    Parameters
    ----------
    dim : int
        Dimension of optimization, which indicates the length of chromosome.
    evaluation_func : function
        Function for optimizing
    gene_min : numpy.ndarray
        List of minimum value of chromosome.
    gene_max : numpy.ndarray
        List of maximum value of chromosome.
    crossover_type : enum
        Type of crossover, like "rex" or "e_undx".
        For more details see :module:`crossover.py`.
    random_type : enum
        Type of probability distribution which is used in generating children.
        For more details see :module:`crossover.py`.
    pop_size : int
        Number of population.
    parent_num : int
        Number of parents which is extracted for crossover.
    child_num : int
        Number of children which is generated.
    seed :
        the seed value of randomness.
    """

    def __init__(self, dim,
                 evaluation_func,
                 gene_min, gene_max,
                 crossover_type,
                 random_type,
                 pop_size, parent_num, child_num,
                 chrom_size,
                 k=None, seed=None):

        numpy.random.seed(seed=seed)

        self.dim = dim
        self.gene_min = gene_min
        self.gene_max = gene_max

        # function for evaluation
        self._eval = evaluation_func

        # function for crossover
        self.cross = cross(crossover_type=crossover_type,
                           random_type=random_type,
                           dim=dim, k=k, seed=seed)
        
        self.pop_size = pop_size
        self.parent_num = parent_num
        self.child_num = child_num
    

    %staticmethod
    def init_population(self):
        self.population = Population(self.pop_size, self.gene_min, self.gene_max, self.dim)
        return self.population
    
    %staticmethod
    def eval(self, *args, **kwargs):
        
