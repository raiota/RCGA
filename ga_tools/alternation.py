
"""
Module for new generation alternation model.
"""

from enum import Enum

import numpy

from .individual import *
from .crossover import *

######################
# Evaluation Problem #
######################

class evalType(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1


def evaltool(evaltype):

    if evaltype == evalType.MINIMIZE:
        return False

    elif evaltype == evalType.MAXIMIZE:
        return True

    else:
        raise ValueError("Argument `evaltype` is invalid.")


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
    evaluation_type : enum
        Type of evaluation, like "minimize problem", or "maximize problem".
        For more details see above.
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
                 evaluation_func, evaluation_type,
                 gene_min, gene_max,
                 crossover_type,
                 random_type,
                 pop_size, parent_num, child_num,
                 k=None, seed=None):

        numpy.random.seed(seed=seed)

        self.dim = dim
        self.gene_min = gene_min
        self.gene_max = gene_max

        # function for evaluation
        self._eval = evaluation_func
        self.evaltype = evaluation_type

        # function for crossover
        self.cross = cross(crossover_type=crossover_type,
                           random_type=random_type,
                           dim=dim, k=k, seed=seed)
        
        self.pop_size = pop_size
        self.parent_num = parent_num
        self.child_num = child_num


    def init_population(self):

        self.population = Population(self.pop_size, self.gene_min, self.gene_max, self.dim)
        # self.population.gen = 1

        return self.population
    

    def eval(self, population=None, *args, **kwargs):

        if population is not None:
            fitnesses = [self._eval(ind, *args, **kwargs) for ind in population]
            for ind, fit in zip(population, fitnesses):
                ind.fitness = fit
        else:
            fitnesses = [self._eval(ind, *args, **kwargs) for ind in self.population]
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness = fit

        return population


    def __extract_parents(self):

        parent_group, self.population = extract(self.population, self.parent_num)
        return parent_group, self.population
        

    def __crossover(self, parent_group):

        children = []

        for _ in range(self.child_num):
            random.shuffle(parent_group)
            a_child = self.cross(*tuple([parent_group[i] for i in range(self.cross.crossover_num)]))
            children.append(a_child)

        return children


    def __select(self, children):

        children.sort(key=lambda ind: ind.fitness, reverse=evaltool(self.evaltype))
        next_children = children[0:self.parent_num]

        return next_children


    def generation_step(self):

        parent_group, _ = self.__extract_parents()
        children_group = [Individual(ind) for ind in self.__crossover(parent_group)]

        evaluated_children = self.eval(population=children_group)
        next_children = self.__select(evaluated_children)

        self.population += next_children
        # self.population.gen += 1

        return self.population