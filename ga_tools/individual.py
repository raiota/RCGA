
import random
import numpy


class Individual(numpy.ndarray):

    fitness = None

    def __new__(cls, chromosome):
        return numpy.asarray(chromosome).view(cls)



def genemake(gene_min, gene_max, size=None):
    """
    Function for Generating Gene

    Parameters
    ----------
    gene_min : array-like
    gene_max : array-like
    size : int
        size of chromosome (the numbers of gene)

    Returns
    ----------
    chromosome : array-like
        Constituted only operated parameters

    Notes
    ----------
    Usage below if you wanna make chromosome [gene1, gene2, gene3, gene4, ..., geneN]
        low : [gene1_min, gene2_min, gene3_min, ..., geneN_min]
        up : [gene1_max, gene2_max, gene3_max, ..., geneN_max]
    """
    try:
        return [random.uniform(a, b) for a, b in zip(gene_min, gene_max)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([gene_min] * size, [gene_max] * size)]


class Population(list):

    def __new__(cls, pop_size, gene_min, gene_max, chromo_size):
        cls = [Individual(genemake(gene_min, gene_max, size=chromo_size)) for _ in range(pop_size)]
        return cls


def extract(container, num):
    """
    Extracting `num` values from `container`

    Parameters
    ----------
    containter : list
    num : int
        Numbers for extracting
    
    Returns
    ----------
    extracted : list
        List from extracted values, which of length is num
    container : list
        Extracted `container`, which of length is len(`container`) - `num`
    """
    random.shuffle(container)
    extracted = []

    for _ in range(num):
        extracted.append(container.pop())
        
    return extracted, container