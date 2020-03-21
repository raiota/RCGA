
"""
Benchmark functions for optimization.
from (https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda)
     (https://github.com/DEAP/deap/blob/master/deap/benchmarks/__init__.py)
"""

import numpy


def ackley(individual):
    """
    Ackley function

    .. range
        - [-32.768, 32.768]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    .. function
        - :math:`f(\\mathbf{x}) = 20 - 20\exp\left(-0.2\sqrt{\\frac{1}{N} \
            \\sum_{i=1}^N x_i^2} \\right) + e - \\exp\\left(\\frac{1}{N}\sum_{i=1}^N \\cos(2\pi x_i) \\right)`
    """
    N = len(individual)
    return 20 - 20*numpy.exp(-0.2*numpy.sqrt(sum(x**2 for x in individual) / N)) \
            + numpy.e - numpy.exp(sum(numpy.cos(2*numpy.pi*x) for x in individual) / N)


def sphere(individual):
    """
    Sphere function

    .. range
        - [-inf, inf]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    .. function
        - :math:`f(\mathbf{x}) = \sum_{i=1}^Nx_i^2`
    """
    return sum(x**2 for x in individual)


def rosenbrock(individual):
    """
    Rosenbrock function

    .. range
        - [-5, 5]
    .. global optimal solution
        - :math:`x_i = 1, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    .. function
        - :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1} (1-x_i)^2 + 100 (x_{i+1} - x_i^2 )^2`
    """
    return sum(100*(y - x**2)**2 + (x - 1)**2 for x, y in zip(individual[1:], individual[:-1]))


def styblinski_tang(individual):
    """
    Styblinski-Tang function

    .. range
        - [-5, 5]
    .. global optimal solution
        - :math:`x_i = -2.903534, \\forall i \in \\lbrace -2.903534 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 39.166165N`
    .. function
    """
    return 0.5 * sum(x**4 - 16*x**2 + 5*x for x in individual)


def ellipsoid(individual):
    """
    Ellipsoid function

    .. range
        - [-5.12, 5.12]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    N = len(individual)
    return sum((1000**((i-1)/(N-1) * x))**2 for i, x in enumerate(individual))


def k_tablet(individual):
    """
    k-tablet function

    .. range
        - [-5.12, 5.12]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    N = len(individual)
    k = int(N/4)
    return sum(x**2 for x in individual[:k]) + sum(100*x**2 for x in individual[k:])


def weighted_sphere(individual):
    """
    Weighted sphere function
    or Hyper ellipsodic function

    .. range
        - [-5.12, 5.12]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    return sum(i*x**2 for i, x in enumerate(individual))


def sum_of_different_power(individual):
    """
    Sum of differen power function

    .. range
        - [-1, 1]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    return sum(abs(x)**(i+1) for i, x in enumerate(individual))


def griewank(individual):
    """
    Griewank function

    .. range
        - [-600, 600]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    return 1 + sum(x**2 for x in individual) / 4000 - numpy.prod([numpy.cos(x / numpy.sqrt(i)) for i, x in enumerate(individual)])


def rastrigin(individual):
    """
    Rastrigin function

    .. range
        - [-5.12, 5.12]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    N = len(individual)
    return 10*N + sum(x**2 - 10*numpy.cos(2*numpy.pi*x) for x in individual)


def schwefel(individual):
    """
    Schwefel function

    .. range
        - [-500, 500]
    .. global optimal solution
        - :math:`x_i = 420.9687, \\forall i \in \\lbrace 420.9687 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = -418.9829N`
    """
    return -1*sum(x*numpy.sin(numpy.sqrt(abs(x))) for x in individual)


def xin_she_yang(individual):
    """
    Xin-She Yang function

    .. range
        - [-2*pi, 2*pi]
    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    return sum(abs(x) for x in individual) * numpy.exp(-1*sum(numpy.sin(x**2) for x in individual))


def zakharov(individual):
    """
    Zakharov function

    .. global optimal solution
        - origin
          :math:`x_i = 0, \\forall i \in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\mathbf{x}) = 0`
    """
    return sum(x for x in individual) + (0.5*sum(x*i for i, x in enumerate(individual)))**2 + (0.5*sum(x*i for i, x in enumerate(individual)))**4