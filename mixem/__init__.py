"""
Simple Expectation-Maximization fitting of mixtures of probability densities.
"""


import mixem.distribution
from mixem.progress import simple_progress
from mixem.em import em
from mixem.model import probability
