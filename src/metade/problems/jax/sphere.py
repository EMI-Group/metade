from functools import partial
import jax
import jax.numpy as jnp
from metade.util import Problem, dataclass, pytree_field
from dataclasses import field

def _sphere_func(x):
    return jnp.sum(x ** 2)

def sphere_func(X):
    return jax.vmap(_sphere_func)(X)

@dataclass
class Sphere(Problem):
    def __init__(self):
        pass

    def evaluate(self, state, X):
        return sphere_func(X), state
