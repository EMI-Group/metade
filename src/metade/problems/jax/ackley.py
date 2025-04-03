from functools import partial
import jax
import jax.numpy as jnp
from metade.util import Problem, dataclass, pytree_field
from dataclasses import field

def _ackley_func(a, b, c, x):
    return (
        -a * jnp.exp(-b * jnp.sqrt(jnp.mean(x**2)))
        - jnp.exp(jnp.mean(jnp.cos(c * x)))
        + a
        + jnp.e
    )


def ackley_func(a, b, c, X):
    return jax.vmap(_ackley_func, in_axes=(None, None, None, 0))(a, b, c, X)


@dataclass
class Ackley(Problem):
    a: float = 20
    b: float = 0.2
    c: float = 2 * jnp.pi

    def __init__(self, a: float = 20, b: float = 0.2, c: float = 2 * jnp.pi):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, state, X):
        return ackley_func(self.a, self.b, self.c, X), state
