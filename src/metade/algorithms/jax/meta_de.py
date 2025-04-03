import jax
from jax.lax import reduce
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import vmap
from metade.util import parse_opt_direction

from metade.util import (
    Problem,
    State,
    jit_class,
)

from jax.experimental.host_callback import id_print


#
def decoder_de(pop):
    # Population transformation of evolver (decoding)
    return {
        "differential_weight": pop[:, 0],
        "cross_probability": pop[:, 1],
        "basevect_prim_type": jnp.floor(pop[:, 2]).astype(int),
        "basevect_sec_type": jnp.floor(pop[:, 3]).astype(int),
        "num_diff_vects": jnp.floor(pop[:, 4]).astype(int),
        "cross_strategy": jnp.floor(pop[:, 5]).astype(int),
    }


def none_transform(pop):
    # Population transformation of executor (decoding)
    return pop


def create_batch_algorithm(base_algorithm, batch_size, num_runs):
    # This method batches base_algorithm to produce batch_size of it,
    # where each base_algorithm is run num_runs times (with different keys) separately
    @jit_class
    class BatchAlgorithm(base_algorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.batch_size = batch_size
            self.num_runs = num_runs

        def setup(self, key):
            state = vmap(vmap(super().setup))(
                jnp.broadcast_to(jax.random.split(key, num=self.num_runs),
                                 (self.batch_size, self.num_runs, 2))
            )
            return state

        def ask(self, state):
            return vmap(vmap(super().ask))(state)

        def tell(self, state, fitness):
            return vmap(vmap(super().tell))(state, fitness)

        def override(self, state, key, params):
            # override the parameters of the base algorithms
            return vmap(
                vmap(super().override, in_axes=(0, 0, None)),
                in_axes=(0, None, 0),
            )(state, jax.random.split(key, num=self.num_runs), params)

    return BatchAlgorithm


@jit_class
class MetaDE(Problem):
    # MetalProblem, which is to evaluate the hyperparameters of the base algorithm on a given problem.
    def __init__(
            self,
            base_algorithm,
            problem,
            batch_size,
            num_runs,
            base_alg_steps,
            override=True,
            base_pop_transform=none_transform,
            base_opt_direction="min",
    ):
        super().__init__()
        self.base_algorithm = base_algorithm
        self.problem = problem
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.override = override
        self.base_alg_steps = base_alg_steps
        self.pop_transform = base_pop_transform
        self.opt_direction = parse_opt_direction(base_opt_direction)

    def setup(self, key):
        return State(key=key)

    def evaluate(self, state, x):
        # x is a population of MetaPSO parameters
        override_key, _ = jax.random.split(state.key, num=2)
        if self.override:
            state = self.base_algorithm.override(state, override_key, x)

        evaluate_batched = vmap(
            vmap(self.problem.evaluate, in_axes=(None, 0), out_axes=(0, None)),
            in_axes=(None, 0),
            out_axes=(0, None),
        )
        # evaluate_batched = vmap(
        #     vmap(self.problem.evaluate))

        pop_transform_batched = vmap(vmap(self.pop_transform))

        def one_step(_i, min_fit_and_pso_state):
            min_fitness, state = min_fit_and_pso_state
            pops, state = self.base_algorithm.ask(state)  # pops (outer_pop_size, num_runs, inner_pop_size, dim)
            pops = pop_transform_batched(pops)  # decoder for specific problem (e.g., neuroevolution)
            fitness, state = evaluate_batched(state, pops)  # fitness (outer_pop_size, num_runs, inner_pop_size)
            fitness = fitness * self.opt_direction
            state = self.base_algorithm.tell(state, fitness)
            min_fitness = jnp.minimum(
                min_fitness, jnp.nanmin(jnp.nanmin(fitness, axis=2), axis=1)
            )
            return min_fitness, state

        base_alg_steps = jax.lax.select(state.power_up, self.base_alg_steps * 5, self.base_alg_steps)

        # the algorithms will be run for base_alg_steps iterations
        min_fitness, state = jax.lax.fori_loop(
            0, base_alg_steps, one_step, (jnp.full((self.batch_size,), jnp.inf), state)
        )

        # return min_fitness, state.update(key=key)
        return min_fitness, state