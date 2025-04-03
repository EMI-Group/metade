import jax
import jax.numpy as jnp
from jax import vmap, lax

from metade.util import (
    Algorithm,
    State,
    jit_class,
)
from metade.util import select_rand_pbest
from metade.util import (
    de_diff_sum,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)
from functools import partial


@jit_class
class ParamDE(Algorithm):
    """Parametric DE"""

    def __init__(
            self,
            lb,
            ub,
            pop_size=100,
            diff_padding_num=9,

            differential_weight=0.3471559,
            cross_probability=0.78762645,
            basevect_prim_type=0,
            basevect_sec_type=2,
            num_diff_vects=3,
            cross_strategy=2,
    ):
        self.num_diff_vects = num_diff_vects
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.pop_size = pop_size
        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.cross_strategy = cross_strategy
        self.diff_padding_num = diff_padding_num
        self.basevect_prim_type = basevect_prim_type
        self.basevect_sec_type = basevect_sec_type

    def setup(self, key):
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.pop_size, self.dim))
        best_index = 0
        start_index = 0
        params_init = {
            "differential_weight": self.differential_weight,
            "cross_probability": self.cross_probability,
            "basevect_prim_type": self.basevect_prim_type,
            "basevect_sec_type": self.basevect_sec_type,
            "num_diff_vects": self.num_diff_vects,
            "cross_strategy": self.cross_strategy,
        }
        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
            params=params_init,
        )

    def ask(self, state):
        key, ask_one_key = jax.random.split(state.key, 2)
        ask_one_keys = jax.random.split(ask_one_key, self.pop_size)
        indices = jnp.arange(self.pop_size) + state.start_index

        trial_vectors = vmap(partial(self._ask_one, state_inner=state))(ask_one_key=ask_one_keys, index=indices)

        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key)

    def _ask_one(self, state_inner, ask_one_key, index):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness
        params = state_inner.params

        """Mutation: base_vect + F * difference_sum"""

        difference_sum, rand_vect_idx = de_diff_sum(
            select_key,
            self.diff_padding_num,
            params["num_diff_vects"],
            index,
            population,
        )

        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]
        pbest_vect = select_rand_pbest(pbest_key, 0.05, population, fitness)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[params["basevect_prim_type"]]
        base_vector_sec = vector_merge[params["basevect_sec_type"]]

        base_vector = base_vector_prim + params["differential_weight"] * (base_vector_sec - base_vector_prim)

        mutation_vector = (base_vector + difference_sum * params["differential_weight"])

        """Crossover: 0 = bin, 1 = exp, 2 = arith"""
        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = lax.switch(
            params["cross_strategy"],
            cross_funcs,
            crossover_key,
            mutation_vector,
            current_vect,
            params["cross_probability"],
        )

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)

        return trial_vector

    def tell(self, state, trial_fitness):
        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim(
            state.population, start_index, self.pop_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.pop_size, axis=0
        )

        compare = trial_fitness <= batch_fitness

        population_update = jnp.where(
            compare[:, jnp.newaxis], state.trial_vectors, batch_pop
        )
        fitness_update = jnp.where(compare, trial_fitness, batch_fitness)

        population = jax.lax.dynamic_update_slice_in_dim(
            state.population, population_update, start_index, axis=0
        )
        fitness = jax.lax.dynamic_update_slice_in_dim(
            state.fitness, fitness_update, start_index, axis=0
        )
        best_index = jnp.argmin(fitness)
        start_index = (state.start_index + self.pop_size) % self.pop_size
        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
        )

    def override(self, state, key, params):
        state = state | ParamDE.setup(self, key)
        return state.update(params=params)