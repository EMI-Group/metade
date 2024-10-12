from tqdm import tqdm
import problems
from jax import random
from flax import linen as nn
import jax.numpy as jnp
import jax
from util import StdSOMonitor, StdWorkflow, TreeAndVector, parse_opt_direction
from algorithms import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE

steps = 20
pop_size = 100
key = jax.random.PRNGKey(42)

class HopperPolicy(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = x.reshape(-1)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        x = nn.Dense(3)(x)
        x = nn.tanh(x)
        return x

model = HopperPolicy()
weights = model.init(random.PRNGKey(42), jnp.zeros((11,)))
adapter = TreeAndVector(weights)
vector_form_weights = adapter.to_vector(weights)

tiny_num = 1e-5
param_lb = jnp.array([0, 0, 0, 0, 1, 0])
param_ub = jnp.array([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num])
algorithm = DE(
    lb=param_lb,
    ub=param_ub,
    pop_size=pop_size,
    base_vector="rand", differential_weight=0.5, cross_probability=0.9
)

BatchDE = create_batch_algorithm(ParamDE, pop_size, 1)
batch_de = BatchDE(
    lb=jnp.full((vector_form_weights.shape[0],), -10.0),
    ub=jnp.full((vector_form_weights.shape[0],), 10.0),
    pop_size=100,
)

base_problem = problems.Brax(
    env_name="hopper",
    policy=jax.jit(model.apply),
    cap_episode=1000,
)

meta_problem = MetaDE(
    batch_de,
    base_problem,
    batch_size=pop_size,
    num_runs=1,
    base_alg_steps=50,
    base_opt_direction="max",
    base_pop_transform=adapter.batched_to_tree,
)

monitor = StdSOMonitor(record_fit_history=False)
workflow = StdWorkflow(
    algorithm=algorithm,
    problem=meta_problem,
    monitor=monitor,
    pop_transform=decoder_de,
    record_pop=True,
)
monitor.set_opt_direction(parse_opt_direction("max"))
key, _ = jax.random.split(key)
state = workflow.init(key)

for i in tqdm(range(steps)):
    power_up = 1 if i == steps - 1 else 0
    state = state.update_child("problem", {"power_up": power_up})
    state = workflow.step(state)

print(monitor.get_best_fitness())


