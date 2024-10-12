import jax.numpy as jnp
import jax
from tqdm import tqdm
from util import StdSOMonitor, StdWorkflow
from algorithms import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE
from evox import problems  # You can import any problem from evox.problems

"""Settings"""
# Problem setting
D = 10  # Dimensionality of the problem

# Meta algorithm settings
BATCH_SIZE = 100
NUM_RUNS = 1  # Must be 1 in MetaDE
key_start = 42

# Outer optimizer settings
STEPS = 50
POP_SIZE = BATCH_SIZE

# Base algorithm settings
BASE_ALG_POP_SIZE = 100
BASE_ALG_STEPS = 100  # Can be increased for specific benchmarks

# DE parameter boundary settings
tiny_num = 1e-5
param_lb = jnp.array([0, 0, 0, 0, 1, 0])
param_ub = jnp.array([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num])

"""Run MetaDE"""
# Initialize the outer DE optimizer (evolver)
evolver = DE(
    lb=param_lb,
    ub=param_ub,
    pop_size=POP_SIZE,
    base_vector="rand",
    differential_weight=0.5,
    cross_probability=0.9
)

# Initialize the base DE algorithm used in the MetaDE framework
BatchDE = create_batch_algorithm(ParamDE, BATCH_SIZE, NUM_RUNS)
batch_de = BatchDE(
    lb=jnp.full((D,), -100),
    ub=jnp.full((D,), 100),
    pop_size=BASE_ALG_POP_SIZE,
)

# Problem to solve
base_problem = problems.numerical.Ackley()
decoder = decoder_de

key = jax.random.PRNGKey(key_start)

# Monitor to track the progress and results
monitor = StdSOMonitor(record_fit_history=False)

# Define the MetaDE problem with the base algorithm and Ackley problem
meta_problem = MetaDE(
    batch_de,
    base_problem,
    batch_size=BATCH_SIZE,
    num_runs=NUM_RUNS,
    base_alg_steps=BASE_ALG_STEPS
)

# Initialize the workflow that coordinates the evolution process
workflow = StdWorkflow(
    algorithm=evolver,
    problem=meta_problem,
    pop_transform=decoder,
    monitor=monitor,
    record_pop=True,
)

# Initialize the workflow state
key, subkey = jax.random.split(key)
state = workflow.init(subkey)

# Power-up control variables
power_up = 0
last_iter = False

# Run the optimization steps
for step in tqdm(range(STEPS)):
    state = state.update_child("problem", {"power_up": power_up})
    state = workflow.step(state)

    if step == STEPS - 1:
        power_up = 1
        if last_iter:
            break
        last_iter = True

# Output the best fitness result
print(f"Best fitness: {monitor.get_best_fitness()}")
