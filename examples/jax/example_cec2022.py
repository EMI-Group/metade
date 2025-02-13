import jax.numpy as jnp
import jax
from tqdm import tqdm
from util import StdSOMonitor, StdWorkflow
from algorithms.jax import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE
from problems.jax import CEC2022TestSuit

## Problem setting
D = 10  # * 10D 20D
FUNC_LIST = jnp.arange(12) + 1
## Meta setting
BATCH_SIZE = 100
NUM_RUNS = 1  # must be 1 in MetaDE
key_start = 42

## Outer optimizer setting
STEPS = 50
POP_SIZE = BATCH_SIZE

## Base algorithm setting
BASE_ALG_POP_SIZE = 100
BASE_ALG_STEPS = 1000

## DE parameter boundary setting
tiny_num = 1e-5
param_lb = jnp.array([0, 0, 0, 0, 1, 0])
param_ub = jnp.array([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num])

evolver = DE(
    lb=param_lb,
    ub=param_ub,
    pop_size=POP_SIZE,
    base_vector="rand",
    differential_weight=0.5,
    cross_probability=0.9,
)

BatchDE = create_batch_algorithm(ParamDE, BATCH_SIZE, NUM_RUNS)
batch_de = BatchDE(
    lb=jnp.full((D,), -100),
    ub=jnp.full((D,), 100),
    pop_size=BASE_ALG_POP_SIZE,
)

for func_num in FUNC_LIST:
    base_problem = CEC2022TestSuit.create(int(func_num))
    decoder = decoder_de
    key = jax.random.PRNGKey(key_start)

    monitor = StdSOMonitor(record_fit_history=False)
    print(type(base_problem).__name__)
    meta_problem = MetaDE(
        batch_de,
        base_problem,
        batch_size=BATCH_SIZE,
        num_runs=NUM_RUNS,
        base_alg_steps=BASE_ALG_STEPS,
    )

    workflow = StdWorkflow(
        algorithm=evolver,
        problem=meta_problem,
        pop_transform=decoder,
        monitor=monitor,
        record_pop=True,
    )
    key, _ = jax.random.split(key)
    state = workflow.init(key)

    power_up = 0
    last_iter = 0

    for i in tqdm(range(STEPS)):
        state = state.update_child("problem", {"power_up": power_up})
        state = workflow.step(state)

        if i == STEPS - 1:
            power_up = 1
            if last_iter:
                break
            last_iter = 1
        steps_iter = i + 1

    print(f"Best_fitness: {monitor.get_best_fitness()}")
