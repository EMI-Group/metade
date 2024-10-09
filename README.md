<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/evox_logo_light.png">
      <img alt="EvoX Logo" height="50" src="./assets/evox_logo_light.png">
  </picture>
  </a>
  <br>
</h1>

<p align="center">
🌟 MetaDE: Evolving Differential Evolution by Differential Evolution 🌟
</p>

<!-- The paper link is waiting to be added.  -->


MetaDE is an advanced evolutionary algorithm framework designed to adaptively optimize Differential Evolution (DE) strategies and hyperparameters using DE itself. MetaDE enables dynamic optimization of mutation and crossover strategies across diverse problem landscapes and leverages GPU-accelerated computations to maximize performance. The framework is built on the principles of meta-evolution, where DE evolves its configurations to optimize complex black-box problems more efficiently.

## Features

- **Meta-level Evolution** 🌱: Uses DE at a meta-level to evolve hyperparameters and strategies of DE applied at a problem-solving level.
- **Parameterized DE (PDE)** 🛠️: A customizable variant of DE that offers dynamic mutation and crossover strategies adaptable to different optimization problems.
- **GPU-accelerated** 🚀: Integrated with a GPU-accelerated framework for fast computation of large-scale problems.
- **End-to-End Optimization** 🔄: MetaDE provides a seamless workflow from hyperparameter tuning to solving optimization problems in a fully automated process.
- **Wide Applicability** 🤖: Supports various benchmarks, including CEC2022, and real-world tasks like evolutionary reinforcement learning in robotics.

## RL Tasks Visualization
Here are demonstrations of MetaDE applied to different simulated robotics environments. In these cases, MetaDE optimizes the parameters of an MLP, which is then used as a policy model to visualize the robot's behavior within the simulation.
<table width="90%">
  <tr>
    <td width="30%">
      <img width="100%" style="display:block; margin:auto;" src="./assets/hopper.gif"></img>
    </td>
    <td width="30%">
      <img width="100%" style="display:block; margin:auto;" src="./assets/swimmer.gif"></img>
    </td>
  </tr>
  <tr>
    <td align="center">
      Hopper
    </td>
    <td align="center">
      Swimmer
    </td>
  </tr>
</table>


- Hopper: Aiming for maximum speed and jumping height.
- Swimmer: Enhancing movement efficiency in fluid environments.

## Requirements
TensorRVEA requires:
- evox (version >= 0.6.0)
- jax (version >= 0.4.16)
- jaxlib (version >= 0.3.0)
- brax (version == 0.10.3)


## Installation
1. Install the correct version of [JAX](https://github.com/google/jax). We recommend `jax >= 0.4.16`.

For cpu version only, you may use:
```
pip install -U jax
```

For nvidia gpus, you may use:
```
pip install -U "jax[cuda12]"
```
For details of installing jax, please check https://github.com/google/jax.


2. Install `metade` from the GitHub source code:
```
pip install git+https://github.com/EMI-Group/metade.git
```

## Quickstart Guide

Here’s a quick example of how to run a MetaDE optimization on a benchmark problem(Ackley):

```python
import jax.numpy as jnp
import jax
import time
from tqdm import tqdm

from util import StdSOMonitor, StdWorkflow
from algorithms import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE
from evox import problems

D = 10
BATCH_SIZE = 100
NUM_RUNS = 1
key_start = 42

STEPS = 50
POP_SIZE = BATCH_SIZE

BASE_ALG_POP_SIZE = 100
BASE_ALG_STEPS = 100

tiny_num = 1e-5
param_lb = jnp.array([0, 0, 0, 0, 1, 0])
param_ub = jnp.array([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num])

evolver = DE(
    lb=param_lb,
    ub=param_ub,
    pop_size=POP_SIZE,
    base_vector="rand",
    differential_weight=0.5,
    cross_probability=0.9
)

BatchDE = create_batch_algorithm(ParamDE, BATCH_SIZE, NUM_RUNS)
batch_de = BatchDE(
    lb=jnp.full((D,), -100),
    ub=jnp.full((D,), 100),
    pop_size=BASE_ALG_POP_SIZE,
)

base_problem = problems.numerical.Ackley()
decoder = decoder_de
key = jax.random.PRNGKey(key_start)

monitor = StdSOMonitor(record_fit_history=False)

meta_problem = MetaDE(
    batch_de,
    base_problem,
    batch_size=BATCH_SIZE,
    num_runs=NUM_RUNS,
    base_alg_steps=BASE_ALG_STEPS
)

workflow = StdWorkflow(
    algorithm=evolver,
    problem=meta_problem,
    pop_transform=decoder,
    monitor=monitor,
    record_pop=True,
)

key, subkey = jax.random.split(key)
state = workflow.init(subkey)

power_up = 0
last_iter = False

for step in tqdm(range(STEPS)):
    state = state.update_child("problem", {"power_up": power_up})
    state = workflow.step(state)

    if step == STEPS - 1:
        power_up = 1
        if last_iter:
            break
        last_iter = True

print(f"Best fitness: {monitor.get_best_fitness()}")

```

## Components

### Evolver
MetaDE employs Differential Evolution (DE) as the evolver to optimize the parameters of its executor.

- **Mutation**: DE's mutation strategies evolve based on feedback from the problem landscape.
- **Crossover**: Different crossover strategies (binomial, exponential, arithmetic) can be used and adapted.
<img src="./assets/evolver.png" alt="Evolver Image" width="90%">

### Executor
The executor is a **Parameterized Differential Evolution (PDE)**, a variant of DE designed to accommodate various mutation and crossover strategies dynamically.

- **Parameterization**: Flexible mutation strategies like `DE/rand/1/bin` or `DE/best/2/exp` can be selected based on problem characteristics.
- **Parallel Execution**: Core operations of PDE are optimized for parallel execution on GPUs.
<img src="./assets/executor.png" alt="Executor Image" width="90%">

### GPU Acceleration
MetaDE integrates with the EvoX framework for distributed, GPU-accelerated evolutionary computation, significantly enhancing performance on large-scale optimization tasks.

## Examples

### Benchmark Problems

MetaDE supports several benchmark suites such as CEC2022. Here's an example to run it on the CEC2022 test suite:

```python
import jax.numpy as jnp
import jax
from tqdm import tqdm
from util import (
    StdSOMonitor,
    StdWorkflow
)
from algorithms import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE
from problems import CEC2022TestSuit

"""Settings"""
D = 10 
FUNC_LIST = jnp.arange(12) + 1
BATCH_SIZE = 100
NUM_RUNS = 1  
key_start = 42

STEPS = 50
POP_SIZE = BATCH_SIZE

BASE_ALG_POP_SIZE = 100
BASE_ALG_STEPS = 100  

tiny_num = 1e-5
param_lb = jnp.array([0, 0, 0, 0, 1, 0])
param_ub = jnp.array([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num])

"""Run"""
evolver = DE(
    lb=param_lb,
    ub=param_ub,
    pop_size=POP_SIZE,
    base_vector="rand", differential_weight=0.5, cross_probability=0.9
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
        base_alg_steps=BASE_ALG_STEPS
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

    """Results"""
    print(f"Best_fitness: {monitor.get_best_fitness()}")
```

### Robotics Control (Evolutionary Reinforcement Learning)

MetaDE can also be used for real-world tasks like robotic control through evolutionary reinforcement learning.Here are some results:

The following animations show the behaviors in Brax environments:

For instance, in controlling a "Swimmer" robot:
```python
from tqdm import tqdm
import problems
from jax import random
from flax import linen as nn
import jax.numpy as jnp
import jax

from util import StdSOMonitor, StdWorkflow, TreeAndVector, parse_opt_direction
from algorithms import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE

steps = 5
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
    cap_episode=500,
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

key, _ = jax.random.split(key)
state = workflow.init(key)

for i in tqdm(range(steps)):
    power_up = 1 if i == steps - 1 else 0
    state = state.update_child("problem", {"power_up": power_up})
    state = workflow.step(state)

print(f"Best fitness: {monitor.get_best_fitness()}")
```

## Results

MetaDE has been benchmarked on the CEC2022 suite and applied to various real-world tasks, demonstrating superior performance compared to traditional DE algorithms and other evolutionary methods.

## Community & Support

- Engage in discussions and share your experiences on [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Join our QQ group (ID: 297969717).

[//]: # (## Citing MetaDE)

[//]: # ()
[//]: # (If you use MetaDE in your research and want to cite it in your work, please use:)

[//]: # (```)

[//]: # (@article{evox,)

[//]: # (  title = {{MetaDE}: {Evolving} {Differential} {Evolution} by {Differential} {Evolution}},)

[//]: # (  author = {Chen, Minyang and Feng, Chenchen and Cheng, Ran},)

[//]: # (  journal = {IEEE Transactions on Evolutionary Computation},)

[//]: # (  year = 2024,)

[//]: # (  doi = {xx.xxxx/TEVC.2024.xxxxxxx})

[//]: # (})

[//]: # (```)
