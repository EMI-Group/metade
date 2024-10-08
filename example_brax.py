from tqdm import tqdm
import problems
from jax import random
from flax import linen as nn
import jax.numpy as jnp
import jax
import imageio
from util import StdSOMonitor, StdWorkflow, TreeAndVector, parse_opt_direction
from algorithms import create_batch_algorithm, decoder_de, MetaDE, ParamDE, DE

steps = 1
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
    base_alg_steps=10,
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
    power_up = 0 if i == steps - 1 else 0
    state = state.update_child("problem", {"power_up": power_up})
    state = workflow.step(state)
# 可视化Bra群体的最佳结果
print(state.get_child_state("algorithm").fitness)
print(monitor.get_best_fitness())
best_weights = state.get_child_state("algorithm").population[state.get_child_state("algorithm").best_index]  # 获取最好个体的权重
key, subkey = jax.random.split(key)

# 可视化并生成帧序列，指定 'rgb_array' 作为输出类型
frames, _ = base_problem.visualize(state, subkey, adapter.to_tree(best_weights), output_type='rgb_array', width=300,
                                   height=300)


# 使用 imageio 保存为GIF
# 注意：frames 是一个包含图像数据的列表，确保每个frame是ndimage格式
imageio.mimsave('hopper_simulation.gif', frames, fps=30)
html_result, state = base_problem.visualize(state, random.key(0), adapter.to_tree(best_weights))
with open("data.html", "w") as file:
    file.write(html_result)

