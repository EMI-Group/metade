import torch
import torch.nn as nn
from tqdm import tqdm

from metade.algorithms.pytorch.de import DE
from metade.algorithms.pytorch.param_de import ParamDE
from metade.problems.torch.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from metade.problems.torch.brax import BraxProblem
from metade.util.workflows import EvalMonitor, StdWorkflow
from metade.util import ParamsAndVector

torch.set_float32_matmul_precision('high')
# ==============================
# Define Global Parameters
# ==============================
Dim = 8  # Input dimension for MLP
Inner_Population = 100  # Population size for the inner optimizer (neuroevolution)
Outer_Population = 100  # Population size for the outer optimizer (HPO search)
HPO_Instances = Outer_Population  # Number of hyperparameter instances to evaluate
Inner_Iterations = 1  # Number of iterations for the inner loop (HPO optimization)
Outer_Iterations = 10  # Number of iterations for outer hyperparameter optimization
tiny_num = 1e-5  # Small value to ensure boundary conditions are respected

# Set device (prefer GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================
# Define Policy Network (Simple MLP)
# ==============================
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(Dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        return torch.tanh(self.features(x))


# Initialize policy network and move to device
model = SimpleMLP().to(device)

# Convert model parameters to a vector using ParamsAndVector
adapter = ParamsAndVector(dummy_model=model)
model_params = dict(model.named_parameters())
pop_center = adapter.to_vector(model_params)

# Define search space bounds for inner optimization [-10, 10]
lower_bound_inner = torch.full_like(pop_center, -10.0)
upper_bound_inner = torch.full_like(pop_center, 10.0)

# ==============================
# Inner Workflow: Using ParamDE to optimize Brax Problem
# ==============================
inner_algo = ParamDE(Inner_Population, lower_bound_inner, upper_bound_inner, device=device)

inner_problem = BraxProblem(
    policy=model,
    env_name="swimmer",
    max_episode_length=500,
    num_episodes=3,
    pop_size=Inner_Population * Outer_Population,
    device=device,
)

inner_monitor = HPOFitnessMonitor()

inner_workflow = StdWorkflow(algorithm=inner_algo, problem=inner_problem, solution_transform=adapter,
                             monitor=inner_monitor, device=device, opt_direction="max")

# ==============================
# Transform Inner Workflow into an HPO Problem
# ==============================
hpo_prob = HPOProblemWrapper(
    iterations=Inner_Iterations,
    num_instances=HPO_Instances,
    workflow=inner_workflow,
    copy_init_state=True
)

# Initialize hyperparameters
params = hpo_prob.get_init_params()
params["algorithm.differential_weight"] = torch.nn.Parameter(torch.rand(HPO_Instances, 1, device=device),
                                                             requires_grad=False)
params["algorithm.cross_probability"] = torch.nn.Parameter(torch.rand(HPO_Instances, 1, device=device),
                                                           requires_grad=False)
params["algorithm.basevect_prim_type"] = torch.nn.Parameter(torch.randint(0, 4, (HPO_Instances,), device=device),
                                                            requires_grad=False)
params["algorithm.basevect_sec_type"] = torch.nn.Parameter(torch.randint(0, 4, (HPO_Instances,), device=device),
                                                           requires_grad=False)
params["algorithm.num_diff_vects"] = torch.nn.Parameter(torch.randint(1, 5, (HPO_Instances,), device=device),
                                                        requires_grad=False)
params["algorithm.cross_strategy"] = torch.nn.Parameter(torch.randint(0, 3, (HPO_Instances,), device=device),
                                                        requires_grad=False)


# ==============================
# Outer Workflow: Using DE to Optimize Hyperparameters
# ==============================
class HyperParamTransform(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return {
            "algorithm.differential_weight": x[:, 0],
            "algorithm.cross_probability": x[:, 1],
            "algorithm.basevect_prim_type": x[:, 2].floor().long(),
            "algorithm.basevect_sec_type": x[:, 3].floor().long(),
            "algorithm.num_diff_vects": x[:, 4].floor().long(),
            "algorithm.cross_strategy": x[:, 5].floor().long(),
        }


# Define search space for hyperparameters
param_lb = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32)  # Lower bounds
param_ub = torch.tensor([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num],
                        dtype=torch.float32)  # Upper bounds

outer_algo = DE(Outer_Population, param_lb, param_ub, device=device)

# Outer monitor: Records the best hyperparameters and fitness values
outer_monitor = EvalMonitor(full_sol_history=False, device=device)

# Define outer workflow using HPOProblemWrapper as the problem
outer_workflow = StdWorkflow(algorithm=outer_algo,
                             problem=hpo_prob,
                             monitor=outer_monitor,
                             solution_transform=HyperParamTransform(),
                             device=device, opt_direction="min")
outer_workflow.init_step()

# ==============================
# Start Evolutionary Optimization for Hyperparameters
# ==============================
for gen in tqdm(range(Outer_Iterations), desc="Optimizing Hyperparameters"):
    outer_workflow.step()
    current_monitor = outer_workflow.get_submodule("monitor")
    print(f"Generation {gen + 1}, Best fitness: {-1 * current_monitor.topk_fitness}")

# ==============================
# Output Final Results
# ==============================
final_monitor = outer_workflow.get_submodule("monitor")
print("Best hyperparameter combinations:\n", final_monitor.topk_solutions)
print("Corresponding fitness values:\n", -1 * final_monitor.topk_fitness)
