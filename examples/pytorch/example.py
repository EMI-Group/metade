import torch
from metade.algorithms.pytorch.de import DE
from metade.algorithms.pytorch.param_de import ParamDE
from metade.problems.torch.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from metade.util.workflows import EvalMonitor, StdWorkflow
from evox.problems.numerical import Sphere  # you can import any problem from evox.problems
from tqdm import tqdm

# ============================
# Define Global Parameters
# ============================
Dim = 10  # Dimensionality of the optimization problem
Inner_Population = 100  # Population size for the inner optimizer
Outer_Population = 100  # Population size for the outer optimizer (HPO search)
HPO_Instances = Outer_Population  # Number of instances in HPO
Inner_Iterations = 100  # Number of iterations for the inner loop (HPO optimization)
Outer_Iterations = 50  # Number of iterations for hyperparameter optimization
tiny_num = 1e-5  # Small value to ensure boundary conditions are respected

# ============================
# Set random seed and device
# ============================
torch.manual_seed(42)
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Initialize Inner Optimization Process
# ============================
inner_algo = ParamDE(Inner_Population, -100 * torch.ones(Dim), 100 * torch.ones(Dim))
inner_prob = Sphere()

inner_monitor = HPOFitnessMonitor()
inner_workflow = StdWorkflow(algorithm=inner_algo, problem=inner_prob, monitor=inner_monitor)

# ============================
# Transform Inner Workflow into an HPO Problem
# ============================
hpo_prob = HPOProblemWrapper(
    iterations=Inner_Iterations, num_instances=HPO_Instances, workflow=inner_workflow, copy_init_state=True
)

params = hpo_prob.get_init_params()

# ============================
# Initialize Hyperparameters for HPO Instances
# ============================
params["algorithm.differential_weight"] = torch.nn.Parameter(torch.rand(HPO_Instances, 1), requires_grad=False)
params["algorithm.cross_probability"] = torch.nn.Parameter(torch.rand(HPO_Instances, 1), requires_grad=False)
params["algorithm.basevect_prim_type"] = torch.nn.Parameter(torch.randint(0, 4, (HPO_Instances,)),
                                                            requires_grad=False)
params["algorithm.basevect_sec_type"] = torch.nn.Parameter(torch.randint(0, 4, (HPO_Instances,)),
                                                           requires_grad=False)
params["algorithm.num_diff_vects"] = torch.nn.Parameter(torch.randint(1, 5, (HPO_Instances,)), requires_grad=False)
params["algorithm.cross_strategy"] = torch.nn.Parameter(torch.randint(0, 3, (HPO_Instances,)), requires_grad=False)


# ============================
# Define Solution Transformation
# ============================
class SolutionTransform(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return {
            "algorithm.differential_weight": x[:, 0],
            "algorithm.cross_probability": x[:, 1],
            "algorithm.basevect_prim_type": x[:, 2].floor().long(),
            "algorithm.basevect_sec_type": x[:, 3].floor().long(),
            "algorithm.num_diff_vects": x[:, 4].floor().long(),
            "algorithm.cross_strategy": x[:, 5].floor().long(),
        }


# ============================
# Define Outer Optimization Process (Hyperparameter Search)
# ============================
param_lb = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32)  # Lower bounds
param_ub = torch.tensor([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num],
                        dtype=torch.float32)  # Upper bounds

outer_algo = DE(Outer_Population, param_lb, param_ub)
monitor = EvalMonitor(full_sol_history=False)

outer_workflow = StdWorkflow(algorithm=outer_algo, problem=hpo_prob, monitor=monitor,
                             solution_transform=SolutionTransform())

# ============================
# Run Optimization for Outer Iterations
# ============================
outer_workflow.init_step()

for _ in tqdm(range(Outer_Iterations), desc="Optimizing Hyperparameters"):
    outer_workflow.step()

# ============================
# Output Final Results
# ============================
monitor = outer_workflow.get_submodule("monitor")

print("Best hyperparameters:\n", monitor.topk_solutions, "\n")
print("Best fitness values:\n", monitor.topk_fitness)
