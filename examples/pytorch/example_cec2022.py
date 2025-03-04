import torch
from algorithms.pytorch.de import DE
from algorithms.pytorch.param_de import ParamDE
from problems.torch.hpo_wrapper import HPOFitnessMonitor, HPOProblemWrapper
from util.workflows import EvalMonitor, StdWorkflow
from problems.torch.cec2022 import CEC2022  # Follow the original code, fitness minimum is 300, 400...
from tqdm import tqdm

# ============================
# Define Global Parameters
# ============================
Dim = 10  # Dimensionality of the optimization problem
Inner_Population = 100  # Population size for the inner optimizer
Outer_Population = 100  # Population size for the outer optimizer (HPO search)
HPO_Instances = Outer_Population  # Number of instances in HPO
Inner_Iterations = 1000  # Number of iterations for the inner loop (HPO optimization)
Outer_Iterations = 50  # Number of iterations for hyperparameter optimization
tiny_num = 1e-5  # Small value to ensure boundary conditions are respected

# ============================
# Set Random Seed and Device
# ============================
torch.manual_seed(42)
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# Define Hyperparameter Transformation Class
# ============================
class SolutionTransform(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return {
            "self.algorithm.differential_weight": x[:, 0],
            "self.algorithm.cross_probability": x[:, 1],
            "self.algorithm.basevect_prim_type": x[:, 2].floor().long(),
            "self.algorithm.basevect_sec_type": x[:, 3].floor().long(),
            "self.algorithm.num_diff_vects": x[:, 4].floor().long(),
            "self.algorithm.cross_strategy": x[:, 5].floor().long(),
        }


# ============================
# Define Parameter Bounds
# ============================
param_lb = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float32)  # Lower bounds
param_ub = torch.tensor([1, 1, 4 - tiny_num, 4 - tiny_num, 5 - tiny_num, 3 - tiny_num],
                        dtype=torch.float32)  # Upper bounds

# ============================
# Iterate Over CEC2022 Problems F1 ~ F12
# ============================
for problem_number in range(1, 13):
    print(f"\n=== Running CEC2022 Problem F{problem_number} ===\n")

    # ============================
    # Initialize Inner Optimization Process
    # ============================
    inner_algo = ParamDE(Inner_Population, -100 * torch.ones(Dim), 100 * torch.ones(Dim))
    inner_prob = CEC2022(problem_number=problem_number, dimension=Dim)

    inner_monitor = HPOFitnessMonitor()
    inner_monitor.setup()

    inner_workflow = StdWorkflow()
    inner_workflow.setup(inner_algo, inner_prob, monitor=inner_monitor)

    # ============================
    # Transform Inner Workflow into an HPO Problem
    # ============================
    hpo_prob = HPOProblemWrapper(
        iterations=Inner_Iterations, num_instances=HPO_Instances, workflow=inner_workflow, copy_init_state=True
    )

    params = hpo_prob.get_init_params()

    # ============================
    # Initialize HPO Parameters
    # ============================
    params["self.algorithm.differential_weight"] = torch.nn.Parameter(torch.rand(HPO_Instances, 1), requires_grad=False)
    params["self.algorithm.cross_probability"] = torch.nn.Parameter(torch.rand(HPO_Instances, 1), requires_grad=False)
    params["self.algorithm.basevect_prim_type"] = torch.nn.Parameter(torch.randint(0, 4, (HPO_Instances,)),
                                                                     requires_grad=False)
    params["self.algorithm.basevect_sec_type"] = torch.nn.Parameter(torch.randint(0, 4, (HPO_Instances,)),
                                                                    requires_grad=False)
    params["self.algorithm.num_diff_vects"] = torch.nn.Parameter(torch.randint(1, 5, (HPO_Instances,)),
                                                                 requires_grad=False)
    params["self.algorithm.cross_strategy"] = torch.nn.Parameter(torch.randint(0, 3, (HPO_Instances,)),
                                                                 requires_grad=False)

    # ============================
    # Initialize Outer Optimization Process (Hyperparameter Search)
    # ============================
    outer_algo = DE(Outer_Population, param_lb, param_ub)
    monitor = EvalMonitor(full_sol_history=False)

    outer_workflow = StdWorkflow()
    outer_workflow.setup(outer_algo, hpo_prob, monitor=monitor, solution_transform=SolutionTransform())

    # ============================
    # Run Hyperparameter Optimization
    # ============================
    outer_workflow.init_step()

    for _ in tqdm(range(Outer_Iterations), desc=f"Optimizing F{problem_number}"):
        outer_workflow.step()

    # ============================
    # Output Final Results
    # ============================
    monitor = outer_workflow.get_submodule("monitor")

    print(f"\n=== Results for CEC2022 F{problem_number} ===")
    print("Best hyperparameters:\n", monitor.topk_solutions)
    print("Best fitness values:\n", monitor.topk_fitness)
