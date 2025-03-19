import torch
from evox.core import Algorithm, Mutable, Parameter
from util import clamp, switch
from util.operators.crossover import (
    DE_differential_sum,
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_exponential_crossover,
)
from util.operators.selection import select_rand_pbest


class ParamDE(Algorithm):

    def __init__(
            self,
            pop_size: int,
            lb: torch.Tensor,
            ub: torch.Tensor,
            diff_padding_num: int = 9,
            differential_weight: float = 0.3471559,
            cross_probability: float = 0.78762645,
            basevect_prim_type: int = 0,
            basevect_sec_type: int = 2,
            num_diff_vects: int = 3,
            cross_strategy: int = 2,
            device: torch.device | None = None,
    ):
        super().__init__()
        device = torch.get_default_device() if device is None else device

        self.pop_size = pop_size
        self.dim = lb.shape[0]
        self.lb = lb[None, :].to(device=device)
        self.ub = ub[None, :].to(device=device)
        self.diff_padding_num = diff_padding_num

        self.differential_weight = Parameter(differential_weight, device=device)
        self.cross_probability = Parameter(cross_probability, device=device)
        self.basevect_prim_type = Parameter(basevect_prim_type, device=device)
        self.basevect_sec_type = Parameter(basevect_sec_type, device=device)
        self.num_diff_vects = Parameter(torch.tensor(num_diff_vects, device=device), device=device)
        self.cross_strategy = Parameter(cross_strategy, device=device)

        pop = torch.rand(pop_size, self.dim, device=device)
        pop = pop * (self.ub - self.lb) + self.lb
        self.pop = Mutable(pop)
        self.fit = Mutable(torch.full((pop_size,), float("inf"), device=device))

    def init_step(self):
        """
        Perform the initial evaluation of the population's fitness and proceed to the first optimization step.

        This method evaluates the fitness of the initial population and then calls the `step` method to perform the first optimization iteration.
        """
        self.fit = self.evaluate(self.pop)
        self.step()

    def step(self):
        device = self.pop.device
        pop_size = self.pop_size
        indices = torch.arange(pop_size, device=device)

        diff_sum, first_diff_idx = DE_differential_sum(
            self.diff_padding_num,
            self.num_diff_vects,
            indices,
            self.pop,
        )

        rand_vect = self.pop[first_diff_idx]
        best_index = torch.argmin(self.fit)
        best_vect = self.pop[best_index].unsqueeze(0).expand(pop_size, self.dim)
        pbest_vect = select_rand_pbest(0.05, self.pop, self.fit)
        current_vect = self.pop

        vector_merge = torch.stack((rand_vect, best_vect, pbest_vect, current_vect), dim=0)
        base_vector_prim = vector_merge[self.basevect_prim_type]
        base_vector_sec = vector_merge[self.basevect_sec_type]

        base_vector = base_vector_prim + self.differential_weight * (base_vector_sec - base_vector_prim)
        mutation_vector = base_vector + diff_sum * self.differential_weight

        # 2. Crossover
        CR_tensor = torch.ones(pop_size, device=device) * self.cross_probability

        trial_vectors = switch(
            self.cross_strategy,
            [
                DE_binary_crossover(mutation_vector, current_vect, CR_tensor),
                DE_exponential_crossover(mutation_vector, current_vect, CR_tensor),
                DE_arithmetic_recombination(mutation_vector, current_vect, CR_tensor),
            ]
        )

        trial_vectors = clamp(trial_vectors, self.lb, self.ub)

        # 3. Selection
        new_fit = self.evaluate(trial_vectors)
        compare = new_fit < self.fit

        self.pop = torch.where(compare.unsqueeze(1), trial_vectors, self.pop)
        self.fit = torch.where(compare, new_fit, self.fit)
