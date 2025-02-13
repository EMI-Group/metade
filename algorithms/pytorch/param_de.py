from typing import Dict, Any, Tuple

import torch
from evox.core import Algorithm, Mutable, jit_class, Parameter, _vmap_fix
from evox.utils import clamp

from evox.operators.crossover import (
    DE_arithmetic_recombination,
    DE_binary_crossover,
    DE_exponential_crossover,
)
from evox.operators.selection import select_rand_pbest

def DE_differential_sum(
    diff_padding_num: int, num_diff_vectors: torch.Tensor, index: torch.Tensor, population: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the difference vectors' sum in differential evolution.

    :param diff_padding_num: The number of padding difference vectors.
    :param num_diff_vectors: The number of difference vectors used in mutation.
    :param index: The index of current individual.
    :param population: The population tensor.

    :return: The difference sum and the index of first difference vector.
    """
    device = population.device
    pop_size = population.size(0)
    if num_diff_vectors.ndim == 0:
        num_diff_vectors = num_diff_vectors.unsqueeze(0)

    select_len = num_diff_vectors.unsqueeze(1) * 2 + 1
    rand_indices = torch.randint(0, pop_size, (pop_size, diff_padding_num), device=device)
    rand_indices = torch.where(rand_indices == index.unsqueeze(1), torch.tensor(pop_size - 1, device=device), rand_indices)

    # pop_permute = population[rand_indices]
    pop_permute = population.index_select(0, rand_indices.flatten())#.unflatten(0, (pop_size, diff_padding_num))
    # TEMP
    pop_permute = _vmap_fix.unwrap_batch_tensor(pop_permute)[0].unflatten(1, (pop_size, diff_padding_num))
    pop_permute = _vmap_fix.wrap_batch_tensor(pop_permute, (0,))
    # TEMP END
    mask = torch.arange(diff_padding_num, device=device).unsqueeze(0) < select_len
    pop_permute_padding = torch.where(mask.unsqueeze(2), pop_permute, torch.zeros_like(pop_permute))

    diff_vectors = pop_permute_padding[:, 1:]
    difference_sum = diff_vectors[:, 0::2].sum(dim=1) - diff_vectors[:, 1::2].sum(dim=1)
    return difference_sum, rand_indices[:, 0]

@jit_class
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
        # 将 lb 和 ub 转换为形状 [1, dim] 的 tensor，并移动至指定设备
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


        trial_vectors = torch.where(
            self.cross_strategy.eq(0),
            DE_binary_crossover(mutation_vector, current_vect, CR_tensor),
            torch.where(
                self.cross_strategy.eq(1),
                DE_exponential_crossover(mutation_vector, current_vect, CR_tensor),
                torch.where(
                    self.cross_strategy.eq(2),
                    DE_arithmetic_recombination(mutation_vector, current_vect, CR_tensor),
                    mutation_vector  # 默认回退为突变向量
                )
            )
        )

        trial_vectors = clamp(trial_vectors, self.lb, self.ub)

        # 3. Selection
        new_fit = self.evaluate(trial_vectors)
        compare = new_fit < self.fit

        # 更新种群和适应度
        self.pop = torch.where(compare.unsqueeze(1), trial_vectors, self.pop)
        self.fit = torch.where(compare, new_fit, self.fit)




