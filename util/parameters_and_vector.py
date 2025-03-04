from math import prod
from typing import Dict, List

import torch
import torch.nn as nn

from evox.core import ModuleBase, jit, jit_class, trace_impl, vmap_impl, _vmap_fix
from evox.core._vmap_fix import tree_flatten, tree_unflatten


@jit_class
class ParamsAndVector(ModuleBase):
    """The class to convert (batched) parameters dictionary to vector(s) and vice versa."""

    def __init__(self, dummy_model: nn.Module):
        """
        Initialize the ParamsAndVector instance.

        :param dummy_model: A PyTorch model whose parameters will be used to initialize the parameter and vector conversion attributes. Must be an initialized PyTorch model.
        """
        super().__init__()
        params = dict(dummy_model.named_parameters())
        flat_params, self.param_spec = tree_flatten(params)

        self._jit_tree_flatten = jit(
            lambda x: tree_flatten(x)[0],
            trace=True,
            lazy=False,
            example_inputs=(params,),
        )
        self._jit_tree_unflatten = jit(
            lambda x: tree_unflatten(x, self.param_spec),
            trace=True,
            lazy=False,
            example_inputs=(flat_params,),
        )

        shapes = [x.shape for x in flat_params]
        start_indices = []
        slice_sizes = []
        index = 0
        for shape in shapes:
            start_indices.append(index)
            size = prod(shape)
            slice_sizes.append(size)
            index += size

        self.shapes = tuple(shapes)
        self.start_indices = tuple(start_indices)
        self.slice_sizes = tuple(slice_sizes)

    def _tree_flatten(self, x: Dict[str, nn.Parameter]) -> List[nn.Parameter]:
        return self._jit_tree_flatten(x)

    def _tree_unflatten(self, x: List[nn.Parameter]) -> Dict[str, nn.Parameter]:
        return self._jit_tree_unflatten(x)

    @trace_impl(_tree_flatten)
    def _trace_tree_flatten(self, x: Dict[str, nn.Parameter]) -> List[nn.Parameter]:
        return tree_flatten(x)[0]

    @trace_impl(_tree_unflatten)
    def _trace_tree_unflatten(self, x: List[nn.Parameter]) -> Dict[str, nn.Parameter]:
        return tree_unflatten(x, self.param_spec)

    def to_vector(self, params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Convert the input parameters dictionary to a single vector.

        :param params: The input parameters dictionary.

        :return: The output vector obtained by concatenating the flattened parameters.
        """

        flat_params: List[nn.Parameter] = self._tree_flatten(params)
        flat_params = [x.reshape(-1) for x in flat_params]
        return torch.concat(flat_params, dim=0)

    def batched_to_vector(self, batched_params: Dict[str, nn.Parameter]) -> torch.Tensor:
        """Convert a batched parameters dictionary to a batch of vectors.

        The input dictionary values must be batched parameters, i.e., they must have the same shape at the first dimension.

        :param batched_params: The input batched parameters dictionary.

        :return: The output vectors obtained by concatenating the flattened batched parameters. The first dimension of the output vector corresponds to the batch size.
        """
        flat_params: List[nn.Parameter] = self._tree_flatten(batched_params)
        flat_params = [x.reshape(x.size(0), -1) for x in flat_params]
        return torch.concat(flat_params, dim=1)

    def to_params(self, vector: torch.Tensor) -> Dict[str, nn.Parameter]:
        """Convert a vector back to a parameters dictionary.

        :param vector: The input vector representing flattened model parameters.

        :return: The reconstructed parameters dictionary.
        """
        flat_params = []
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            flat_params.append(vector.narrow(dim=0, start=start_index, length=slice_size).reshape(shape))
        return self._tree_unflatten(flat_params)

    def batched_to_params(self, vectors: torch.Tensor) -> Dict[str, nn.Parameter]:
        """Convert a batch of vectors back to a batched parameters dictionary.

        :param vectors: The input batch of vectors representing flattened model parameters. The first dimension of the tensor corresponds to the batch size.

        :return: The reconstructed batched parameters dictionary whose tensors' first dimensions correspond to the batch size.
        """
        flat_params = []
        batch_size = vectors.size(0)
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            p = vectors.narrow(dim=1, start=start_index, length=slice_size)
            flat_params.append(p.view(batch_size, *shape))
        return self._tree_unflatten(flat_params)

    @vmap_impl(batched_to_params)
    def _vmap_batched_to_params(self, vectors: torch.Tensor) -> Dict[str, nn.Parameter]:
        """Convert a batch of vectors back to a batched parameters dictionary.

        :param vectors: The input batch of vectors representing flattened model parameters. The first dimension of the tensor corresponds to the batch size.

        :return: The reconstructed batched parameters dictionary whose tensors' first dimensions correspond to the batch size.
        """
        flat_params = []
        batch_size = vectors.size(0)
        for start_index, slice_size, shape in zip(self.start_indices, self.slice_sizes, self.shapes):
            p = vectors.narrow(dim=1, start=start_index, length=slice_size)
            ori_p, dims, sizes = _vmap_fix.unwrap_batch_tensor(p)
            if len(dims) == 0:
                flat_params.append(p.view(batch_size, *shape))
            else:
                new_shape = [batch_size, slice_size]
                shape_dim = 1
                for dd, ss in zip(dims, sizes):
                    new_shape.insert(dd, ss)
                    shape_dim += 1 if dd <= shape_dim else 0
                new_shape = new_shape[:shape_dim] + list(shape) + (new_shape[shape_dim+1:] if shape_dim+1 < len(new_shape) else [])
                ori_p = ori_p.view(*new_shape)
                p = _vmap_fix.wrap_batch_tensor(ori_p, dims)
                flat_params.append(p)
        return self._tree_unflatten(flat_params)

    def forward(self, x: torch.Tensor) -> Dict[str, nn.Parameter]:
        """The forward function for the `ParamsAndVector` module is an alias of `batched_to_params` to cope with `StdWorkflow`."""
        return self.batched_to_params(x)
