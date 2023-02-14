import os
from typing import List, Union, Optional

import numpy as np
import torch

from equistore import Labels, TensorBlock, TensorMap


# ===== tensors to torch fxns


def tensors_to_torch(
    tensors: List[TensorMap],
    requires_grad: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> list:
    """
    Takes a list of TensorMap objects and returns a new list of TensorMap
    objects where the values arrays of each block have been converted to
    torch.tensor objects of specified dtype (recommended and default
    torch.float64). If ``requires_grad=True``, torch's autograd algorithm will
    record operations on the values tensors when they are executed.

    :param tensors: a ``list`` of ``TensorMaps`` whose block values should be
        converted to torch.
    :param requires_grad: bool, whether or not to torch's autograd should record
        operations on this tensor.
    :param dtype: ``torch.dtype``, the base dtype of the resulting torch tensor,
        i.e. ``torch.float64``.
    :param device: ``torch.device``, the device on which the resulting torch
        tensor should be stored, i.e. ``torch.device("cpu")``.
    """

    return [
        tensor_to_torch(
            tensor,
            requires_grad=requires_grad,
            dtype=dtype,
            device=device,
        )
        for tensor in tensors
    ]


def tensor_to_torch(
    tensor: TensorMap,
    requires_grad: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorMap:
    """
    Creates a new :py:class:`TensorMap` where block values are
    PyTorch :py:class:`torch.tensor` objects. Assumes the block
    values are already as a type that is convertible to a
    :py:class:`torch.tensor`, such as a numpy array or RustNDArray.
    The resulting torch tensor dtypes are enforced as
    :py:class:`torch.float64`.

    :param tensor: input :py:class:`TensorMap`, with block values
        as ndarrays.
    :param requires_grad: bool, whether or not to torch's autograd should record
        operations on this tensor.
    :param dtype: ``torch.dtype``, the base dtype of the resulting torch tensor,
        i.e. ``torch.float64``.
    :param device: ``torch.device``, the device on which the resulting torch
        tensor should be stored, i.e. ``torch.device("cpu")``.

    :return: a :py:class:`TensorMap` where the values tensors of each
        block are now of type :py:class:`torch.tensor`.
    """

    return TensorMap(
        keys=tensor.keys,
        blocks=[
            block_to_torch(
                block,
                requires_grad=requires_grad,
                dtype=dtype,
                device=device,
            )
            for _, block in tensor
        ],
    )


def block_to_torch(
    block: TensorBlock,
    requires_grad: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorBlock:
    """
    Creates a new :py:class:`TensorBlock` where block values are PyTorch
    :py:class:`torch.tensor` objects. Assumes the block values are already as a
    type that is convertible to a :py:class:`torch.tensor`, such as a numpy
    array or RustNDArray. The resulting torch tensor dtypes are enforced as
    :py:class:`torch.float64`.

    :param block: input :py:class:`TensorBlock`, with block values as ndarrays.
    :param requires_grad: bool, whether or not to torch's autograd should record
        operations on this tensor.
    :param dtype: ``torch.dtype``, the base dtype of the resulting torch tensor,
        i.e. ``torch.float64``.
    :param device: ``torch.device``, the device on which the resulting torch
        tensor should be stored, i.e. ``torch.device("cpu")``.

    :return: a :py:class:`TensorBlock` whose values tensor is now of type
        :py:class:`torch.tensor`.
    """

    # Create new block, with the values tensor converted to a torch tensor.
    new_block = TensorBlock(
        values=torch.tensor(block.values, requires_grad=requires_grad, dtype=dtype).to(
            device.type
        ),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    # Add gradients to each block, again where the values tensor are torch
    # tensors.
    for parameter, gradient in block.gradients():
        new_block.add_gradient(
            parameter,
            torch.tensor(gradient.data, requires_grad=requires_grad, dtype=dtype).to(
                device.type
            ),
            gradient.samples,
            gradient.components,
        )

    return new_block


def random_tensor_like(tensor: Union[TensorMap, TensorBlock, torch.Tensor]):
    """
    Returns a torch-based TensorMap, TensorBlock, or torch Tensor with the same
    shape and type as the input ``tensor``, but with random values.
    """
    # Input is a TensorMap, return a TensorMap
    if isinstance(tensor, TensorMap):
        new_blocks = []
        for key, block in tensor:

            new_block = TensorBlock(
                samples=block.samples,
                components=[c for c in block.components],
                properties=block.properties,
                values=torch.rand(block.values.shape),
            )
            new_blocks.append(new_block)
        return TensorMap(
            keys=tensor.keys,
            blocks=new_blocks,
        )
    # Input is a TensorBlock, return a TensorBlock
    elif isinstance(tensor, TensorBlock):
        return TensorBlock(
            samples=tensor.samples,
            components=[c for c in tensor.components],
            properties=tensor.properties,
            values=torch.rand(tensor.values.shape),
        )
    # Input is a torch Tensor, return a torch Tensor
    elif isinstance(tensor, torch.Tensor):
        return torch.rand(tensor.shape)
    else:
        raise TypeError("must pass either a TensorMap, TensorBlock, or torch Tensor")


# ===== tensors to numpy fxns


def tensor_to_numpy(tensor: TensorMap) -> TensorMap:
    """
    Takes a TensorMap object whose block values are torch.tensor objects and
    converts them to numpy arrays of dtype np.float64. Returns a new TensorMap
    object.
    """
    return TensorMap(
        keys=tensor.keys,
        blocks=[block_to_numpy(block) for _, block in tensor],
    )


def block_to_numpy(block: TensorBlock) -> TensorBlock:
    """
    Takes a TensorBlock object whose values are torch.tensor objects and
    converts them to numpy arrays of dtype np.float64. Returns a new TensorBlock
    object.
    """
    # Create new block, with the values tensor converted to a torch tensor.
    new_block = TensorBlock(
        values=np.array(block.values.detach().numpy(), dtype=np.float64),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    # Add gradients to each block, again where the values tensor are torch
    # tensors.
    for parameter, gradient in block.gradients():
        new_block.add_gradient(
            parameter,
            np.array(gradient.data.detach().numpy(), dtype=np.float64),
            gradient.samples,
            gradient.components,
        )

    return new_block


def make_contiguous_numpy(tensor: TensorMap) -> TensorMap:
    """
    Takes a TensorMap whose block values are ndarrays and ensures they are
    contiguous. Allows tensors produced by slicing/splitting to be saved to file
    using the equistore.io.save method.
    """

    new_blocks = []
    for key, block in tensor:
        new_block = TensorBlock(
            samples=block.samples,
            components=block.components,
            properties=block.properties,
            values=np.ascontiguousarray(block.values),
        )
        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                samples=gradient.samples,
                components=gradient.components,
                data=np.ascontiguousarray(gradient.data),
            )
        new_blocks.append(new_block)

    return TensorMap(
        keys=tensor.keys,
        blocks=new_blocks,
    )


# ===== fxns converting types of Labels object entries:
# i.e. np.void <--> tuple <--> str


def key_to_str(key: Union[np.void, tuple]) -> str:
    """
    Takes a single numpy void key (i.e. an element of a Labels object) or a
    tuple and converts it to a string. Return string has the form
    f"{key[0]}_{key[1]}_...", where element values are separated by undescores.
    """
    return "_".join([str(k) for k in key])


def key_npvoid_to_tuple(key: np.void) -> tuple:
    """
    Takes a single numpy void key (i.e. an element of a Labels object) and
    converts it to a tuple.
    """
    return tuple(k for k in key)


def key_tuple_to_npvoid(key: tuple, names: List[str]) -> np.void:
    """
    Takes a key as a tuple and the associated names of the values in that tuple
    and returns a numpy void object that can be used to access blocks in a
    TensorMap, as well as values in a dict that are indexed by these numpy void
    keys.
    """
    # We need to create a TensorMap object here, as this allows a hashable object that
    # can be used to access dict values to be returned.
    tensor = TensorMap(
        keys=Labels(
            names=names,
            values=np.array(
                [key],
                dtype=np.int32,
            ),
        ),
        blocks=[
            TensorBlock(
                values=np.full((1, 1), 0.0),
                samples=Labels(
                    ["s"],
                    np.array(
                        [
                            [0],
                        ],
                        dtype=np.int32,
                    ),
                ),
                components=[],
                properties=Labels(["p"], np.array([[0]], dtype=np.int32)),
            )
        ],
    )
    return tensor.keys[0]


# ===== fxns for equistore Labels objects comparisons


def searchable_labels(labels: Labels):
    """
    Returns the input Labels object but after being used to construct a
    TensorBlock, so that look-ups can be performed.
    """
    return TensorBlock(
        values=np.full((len(labels), 1), 0.0),
        samples=labels,
        components=[],
        properties=Labels(["p"], np.array([[0]], dtype=np.int32)),
    ).samples


def labels_equal(a: Labels, b: Labels, correct_order: bool = True):
    """
    For 2 :py:class:`Labels` objects ``a`` and ``b``, returns true if they are
    exactly equivalent in names and values. If ``correct_order=True`` (the
    default), checks also that the order of the elements in the Labels is
    exactly equivalent.
    """
    # They can only be equivalent if the same length
    if len(a) != len(b):
        return False
    if not correct_order:
        a, b = np.sort(a), np.sort(b)
    return np.all(np.array(searchable_labels(a) == searchable_labels(b)))


def labels_intersection(a: Labels, b: Labels):
    """
    For 2 :py:class:`Labels` objects ``a`` and ``b``, returns a new Labels
    object of the indices that they share, i.e. the intersection.
    """
    # Create dummy TensorBlocks with a and b as the samples labels
    a = searchable_labels(a)
    b = searchable_labels(b)
    # Find the intersection
    intersection_idxs = [i for i in [a.position(b_i) for b_i in b] if i is not None]
    return a[intersection_idxs]


def get_feature_labels(tensor: Union[TensorMap, TensorBlock]) -> Union[dict, Labels]:
    """
    Returns the feature labels for the input ``tensor``. If passing a TensorMap,
    this function returns a dict of Labels object, indexed by the block keys,
    with values that correspond to the properties Labels of each block. If
    passing just a TensorBlock, a Labels object corresponding to the input
    ``tensor`` properties Labels is returned.
    """
    if isinstance(tensor, TensorMap):
        return {key: block.properties for key, block in tensor}
    elif isinstance(tensor, TensorBlock):
        return tensor.properties
    else:
        raise TypeError("``tensor`` must be either a TensorMap or TensorBlock")


# ===== TensorMap + TensorBlock functions


def num_elements_tensor(tensor: Union[TensorMap, TensorBlock, torch.Tensor]) -> int:
    """
    Returns the total number of elements in the input tensor.

    If the input tensor is a TensorMap the number of elements is given by the
    sum of the product of the dimensions for each block.

    If the input tensor is a TensorBlock or a torch.Tensor, the number of
    elements is just given by the product of the dimensions.
    """
    if isinstance(tensor, TensorMap):
        return torch.sum([torch.prod(block.values.shape) for _, block in tensor])
    elif isinstance(tensor, TensorBlock):
        return torch.prod(tensor.values.shape)
    elif isinstance(tensor, torch.Tensor):
        return torch.prod(tensor.size)
    else:
        raise TypeError("must pass either a TensorMap, TensorBlock, or torch.Tensor")


def delta_tensor(input: TensorMap, target: TensorMap, absolute: bool = False):
    """
    Returns a :py:class:`TensorMap` whose block values are the difference
    between the block values of the input and target tensors. The keys,
    samples, components and properties Labels must be the same for both
    tensors. For each :py:class:`TensorBlock` in the ``input`` tensor, the
    values of the corresponding :py:class:`TensorBlock` in the ``target``
    tensor is subtracted, i.e. input_block.values - target_block.values for
    each block in the input/target TensorMaps.

    If ``absolute`` is true, the absolute difference |input - target|.
    """
    # Check tensor metadata is equivalent
    equivalent_metadata(input, target)

    # Calculate the delta tensor
    blocks = []
    for key in target.keys:
        values = input[key].values - target[key].values
        if absolute:
            values = abs(values)
        blocks.append(
            TensorBlock(
                samples=target[key].samples,
                components=target[key].components,
                properties=target[key].properties,
                values=values,
            )
        )
    delta = TensorMap(keys=target.keys, blocks=blocks)
    return delta


def equivalent_metadata(
    a: TensorMap,
    b: TensorMap,
    keys: bool = True,
    samples: bool = True,
    components: bool = True,
    properties: bool = True,
):
    """
    For each of the Labels objects ``keys``, ``samples``, ``components`` and
    ``properties``, if passed as true, checks that these metadata are equivalent
    between the 2 TensorMaps ``a`` and ``b``. Keys are checked for equivalent
    values, ignoring order. Samples, components, and properties are checked for
    exact equivalent, including order.

    If any of the tests fail, a ValueError is raised. Note: gradient comparisons
    aren't yet implemented.
    """
    if keys:
        if not labels_equal(a.keys, b.keys, correct_order=False):
            raise ValueError("input TensorMaps have different key Labels")
    for key in a.keys:
        # Check samples
        if samples:
            if not labels_equal(a[key].samples, b[key].samples, correct_order=True):
                raise ValueError(
                    "samples Labels for ``a`` and ``b`` must be exactly equivalent."
                    + f" Offending block at key {key}"
                )
        # Check components
        if components:
            for c_i in range(len(a[key].components)):
                if not labels_equal(
                    a[key].components[c_i], b[key].components[c_i], correct_order=True
                ):
                    raise ValueError(
                        "components Labels for ``a`` and ``b`` must be exactly equivalent."
                        + f" Offending block at key {key}"
                    )
        # Check properties
        if properties:
            if not labels_equal(
                a[key].properties, b[key].properties, correct_order=True
            ):
                raise ValueError(
                    "properties Labels for ``a`` and ``b`` must be exactly equivalent."
                    + f" Offending block at key {key}"
                )


def rename_tensor(
    tensor: TensorMap,
    keys_names: Optional[List[str]] = None,
    samples_names: Optional[List[str]] = None,
    components_names: Optional[List[str]] = None,
    properties_names: Optional[List[str]] = None,
) -> TensorMap:
    """
    Constructs and returns a new TensorMap where the names of the Labels
    metadata for the keys, samples, components, and/or properties have been
    modified according to the new names. Note: does not yet handle gradients.
    """
    new_keys = tensor.keys
    # Modify key names
    if keys_names is not None:
        if len(keys_names) != len(tensor.keys.names):
            raise ValueError(
                "must pass the same number of new keys names as there are old ones"
            )
        new_keys = Labels(
            names=keys_names,
            values=np.array([[i for i in k] for k in tensor.keys]),
        )
    if samples_names is None and components_names is None and properties_names is None:
        new_blocks = [tensor[key].copy() for key in tensor.keys]
    else:
        new_blocks = [
            rename_block(
                tensor[key],
                samples_names=samples_names,
                components_names=components_names,
                properties_names=properties_names,
            )
            for key in tensor.keys
        ]
    return TensorMap(keys=new_keys, blocks=new_blocks)


def rename_block(
    block: TensorBlock,
    samples_names: Optional[List[str]] = None,
    components_names: Optional[List[str]] = None,
    properties_names: Optional[List[str]] = None,
) -> TensorBlock:
    """
    Constructs and returns a new TensorBlock where the names of the Labels
    metadata for samples, components, and/or properties have been modified
    according to the new names. Note: does not yet handle gradients.
    """
    new_samples = block.samples
    new_components = block.components
    new_properties = block.properties
    # Modify samples names
    if samples_names is not None:
        if len(samples_names) != len(block.samples.names):
            raise ValueError(
                "must pass the same number of new samples names as there are old ones"
            )
        samp_values = np.array(
            [i for s in block.samples for i in s], dtype=np.int32
        ).reshape(-1, len(samples_names))
        new_samples = Labels(
            names=samples_names,
            values=samp_values,
        )
    # Modify components names
    if components_names is not None:
        if len(block.components) != len(components_names):
            raise ValueError("must pass same number of new components as old")
        new_components = []
        for c_i in range(len(block.components)):
            if len(components_names[c_i]) != len(block.components[c_i].names):
                raise ValueError(
                    "must pass the same number of new components names as there are old ones"
                )
            comp_values = np.array(
                [j for i in block.components[c_i] for j in i], dtype=np.int32
            ).reshape(-1, len(components_names[c_i]))
            new_components.append(
                Labels(
                    names=components_names[c_i],
                    values=comp_values,
                )
            )
    # Modify properties names
    if properties_names is not None:
        if len(properties_names) != len(block.properties.names):
            raise ValueError(
                "must pass the same number of new properties names as there are old ones"
            )
        prop_values = np.array(
            [i for p in block.properties for i in p], dtype=np.int32
        ).reshape(-1, len(properties_names))
        new_properties = Labels(
            names=properties_names,
            values=prop_values,
        )

    return TensorBlock(
        samples=new_samples,
        components=new_components,
        properties=new_properties,
        values=block.values,
    )


def pad_with_empty_blocks(
    input: TensorMap, target: TensorMap, slice_axis: str = "samples"
) -> TensorMap:
    """
    Takes an ``input`` TensorMap with fewer blocks than the ``target``
    TensorMap. For every key present in ``target`` but not ``input``, an empty
    block is created by slicing the ``target`` block to zero dimension along the
    ``slice_axis``, which is either "samples" or "properties". For every key
    present in both ``target`` and ``input``, the block is copied exactly from
    ``input``. A new TensorMap, with the same number of blocks at ``target``,
    but all the original data from ``input``, is returned.
    """
    blocks = []
    for key in target.keys:
        if key in input.keys:
            # Keep the block
            blocks.append(input[key].copy())
        else:
            samples = target[key].samples
            properties = target[key].properties
            # Create an empty sliced block
            if slice_axis == "samples":
                values = target[key].values[:0]
                samples = samples[:0]
            else:  # properties
                target[key].values[..., :0]
                properties = properties[:0]
            blocks.append(
                TensorBlock(
                    samples=samples,
                    components=target[key].components,
                    properties=properties,
                    values=values,
                )
            )
    return TensorMap(keys=target.keys, blocks=blocks)


def equal_metadata(a: TensorMap, b: TensorMap):
    """Checks for equivalence in metadata between 2 TensorMaps"""
    # Check for equivalence in keys Labels (order not important)
    keys_a = a.keys
    keys_b = b.keys
    assert labels_equal(keys_a, keys_b, correct_order=False)

    # Check for exact equivalence (including exact order) between samples and
    # components Labels for each block in a and b
    for key in a.keys:
        samples_a = a[key].samples
        samples_b = b[key].samples
        if not labels_equal(samples_a, samples_b, correct_order=True):
            raise ValueError(
                f"a and b blocks at key {key} have inequivalent samples Labels"
            )
        components_a = a[key].components
        components_b = b[key].components
        assert len(components_a) == len(components_b)
        for c_i in range(len(components_a)):
            if not labels_equal(components_a[c_i], components_b[c_i]):
                raise ValueError(
                    f"a and b blocks at key {key} have inequivalent components Labels"
                )


# ===== other utility functions


def get_log_subset_sizes(
    n_max: int,
    n_subsets: int,
    base: Optional[float] = np.e,
) -> np.array:
    """
    Returns an ``n_subsets`` length array of subset sizes equally spaced along a
    log of specified ``base`` (default base e) scale from 0 up to ``n_max``.
    Elements of the returned array are rounded to integer values. The final
    element of the returned array may be less than ``n_max``.
    """
    # Generate subset sizes evenly spaced on a log (base e) scale
    subset_sizes = np.logspace(
        np.log(n_max / n_subsets),
        np.log(n_max),
        num=n_subsets,
        base=base,
        endpoint=True,
        dtype=int,
    )
    return subset_sizes
