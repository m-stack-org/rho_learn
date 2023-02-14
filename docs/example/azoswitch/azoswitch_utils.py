"""
A module containing helper functions specific to the example workflow.
"""
import os
import numpy as np

from equistore import io, Labels, TensorBlock, TensorMap


def clean_azoswitch_lambda_soap(input: TensorMap) -> TensorMap:
    """
    Cleans the lambda-SOAP structural descriptor for the azoswitch database by
    a) dropping all blocks with odd parity (i.e. sigma = -1), b) dropping the
    block corresponding to lambda channel 5 and species Hydrogen, and c)
    modifies the keys of the blocks such that the 'order_nu' name is omitted.
    """
    # Record at which index the key names appear in the input TensorMap key
    # Labels, makes indexing agnostic to ordering.
    key_idxs = {key_name: idx for idx, key_name in enumerate(input.keys.names)}

    # Iterate over key: block pairs in the lambda-SOAP descriptor and store the
    # indices of the keys that should be kept in the return TensorMap.
    idxs_to_keep = []
    for idx, key in enumerate(input.keys):

        # Skip the block for l=5, hydrogen
        if (
            key[key_idxs["spherical_harmonics_l"]] == 5
            and key[key_idxs["species_center"]] == 1
        ):
            continue

        # Skip the blocks with parity = -1
        if key[key_idxs["inversion_sigma"]] == -1:
            continue

        # Store the index of the key to keep if passed the above conditionss
        idxs_to_keep.append(idx)

    # Keep only the key names indicating lambda and species, i.e. drop the
    # 'order_nu' and 'inversion_sigma'
    new_names = ["spherical_harmonics_l", "species_center"]

    # Slice the keys Labels object to generate keys of blocks to keep
    new_keys = Labels(
        names=new_names,
        values=np.array(
            [
                [k[key_idxs[new_names[0]]], k[key_idxs[new_names[1]]]]
                for k in input.keys[idxs_to_keep]
            ]
        ),
    )
    new_blocks = [input[k].copy() for k in input.keys[idxs_to_keep]]

    # Construct and return the cleaned TensorMap
    return TensorMap(keys=new_keys, blocks=new_blocks)


def recombine_coulomb_matrices(data_dir: str, num_sub_matrices: int):
    """
    The TensorMap of coulomb matrices for the azoswitch workflow example is too
    large to store as a single file in the GitHub repository. As such, the
    TensorMap has been split into 3 smaller TensorMaps and saved under the names
    "coulomb_matrices_i.npz", where i = 0, 1, or 2, each comprised of 120
    blocks. This function recombines them into a single TensorMap, saves it to
    the same directory under the filename "coulomb_matrices_full.npz", and
    returns it.
    """
    # Load the individual TensorMaps into a list
    cm_list = [
        io.load(
            os.path.join(data_dir, "coulomb_matrices/", f"coulomb_matrices_{i}.npz")
        )
        for i in range(num_sub_matrices)
    ]

    # Combine them
    combined_keys, combined_blocks = [], []
    for tensor in cm_list:
        for key, block in tensor:
            combined_keys.append([k for k in key])
            combined_blocks.append(block.copy())

    # Create a Labels object for the combined keys
    combined_keys = Labels(names=cm_list[0].keys.names, values=np.array(combined_keys))

    # Build the combined TensorMap and save to file
    combined_tensor = TensorMap(
        keys=combined_keys,
        blocks=combined_blocks,
    )
    io.save(os.path.join(data_dir, "coulomb_matrices.npz"), combined_tensor)

    return combined_tensor

def drop_structure_label(tensor: TensorMap):
    """
    Drops 'structure' from the samples Labels of each block in the input
    ``tensor`` and returns a new TensorMap.
    """
    new_blocks = []
    for key in tensor.keys:
        new_block = TensorBlock(
            samples=Labels(
                names=[
                    "center",
                ],
                values=tensor[key].samples["center"].reshape(-1, 1),
            ),
            components=[c for c in tensor[key].components],
            properties=tensor[key].properties,
            values=tensor[key].values,
        )
        new_blocks.append(new_block)
    return TensorMap(
        keys=tensor.keys,
        blocks=new_blocks,
    )