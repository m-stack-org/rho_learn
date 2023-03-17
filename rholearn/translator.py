"""
Module for performing data format translations between various packages and
rholearn.
"""

import equistore
from equistore import Labels, TensorBlock, TensorMap


def salted_coeffs_to_tensormap(coef_dir: str, n_structures: int) -> TensorMap:
    """
    Convert the flat vector of SALTED electron density coefficients to a
    TensorMap. Assumes the order of the coefficients, for a given water
    molecule, is:

    .. math::

        c_v = \sum_{i \in atoms(A)} \sum_{l}^{l_max} \sum_{n}^{n_max} \sum_{m
        \in [-l, +l]} c^i_{lnm}
    """

    # Generate an empty TensorMap of the correct dimensions
    key_values = []
    for symbol in np.unique(species_symbols):
        species_lmax = lmax[symbol]
        species_center = sym_to_num[symbol]
        for l in range(species_lmax + 1):
            key_values.append([l, species_center])
    keys = Labels(
        names=["spherical_harmonics_l", "species_center"], values=np.array(key_values)
    )

    blocks = []
    for key in keys:
        # print(key)
        l, a = key
        species_symbol = num_to_sym[a]
        samples = Labels(
            names=["structure", "center"],
            values=np.array(
                [
                    [A, i]
                    for A, i in product(
                        range(n_structures),
                        np.where(species_symbols == species_symbol)[0],
                    )
                ]
            ),
        )
        components = [
            Labels(
                names=["spherical_harmonics_m"],
                values=np.arange(-l, l + 1).reshape(-1, 1),
            )
        ]
        properties = Labels(
            names=["n"], values=np.arange(nmax[(species_symbol, l)]).reshape(-1, 1)
        )
        # print(samples, components, properties)
        block = TensorBlock(
            samples=samples,
            components=components,
            properties=properties,
            values=np.zeros(
                (len(samples), *[len(c) for c in components], len(properties)),
                dtype=float,
            ),
        )
        blocks.append(block)

    # Build the TensorMap
    tensor = TensorMap(keys=keys, blocks=blocks)

    # Fill the TensorMap with the coefficients
    for A in range(n_structures):
        # Load the coefficients vector
        coef = np.load(
            os.path.join(
                coef_dir, f"coefficients_conf{A}.npy"
            )
        )
        flat_index = 0
        for i in range(natoms):
            atomic_num = centers[i]
            for l in range(lmax[species_symbols[i]] + 1):
                for n in range(nmax[(species_symbols[i], l)]):
                    for m in range(-l, l + 1):
                        block = tensor.block(
                            spherical_harmonics_l=l, species_center=atomic_num
                        )
                        samples_idx = block.samples.position(label=(A, i))
                        components_idx = block.components[0].position(label=(m,))
                        properties_idx = block.properties.position(label=(n,))
                        block.values[
                            samples_idx, components_idx, properties_idx
                        ] = coef[flat_index]
                        flat_index += 1
    return tensor