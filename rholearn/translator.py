"""
Module for performing data format translations between various packages and
rholearn.
"""
from itertools import product
import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap

from rholearn import basis


def salted_coeffs_to_tensormap(
    frames: list, coeffs: np.ndarray, basis_str: str
) -> TensorMap:
    """
    Convert the flat vector of SALTED electron density coefficients to a
    TensorMap. Assumes the order of the coefficients, for a given molecule, is:

    .. math::

        c_v = \sum_{i \in atoms(A)} \sum_{l}^{l_max} \sum_{n}^{n_max} \sum_{m
        \in [-l, +l]} c^i_{lnm}

    :param frames: List of ASE Atoms objects for each xyz structure in the data
        set.
    :param coeffs: np.ndarray of shape (n_structures, n_coefficients), where
        each flat vector of coefficients for a single structure is ordered with
        indices iterating over i, l, n, m, in that nested order - as in the
        equation above.
    :param basis_str: str of the basis set that the density has been expanded
        onto, e.g. "RI-cc-pvqz".

    :return: TensorMap of the  density coefficients, with the appropriate
        indices named as follows: Keys: l -> "spherical_harmonics_l", a ->
        "species_center". Samples: A -> "structure", i -> "center". Components:
        m -> "spherical_harmonics_m". Properties: n -> "n".
    """
    # Check that there is one ceofficient vector per frame
    n_structures = len(frames)
    assert n_structures == coeffs.shape[0]

    # Define some useful variables
    natoms = len(frames[0].get_positions())
    species_symbols = np.array(frames[0].get_chemical_symbols())
    sym_to_num = {"H": 1, "C": 6, "O": 8, "N": 7}
    num_to_sym = {1: "H", 6: "C", 8: "O", 7: "N"}
    centers = np.array([sym_to_num[s] for s in species_symbols])
    species_centers = np.unique(centers)

    # Retrieve basis set dimensions in l and n
    lmax, nmax = basis.basiset(basis_str)

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
        coef = coeffs[A]
        flat_index = 0
        # Iterate over i, l, n, m, (in that nested order)
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


def salted_overlaps_to_tensormap(
    frames: list, overlaps: np.ndarray, basis_str: str
) -> TensorMap:
    """
    Convert the flat vector of SALTED electron density coefficients to a
    TensorMap. Assumes the order of the coefficients, for a given molecule, is:

    .. math::

        c_v = \sum_{i \in atoms(A)} \sum_{l}^{l_max} \sum_{n}^{n_max} \sum_{m
        \in [-l, +l]} c^i_{lnm}

    :param frames: List of ASE Atoms objects for each xyz structure in the data
        set.
    :param overlaps: np.ndarray of shape (n_structures, n_coefficients,
        n_coefficients), where each 2D-matrix of overlap elements for a single
        structure is ordered with indices iterating over i1, l1, n1, m1, along
        one axis and i2, l2, n2, m2, in that nested order - as in the equation
        above.
    :param basis_str: str of the basis set that the density has been expanded
        onto, e.g. "RI-cc-pvqz".

    :return: TensorMap of the density basis set overlap matrix elements, with
        the appropriate indices named as follows: Keys: l1 ->
        "spherical_harmonics_l1", l2 -> "spherical_harmonics_l2", a1 ->
        "species_center_1", a2 -> "species_center_2". Samples: A -> "structure",
        i1 -> "center_1", i2 -> "center_2". Components: m1 ->
        "spherical_harmonics_m2", m1 -> "spherical_harmonics_m2". Properties: n1
        -> "n1", n2 -> "n2".
    """
    # Check that there is one ceofficient vector per frame
    n_structures = len(frames)
    assert n_structures == coeffs.shape[0]

    # Define some useful variables
    natoms = len(frames[0].get_positions())
    species_symbols = np.array(frames[0].get_chemical_symbols())
    sym_to_num = {"H": 1, "C": 6, "O": 8, "N": 7}
    num_to_sym = {1: "H", 6: "C", 8: "O", 7: "N"}
    centers = np.array([sym_to_num[s] for s in species_symbols])
    species_centers = np.unique(centers)

    # Retrieve basis set dimensions in l and n
    lmax, nmax = basis.basiset(basis_str)

    # Generate an empty TensorMap of the correct dimensions
    key_values = []
    for symbol_1 in np.unique(species_symbols):
        species_lmax1 = lmax[symbol_1]
        species_center_1 = sym_to_num[symbol_1]
        for symbol_2 in np.unique(species_symbols):
            species_lmax2 = lmax[symbol_2]
            species_center_2 = sym_to_num[symbol_2]
            for l1 in range(species_lmax1 + 1):
                for l2 in range(species_lmax2 + 1):
                    key_values.append([l1, l2, species_center_1, species_center_2])
    keys = Labels(
        names=[
            "spherical_harmonics_l1",
            "spherical_harmonics_l2",
            "species_center_1",
            "species_center_2",
        ],
        values=np.array(key_values),
    )
    blocks = []
    for key in keys:
        l1, l2, a1, a2 = key
        species_symbol_1 = num_to_sym[a1]
        species_symbol_2 = num_to_sym[a2]
        samples = Labels(
            names=["structure", "center_1", "center_2"],
            values=np.array(
                [
                    [A, i1, i2]
                    for A, i1, i2 in product(
                        range(n_structures),
                        np.where(species_symbols == species_symbol_1)[0],
                        np.where(species_symbols == species_symbol_2)[0],
                    )
                ]
            ),
        )
        components = [
            Labels(
                names=["spherical_harmonics_m1"],
                values=np.arange(-l1, l1 + 1).reshape(-1, 1),
            ),
            Labels(
                names=["spherical_harmonics_m2"],
                values=np.arange(-l2, l2 + 1).reshape(-1, 1),
            ),
        ]
        properties = Labels(
            names=["n1", "n2"],
            values=np.array(
                [
                    [n1, n2]
                    for n1, n2 in product(
                        np.arange(nmax[(species_symbol_1, l1)]),
                        np.arange(nmax[(species_symbol_2, l2)]),
                    )
                ]
            ),
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
        overlap = overlaps[A]

        # Iterate over i1, l1, n1, m1, (in that nested order)
        flat_index_1 = 0
        for i1 in range(natoms):
            atomic_num_1 = centers[i1]
            for l1 in range(lmax[species_symbols[i1]] + 1):
                for n1 in range(nmax[(species_symbols[i1], l1)]):
                    for m1 in range(-l1, l1 + 1):

                        # Iterate over i2, l2, n2, m2, (in that nested order)
                        flat_index_2 = 0
                        for i2 in range(natoms):
                            atomic_num_2 = centers[i2]
                            for l2 in range(lmax[species_symbols[i2]] + 1):
                                for n2 in range(nmax[(species_symbols[i2], l2)]):
                                    for m2 in range(-l2, l2 + 1):

                                        block = tensor.block(
                                            spherical_harmonics_l1=l1,
                                            spherical_harmonics_l2=l2,
                                            species_center_1=atomic_num_1,
                                            species_center_2=atomic_num_2,
                                        )
                                        samples_idx = block.samples.position(
                                            label=(A, i1, i2)
                                        )
                                        components_idx_1 = block.components[0].position(
                                            label=(m1,)
                                        )
                                        components_idx_2 = block.components[1].position(
                                            label=(m2,)
                                        )
                                        properties_idx = block.properties.position(
                                            label=(n1, n2)
                                        )
                                        block.values[
                                            samples_idx,
                                            components_idx_1,
                                            components_idx_2,
                                            properties_idx,
                                        ] = overlap[flat_index_1, flat_index_2]
                                        flat_index_2 += 1
                        flat_index_1 += 1
    return tensor
