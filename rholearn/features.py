"""
Generates features vectors for equivariant structural representations.
Currently implemented:
    - lambda-SOAP
"""
import os
import pickle
from typing import List, Optional

import numpy as np

import rascaline

# HACK: force loading the version of equistore inside rascaline
rascaline._c_lib._get_library()
from rascaline import SphericalExpansion

import equistore
from equistore import io, Labels, TensorMap

from rholearn.spherical import (
    ClebschGordanReal,
    acdc_standardize_keys,
    cg_increment,
)
from rholearn.io import check_or_create_dir


def lambda_soap_vector(
    frames: list,
    rascal_hypers: dict,
    save_dir: Optional[str] = None,
    neighbor_species: Optional[List[int]] = None,
) -> TensorMap:
    """
    Takes a list of frames of ASE loaded structures and a dict of Rascaline
    hyperparameters and generates a lambda-SOAP (i.e. nu=2) representation of
    the data.

    :param frames: a list of structures generated by the ase.io function.
    :param rascal_hypers: a dict of hyperparameters used to calculate the atom
        density correlation calculated with the Rascaline SphericalExpansion
        calculator.
    :param save_dir: a str of the absolute path to the directory where the
        TensorMap of the calculated lambda-SOAP representation and pickled
        ``rascal_hypers`` dict should be written. If none, the TensorMap will
        not be saved.
    :param neighbor_species: a list of int that correspond to the atomic
        charges of all the neighbour species that you want to be in your
        properties (or features) dimension. This list may contain charges for
        atoms that don't appear in ``frames``, but are included anyway so that
        the one can enforce consistent properties dimension size with other
        lambda SOAP feature vectors.
    """
    # Create save directory
    if save_dir is not None:
        check_or_create_dir(save_dir)

    # Generate Rascaline hypers and Clebsch-Gordon coefficients
    calculator = SphericalExpansion(**rascal_hypers)
    cg = ClebschGordanReal(l_max=rascal_hypers["max_angular"])

    # Generate descriptor via Spherical Expansion
    acdc_nu1 = calculator.compute(frames)

    # nu=1 features
    acdc_nu1 = acdc_standardize_keys(acdc_nu1)

    # Move "species_neighbor" sparse keys to properties with enforced atom
    # charges if ``neighbor_species`` is specified. This is required as the CG
    # iteration code currently does not handle neighbour species padding
    # automatically.
    keys_to_move = "species_neighbor"
    if neighbor_species is not None:
        keys_to_move = Labels(
            names=(keys_to_move,),
            values=np.array(neighbor_species).reshape(-1, 1),
        )
    acdc_nu1 = acdc_nu1.keys_to_properties(keys_to_move=keys_to_move)

    # Combined nu=1 features to generate nu=2 features. lambda-SOAP is defined
    # as just the nu=2 features.
    acdc_nu2 = cg_increment(
        acdc_nu1,
        acdc_nu1,
        clebsch_gordan=cg,
        lcut=rascal_hypers["max_angular"],
        other_keys_match=["species_center"],
    )

    # Write to file
    if save_dir is not None:
        # Rascaline hypers
        with open(os.path.join(save_dir, "rascal_hypers.pickle"), "wb") as handle:
            pickle.dump(rascal_hypers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Lambda-SOAP
        io.save(os.path.join(save_dir, "lambda_soap.npz"), acdc_nu2)

    return acdc_nu2


def lambda_soap_kernel(lsoap_vector: TensorMap) -> TensorMap:
    """
    Takes a lambda-SOAP feature vector (as a TensorMap) and takes the relevant
    inner products to form a lambda-SOAP kernel, returned as a TensorMap.
    """
    return
