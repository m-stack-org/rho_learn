"""
Module to make an electron density prediction on an xyz file using a pretrained
model.
"""
from typing import Optional

import ase.io
import numpy as np
import torch

import qstack
from qstack import equio
from rholearn import io, features, utils


def predict_density(
    xyz_path: str,
    rascal_hypers: dict,
    model_path: str,
    basis: str,
    invariant_means: Optional[str] = None,
):
    """
    Loads a xyz file of structure(s) at `xyz_path` and uses the `rascal_hypers`
    to generate a lambda-SOAP structural representation. Loads the the
    pretrained torch model from `model_path` and uses it to make a prediction on
    the electron density. Returns the prediction both as a TensorMap and as a
    vector of coefficients.

    :param xyz_path: Path to xyz file containing structure to predict density
        for.
    :param rascal_hypers: dict of rascaline hyperparameters to use when
        computing the lambda-SOAP representation of the input structure.
    :param model_path: path to the trained rholearn/torch model to use for
        prediction.
    :param basis: the basis set, i.e. "ccpvqz jkfit", to use when constructing
        the vectorised density coefficients. Must be the same as the basis used
        to calculate the training data.
    :param invariant_means: if the invariant blocks have been standardized by
        subtraction of the mean of their features, the mean needs to be added
        back to the prediction. If so, `invariant_means` should be the path to
        the TensorMap containing these means. Otherwise, pass as None (default).
    """

    # Load xyz file to ASE
    frame = ase.io.read(xyz_path)

    # Create a molecule object with Q-Stack
    mol = qstack.compound.xyz_to_mol(xyz_path, basis=basis)

    # Generate lambda-SOAP representation
    input = features.lambda_soap_vector([frame], rascal_hypers, even_parity_only=True)

    # Load model from file
    model = io.load_torch_object(
        model_path, device=torch.device("cpu"), torch_obj_str="model"
    )

    # Drop blocks from input that aren't present in the model
    input = utils.drop_blocks(input, keys=np.setdiff1d(input.keys, model.keys))

    # Convert the input TensorMap to torch
    input = utils.tensor_to_torch(
        input, requires_grad=False, dtype=torch.float64, device=torch.device("cpu")
    )

    # Make a prediction
    with torch.no_grad():
        out_pred = model(input)

    # Add back the feature means to the invariant (l=0) blocks if the model was trained
    # against electron densities with standardized invariants
    if invariant_means is not None:
        out_pred = features.standardize_invariants(out_pred, invariant_means, reverse=True)

    # Dropt the structure label from the TensorMap
    out_pred = utils.drop_metadata_name(out_pred, axis="samples", name="structure")

    # Rename the TensorMap keys to match LCMD convention
    out_pred = utils.rename_tensor(out_pred, keys_names=["spherical_harmonics_l", "element"])

    # Convert TensorMap to Q-Stack coeffs
    vect_coeffs = qstack.equio.tensormap_to_vector(mol, out_pred)

    return vect_coeffs