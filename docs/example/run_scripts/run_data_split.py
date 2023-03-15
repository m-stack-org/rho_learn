import os
import ase.io
import numpy as np

import equistore   # storage format for atomistic ML
from equistore import Labels

from equisolve.utils import split_data

import rholearn    # torch-based density leaning
from rholearn import io, features, utils


RHOLEARN_DIR = "/Users/joe.abbott/Documents/phd/code/rho/rho_learn/"  # for example
data_dir = os.path.join(RHOLEARN_DIR, "docs/example/water/data")


# Read the water molecules from file
n_structures = 1000
frames = ase.io.read(
    os.path.join(data_dir, "water_monomers_1k.xyz"), index=f":{n_structures}"
)

# Define and save rascaline hypers
rascal_hypers = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 6,  # Exclusive
    "max_angular": 5,  # Inclusive
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}
io.pickle_dict(os.path.join(data_dir, "rascal_hypers.pickle"), rascal_hypers)

# Compute lambda-SOAP: uses rascaline to compute a SphericalExpansion
# Runtime approx 25 seconds
input = features.lambda_soap_vector(
    frames, rascal_hypers, even_parity_only=True
)

# Drop the block for l=5, Hydrogen as this isn't included in the output electron density
input = equistore.drop_blocks(input, keys=Labels(input.keys.names, np.array([[5, 1]])))

# Save lambda-SOAP descriptor to file
equistore.save(os.path.join(data_dir, "lambda_soap.npz"), input)

# Load the electron density data
output = equistore.load(os.path.join(data_dir, "e_densities.npz"))

# Check that the metadata of input and output match along the samples and components axes
assert equistore.equal_metadata(input, output, check=["samples", "components"])

# Split the data into training, validation, and test sets
[[in_train, in_test, in_val], [out_train, out_test, out_val]], grouped_labels = split_data(
    [input, output],
    axis="samples",
    names="structure",
    n_groups=3,
    group_sizes=[800, 199, 1],
    seed=10,
)
tm_files = {
    "in_train.npz": in_train,
    "in_test.npz": in_test,
    "out_train.npz": out_train,
    "out_test.npz": out_test,
    "in_val.npz": in_val,
    "out_val.npz": out_val,
}
# Save the TensorMaps to file
for name, tm in tm_files.items():
    equistore.save(os.path.join(data_dir, name), tm)