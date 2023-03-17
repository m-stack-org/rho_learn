import os
import pprint
import numpy as np
import torch

import equistore

import rholearn
from rholearn import io, features, pretraining, training, utils
from settings import DATA_SETTINGS, ML_SETTINGS


# Create simulation run directory and save simulation
io.check_or_create_dir(ML_SETTINGS["run_dir"])
with open(os.path.join(ML_SETTINGS["run_dir"], "ml_settings.txt"), "a+") as f:
    f.write(f"ML Settings:\n{pprint.pformat(ML_SETTINGS)}\n\n")

# IMPORTANT! - set the torch default dtype
torch.set_default_dtype(ML_SETTINGS["torch"]["dtype"])

# Pre-construct the appropriate torch objects (i.e. models, loss fxns)
pretraining.construct_torch_objects_in_train_dir(
    DATA_SETTINGS["data_dir"], ML_SETTINGS["run_dir"], ML_SETTINGS, 
)

# Define the training subdirectory
train_rel_dir = ""
train_run_dir = os.path.join(ML_SETTINGS["run_dir"], train_rel_dir)

# Load training data and torch objects
data, model, loss_fn, optimizer = pretraining.load_training_objects(
    train_rel_dir, DATA_SETTINGS["data_dir"], ML_SETTINGS, ML_SETTINGS["training"]["restart_epoch"]
)

# Unpack the data
in_train, in_test, out_train, out_test = data

# Execute model training
training.train(
    in_train=in_train,
    out_train=out_train,
    in_test=in_test,
    out_test=out_test,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    n_epochs=ML_SETTINGS["training"]["n_epochs"],
    save_interval=ML_SETTINGS["training"]["save_interval"],
    save_dir=train_run_dir,
    restart=ML_SETTINGS["training"]["restart_epoch"],
)