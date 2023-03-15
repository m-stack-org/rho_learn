import os
import pprint
import numpy as np
import torch

import equistore

import rholearn
from rholearn import io, features, pretraining, training, utils

RHOLEARN_DIR = "/Users/joe.abbott/Documents/phd/code/rho/rho_learn/"  # for example
data_dir = os.path.join(RHOLEARN_DIR, "docs/example/water/data")


run_dir = os.path.join(RHOLEARN_DIR, "docs/example/water/runs")
io.check_or_create_dir(run_dir)

settings = {
    "io": {
        "data_dir": os.path.join(data_dir),
        "run_dir": os.path.join(run_dir, "00_nonlinear_1"),
    },
    "torch": {
        "requires_grad": True,  # needed to track gradients
        "dtype": torch.float64,  # recommended
        "device": torch.device("cpu"),  # which device to load tensors to
    },
    "model": {
        "type": "nonlinear",  # linear or nonlinear
        "args": {
            "hidden_layer_widths": [8, 8],
            "activation_fn": "SiLU"
        },
    },
    "optimizer": {
        "algorithm": torch.optim.LBFGS,
        "args": {
            "lr": 0.25,
        },
    },
    "loss": {
        "fn": "MSELoss",  # CoulombLoss or MSELoss
        "args": {
            "reduction": "sum",  # reduction can be used with MSELoss
        },
    },
    "training": {
        "n_epochs": 50,  # number of total epochs to run
        "save_interval": 50,  # save model and optimizer state every x intervals
        "restart_epoch": None,  # None, or the epoch checkpoint number if restarting
        "standardize_invariant_features": True,
    },
}
# Save settings
io.check_or_create_dir(settings["io"]["run_dir"])
io.pickle_dict(os.path.join(settings["io"]["run_dir"], "train_settings.pickle"), settings)
with open(os.path.join(settings["io"]["run_dir"], "train_settings.txt"), "w+") as f:
    f.write(f"Settings:\n\n{pprint.pformat(settings)}")

# IMPORTANT! - set the torch default dtype
torch.set_default_dtype(settings["torch"]["dtype"])

# Construct the appropriate torch objects (i.e. models, loss fxns) prior to training
# pretraining.construct_torch_objects(settings)
pretraining.construct_torch_objects_in_train_dir(
    settings, settings["io"]["data_dir"], settings["io"]["run_dir"]
)

# Define the training subdirectory
train_rel_dir = ""
train_run_dir = os.path.join(settings["io"]["run_dir"], train_rel_dir)

# Load training data and torch objects
data, model, loss_fn, optimizer = pretraining.load_training_objects(
    settings, train_rel_dir, settings["training"]["restart_epoch"]
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
    n_epochs=settings["training"]["n_epochs"],
    save_interval=settings["training"]["save_interval"],
    save_dir=train_run_dir,
    restart=settings["training"]["restart_epoch"],
)