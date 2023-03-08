import os
from typing import Union

import numpy as np
import torch

from equistore import Labels, TensorBlock, TensorMap
from rholearn import io


def train(
    in_train: Union[TensorMap, TensorBlock, torch.Tensor],
    in_test: Union[TensorMap, TensorBlock, torch.Tensor],
    out_train: Union[TensorMap, TensorBlock, torch.Tensor],
    out_test: Union[TensorMap, TensorBlock, torch.Tensor],
    model: torch.nn.Module,
    loss_fn: Union[dict, torch.nn.Module],
    optimizer: torch.nn.Module,
    n_epochs: int,
    save_interval: int,
    save_dir: str,
    restart: int = None,
):
    """
    Performs ``model`` training over the specified number of epochs
    ``n_epochs``. In the specified ``save_dir``, saves the train and test losses
    every epoch, and the torch model every ``save_interval`` number of epochs,
    at relative path "epoch_i/model.pt".
    """
    # Check input args
    _check_args_train(
        in_train,
        in_test,
        out_train,
        out_test,
        model,
        loss_fn,
        optimizer,
        n_epochs,
        save_interval,
        save_dir,
        restart,
    )

    # Create a results directory
    io.check_or_create_dir(save_dir)

    # Define the range of epochs and initialize losses, depending on whether the
    # simulation is being restarted or not
    if restart is None:
        epochs = range(1, n_epochs + 1)
        losses_train = []
        losses_test = []
    else:
        epochs = range(restart + 1, n_epochs + 1)
        losses = np.load(os.path.join(save_dir, "losses.npz"))
        losses_train = list(losses["train"][:restart])
        losses_test = list(losses["test"][:restart])

    for epoch in epochs:

        # Execute a single training step
        loss_train, loss_test = training_step(
            in_train=in_train,
            in_test=in_test,
            out_train=out_train,
            out_test=out_test,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        try:
            losses_train.append(loss_train.detach().numpy())
            losses_test.append(loss_test.detach().numpy())
        except TypeError:
            losses_train.append(loss_train.cpu().detach().numpy())
            losses_test.append(loss_test.cpu().detach().numpy())

        # Write losses to file
        assert len(losses_train) == epoch
        np.savez(
            os.path.join(save_dir, "losses.npz"),
            train=losses_train,
            test=losses_test,
        )

        # Generate log msg and update progress log
        n_msgs = 200
        if n_epochs <= n_msgs or epoch % (n_epochs / n_msgs) == 0:
            msg = (
                f"epoch {epoch}"
                f" train {np.round(losses_train[-1], 10)}"
                f" test {np.round(losses_test[-1], 10)}"
            )
            print(msg)
            with open(os.path.join(save_dir, "log.txt"), "a+") as f:
                f.write("\n" + msg)

        # Write model to file and update log at the specified epoch intervals
        if epoch % save_interval == 0:

            # Create a results directory for this specific epoch
            epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
            io.check_or_create_dir(epoch_dir)

            # Save model
            io.save_torch_object(
                torch_obj=model,
                path=os.path.join(epoch_dir, "model.pt"),
                torch_obj_str="model",
            )

            # Save optimizer state dict
            torch.save(optimizer.state_dict(), os.path.join(epoch_dir, "optimizer.pt"))


def training_step(
    in_train: Union[TensorMap, TensorBlock, torch.Tensor],
    in_test: Union[TensorMap, TensorBlock, torch.Tensor],
    out_train: Union[TensorMap, TensorBlock, torch.Tensor],
    out_test: Union[TensorMap, TensorBlock, torch.Tensor],
    model: torch.nn.Module,
    loss_fn: Union[torch.nn.Module, dict],
    optimizer: torch.nn.Module,
):
    """
    Performs a single training step, i.e. zeroes optimizer gradients, makes a
    forward prediction, calculates loss, updates model weights. Returns the
    train and test losses.

    If predicting on the TensorMap level, ``model`` must be a ``GlobalModel``
    object from the ``rholearn.models`` module, with ``model.models`` and
    ``model.out_features_labels`` attributes.

    If predicting on the TensorBlock level, ``model`` must be a ``LocalModel``
    object from the ``rholearn.models`` module, with ``model.model`` and
    ``model.out_features_labels`` attributes.

    If predicting on a torch Tensor, ``model`` must be an ordinary
    torch.nn.Module model.

    The ``loss_fn`` can either be passed as a torch.nn object, in which case it
    is used to calculate the loss for both the train and test data, or as a dict
    containing 2 loss objects indexed by keys "train" and "test".
    """
    # Define a closure function that zeroes the optimizer gradients, makes a
    # forward prediction on the training data, calculates the loss and its
    # gradient wrt model parameters
    def closure():
        # Zero optimizer gradients
        optimizer.zero_grad()
        # Make a training prediction with the model
        out_train_pred = model(in_train)
        # Calculate the train loss
        if isinstance(loss_fn, dict):
            loss = loss_fn["train"](input=out_train_pred, target=out_train)
        else:
            loss = loss_fn(input=out_train_pred, target=out_train)
        # Calculate the gradient of the loss wrt model parameters
        loss.backward(retain_graph=True)

        return loss

    # Update model weights and get the train loss
    loss_train = optimizer.step(closure)

    # Make a test prediction and calculate the loss
    with torch.no_grad():
        out_test_pred = model(in_test)
        if isinstance(loss_fn, dict):
            loss_test = loss_fn["test"](input=out_test_pred, target=out_test)
        else:
            loss_test = loss_fn(input=out_test_pred, target=out_test)

    # Return the train and test loss
    return loss_train, loss_test


def _check_args_train(
    in_train: Union[TensorMap, TensorBlock, torch.Tensor],
    in_test: Union[TensorMap, TensorBlock, torch.Tensor],
    out_train: Union[TensorMap, TensorBlock, torch.Tensor],
    out_test: Union[TensorMap, TensorBlock, torch.Tensor],
    model: torch.nn.Module,
    loss_fn: Union[dict, torch.nn.Module],
    optimizer: torch.nn.Module,
    n_epochs: int,
    save_interval: int,
    save_dir: str,
    restart: int = None,
):
    """
    Checks the input arguments for the ``train`` function
    """
    # Check input data types
    if not isinstance(in_train, (TensorMap, TensorBlock, torch.Tensor)):
        raise TypeError(
            "must pass ``in_train`` as either a TensorMap, TensorBlock or torch.Tensor"
        )
    if not isinstance(in_test, (TensorMap, TensorBlock, torch.Tensor)):
        raise TypeError(
            "must pass ``in_test`` as either a TensorMap, TensorBlock or torch.Tensor"
        )
    if not isinstance(out_train, (TensorMap, TensorBlock, torch.Tensor)):
        raise TypeError(
            "must pass ``out_train`` as either a TensorMap, TensorBlock or torch.Tensor"
        )
    if not isinstance(out_test, (TensorMap, TensorBlock, torch.Tensor)):
        raise TypeError(
            "must pass ``out_test`` as either a TensorMap, TensorBlock or torch.Tensor"
        )
    if restart is not None:
        if not isinstance(restart, int):
            raise TypeError(
                "must pass ``restart`` as an int corresponding to the checkpoint"
                " epoch from which the simulation should be restarted."
            )
    # If training on the TensorMap level
    if isinstance(in_train, TensorMap):
        try:
            model.models
        except AttributeError:
            raise AttributeError(
                "if predicting at the TensorMap level, ``model`` must have a"
                " ``models`` attribute that is a dict of torch models for"
                " each block."
            )
        if not isinstance(model.models, dict):
            raise TypeError(
                "if predicting on a TensorMap, ``model`` must be passed as a"
                " rholearn.models.GlobalModel object whose ``model.models``"
                " attribute is a dict of torch models for each block,"
                " indexed by TensorMap keys"
            )
        try:
            model.out_feature_labels
        except AttributeError:
            raise AttributeError(
                "if predicting at the TensorMap level, ``model`` must have a"
                " ``out_feature_labels`` attribute that is a dict of Labels for"
                " the features/properties of each block."
            )
        if not isinstance(model.out_feature_labels, dict):
            raise TypeError(
                "if predicting on a TensorMap, ``model`` must be passed as a"
                " rholearn.models.GlobalModel object whose ``out_feature_labels``"
                " attribute is a dict of Labels objects for each block, indexed"
                " by TensorMap keys"
            )
    # If training on the TensorBlock level
    elif isinstance(in_train, TensorBlock):
        try:
            model.model
        except AttributeError:
            raise AttributeError(
                "if predicting at the TensorBlock level, ``model`` must have a"
                " ``model`` attribute that is a torch model."
            )
        if not isinstance(model.model, torch.nn.Module):
            raise TypeError(
                "if predicting on a TensorBlock, ``model`` must be passed as a"
                " rholearn.models.LocalModel object whose ``model``"
                " attribute is a torch model"
            )
        try:
            model.out_feature_labels
        except AttributeError:
            raise AttributeError(
                "if predicting at the TensorBlock level, ``model`` must have a"
                " ``out_feature_labels`` attribute that is a Labels object for"
                " the features/properties of the input block"
            )
        if not isinstance(model.out_feature_labels, Labels):
            raise TypeError(
                "if predicting on a TensorBlock, ``model`` must be passed as a"
                " rholearn.models.LocalModel object whose ``out_feature_labels``"
                " attribute is a Labels object for the features/properties of the"
                " input block"
            )
    # Make prediction at the torch Tensor level
    elif isinstance(in_train, torch.Tensor):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "if predicting on a torch Tensor, ``model`` must be a torch model"
            )
    # Check model
    tmp_key = list(model.models.keys())[0]
    if not isinstance(tmp_key, np.void):
        raise ValueError(
            f"``model.models`` keys must be numpy.void objects, not {type(tmp_key)}"
        )
    tmp_key = list(model.in_feature_labels.keys())[0]
    if not isinstance(tmp_key, np.void):
        raise ValueError(
            "``model.in_feature_labels`` keys must be numpy.void objects,"
            f" not {type(tmp_key)}"
        )
    tmp_key = list(model.out_feature_labels.keys())[0]
    if not isinstance(tmp_key, np.void):
        raise ValueError(
            "``model.out_feature_labels`` keys must be numpy.void objects,"
            f" not {type(tmp_key)}"
        )
    # Check loss fn
    if isinstance(loss_fn, dict):
        for k in loss_fn.keys():
            if k not in ["train", "test"]:
                raise ValueError(
                    "if passing ``loss_fn`` as a dict, it must contain"
                    " 2 loss function objects indexed by keys 'train' and 'test'"
                )
    else:
        if not isinstance(loss_fn, torch.nn.Module):
            raise TypeError(
                "``loss_fn`` must be passed as either a dict or torch.nn.Module"
            )
    # Check optimizer
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("``optimizer`` must be a torch.optim.Optimizer class")
    # Check n_epochs
    if not isinstance(n_epochs, int):
        raise TypeError("must pass ``n_epochs`` as an int")
    # Check save_interval
    if not isinstance(save_interval, int):
        raise TypeError("must pass ``save_interval`` as an int")
    if save_interval > n_epochs:
        raise ValueError(
            "the ``save_interval`` cannot be greater than the total num of epochs"
        )
    if n_epochs % save_interval != 0:
        raise ValueError("must pass ``save_interval`` as a factor of ``n_epochs``")
    # Check save dir
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"directory {save_dir} does not exist")


def _get_dict_key_type(d):
    return type(list(d.keys())[0])
