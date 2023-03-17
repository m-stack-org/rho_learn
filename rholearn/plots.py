"""
Functions to buidl analysis plots using matplotlib
"""
from itertools import product
from typing import List, Dict, Union

import numpy as np
import matplotlib.pyplot as plt
import mpltex

import equistore
from equistore import TensorMap

from rholearn import utils


def loss_vs_epoch(
    data: Union[List[Dict[int, np.ndarray]], List[np.ndarray]],
    mutliple_traces: bool = True,
    sharey: bool = False,
):
    """
    Takes arrays for loss values and returns a loglog plot of epoch vs loss.
    Assumes that the arrays passed for the losses correspond to a loss value for
    each epoch.

    If ``multiple_traces=True``, assumes the ``data`` dict passed will be of the
    form:

    {
        trace_0 <int>: losses_0 <np.ndarray>,
        trace_1 <int>: losses_1 <np.ndarray>,
        ...
    }

    and multiple traces will be plotted on the same axis. Otherwise if
    ``multiple_traces=False``, assumes the ``data`` will just be a numpy ndarray
    and a single trace will be plotted on the axis.

    If ``data`` is passed as a list of the above data formats, n horizontal
    subplots for each entry in the list will be created.
    """
    if not isinstance(data, list):
        data = [data]
    fig, ax = plt.subplots(1, len(data), sharey=sharey)
    for col, d in enumerate(data):
        linestyles = mpltex.linestyle_generator(markers=[])
        if mutliple_traces:
            subsets = np.sort(list(d.keys()))
            x = np.arange(len(d[0]))
            Y = [d[key] for key in subsets]
        else:
            x = np.arange(len(d))
            Y = [d]

        for i, y in enumerate(Y):
            ax[col].loglog(x, y, label=str(i), **next(linestyles))

        ax[col].set_xlabel("epoch")

    return fig, ax


def learning_curve(data, subset_sizes: np.array, point: str = "best"):
    """
    Plots the learning curves as a function of training subset size. From the
    array of loss values for each epoch either selects the "best" (i.e. lowest)
    or the "final" (i.e. the last epoch) loss value for each subset.

    Accepts as input a dict of the form:

    {
        trace_0 <int>: losses_0 <np.ndarray>, trace_1 <int>: losses_1
        <np.ndarray>, ...
    }

    or a list of these. If passing a list of these, a trace for each element
    will be plotted on the same plot.
    """
    if not isinstance(data, list):
        data = [data]
    fig, ax = plt.subplots()
    linestyles = mpltex.linestyle_generator(lines=["-"], hollow_styles=[])
    for col, d in enumerate(data):
        subsets = np.sort(list(d.keys()))
        x = subset_sizes
        if point == "final":
            y = [d[subset][-1] for subset in subsets]
        elif point == "best":
            y = [np.min(d[subset]) for subset in subsets]
        else:
            raise ValueError("``point`` must be 'final' or 'best'")
        if len(x) != len(y):
            raise ValueError(
                f"number of subset_sizes (x={x}) passed not equal to number of"
                + f" loss values gathered (y={y})."
            )
        ax.loglog(x, y, **next(linestyles))
    ax.set_xlabel(r"number of training structures")

    return fig, ax


def parity_plot(
    target: TensorMap,
    predicted: TensorMap,
    color_by: str = "spherical_harmonics_l",
):
    """
    Returns a parity plot of the target (x axis) and predicted (y axis)
    values. Plots also a grey dashed y=x line. The keys of the input TensorMap
    ``color_by`` decides what to colour the data by.
    """
    # Check that the metadata is equivalent between the 2 TensorMaps
    equistore.equal_metadata(target, predicted)
    # Build the parity plot
    fig, ax = plt.subplots()
    linestyles = mpltex.linestyle_generator(
        lines=[], markers=["o"], hollow_styles=[False]
    )

    key_ranges = {key: np.unique(target.keys[key]) for key in target.keys.names}
    color_by_range = key_ranges.pop(color_by)
    key_names = list(key_ranges.keys())

    for color_by_idx in color_by_range:
        x, y = np.array([]), np.array([])

        for combo in list(product(*[key_ranges[key] for key in key_names])):
            # other_keys = {key_names[i]: combo[i] for i in range(len(key_names))}
            other_keys = {name_i: combo_i for name_i, combo_i in zip(key_names, combo)}
            try:
                target_block = target.block(**{color_by: color_by_idx}, **other_keys)
                pred_block = predicted.block(**{color_by: color_by_idx}, **other_keys)
            except ValueError as e:
                assert str(e).startswith(
                    "Couldn't find any block matching the selection"
                )
                print(f"key not found for {color_by} = {color_by_idx} and {other_keys}")
            try:
                x = np.append(x, target_block.values.detach().numpy().flatten())
                y = np.append(y, pred_block.values.detach().numpy().flatten())
            except AttributeError:
                x = np.append(x, target_block.values.flatten())
                y = np.append(y, pred_block.values.flatten())
        ax.plot(x, y, label=f"{color_by} = {color_by_idx}", **next(linestyles))
    # Plot a y=x grey dashed line
    ax.axline((-1e-5, -1e-5), (1e-5, 1e-5), color="grey", linestyle="dashed")
    fig.tight_layout()

    return fig, ax


@mpltex.presentation_decorator
def save_fig_mpltex(fig, path_to_filename: str):
    """
    Uses mpltex (https://github.com/liuyxpp/mpltex) to save the matplotlib
    ``fig`` to file at ``path_to_filename`` (with no extension) with their
    decorator that produces a publication-quality figure as a pdf.
    """
    fig.savefig(path_to_filename)
