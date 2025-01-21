import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Literal
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
import seaborn as sns

from ..display.plot_options import plot_options


def plot_obs_vs_pred(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    model_name: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Returns a figure that is a scatter plot of the observed and predicted y
    values. Predicted values on x axis, observed values on y axis.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted y values.

    y_true : np.ndarray
        The observed y values.

    model_name : str | None.
        Default: None. The name of the model to display in the title.
        If None, model name is not included in the title.

    figsize : tuple[float, float]
        Default: (5, 5). The size of the figure. Only used if ax is None.

    ax : plt.Axes | None
        Default: None. The axes to plot on. If None, a new figure is created.

    Returns
    -------
    plt.Figure
        Figure of the scatter plot.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    min_val = np.min(np.hstack((y_pred, y_true)))
    max_val = np.max(np.hstack((y_pred, y_true)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color=plot_options._reference_line_color,
        linewidth=plot_options._line_width,
    )

    ax.scatter(y_pred, y_true, s=plot_options._dot_size, color=plot_options._dot_color)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

    if model_name is not None:
        ax.set_title(
            f"{model_name}: Observed vs Predicted | "
            f"r = {round(pearsonr(y_pred, y_true)[0], 3)}"
        )
    else:
        ax.set_title(
            "Observed vs Predicted | " + f"r = {round(pearsonr(y_pred, y_true)[0], 3)}"
        )
    ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)
    ax.yaxis.get_offset_text().set_fontsize(ax.yaxis.get_ticklabels()[0].get_fontsize())
    ax.xaxis.get_offset_text().set_fontsize(ax.xaxis.get_ticklabels()[0].get_fontsize())

    ax.title.set_fontsize(plot_options._title_font_size)
    ax.xaxis.label.set_fontsize(plot_options._axis_title_font_size)
    ax.yaxis.label.set_fontsize(plot_options._axis_title_font_size)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=plot_options._axis_major_ticklabel_font_size,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        labelsize=plot_options._axis_minor_ticklabel_font_size,
    )

    if fig is not None:
        fig.tight_layout()
        plt.close()
    return fig


def plot_roc_curve(
    y_score: np.ndarray,
    y_true: np.ndarray,
    model_name: str | None = None,
    label_curve: bool = False,
    color: str | Any = None,
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Returns a figure that is the ROC curve for the model.

    Parameters
    ----------
    y_score : np.ndarray
        Predicted probabilities or decision scores.

    y_true : np.ndarray
        True binary labels.

    model_name : str | None
        Default: None. The name of the model to display in the title.
        If None, model name is not included in the title.

    label_curve : bool
        Default: False. Whether to label the ROC curve with model name and AUC.

    color : str | Any
        Default: None. The color of the ROC curve. If None,
        the plot options line color is used.

    figsize : tuple[float, float]
        Default: (5, 5). The size of the figure. Only used if ax is None.

    ax : plt.Axes | None
        Default: None. The axes to plot on. If None, a new figure is created.

    Returns
    -------
    plt.Figure
        Figure of the ROC curve.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    if color is None:
        color = plot_options._line_color

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=plot_options._reference_line_color,
        linewidth=plot_options._line_width,
    )
    if label_curve:
        if model_name is not None:
            ax.plot(
                fpr,
                tpr,
                color=color,
                label=f"{model_name} | AUC = {roc_auc:.3f}",
            )
        else:
            ax.plot(fpr, tpr, color=color, label=f"AUC = {roc_auc:.3f}")
    else:
        ax.plot(fpr, tpr, color=color)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    if not label_curve:
        if model_name is not None:
            ax.set_title(f"{model_name}: ROC Curve | AUC = {roc_auc:.3f}")
        else:
            ax.set_title(f"ROC Curve | AUC = {roc_auc:.3f}")

    ax.title.set_fontsize(plot_options._title_font_size)
    ax.xaxis.label.set_fontsize(plot_options._axis_title_font_size)
    ax.yaxis.label.set_fontsize(plot_options._axis_title_font_size)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=plot_options._axis_major_ticklabel_font_size,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        labelsize=plot_options._axis_minor_ticklabel_font_size,
    )

    legend = ax.legend_
    if legend is not None:
        legend.set_title(
            legend.get_title().get_text(),
            prop={"size": plot_options._axis_title_font_size},
        )
        for text in legend.get_texts():
            text.set_fontsize(plot_options._axis_title_font_size)

    if fig is not None:
        fig.tight_layout()
        plt.close()
    return fig


def plot_confusion_matrix(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    model_name: str | None = None,
    cmap: Any = "Blues",
    annotation_type: Literal["count", "percent"] = "count",
    figsize: tuple[float, float] = (5, 5),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Returns a figure that is the confusion matrix for the model.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted binary or multiclass labels.

    y_true : np.ndarray
        True binary or multiclass labels.

    model_name : str | None
        Default: None. The name of the model to display in the title.
        If None, model name is not included in the title.

    cmap : Any
        Default: "Blues". The colormap to use for the heatmap.
        Must be a valid Matplotlib colormap.

    annotation_type : Literal["count", "percent"]
        Default: "count". The type of annotation to display in the heatmap cells.

    figsize : tuple[float, float]
        Default: (5, 5). The size of the figure. Only used if ax is None.

    ax : plt.Axes | None
        Default: None. The axes to plot on. If None, a new figure is created.

    Returns
    -------
    plt.Figure
        Figure of the confusion matrix.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    all_labels = np.unique(np.concatenate((y_true, y_pred)))
    all_labels.sort()

    if annotation_type == "percent":
        confusion_matrix = pd.crosstab(
            y_true, y_pred, rownames=["True"], colnames=["Predicted"], dropna=False
        )
        confusion_matrix = confusion_matrix.reindex(
            index=all_labels, columns=all_labels, fill_value=0
        )
        confusion_matrix = confusion_matrix / confusion_matrix.sum().sum()
        fmt = ".2f"

    else:
        confusion_matrix = pd.crosstab(
            y_true, y_pred, rownames=["True"], colnames=["Predicted"], dropna=False
        )
        confusion_matrix = confusion_matrix.reindex(
            index=all_labels, columns=all_labels, fill_value=0
        )
        fmt = "d"

    sns.heatmap(
        confusion_matrix,
        annot=True,
        annot_kws={"size": plot_options._axis_title_font_size},
        fmt=fmt,
        cmap=cmap,
        ax=ax,
        cbar=False,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

    if model_name is not None:
        ax.set_title(
            f"{model_name}: Confusion Matrix | "
            f"Accuracy = {np.mean(y_pred == y_true):.3f}"
        )
    else:
        ax.set_title(
            f"Confusion Matrix | " f"Accuracy = {np.mean(y_pred == y_true):.3f}"
        )

    if fig is not None:
        fig.tight_layout()
        plt.close()
    return fig


def decrease_font_sizes_axs(
    axs,
    title_font_size_decrease: int,
    axes_label_font_size_decrease: int,
    ticks_font_size_decrease_from_label: int = 0,
):
    """
    Decreases the font sizes of titles, axes labels, and ticks in a set of axes.

    Parameters
    ----------
    axs. Array of plt.Axes.
    title_font_size_decrease : int.
    axes_label_font_size_decrease : int.
    ticks_font_size_decrease_from_label : int.
    """
    for ax in axs.flat:
        ax.title.set_fontsize(ax.title.get_fontsize() - title_font_size_decrease)
        ax.xaxis.label.set_fontsize(
            ax.xaxis.label.get_fontsize() - axes_label_font_size_decrease
        )
        ax.yaxis.label.set_fontsize(
            ax.yaxis.label.get_fontsize() - axes_label_font_size_decrease
        )
        tick_params = ax.tick_params(
            axis="both",
            which="major",
            labelsize=ax.xaxis.label.get_fontsize()
            - ticks_font_size_decrease_from_label,
        )
        if tick_params is not None:
            tick_params.set_fontsize(
                tick_params.get("size") - ticks_font_size_decrease_from_label
            )

        new_offset_size_diff = (
            axes_label_font_size_decrease + ticks_font_size_decrease_from_label
        )
        ax.xaxis.offsetText.set_fontsize(
            ax.xaxis.offsetText.get_fontsize() - new_offset_size_diff
        )
        ax.yaxis.offsetText.set_fontsize(
            ax.yaxis.offsetText.get_fontsize() - new_offset_size_diff
        )
