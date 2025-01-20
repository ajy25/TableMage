import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from adjustText import adjust_text
import pandas as pd
import numpy as np
from ...display.plot_options import plot_options
from ..lmutils.constants import MAX_N_OUTLIERS_TEXT
from ..lmutils.funcs import reverse_argsort


def plot_residuals_vs_fitted(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    standardized: bool,
    outliers_idx: np.ndarray,
    outliers_mask: np.ndarray,
    show_outliers: bool = True,
    include_text: bool = False,
    figsize: tuple[float, float] = (5.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots the residuals vs fitted values.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.

    residuals : np.ndarray
        The residuals of the model.

    standardized : bool
        Whether to standardize the residuals.

    outliers_idx : np.ndarray
        The indices (pandas Series converted to NumPy) of the outliers
        (rows of DataFrame).

    outliers_mask : np.ndarray
        The mask of the outliers.

    show_outliers : bool
        Whether to show the outliers, by default True.

    include_text : bool
        Whether to include text for the outliers, by default False.

    figsize : tuple[float, float] | None
        The size of the figure, by default (5.0, 5.0).

    ax : plt.Axes | None
        The axes to plot on, by default None.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if standardized:
        residuals = residuals / np.std(residuals)

    n_outliers = 0 if outliers_idx is None else len(outliers_idx)

    ax.axhline(
        y=0,
        color=plot_options._reference_line_color,
        linestyle="--",
        linewidth=plot_options._line_width,
    )
    if show_outliers and n_outliers > 0:
        ax.scatter(
            y_pred[~outliers_mask],
            residuals[~outliers_mask],
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )
        ax.scatter(
            y_pred[outliers_mask],
            residuals[outliers_mask],
            s=plot_options._dot_size,
            color="red",
        )
        if include_text and n_outliers <= MAX_N_OUTLIERS_TEXT:
            annotations = []
            for i, label in enumerate(outliers_idx):
                annotations.append(
                    ax.annotate(
                        label,
                        (
                            y_pred[outliers_mask][i],
                            residuals[outliers_mask][i],
                        ),
                        color="red",
                        fontsize=plot_options._axis_minor_ticklabel_font_size,
                    )
                )
            adjust_text(annotations, ax=ax)
    else:
        ax.scatter(
            y_pred,
            residuals,
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )

    ax.set_xlabel("Fitted")
    if standardized:
        ax.set_ylabel("Standardized Residuals")
        ax.set_title("Standardized Residuals vs Fitted")
    else:
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted")
    ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)

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


def plot_pearson_vs_observation(
    std_residuals: np.ndarray,
    df_idx: pd.Series,
    outliers_idx: np.ndarray,
    outliers_mask: np.ndarray,
    show_outliers: bool = True,
    include_text: bool = False,
    figsize: tuple[float, float] = (5.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots Pearson residuals against observation number.

    Parameters
    ----------
    std_residuals : np.ndarray
        The standardized residuals (Pearson residuals).

    df_idx : pd.Series
        The index of the dataset (e.g., observation numbers).

    outliers_idx : np.ndarray
        The indices of the outliers.

    outliers_mask : np.ndarray
        The mask of the outliers.

    show_outliers : bool
        Whether to highlight outliers in the plot, by default True.

    include_text : bool
        Whether to annotate the outliers with their observation numbers, by default False.

    figsize : tuple[float, float]
        The size of the figure, by default (5.0, 5.0).

    ax : plt.Axes | None
        The axes to plot on, by default None.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    n_outliers = 0 if outliers_idx is None else len(outliers_idx)

    ax.axhline(
        y=0,
        color=plot_options._reference_line_color,
        linestyle="--",
        linewidth=plot_options._line_width,
    )
    if show_outliers and n_outliers > 0:
        ax.scatter(
            df_idx[~outliers_mask],
            std_residuals[~outliers_mask],
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )
        ax.scatter(
            df_idx[outliers_mask],
            std_residuals[outliers_mask],
            s=plot_options._dot_size,
            color="red",
        )
        if include_text and n_outliers <= MAX_N_OUTLIERS_TEXT:
            annotations = []
            for i, label in enumerate(outliers_idx):
                annotations.append(
                    ax.annotate(
                        label,
                        (
                            df_idx[outliers_mask][i],
                            std_residuals[outliers_mask][i],
                        ),
                        color="red",
                        fontsize=plot_options._axis_minor_ticklabel_font_size,
                    )
                )
            adjust_text(annotations, ax=ax)
    else:
        ax.scatter(
            df_idx,
            std_residuals,
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )

    ax.set_xlabel("Observation Number")
    ax.set_ylabel("Standardized Residuals")
    ax.set_title("Pearson Residuals vs. Observation Number")
    ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)

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

def plot_residuals_vs_var(
    predictor: str,
    X_eval_df: pd.DataFrame,
    residuals: np.ndarray,
    standardized: bool,
    outliers_idx: np.ndarray,
    outliers_mask: np.ndarray,
    show_outliers: bool = True,
    include_text: bool = False,
    figsize: tuple[float, float] = (5.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Returns a figure that is a residuals vs fitted (y_pred) plot.

    Parameters
    ----------
    predictor : str
        The predictor variable to plot residuals against.

    X_eval_df : pd.DataFrame
        The evaluation dataset.

    residuals : np.ndarray
        The residuals of the model.

    standardized : bool
        Whether to standardize the residuals.

    outliers_idx : np.ndarray
        The indices (pandas Series converted to NumPy) of the outliers
        (rows of DataFrame).

    outliers_mask : np.ndarray
        The mask of the outliers.

    show_outliers : bool
        Whether to show the outliers, by default True.

    include_text : bool
        Whether to include text for the outliers, by default False.

    figsize : tuple[float, float] | None
        The size of the figure, by default (5.0, 5.0).

    ax : plt.Axes | None
        The axes to plot on, by default None.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if standardized:
        residuals = residuals / np.std(residuals)

    x_vals = X_eval_df[predictor].to_numpy()
    n_outliers = 0 if outliers_idx is None else len(outliers_idx)

    ax.axhline(
        y=0,
        color=plot_options._reference_line_color,
        linestyle="--",
        linewidth=plot_options._line_width,
    )
    if show_outliers and n_outliers > 0:
        ax.scatter(
            x_vals[~outliers_mask],
            residuals[~outliers_mask],
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )
        ax.scatter(
            x_vals[outliers_mask],
            residuals[outliers_mask],
            s=plot_options._dot_size,
            color="red",
        )
        if include_text and n_outliers <= MAX_N_OUTLIERS_TEXT:
            annotations = []
            for i, label in enumerate(outliers_idx):
                annotations.append(
                    ax.annotate(
                        label,
                        (
                            x_vals[outliers_mask][i],
                            residuals[outliers_mask][i],
                        ),
                        color="red",
                        fontsize=plot_options._axis_minor_ticklabel_font_size,
                    )
                )
            adjust_text(annotations, ax=ax)
    else:
        ax.scatter(
            x_vals,
            residuals,
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )

    ax.set_xlabel(predictor)
    if standardized:
        ax.set_ylabel("Standardized Residuals")
        ax.set_title(f"Standardized Residuals vs {predictor}")
    else:
        ax.set_ylabel("Residuals")
        ax.set_title(f"Residuals vs {predictor}")
    ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)

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


def plot_residuals_hist(
    residuals: np.ndarray,
    standardized: bool,
    density: bool = False,
    figsize: tuple[float, float] = (5.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots the histogram of the residuals.

    Parameters
    ----------
    residuals : np.ndarray
        The residuals of the model.

    standardized : bool
        Whether to standardize the residuals.

    density : bool
        Whether to plot the density, by default False.

    figsize : tuple[float, float] | None
        The size of the figure, by default (5.0, 5.0).

    ax : plt.Axes | None
        The axes to plot on, by default None.

    Returns
    -------
    plt.Figure
    """
    if density:
        stat = "density"
    else:
        stat = "count"

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if standardized:
        residuals = residuals / np.std(residuals)

    sns.histplot(
        residuals,
        bins="auto",
        color=plot_options._bar_color,
        edgecolor=plot_options._bar_edgecolor,
        stat=stat,
        ax=ax,
        kde=True,
        alpha=plot_options._bar_alpha,
    )
    if standardized:
        ax.set_title("Distribution of Standardized Residuals")
        ax.set_xlabel("Standardized Residuals")
    else:
        ax.set_title("Distribution of Residuals")
        ax.set_xlabel("Residuals")
    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Frequency")
    ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)

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


def plot_scale_location(
    y_pred: np.ndarray,
    std_residuals: np.ndarray,
    outliers_idx: np.ndarray = None,
    outliers_mask: np.ndarray = None,
    show_outliers: bool = True,
    include_text: bool = False,
    figsize: tuple[float, float] = (5.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots the square root of the absolute standardized residuals vs fitted values.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.

    std_residuals : np.ndarray
        The standardized residuals.

    outliers_idx : np.ndarray
        The indices (pandas Series converted to NumPy) of the outliers
        (rows of DataFrame).

    outliers_mask : np.ndarray
        The mask of the outliers.

    show_outliers : bool
        Whether to show the outliers, by default True.

    include_text : bool
        Whether to include text for the outliers, by default False.

    figsize : tuple[float, float] | None
        The size of the figure, by default (5.0, 5.0).

    ax : plt.Axes | None
        The axes to plot on, by default None.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    residuals = np.sqrt(np.abs(std_residuals))
    n_outliers = 0 if outliers_idx is None else len(outliers_idx)

    ax.axhline(
        y=0,
        color=plot_options._reference_line_color,
        linestyle="--",
        linewidth=plot_options._line_width,
    )
    if show_outliers and n_outliers > 0:
        ax.scatter(
            y_pred[~outliers_mask],
            residuals[~outliers_mask],
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )
        ax.scatter(
            y_pred[outliers_mask],
            residuals[outliers_mask],
            s=plot_options._dot_size,
            color="red",
        )
        if include_text and n_outliers <= MAX_N_OUTLIERS_TEXT:
            annotations = []
            for i, label in enumerate(outliers_idx):
                annotations.append(
                    ax.annotate(
                        label,
                        (
                            y_pred[outliers_mask][i],
                            residuals[outliers_mask][i],
                        ),
                        color="red",
                        fontsize=plot_options._axis_minor_ticklabel_font_size,
                    )
                )
            adjust_text(annotations, ax=ax)

    else:
        ax.scatter(
            y_pred,
            residuals,
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )

    ax.set_xlabel("Fitted")
    ax.set_ylabel("sqrt(Standardized Residuals)")
    ax.set_title("Scale-Location")
    ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)

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


def plot_residuals_vs_leverage(
    leverage: np.ndarray,
    residuals: np.ndarray,
    standardized: bool,
    outliers_idx: np.ndarray,
    outliers_mask: np.ndarray,
    show_outliers: bool = True,
    include_text: bool = False,
    figsize: tuple[float, float] = (5.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots the residuals vs leverage.

    Parameters
    ----------
    leverage : np.ndarray
        The leverage values.

    residuals : np.ndarray
        The residuals of the model.

    standardized : bool
        Whether to standardize the residuals.

    outliers_idx : np.ndarray
        The indices (pandas Series converted to NumPy) of the outliers
        (rows of DataFrame).

    outliers_mask : np.ndarray
        The mask of the outliers.

    show_outliers : bool
        Whether to show the outliers, by default True.

    include_text : bool
        Whether to include text for the outliers, by default False.

    figsize : tuple[float, float] | None
        The size of the figure, by default (5.0, 5.0).

    ax : plt.Axes | None
        The axes to plot on, by default None.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if standardized:
        residuals = residuals / np.std(residuals)

    n_outliers = 0 if outliers_idx is None else len(outliers_idx)

    ax.axhline(
        y=0,
        color=plot_options._reference_line_color,
        linestyle="--",
        linewidth=plot_options._line_width,
    )
    if show_outliers and n_outliers > 0:
        ax.scatter(
            leverage[~outliers_mask],
            residuals[~outliers_mask],
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )
        ax.scatter(
            leverage[outliers_mask],
            residuals[outliers_mask],
            s=plot_options._dot_size,
            color="red",
        )
        if include_text and n_outliers <= MAX_N_OUTLIERS_TEXT:
            annotations = []
            for i, label in enumerate(outliers_idx):
                annotations.append(
                    ax.annotate(
                        label,
                        (
                            leverage[outliers_mask][i],
                            residuals[outliers_mask][i],
                        ),
                        color="red",
                        fontsize=plot_options._axis_minor_ticklabel_font_size,
                    )
                )
            adjust_text(annotations, ax=ax)

    else:
        ax.scatter(
            leverage,
            residuals,
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )

    ax.set_xlabel("Leverage")
    if standardized:
        ax.set_ylabel("Standardized Residuals")
        ax.set_title("Standardized Residuals vs Leverage")
    else:
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Leverage")
    ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)

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


def plot_qq(
    df_idx: pd.Series,
    residuals: np.ndarray,
    standardized: bool = False,
    outliers_idx: np.ndarray = None,
    outliers_mask: np.ndarray = None,
    show_outliers: bool = True,
    include_text: bool = False,
    figsize: tuple[float, float] = (5.0, 5.0),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plots a QQ plot of the residuals (for normality check).

    Parameters
    ----------
    df_idx : pd.Series
        The index of the evaluation dataset.

    residuals : np.ndarray
        The residuals of the model.

    standardized : bool
        Whether to standardize the residuals, by default False.

    outliers_idx : np.ndarray
        The indices of the outliers.

    outliers_mask : np.ndarray
        The mask of the outliers.

    show_outliers : bool
        Whether to show the outliers, by default True.

    include_text : bool
        Whether to include text for the outliers, by default False.

    figsize : tuple[float, float] | None
        The size of the figure, by default (5.0, 5.0).

    ax : plt.Axes | None
        The axes to plot on, by default None.

    Returns
    -------
    plt.Figure
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if standardized:
        residuals = residuals / np.std(residuals)

    tup1, tup2 = stats.probplot(residuals, dist="norm")
    theoretical_quantitles, ordered_vals = tup1
    slope, intercept, _ = tup2

    ax.set_title("Q-Q Plot")
    ax.set_xlabel("Theoretical Quantiles")

    if standardized:
        ax.set_ylabel("Standardized Residuals")
    else:
        ax.set_ylabel("Residuals")

    min_val = np.min(theoretical_quantitles)
    max_val = np.max(theoretical_quantitles)

    n_outliers = 0 if outliers_idx is None else len(outliers_idx)

    ax.plot(
        [min_val, max_val],
        [min_val * slope + intercept, max_val * slope + intercept],
        color=plot_options._reference_line_color,
        linestyle="--",
        linewidth=plot_options._line_width,
    )

    if show_outliers and n_outliers > 0:
        residuals_sorted_idx = reverse_argsort(np.argsort(residuals))

        residuals_df = pd.DataFrame(residuals, columns=["residuals"])
        residuals_df["label"] = df_idx
        residuals_df["is_outlier"] = outliers_mask
        residuals_df["theoretical_quantile"] = theoretical_quantitles[
            residuals_sorted_idx
        ]
        residuals_df["ordered_value"] = ordered_vals[residuals_sorted_idx]
        residuals_df_outliers = residuals_df[residuals_df["is_outlier"]]
        residuals_df_not_outliers = residuals_df[~residuals_df["is_outlier"]]

        ax.scatter(
            residuals_df_not_outliers["theoretical_quantile"],
            residuals_df_not_outliers["ordered_value"],
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )
        ax.scatter(
            residuals_df_outliers["theoretical_quantile"],
            residuals_df_outliers["ordered_value"],
            s=plot_options._dot_size,
            color="red",
        )
        if include_text and n_outliers <= MAX_N_OUTLIERS_TEXT:
            annotations = []
            for _, row in residuals_df_outliers.iterrows():
                annotations.append(
                    ax.annotate(
                        row["label"],
                        (row["theoretical_quantile"], row["ordered_value"]),
                        color="red",
                        fontsize=plot_options._axis_minor_ticklabel_font_size,
                    )
                )
            adjust_text(annotations, ax=ax)

    else:
        ax.scatter(
            theoretical_quantitles,
            ordered_vals,
            s=plot_options._dot_size,
            color=plot_options._dot_color,
        )

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
