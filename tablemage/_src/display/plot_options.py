import seaborn as sns
from typing import Literal


class _PlotOptions:
    """Class for setting and tracking options for plotting."""

    def __init__(self):
        """Initializes the a _PlotOptions object with default settings."""
        self.set_to_defaults()

    def set_style(
        self, style: Literal["whitegrid", "darkgrid", "white", "dark", "ticks"]
    ):
        """Updates the style of the plots.

        Parameters
        ----------
        style : str
            The style to use for the plots.
            Must be one of: "whitegrid", "darkgrid", "white", "dark", "ticks".
        """
        self._style = style
        self._on_sns_update()

    def set_context(self, context: Literal["paper", "notebook", "talk", "poster"]):
        """Updates the context of the plots.

        Parameters
        ----------
        context : str
            The context to use for the plots.
            Must be one of: "paper", "notebook", "talk", "poster".
        """
        self._context = context
        self._on_sns_update()

    def set_dot_size(self, dot_size: int):
        """Updates the size of the dots in scatter plots."""
        self._dot_size = dot_size

    def set_dot_color(self, color):
        """Updates the color of the dots in scatter plots.

        Parameters
        ----------
        color : str
            Must be a valid RGBA color string.
            Can be a named color, a hex color, or a tuple of RGBA values.
        """
        self._dot_color = color

    def set_bar_alpha(self, alpha: float):
        """Updates the transparency of bars/bins."""
        self._bar_alpha = alpha

    def set_bar_color(self, color):
        """Updates the color of the bars/bins.

        Parameters
        ----------
        color : str
            Must be a valid RGBA color string.
            Can be a named color, a hex color, or a tuple of RGBA values.
        """
        self._bar_color = color

    def set_bar_edgecolor(self, color):
        """Updates the color of the edges of the bars/bins.

        Parameters
        ----------
        color : str
            Must be a valid RGBA color string.
            Can be a named color, a hex color, or a tuple of RGBA values.
        """
        self._bar_edgecolor = color

    def set_color_palette(self, color_palette):
        """Updates the color palette to use for plots.

        Parameters
        ----------
        color_palette
            A valid seaborn color palette, e.g. sns.color_palette("muted")
        """
        self._color_palette = color_palette

    def set_color_map(self, cmap: str):
        """Updates the color map to use for plots.

        Parameters
        ----------
        cmap : str
            A valid matplotlib color map.
            Examples:
            - "viridis"
            - "plasma"
            - "inferno"
            - "magma"
            - "cividis"
            - "Greys"
            - "Purples"
            - "Blues"
            - "Greens"
            - "Oranges"
            - "Reds"
            - "YlOrBr"
            - "YlOrRd"
            - "OrRd"
            - "PuRd"
            - "RdPu"
            - "BuPu"
            - "GnBu"
            - "PuBu"
            - "YlGnBu"
            - "PuBuGn"
        """
        self._cmap = cmap

    def set_font_sizes(
        self,
        title: int | None = None,
        axis_title: int | None = None,
        major_ticklabel: int | None = None,
        minor_ticklabel: int | None = None,
    ):
        """Updates the font sizes of the plot elements.

        Parameters
        ----------
        title : int | None
            Default: None. If None, no change. Font size of the title of the plot.

        axis_title : int | None
            Default: None. If None, no change. Font size of the axis titles of the plot.

        major_ticklabel : int | None
            Default: None. If None, no change.
            Font size of the major axis ticklabels of the plot.

        minor_ticklabel : int | None
            Default: None. If None, no change.
            Font size of the minor axis ticklabels of the plot.
        """
        if title is not None:
            self._title_font_size = title

        if axis_title is not None:
            self._axis_title_font_size = axis_title

        if major_ticklabel is not None:
            self._axis_major_ticklabel_font_size = major_ticklabel

        if minor_ticklabel is not None:
            self._axis_minor_ticklabel_font_size = minor_ticklabel

    def set_scilimits(self, scilimits: tuple[int, int]):
        """Updates the scientific limits for the axis labels.

        Parameters
        ----------
        scilimits : tuple[int, int]
            Tuple of the form (min, max) where min and max are integers.
        """
        self._scilimits = scilimits

    def set_to_defaults(self):
        """Resets all plot options to their default values."""
        self._style: Literal[
            "whitegrid", "darkgrid", "white", "dark", "ticks"
        ] = "whitegrid"
        """Style to use for plots. (Seaborn)"""

        self._context: Literal["paper", "notebook", "talk", "poster"] = "paper"
        """Context to use for plots. (Seaborn)"""

        self._dot_size = 2
        """Size of the dots in scatter plots. ("size" param in Seaborn, \
        "s" param in Matplotlib)
        """

        self._dot_s = 3
        """Size of the dots in scatter plots. ("s" param in Seaborn)"""

        self._dot_color = "black"
        """Color of the dots in scatter plots. (Matplotlib)"""

        self._dot_facecolor = "none"
        """Face color of the dots in scatter plots. (Seaborn)"""

        self._dot_edgecolor = "black"
        """Edge color of the dots in scatter plots. (Seaborn)"""

        self._line_width = 1.0
        """Width of the lines in line plots. (Matplotlib)"""

        self._line_color = "black"
        """Color of the lines in line plots. (Matplotlib)"""

        self._reference_line_color = "gray"

        self._bar_alpha = 0.3
        """Transparency of bars/bins. (Matplotlib)"""

        self._bar_color = "black"
        """Color of the bars/bins. (Matplotlib)"""

        self._bar_edgecolor = "none"
        """Color of the edges of the bars/bins. (Matplotlib)"""

        self._color_palette = sns.color_palette("muted")
        """Color palette to use for plots. (Seaborn)"""

        self._title_font_size = 10
        """Font size of the title of the plot. (Matplotlib)"""

        self._axis_title_font_size = 8
        """Font size of the axis titles of the plot. (Matplotlib)"""

        self._axis_major_ticklabel_font_size = 7
        """Font size of the major axis ticklabels of the plot. (Matplotlib)"""

        self._axis_minor_ticklabel_font_size = 6
        """Font size of the minor axis ticklabels of the plot. (Matplotlib)"""

        self._scilimits = (-3, 3)
        """Scientific limits for the axis labels. (Matplotlib)"""

        self._cmap = "Grays"

        self._on_sns_update()

    def _on_sns_update(self):
        """Updates the style and context of seaborn plots."""
        sns.set_style(self._style)
        sns.set_context(self._context)


plot_options = _PlotOptions()
