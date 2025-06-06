# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/11.10
# __author:       Lang Zhu
# __corporation:  OriginQuantum
"""
Plot Util.
"""

import io
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure


def get_fig(figsize: Tuple[int, int] = (10, 6),
            default_figure_canvas=FigureCanvasSVG) -> Figure:
    """Return a matplotlib axes that can be used in a child thread.

    Analysis/plotting is done in a separate thread (so it doesn't block the
    main thread), but matplotlib doesn't support GUI mode in a child thread.
    This function creates a separate Figure and attaches a non-GUI
    SVG canvas to it.

    Returns:
        matplotlib.figure.Figure
    """

    figure = Figure(figsize=figsize)
    _ = default_figure_canvas(figure)
    return figure


def get_ax(figsize: Tuple[int, int] = (10, 6),
           subplots: Tuple[int, int] = (1, 1),
           default_figure_canvas=FigureCanvasSVG) -> Axes:
    """Return a matplotlib axes that can be used in a child thread.

    Analysis/plotting is done in a separate thread (so it doesn't block the
    main thread), but matplotlib doesn't support GUI mode in a child thread.
    This function creates a separate Figure and attaches a non-GUI
    SVG canvas to it.

    Returns:
        matplotlib.axes.Axes: A matplotlib axes that can be used in a child thread.
    """
    figure = Figure(figsize=figsize)
    _ = default_figure_canvas(figure)
    return figure.subplots(*subplots)


def plot_to_bytes(figure: "pyplot.Figure", format_plot="png") -> bytes:
    """Convert a pyplot Figure to SVG in bytes.

    Args:
        figure: Figure to be converted

    Returns:
        Figure in bytes.
    """
    buf = io.BytesIO()
    opaque_color = list(figure.get_facecolor())
    opaque_color[3] = 1.0  # set alpha to opaque
    figure.savefig(buf,
                   format=format_plot,
                   dpi=500,
                   facecolor=tuple(opaque_color),
                   edgecolor="none",
                   bbox_inches="tight")
    buf.seek(0)
    figure_data = buf.read()
    buf.close()
    return figure_data
