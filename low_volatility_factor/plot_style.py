"""Shared rendering policy for low-volatility article figures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PlotConfig


def dark_plot_config(plot_config: PlotConfig) -> PlotConfig:
    """Return the same visual grammar tuned for the website's dark surface."""

    return replace(
        plot_config,
        decile_profile_color="#8B949E",
        low_volatility_color="#78A0C4",
        high_volatility_color="#A093B8",
        naive_long_short_color="#8B949E",
        volatility_scaled_color="#78A0C4",
        realized_beta_color="#B8C1C9",
        ex_ante_beta_color="#66717C",
        text_color="#C9D1D9",
        muted_text_color="#8B949E",
        grid_color="#30363D",
        background_color="#0D1117",
        zero_line_color="#6E7681",
    )


def finish_figure(
    figure: plt.Figure,
    path: Path,
    plot_config: PlotConfig,
    *,
    tight_layout: bool = True,
    tick_label_size: float = 10.6,
    axis_label_size: float = 12.1,
    title_size: float = 12.1,
    legend_size: float = 10.9,
) -> None:
    """Apply the article style, save one figure, and release its memory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor(plot_config.background_color)
    for axis in figure.axes:
        axis.set_facecolor(plot_config.background_color)
        axis.tick_params(
            colors=plot_config.muted_text_color,
            labelsize=tick_label_size,
            length=0,
            width=0,
        )
        axis.xaxis.label.set_color(plot_config.muted_text_color)
        axis.xaxis.label.set_fontsize(axis_label_size)
        axis.yaxis.label.set_color(plot_config.muted_text_color)
        axis.yaxis.label.set_fontsize(axis_label_size)
        axis.title.set_color(plot_config.text_color)
        axis.title.set_fontweight("normal")
        axis.title.set_fontsize(title_size)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_color(plot_config.text_color)
                label.set_fontsize(legend_size)
    if tight_layout:
        figure.tight_layout()
    figure.savefig(
        path,
        format=path.suffix.removeprefix("."),
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor=plot_config.background_color,
    )
    plt.close(figure)


def clean_axis(axis: plt.Axes, plot_config: PlotConfig) -> None:
    """Remove chart furniture and retain only a quiet horizontal grid."""

    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="y", color=plot_config.grid_color, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.margins(x=0)


def render_figure(
    figures: Path,
    stem: str,
    renderer: Callable[..., None],
    *,
    variant_suffix: str = "",
) -> None:
    """Render the shared SVG layout used at every viewport width."""

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
        }
    ):
        renderer(path=figures / f"{stem}{variant_suffix}.svg")
