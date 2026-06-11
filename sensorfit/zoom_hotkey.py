"""Hotkey-based zoom for matplotlib screens.

Press ``z`` to toggle a zoom-rectangle mode (next click-drag selects a region
to zoom into).  Press ``r`` to reset the axes back to the full data extent.

Both work in addition to matplotlib's built-in toolbar buttons — they're
just there as keyboard shortcuts for users who prefer not to take their hand
off the mouse / keyboard to find the toolbar.

Usage
-----

    fig, ax = plt.subplots()
    ax.plot(t, y)
    install_zoom_keys(fig, ax)
    plt.show()

For multi-axes figures pass a list of axes; ``r`` resets all of them, while
``z`` only operates on the axes the mouse is over (the underlying
RectangleSelector is bound per-axes).
"""

from __future__ import annotations

from typing import Iterable

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.widgets import RectangleSelector


def install_zoom_keys(fig: Figure, axes: Axes | Iterable[Axes]) -> dict:
    """Wire ``z`` (toggle zoom) and ``r`` (reset view) into ``fig``.

    Returns a dict carrying the active state and the per-axes
    ``RectangleSelector`` instances — useful only for tests.  Callers don't
    need to use it.
    """
    if isinstance(axes, Axes):
        ax_list: list[Axes] = [axes]
    else:
        ax_list = list(axes)

    # Capture each axes' initial view so 'r' restores it even after
    # programmatic zooms.
    initial_views = {
        id(ax): (ax.get_xlim(), ax.get_ylim()) for ax in ax_list
    }

    state = {"zoom_active": False, "selectors": {}}

    def _on_select(eclick, erelease, *, ax: Axes) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, x1 = sorted((float(eclick.xdata), float(erelease.xdata)))
        y0, y1 = sorted((float(eclick.ydata), float(erelease.ydata)))
        # Avoid zooming into a degenerate region from an accidental click
        if (x1 - x0) < 1e-12 or (y1 - y0) < 1e-12:
            return
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        fig.canvas.draw_idle()

    # Build one inactive RectangleSelector per axes.
    for ax in ax_list:
        selector = RectangleSelector(
            ax,
            lambda eclick, erelease, _ax=ax: _on_select(eclick, erelease, ax=_ax),
            useblit=True,
            button=[1],  # left mouse only
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=False,
        )
        selector.set_active(False)
        state["selectors"][id(ax)] = selector

    def _set_zoom(active: bool) -> None:
        state["zoom_active"] = active
        for sel in state["selectors"].values():
            sel.set_active(active)
        # Visual cue: change cursor for the canvas.
        try:
            fig.canvas.set_cursor(2 if active else 1)  # 2=crosshair, 1=hand
        except Exception:
            pass
        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        if event.key in ("z", "Z"):
            _set_zoom(not state["zoom_active"])
        elif event.key in ("r", "R"):
            for ax in ax_list:
                xlim, ylim = initial_views[id(ax)]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            # Turn zoom mode off after a reset so the next click is normal
            if state["zoom_active"]:
                _set_zoom(False)
            else:
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    return state
