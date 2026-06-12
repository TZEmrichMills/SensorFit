"""Hotkey-based zoom for matplotlib screens.

Press ``z`` to toggle matplotlib's built-in zoom-rectangle mode (the same
mode as the toolbar magnifier button).  Press ``r`` to reset the view back
to the original limits.

The previous implementation tried to drive a ``RectangleSelector`` with
``useblit=True`` itself.  That approach interacted badly with screens
that have a ``button_press_event`` handler — the blit-saved background
re-rendered with whatever limits the axes were at when the selector was
created, causing the y-axis to "snap" on every click.

This rewrite simply delegates to the navigation toolbar already attached
to every figure, which correctly intercepts click events, manages the
rubber-band rectangle, and updates the view.  All existing screens
already check ``toolbar.mode`` in their click handlers to skip clicks
during zoom/pan, so no other code needs to change.
"""

from __future__ import annotations

from typing import Iterable

from matplotlib.figure import Figure
from matplotlib.axes import Axes


def install_zoom_keys(fig: Figure, axes: Axes | Iterable[Axes]) -> dict:
    """Wire ``z`` (toggle zoom) and ``r`` (reset view) into ``fig``.

    Returns a dict for compatibility with the old API; callers don't
    need to use it.
    """
    if isinstance(axes, Axes):
        ax_list: list[Axes] = [axes]
    else:
        ax_list = list(axes)

    # Capture each axes' initial view at install time so 'r' restores it
    # even after programmatic zooms.
    initial_views = {id(ax): (ax.get_xlim(), ax.get_ylim()) for ax in ax_list}

    state = {"installed": True}

    def _on_key(event) -> None:
        toolbar = fig.canvas.toolbar
        if event.key in ("z", "Z"):
            # Toolbar's zoom() toggles between zoom-rect mode and normal mode.
            # When active, clicks on the canvas draw a rubber-band rectangle
            # and the existing screen-level on_click handlers correctly skip
            # them (they all check toolbar.mode).
            if toolbar is not None and hasattr(toolbar, "zoom"):
                try:
                    toolbar.zoom()
                except Exception:
                    pass
        elif event.key in ("r", "R"):
            # Restore the initial view we captured at install time.
            for ax in ax_list:
                xlim, ylim = initial_views[id(ax)]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            # Also defuse any active zoom/pan mode so the next click acts
            # normally.
            if (
                toolbar is not None
                and getattr(toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan")
                and hasattr(toolbar, "zoom")
            ):
                try:
                    if getattr(toolbar, "mode", "").startswith("zoom"):
                        toolbar.zoom()
                    elif getattr(toolbar, "mode", "").startswith("pan"):
                        if hasattr(toolbar, "pan"):
                            toolbar.pan()
                except Exception:
                    pass
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    return state
