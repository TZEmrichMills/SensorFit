"""Single-window session manager for SensorFit interactive UI.

Problem this solves
-------------------
The original code created a fresh OS-level matplotlib window for every
interactive step (``plt.subplots`` → ``plt.show`` → ``plt.close``), then
opened a new one for the next step. A typical run cycled 10-20+ windows.

Rapid destroy-create of native windows is a known trigger for DisplayLink
driver crashes (each transition is a GPU-context handoff that DisplayLink
mirrors, and the driver races on teardown). It also inflates memory
churn and makes cross-step size mismatches jarring.

Design
------
One process-wide ``WindowManager`` opens a single figure on first use and
reuses it for every subsequent step. Each step calls ``reset(figsize=...)``
to clear the previous step's contents and (optionally) resize the same
window; it never closes it. Instead of the old ``plt.show()`` +
``plt.close()`` pair, callbacks call ``stop()`` to release the blocking
event loop while leaving the window alive.

The requested figsize is clamped to what fits on the primary display so
users on smaller laptops still see the buttons at the bottom.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt


# ────────────────────────────────────────────────────────────────────────
# Screen-size detection
# ────────────────────────────────────────────────────────────────────────

def _detect_screen_inches() -> Tuple[float, float]:
    """Return (max_width_in, max_height_in) that comfortably fits on the
    primary display, leaving room for the OS chrome (menu bar, dock,
    Windows taskbar, matplotlib toolbar).

    Falls back to a safe 12 × 7.5 if detection fails.
    """
    # Try Tk first — cross-platform and cheap.
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w_px = root.winfo_screenwidth()
        h_px = root.winfo_screenheight()
        dpi = float(root.winfo_fpixels("1i"))
        root.destroy()
        if dpi > 0 and w_px > 0 and h_px > 0:
            w_in = w_px / dpi
            h_in = h_px / dpi
            # Leave ~1.0" side margins, ~1.6" vertical (menu bar + dock +
            # matplotlib toolbar). Clamp to sane bounds.
            usable_w = max(8.0, min(w_in - 1.0, 20.0))
            usable_h = max(5.5, min(h_in - 1.6, 12.0))
            return usable_w, usable_h
    except Exception:
        pass

    # Fallback: assume a 13" MacBook-ish display.
    return 12.0, 7.5


def _clamp_figsize(
    requested: Tuple[float, float],
    ceiling: Tuple[float, float],
) -> Tuple[float, float]:
    """Clamp a requested (w, h) to the screen ceiling, preserving aspect
    ratio if either dimension exceeds it."""
    rw, rh = requested
    cw, ch = ceiling
    if rw <= cw and rh <= ch:
        return rw, rh
    scale_w = cw / rw if rw > cw else 1.0
    scale_h = ch / rh if rh > ch else 1.0
    scale = min(scale_w, scale_h)
    return rw * scale, rh * scale


# ────────────────────────────────────────────────────────────────────────
# WindowManager
# ────────────────────────────────────────────────────────────────────────

class WindowManager:
    """Owns a single reusable matplotlib figure for the whole app run."""

    def __init__(self) -> None:
        self._fig: Optional[plt.Figure] = None
        self._screen_ceiling: Optional[Tuple[float, float]] = None

    # -- lifecycle ------------------------------------------------------

    @property
    def screen_ceiling(self) -> Tuple[float, float]:
        if self._screen_ceiling is None:
            self._screen_ceiling = _detect_screen_inches()
        return self._screen_ceiling

    def reset(
        self,
        figsize: Tuple[float, float] = (11.0, 6.5),
        title: str = "SensorFit",
    ) -> plt.Figure:
        """Return a cleared figure sized for this step.

        On the first call this opens the window; on subsequent calls it
        clears the previous step's axes/widgets and resizes the *same*
        window — no OS-level destroy/create.
        """
        w, h = _clamp_figsize(figsize, self.screen_ceiling)

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            # Either first use, or the user closed the window with the OS
            # close button. Reopen fresh.
            self._fig = plt.figure(figsize=(w, h))
        else:
            self._fig.clf()
            try:
                self._fig.set_size_inches(w, h, forward=True)
            except Exception:
                pass

        fig = self._fig
        try:
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

        return fig

    def run(self) -> None:
        """Block until a callback calls ``stop()``. Replaces the old
        ``plt.show(); plt.close(fig)`` pair."""
        if self._fig is None:
            return
        # Ensure the OS window is actually displayed. On first call the
        # figure exists but hasn't been shown; on subsequent calls after
        # ``clf()`` some backends need a nudge to raise the window.
        try:
            mgr = self._fig.canvas.manager
            if mgr is not None:
                mgr.show()
        except Exception:
            pass
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass
        # start_event_loop blocks matplotlib's own event loop without
        # tearing down the window on exit. Callbacks call stop_event_loop.
        try:
            self._fig.canvas.start_event_loop(timeout=0)
        except Exception:
            # Fall back to plt.show() if the backend can't nest event
            # loops. This preserves behaviour on unusual backends.
            plt.show()

    def stop(self) -> None:
        """Release ``run()``'s blocking event loop, leaving the window
        open for the next step to reuse."""
        if self._fig is None:
            return
        try:
            self._fig.canvas.stop_event_loop()
        except Exception:
            # If the backend doesn't support the event-loop hooks and we
            # fell back to plt.show(), the callback should call
            # plt.close(fig) instead. But we can't tell which path was
            # taken here, so we no-op and let the caller decide.
            pass

    def finish(self) -> None:
        """Close the window at the end of the whole run."""
        if self._fig is not None and plt.fignum_exists(self._fig.number):
            plt.close(self._fig)
        self._fig = None


# ────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ────────────────────────────────────────────────────────────────────────

_WINDOW: Optional[WindowManager] = None


def get_window() -> WindowManager:
    global _WINDOW
    if _WINDOW is None:
        _WINDOW = WindowManager()
    return _WINDOW


def finish_window() -> None:
    """Close the shared window at the end of a processing session."""
    global _WINDOW
    if _WINDOW is not None:
        _WINDOW.finish()
        _WINDOW = None
