#!/usr/bin/env python3
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

import game_of_life_vect


def main() -> int:
    # More complex demo than a single glider: Gosper glider gun.
    dims = (400, 400)
    glider_gun = [
        (51, 76), (52, 74), (52, 76), (53, 64), (53, 65), (53, 72), (53, 73), (53, 86), (53, 87),
        (54, 63), (54, 67), (54, 72), (54, 73), (54, 86), (54, 87), (55, 52), (55, 53), (55, 62),
        (55, 68), (55, 72), (55, 73), (56, 52), (56, 53), (56, 62), (56, 66), (56, 68), (56, 69),
        (56, 74), (56, 76), (57, 62), (57, 68), (57, 76), (58, 63), (58, 67), (59, 64), (59, 65),
    ]

    grid = game_of_life_vect.Grille(dims, init_pattern=glider_gun)

    frames = 140
    steps_per_frame = 1

    # Zoom/crop to improve readability in the report.
    # This ROI shows the gun and the first emitted gliders.
    y0, y1 = 30, 150
    x0, x1 = 35, 260

    # Render at higher resolution; keep file size manageable.
    dpi = 140
    fig_w, fig_h = 9.0, 5.0

    with imageio.get_writer("gol_demo.gif", mode="I", duration=0.06, loop=0) as writer:
        for _ in range(frames):
            for _ in range(steps_per_frame):
                grid.compute_next_iteration()

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            view = grid.cells[y0:y1, x0:x1]
            ax.imshow(view, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1, aspect="auto")
            ax.set_axis_off()
            fig.tight_layout(pad=0)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            rgba = np.asarray(fig.canvas.buffer_rgba()).reshape((h, w, 4))
            rgb = rgba[:, :, :3].copy()
            writer.append_data(rgb)
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
