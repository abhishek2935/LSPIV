import matplotlib.pyplot as plt
import numpy as np


def plot_speed_heatmap(u, v):
    speed = np.sqrt(u**2 + v**2)

    if np.all(np.isnan(speed)):
        print("⚠️ Speed heatmap skipped: all NaNs")
        return

    plt.figure(figsize=(10, 6))
    im = plt.imshow(speed, cmap="jet")
    plt.colorbar(im, label="Speed (m/s)")
    plt.title("Velocity magnitude (m/s)")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_quiver(u, v, step=3):
    speed = np.sqrt(u**2 + v**2)

    if np.nanmax(speed) == 0:
        print("⚠️ Quiver skipped: zero velocity field")
        return

    mean_speed = np.nanmean(speed)
    scale = 1 / mean_speed

    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]

    plt.figure(figsize=(10, 6))
    plt.quiver(
        x[::step, ::step],
        y[::step, ::step],
        u[::step, ::step],
        -v[::step, ::step],
        scale=scale
    )
    plt.gca().invert_yaxis()
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Velocity field (quiver)")
    plt.tight_layout()
    plt.show()


def plot_streamlines(u, v, density=1.5):
    speed = np.sqrt(u**2 + v**2)

    if np.nanmax(speed) == 0:
        print("⚠️ Streamlines skipped: zero velocity field")
        return

    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]

    plt.figure(figsize=(10, 6))
    plt.streamplot(
        x, y,
        u, -v,
        color=speed,
        cmap="jet",
        density=density,
        arrowsize=1.5
    )
    plt.colorbar(label="Speed (m/s)")
    plt.gca().invert_yaxis()
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Streamlines")
    plt.tight_layout()
    plt.show()
