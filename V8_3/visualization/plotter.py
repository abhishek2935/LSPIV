import matplotlib.pyplot as plt
import numpy as np
import cv2


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
        v[::step, ::step],
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
        u, v,
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



####

def render_streamlines_to_image(u, v, roi_shape, density=1.5):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    ny, nx = u.shape
    y, x = np.mgrid[0:ny, 0:nx]

    fig, ax = plt.subplots(figsize=(nx / 10, ny / 10), dpi=100)

    ax.streamplot(
        x, y,
        u, v,
        color="black",
        density=density,
        linewidth=1
    )

    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.axis("off")

    fig.canvas.draw()

    # --- Backend-safe image extraction ---
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()  # drop alpha channel

    plt.close(fig)

    # --- Resize overlay to ROI image size ---
    overlay_resized = cv2.resize(
        img,
        (roi_shape[1], roi_shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    print("Overlay image shape:", overlay_resized.shape)

    return overlay_resized


def overlay_streamlines_on_frame(
    original_frame,
    u,
    v,
    M,
    roi_shape,
    alpha=0.7
):
    """
    Inverse-warp streamlines from ROI space back to original frame
    and overlay them.
    """

    # --- Render streamlines in ROI space ---
    roi_overlay = render_streamlines_to_image(u, v, roi_shape)

    # --- Inverse perspective transform ---
    M_inv = np.linalg.inv(M)

    overlay_original = cv2.warpPerspective(
        roi_overlay,
        M_inv,
        (original_frame.shape[1], original_frame.shape[0])
    )

    # --- Alpha blending ---
    overlay_gray = cv2.cvtColor(overlay_original, cv2.COLOR_BGR2GRAY)
    mask = overlay_gray > 0

    result = original_frame.copy()
    result[mask] = cv2.addWeighted(
        original_frame[mask],
        1 - alpha,
        overlay_original[mask],
        alpha,
        0
    )

    return result