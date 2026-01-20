import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import cm, colors
from matplotlib.streamplot import StreamplotSet


def plot_speed_heatmap(u, v):
    """Plot velocity magnitude as a heatmap."""
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
    """Plot velocity vectors as quiver plot."""
    speed = np.sqrt(u**2 + v**2)
    if np.nanmax(speed) == 0:
        print("⚠️ Quiver skipped: zero velocity field")
        return
    
    mean_speed = np.nanmean(speed)
    scale = 1 / mean_speed if mean_speed > 0 else 1
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    
    plt.figure(figsize=(10, 6))
    plt.quiver(
        x[::step, ::step], y[::step, ::step],
        u[::step, ::step], v[::step, ::step],
        scale=scale
    )
    plt.gca().invert_yaxis()
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Velocity field (quiver)")
    plt.tight_layout()
    plt.show()


def plot_streamlines(u, v, density=1.5):
    """Plot streamlines colored by velocity magnitude."""
    speed = np.sqrt(u**2 + v**2)
    if np.nanmax(speed) == 0:
        print("⚠️ Streamlines skipped: zero velocity field")
        return
    
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    
    plt.figure(figsize=(10, 6))
    plt.streamplot(
        x, y, u, v,
        color=speed, cmap="jet",
        density=density, arrowsize=1.5
    )
    plt.colorbar(label="Speed (m/s)")
    plt.gca().invert_yaxis()
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Streamlines")
    plt.tight_layout()
    plt.show()



def render_colored_streamlines_to_image(u, v, roi_shape, density=1.2):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    ny, nx = u.shape
    y, x = np.mgrid[0:ny, 0:nx]
    speed = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(nx / 10, ny / 10), dpi=100)

    # Generate streamlines WITHOUT arrows
    sp = ax.streamplot(
        x, y,
        u, v,
        density=density,
        linewidth=1,
        arrowsize=0
    )

    # Prepare colormap
    vmin = np.nanmin(speed)
    vmax = np.nanmax(speed)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.jet.reversed()

    # Remove auto-generated lines
    sp.lines.remove()

    # Draw streamlines manually
    for path in sp.lines.get_paths():
        verts = path.vertices
        if len(verts) < 10:
            continue

        xs = np.clip(verts[:, 0].astype(int), 0, nx - 1)
        ys = np.clip(verts[:, 1].astype(int), 0, ny - 1)

        local_speed = speed[ys, xs]
        mean_speed = np.nanmean(local_speed)

        if np.isnan(mean_speed):
            continue

        color = cmap(norm(mean_speed))

        ax.plot(
            verts[:, 0],
            verts[:, 1],
            color=color,
            linewidth=1
        )

        # ---- Direction arrows (manual, stable) ----
        step = max(len(verts) // 5, 1)
        for i in range(step, len(verts) - step, step):
            dx = verts[i + 1, 0] - verts[i, 0]
            dy = verts[i + 1, 1] - verts[i, 1]

            if dx == 0 and dy == 0:
                continue

            ax.arrow(
                verts[i, 0],
                verts[i, 1],
                dx,
                dy,
                color=color,
                head_width=0.8,
                head_length=1.2,
                length_includes_head=True
            )

    # Axes handling
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.axis("off")

    # ---- Colorbar ----
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Velocity magnitude (m/s)")

    # Rasterize
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()
    plt.close(fig)

    # Resize to ROI image size
    overlay_resized = cv2.resize(
        img,
        (roi_shape[1], roi_shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    return overlay_resized


def overlay_streamlines_on_frame(original_frame, u, v, M, roi_shape, alpha=0.7):
    """
    Overlay perspective-warped streamlines on original frame.
    
    Args:
        original_frame: BGR input frame
        u, v: Velocity fields in ROI space
        M: Forward homography matrix (ROI → frame)
        roi_shape: ROI shape (height, width)
        alpha: Overlay transparency (0-1)
    
    Returns:
        BGR frame with streamlines overlay
    """
    # Render streamlines in ROI space
    roi_overlay = render_colored_streamlines_to_image(u, v, roi_shape)
    
    # Inverse perspective transform (ROI → original frame)
    M_inv = np.linalg.inv(M)
    overlay_original = cv2.warpPerspective(
        roi_overlay, M_inv,
        (original_frame.shape[1], original_frame.shape[0])
    )
    
    # Create mask from overlay
    overlay_gray = cv2.cvtColor(overlay_original, cv2.COLOR_BGR2GRAY)
    mask = overlay_gray > 10  # Threshold for visibility
    
    # Alpha blend where overlay is present
    result = original_frame.copy()
    result[mask] = cv2.addWeighted(
        original_frame[mask], 1 - alpha,
        overlay_original[mask], alpha, 0
    )
    
    return result
