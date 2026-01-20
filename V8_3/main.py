import numpy as np
import cv2 


from data_io.video_loader import read_first_frame, frame_generator
from image_processing.roi_selector import ROISelector
from image_processing.transforms import get_perspective_transform
from image_processing.frame_warper import FrameWarper
from piv.piv_core import compute_piv
from piv.scaling import meters_per_pixel_from_width, scale_velocity
from visualization.plotter import plot_speed_heatmap, plot_quiver, plot_streamlines, overlay_streamlines_on_frame 


# ================= USER PARAMETERS =================
VIDEO_PATH = "Videos/good_river.mp4"
FPS = 30
ROI_WIDTH_METERS = 100.0
MAX_PIV_FRAMES = 80   # set to e.g. 200 for testing
# ==================================================

dt = 1.0 / FPS

# --- Read first frame ---
first_frame = read_first_frame(VIDEO_PATH)
if first_frame is None:
    raise RuntimeError("Failed to read first frame")

# --- Select ROI ---
selector = ROISelector()
roi_pts = selector.select(first_frame)

# --- Perspective transform ---
M, warp_size = get_perspective_transform(roi_pts)
warper = FrameWarper(M, warp_size)

# --- Warp all frames ---
roi_frames = []
for i, frame in enumerate(frame_generator(VIDEO_PATH)):
    roi_frames.append(warper.warp(frame))

    if MAX_PIV_FRAMES and len(roi_frames) >= MAX_PIV_FRAMES + 1:
        break

print(f"Total ROI frames: {len(roi_frames)}")

# --- Scaling ---
roi_width_px = roi_frames[0].shape[1]
meters_per_pixel = meters_per_pixel_from_width(
    ROI_WIDTH_METERS,
    roi_width_px
)

print(f"Meters per pixel: {meters_per_pixel}")
print(f"dt (s): {dt}")

# --- Compute PIV ---
velocity_fields = []

for i in range(len(roi_frames) - 1):
    print(f"PIV {i+1}/{len(roi_frames)-1}", end="\r")

    u_px, v_px = compute_piv(
        roi_frames[i],
        roi_frames[i + 1],
        win_size = 32 ,
        overlap= 16 , 
        dt=1  # PIV in pixels/frame
    )

    u, v = scale_velocity(u_px, v_px, meters_per_pixel, dt)

    # --- Diagnostics ---
    print(
        f"\nFrame {i}: "
        f"u[min,max]=({np.nanmin(u):.3f},{np.nanmax(u):.3f}) "
        f"v[min,max]=({np.nanmin(v):.3f},{np.nanmax(v):.3f}) "
        f"NaNs=({np.isnan(u).sum()},{np.isnan(v).sum()})"
    )

    velocity_fields.append((u, v))

print(f"\nTotal PIV fields: {len(velocity_fields)}")


# --- Time averaging ---
print("\nComputing time-averaged velocity field...")

u_stack = np.array([uv[0] for uv in velocity_fields])
v_stack = np.array([uv[1] for uv in velocity_fields])

u_mean = np.nanmean(u_stack, axis=0)
v_mean = np.nanmean(v_stack, axis=0)

# --- VALID FRACTION DIAGNOSTIC (ADD THIS HERE) ---
valid_fraction = np.sum(~np.isnan(u_stack), axis=0) / u_stack.shape[0]
print("Mean valid vector fraction:", np.nanmean(valid_fraction))

print(
    "Averaged field stats:",
    f"u[min,max]=({np.nanmin(u_mean):.3f},{np.nanmax(u_mean):.3f})",
    f"v[min,max]=({np.nanmin(v_mean):.3f},{np.nanmax(v_mean):.3f})"
)


# --- Plot first valid field ---
u0, v0 = u_mean, v_mean

plot_speed_heatmap(u0, v0)
plot_quiver(u0, v0)
plot_streamlines(u0, v0)

overlay = overlay_streamlines_on_frame(
    original_frame=first_frame,
    u=u_mean,
    v=v_mean,
    M=M,
    roi_shape=roi_frames[0].shape,
    alpha=0.7
)

cv2.imshow("Streamlines Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
