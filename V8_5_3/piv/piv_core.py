import numpy as np
from openpiv import tools, pyprocess, validation, filters
import cv2
from typing import Generator, Tuple, Callable, Optional

def compute_piv(frame_a, frame_b,
                win_size=16,
                overlap=8,
                dt=1):

    frame_a = tools.rgb2gray(frame_a)
    frame_b = tools.rgb2gray(frame_b)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame_a = clahe.apply(frame_a.astype(np.uint8))
    frame_b = clahe.apply(frame_b.astype(np.uint8))


    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=win_size,
        overlap=overlap,
        dt=dt,
        search_area_size=win_size,
        sig2noise_method='peak2peak'
    )

    # Signal-to-noise validation
    mask = validation.sig2noise_val(sig2noise, threshold=1.0)

    # Mark invalid vectors
    bad_ratio = np.sum(mask) / mask.size
    print(f"Bad vectors ratio: {bad_ratio:.2f}")

    if bad_ratio < 0.8:
        u, v = filters.replace_outliers(
            u, v,
            flags=mask,
            method='localmean',
            max_iter=3,
            kernel_size=2
        )
    else:
        print("O_O Validation skipped: too many bad vectors")

    return u, v


def streaming_piv(
    frame_generator: Generator,
    warper,
    win_size: int = 32,
    overlap: int = 16,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    verbose: bool = False
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Memory-efficient streaming PIV that processes frame pairs on-the-fly.

    Only keeps 2 frames in memory at a time, yielding velocity fields
    as they are computed.

    Args:
        frame_generator: Generator yielding video frames
        warper: FrameWarper instance for perspective transformation
        win_size: PIV interrogation window size
        overlap: Window overlap in pixels
        max_frames: Maximum number of frames to process (None for all)
        progress_callback: Optional callback(current, total) for progress updates
        verbose: Print diagnostic information

    Yields:
        Tuple[np.ndarray, np.ndarray]: (u, v) velocity field for each frame pair
    """
    prev_frame = None
    frame_count = 0
    pair_count = 0

    for frame in frame_generator:
        # Apply perspective warp
        warped = warper.warp(frame)

        if prev_frame is not None:
            # Compute PIV for this frame pair
            u, v = compute_piv(
                prev_frame,
                warped,
                win_size=win_size,
                overlap=overlap,
                dt=1
            )

            pair_count += 1

            if progress_callback:
                total = max_frames - 1 if max_frames else pair_count
                progress_callback(pair_count, total)

            if verbose:
                print(f"PIV pair {pair_count}: u_range=[{np.nanmin(u):.3f}, {np.nanmax(u):.3f}]")

            yield u, v

        # Keep only current frame for next iteration
        prev_frame = warped
        frame_count += 1

        # Check max frames limit
        if max_frames and frame_count >= max_frames:
            break
