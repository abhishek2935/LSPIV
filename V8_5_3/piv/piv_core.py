import numpy as np
from openpiv import tools, pyprocess, validation, filters
import cv2

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


    # Replace outliers using mask
    u, v = filters.replace_outliers(
        u, v,
        flags=mask,
        method='localmean',
        max_iter=3,
        kernel_size=2
    )

    return u, v
