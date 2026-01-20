import numpy as np
from openpiv import tools, pyprocess, validation, filters

def compute_piv(frame_a, frame_b,
                win_size=32,
                overlap=16,
                dt=1):

    frame_a = tools.rgb2gray(frame_a)
    frame_b = tools.rgb2gray(frame_b)

    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=win_size,
        overlap=overlap,
        dt=dt,
        search_area_size=win_size,
        sig2noise_method='peak2peak'
    )

    '''# Signal-to-noise validation
    mask = validation.sig2noise_val(sig2noise, threshold=1.3)

    # Mark invalid vectors
    u[mask] = np.nan
    v[mask] = np.nan

    # Replace outliers using mask
    u, v = filters.replace_outliers(
        u, v,
        flags=mask,
        method='localmean',
        max_iter=3,
        kernel_size=2
    )'''

    return u, v
