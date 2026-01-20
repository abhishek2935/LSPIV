def meters_per_pixel_from_width(roi_width_m, roi_width_px):
    return roi_width_m / roi_width_px


def scale_velocity(u, v, meters_per_pixel, dt):
    factor = meters_per_pixel / dt
    return u * factor, v * factor
