"""
Interactive ROI selection component.

Allows users to define a quadrilateral ROI by entering coordinates or using sliders.
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2


def render_roi_canvas(first_frame: np.ndarray, key: str = "roi_canvas"):
    """
    Render ROI selection interface.

    Args:
        first_frame: First video frame (BGR)
        key: Unique key for widgets

    Returns:
        None (points stored in session state)
    """
    st.subheader("Step 2: Select ROI")
    st.write("Define the 4 corners of your region of interest.")
    st.write("Points should be in order: **top-left, top-right, bottom-right, bottom-left**")

    # Get frame dimensions
    height, width = first_frame.shape[:2]

    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # Initialize default points if not set
    if 'roi_points_input' not in st.session_state:
        # Default to a centered rectangle covering ~60% of frame
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        st.session_state['roi_points_input'] = [
            [margin_x, margin_y],                    # Top-left
            [width - margin_x, margin_y],            # Top-right
            [width - margin_x, height - margin_y],   # Bottom-right
            [margin_x, height - margin_y]            # Bottom-left
        ]

    # Point labels
    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    # Create columns for point inputs
    st.write("**Enter ROI corner coordinates:**")

    points = []
    cols = st.columns(4)

    for i, (col, label) in enumerate(zip(cols, labels)):
        with col:
            st.write(f"**{label}**")
            default_x = st.session_state['roi_points_input'][i][0]
            default_y = st.session_state['roi_points_input'][i][1]

            x = st.number_input(
                f"X",
                min_value=0,
                max_value=width,
                value=int(default_x),
                key=f"{key}_x_{i}"
            )
            y = st.number_input(
                f"Y",
                min_value=0,
                max_value=height,
                value=int(default_y),
                key=f"{key}_y_{i}"
            )
            points.append([x, y])

    # Update stored points
    st.session_state['roi_points_input'] = points

    # Draw ROI preview on frame
    preview = frame_rgb.copy()
    pts = np.array(points, dtype=np.int32)

    # Draw filled polygon with transparency
    overlay = preview.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    preview = cv2.addWeighted(overlay, 0.2, preview, 0.8, 0)

    # Draw polygon outline
    cv2.polylines(preview, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw corner points with labels
    for i, (pt, label, color) in enumerate(zip(pts, labels, colors)):
        cv2.circle(preview, tuple(pt), 8, color, -1)
        cv2.circle(preview, tuple(pt), 8, (255, 255, 255), 2)
        # Add label
        label_pos = (pt[0] + 10, pt[1] - 10)
        cv2.putText(
            preview, f"{i+1}",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

    # Display preview
    st.write("**ROI Preview:**")
    st.image(preview, use_container_width=True)

    # Display coordinate summary
    st.write("**Selected Coordinates:**")
    coord_text = " | ".join([f"{labels[i]}: ({points[i][0]}, {points[i][1]})" for i in range(4)])
    st.code(coord_text)

    return None  # Points accessed via get_roi_points


def get_roi_points(canvas_result=None, scale: float = 1.0) -> np.ndarray:
    """
    Get ROI points from session state.

    Args:
        canvas_result: Unused (kept for compatibility)
        scale: Unused (kept for compatibility)

    Returns:
        np.ndarray: ROI points (4, 2) or None
    """
    if 'roi_points_input' not in st.session_state:
        return None

    points = st.session_state['roi_points_input']
    if len(points) != 4:
        return None

    return np.array(points, dtype=np.float32)


def render_roi_preview(first_frame: np.ndarray, roi_points: np.ndarray):
    """
    Render preview of selected ROI overlay on frame.

    Args:
        first_frame: First video frame (BGR)
        roi_points: ROI corner points (4, 2)
    """
    if roi_points is None or len(roi_points) != 4:
        return

    # This is now handled in render_roi_canvas
    pass


def validate_roi(roi_points: np.ndarray) -> tuple:
    """
    Validate ROI points.

    Args:
        roi_points: ROI corner points

    Returns:
        tuple: (is_valid, error_message)
    """
    if roi_points is None:
        return False, "No ROI points selected"

    if len(roi_points) != 4:
        return False, f"Need exactly 4 points, got {len(roi_points)}"

    # Check for degenerate polygon (all points too close)
    dists = []
    for i in range(4):
        d = np.linalg.norm(roi_points[i] - roi_points[(i + 1) % 4])
        dists.append(d)

    if min(dists) < 10:
        return False, "ROI edges too small. Please select a larger region."

    # Check polygon area (should be positive for proper winding)
    area = 0.5 * abs(
        (roi_points[0][0] - roi_points[2][0]) * (roi_points[1][1] - roi_points[3][1]) -
        (roi_points[1][0] - roi_points[3][0]) * (roi_points[0][1] - roi_points[2][1])
    )

    if area < 100:
        return False, "ROI area too small. Please select a larger region."

    return True, ""
