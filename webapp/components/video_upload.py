"""
Video upload component for LSPIV web application.

Handles video file upload and first frame preview.
"""

import streamlit as st
import tempfile
import os
import cv2
from PIL import Image
import numpy as np


def render_video_upload():
    """
    Render video upload widget and preview.

    Returns:
        tuple: (video_path, first_frame) if video uploaded, (None, None) otherwise
    """
    st.subheader("Step 1: Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for LSPIV analysis"
    )

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Read first frame
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()

        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if not ret or first_frame is None:
            st.error("Failed to read video file. Please try a different file.")
            return None, None

        # Display video info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("FPS", f"{fps:.1f}")
        with col2:
            st.metric("Frames", frame_count)
        with col3:
            st.metric("Width", width)
        with col4:
            st.metric("Height", height)

        # Display first frame preview
        st.write("**First Frame Preview:**")
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True)

        # Update suggested FPS in session state
        if 'fps' not in st.session_state or st.session_state.get('fps_auto_set', False) is False:
            st.session_state['fps'] = fps
            st.session_state['fps_auto_set'] = True

        return video_path, first_frame

    return None, None


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        dict: Video metadata (fps, frame_count, width, height, duration)
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    cap.release()
    return info
