"""
Parameter panel component for LSPIV web application.

Provides widgets for configuring analysis parameters in the sidebar.
"""

import streamlit as st


def render_parameter_panel() -> dict:
    """
    Render parameter input panel in sidebar.

    Returns:
        dict: Dictionary of parameter values
    """
    st.sidebar.header("Analysis Parameters")

    # Frame rate
    fps = st.sidebar.number_input(
        "Video FPS",
        min_value=1.0,
        max_value=240.0,
        value=st.session_state.get('fps', 30.0),
        step=1.0,
        help="Frame rate of the video (frames per second)"
    )

    st.sidebar.divider()

    # Physical scaling
    st.sidebar.subheader("Physical Scaling")
    roi_width_meters = st.sidebar.number_input(
        "ROI Width (meters)",
        min_value=0.1,
        max_value=1000.0,
        value=st.session_state.get('roi_width_meters', 10.0),
        step=0.5,
        help="Physical width of the ROI in meters (for velocity scaling)"
    )

    st.sidebar.divider()

    # PIV parameters
    st.sidebar.subheader("PIV Settings")

    win_size = st.sidebar.select_slider(
        "Window Size (pixels)",
        options=[16, 24, 32, 48, 64, 96, 128],
        value=st.session_state.get('win_size', 32),
        help="Interrogation window size for PIV"
    )

    # Calculate valid overlap values (must be less than window size)
    overlap_options = [v for v in [8, 12, 16, 24, 32, 48, 64] if v < win_size]
    default_overlap = min(overlap_options, key=lambda x: abs(x - win_size // 2))

    overlap = st.sidebar.select_slider(
        "Overlap (pixels)",
        options=overlap_options,
        value=st.session_state.get('overlap', default_overlap),
        help="Window overlap in pixels (typically 50% of window size)"
    )

    st.sidebar.divider()

    # Processing limits
    st.sidebar.subheader("Processing")

    max_frames = st.sidebar.number_input(
        "Max Frames to Analyze",
        min_value=3,
        max_value=500,
        value=st.session_state.get('max_frames', 50),
        step=10,
        help="Maximum number of frames to process"
    )

    st.sidebar.divider()

    # Visualization
    st.sidebar.subheader("Visualization")

    streamline_density = st.sidebar.slider(
        "Streamline Density",
        min_value=0.5,
        max_value=3.0,
        value=st.session_state.get('streamline_density', 1.5),
        step=0.1,
        help="Density of streamlines in visualization"
    )

    # Store parameters in session state
    params = {
        'fps': fps,
        'roi_width_meters': roi_width_meters,
        'win_size': win_size,
        'overlap': overlap,
        'max_frames': max_frames,
        'streamline_density': streamline_density,
    }

    # Update session state
    for key, value in params.items():
        st.session_state[key] = value

    return params


def render_parameter_summary(params: dict):
    """
    Render a summary of current parameters.

    Args:
        params: Dictionary of parameter values
    """
    st.sidebar.divider()
    st.sidebar.subheader("Current Settings")

    summary = f"""
    - **FPS:** {params['fps']:.1f}
    - **ROI Width:** {params['roi_width_meters']:.1f} m
    - **Window:** {params['win_size']}px
    - **Overlap:** {params['overlap']}px
    - **Max Frames:** {params['max_frames']}
    - **Streamline Density:** {params['streamline_density']:.1f}
    """
    st.sidebar.markdown(summary)


def get_default_parameters() -> dict:
    """
    Get default parameter values.

    Returns:
        dict: Default parameter values
    """
    return {
        'fps': 30.0,
        'roi_width_meters': 10.0,
        'win_size': 32,
        'overlap': 16,
        'max_frames': 50,
        'streamline_density': 1.5,
    }
