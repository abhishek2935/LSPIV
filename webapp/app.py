"""
LSPIV Web Application

A Streamlit-based web interface for Large Scale Particle Image Velocimetry analysis.
Allows users to upload videos, select ROI interactively, and visualize water flow analysis results.
"""

import streamlit as st
import sys
import os
import numpy as np

# Add project directories to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'V8_5_3'))

from V8_5_3.core.pipeline import LSPIVPipeline
from components.video_upload import render_video_upload
from components.roi_canvas import render_roi_canvas, get_roi_points, render_roi_preview, validate_roi
from components.parameter_panel import render_parameter_panel
from components.results_display import render_results, render_raw_data_download
from utils.session_state import init_session_state, get_state, set_state


def main():
    """Main application entry point."""

    # Page configuration
    st.set_page_config(
        page_title="LSPIV Analysis",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("🌊 LSPIV Water Flow Analysis")
    st.markdown("""
    Upload a video, select a region of interest, and analyze surface water flow velocities
    using Large Scale Particle Image Velocimetry (LSPIV).
    """)

    # Sidebar parameters
    params = render_parameter_panel()

    # Main content area
    main_container = st.container()

    with main_container:
        # Step 1: Video Upload
        video_path, first_frame = render_video_upload()

        if video_path is not None and first_frame is not None:
            set_state('video_path', video_path)
            set_state('first_frame', first_frame)
            set_state('video_uploaded', True)

            st.divider()

            # Step 2: ROI Selection
            render_roi_canvas(first_frame)

            # Extract and validate ROI points
            roi_points = get_roi_points()

            if roi_points is not None:
                is_valid, error_msg = validate_roi(roi_points)

                if is_valid:
                    set_state('roi_points', roi_points)
                    set_state('roi_set', True)
                    st.success("ROI defined with 4 points")
                else:
                    st.warning(error_msg)

            st.divider()

            # Step 3: Process
            st.subheader("Step 3: Process")

            roi_ready = get_state('roi_set', False) and get_state('roi_points') is not None

            if not roi_ready:
                st.info("Please select 4 ROI points before processing.")
            else:
                col1, col2 = st.columns([1, 4])

                with col1:
                    process_button = st.button(
                        "🚀 Start Analysis",
                        type="primary",
                        disabled=get_state('processing', False)
                    )

                with col2:
                    if get_state('processed', False):
                        if st.button("🔄 Reset & Reprocess"):
                            set_state('processed', False)
                            set_state('results', None)
                            st.rerun()

                if process_button and not get_state('processing', False):
                    set_state('processing', True)

                    # Create progress placeholders
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Initialize pipeline
                        pipeline = LSPIVPipeline()
                        pipeline.load_video(video_path)
                        pipeline.set_roi(get_state('roi_points'))
                        pipeline.set_parameters(
                            fps=params['fps'],
                            roi_width_meters=params['roi_width_meters'],
                            win_size=params['win_size'],
                            overlap=params['overlap'],
                            max_frames=params['max_frames'],
                            streamline_density=params['streamline_density']
                        )

                        # Progress callback
                        def update_progress(current, total, status):
                            progress = current / total if total > 0 else 0
                            progress_bar.progress(progress)
                            status_text.text(status)

                        # Run processing
                        results = pipeline.process(progress_callback=update_progress)

                        # Store results
                        set_state('results', results)
                        set_state('pipeline', pipeline)
                        set_state('processed', True)
                        set_state('processing', False)

                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")
                        st.success(f"Analysis complete! Processed {results.frames_processed} frame pairs.")

                    except Exception as e:
                        set_state('processing', False)
                        st.error(f"Processing failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

            # Step 4: Results Display
            if get_state('processed', False) and get_state('results') is not None:
                st.divider()
                results = get_state('results')
                pipeline = get_state('pipeline')

                render_results(results, pipeline)

                st.divider()
                render_raw_data_download(results)


def render_about():
    """Render about section in sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About LSPIV

    Large Scale Particle Image Velocimetry (LSPIV) is a
    remote sensing technique for measuring surface water
    velocities using video imagery.

    **Workflow:**
    1. Upload a video of water surface
    2. Select ROI (region of interest)
    3. Configure analysis parameters
    4. View velocity field visualizations

    ---
    *Built with Streamlit and OpenPIV*
    """)


if __name__ == "__main__":
    main()
    render_about()
