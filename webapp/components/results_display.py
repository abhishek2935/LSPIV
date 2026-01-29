"""
Results display component for LSPIV web application.

Provides visualization tabs and statistics display for LSPIV results.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Add V8_5_3 to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'V8_5_3'))


def render_results(results, pipeline):
    """
    Render results display with tabs for different visualizations.

    Args:
        results: LSPIVResults object from pipeline.process()
        pipeline: LSPIVPipeline instance for rendering visualizations
    """
    st.subheader("Step 4: Results")

    # Statistics summary
    render_statistics(results)

    st.divider()

    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Streamlines Overlay",
        "Speed Heatmap",
        "Quiver Plot",
        "Streamlines Plot"
    ])

    with tab1:
        render_streamlines_overlay(results, pipeline)

    with tab2:
        render_speed_heatmap(results, pipeline)

    with tab3:
        render_quiver_plot(results, pipeline)

    with tab4:
        render_streamlines_plot(results, pipeline)


def render_statistics(results):
    """
    Render statistics summary cards.

    Args:
        results: LSPIVResults object
    """
    st.write("### Flow Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mean Velocity",
            f"{results.mean_velocity:.3f} m/s",
            help="Time-averaged mean velocity magnitude"
        )

    with col2:
        st.metric(
            "Max Velocity",
            f"{results.max_velocity:.3f} m/s",
            help="Maximum velocity magnitude"
        )

    with col3:
        st.metric(
            "Valid Vectors",
            f"{results.valid_vector_fraction * 100:.1f}%",
            help="Percentage of valid velocity vectors"
        )

    with col4:
        st.metric(
            "Flow Direction",
            f"{results.dominant_direction:.1f}°",
            help="Dominant flow direction (degrees from horizontal)"
        )

    # Additional info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Frames Processed",
            results.frames_processed,
        )

    with col2:
        st.metric(
            "Processing Time",
            f"{results.processing_time:.1f}s",
        )

    with col3:
        st.metric(
            "Resolution",
            f"{results.meters_per_pixel * 1000:.2f} mm/px",
            help="Spatial resolution (millimeters per pixel)"
        )


def render_streamlines_overlay(results, pipeline):
    """
    Render streamlines overlay on original frame.

    Args:
        results: LSPIVResults object
        pipeline: LSPIVPipeline instance
    """
    st.write("**Streamlines Overlay on Original Frame**")

    try:
        overlay = pipeline.render_streamlines_overlay(results)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        st.image(overlay_rgb, use_container_width=True)

        # Download button
        img_pil = Image.fromarray(overlay_rgb)
        import io
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        st.download_button(
            label="Download Streamlines Image",
            data=buf.getvalue(),
            file_name="lspiv_streamlines_overlay.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error rendering streamlines overlay: {e}")


def render_speed_heatmap(results, pipeline):
    """
    Render velocity magnitude heatmap.

    Args:
        results: LSPIVResults object
        pipeline: LSPIVPipeline instance
    """
    st.write("**Velocity Magnitude Heatmap**")

    try:
        heatmap = pipeline.render_speed_heatmap(results)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        st.image(heatmap_rgb, use_container_width=True)

        # Download button
        img_pil = Image.fromarray(heatmap_rgb)
        import io
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        st.download_button(
            label="Download Heatmap Image",
            data=buf.getvalue(),
            file_name="lspiv_speed_heatmap.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error rendering heatmap: {e}")


def render_quiver_plot(results, pipeline):
    """
    Render velocity vector quiver plot.

    Args:
        results: LSPIVResults object
        pipeline: LSPIVPipeline instance
    """
    st.write("**Velocity Vector Field (Quiver Plot)**")

    # Step size control
    step = st.slider("Vector spacing", 1, 10, 3, key="quiver_step")

    try:
        quiver = pipeline.render_quiver(results, step=step)
        quiver_rgb = cv2.cvtColor(quiver, cv2.COLOR_BGR2RGB)
        st.image(quiver_rgb, use_container_width=True)

        # Download button
        img_pil = Image.fromarray(quiver_rgb)
        import io
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        st.download_button(
            label="Download Quiver Plot",
            data=buf.getvalue(),
            file_name="lspiv_quiver_plot.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error rendering quiver plot: {e}")


def render_streamlines_plot(results, pipeline):
    """
    Render matplotlib streamlines plot.

    Args:
        results: LSPIVResults object
        pipeline: LSPIVPipeline instance
    """
    st.write("**Streamlines Plot**")

    try:
        streamlines = pipeline.render_streamlines_plot(results)
        streamlines_rgb = cv2.cvtColor(streamlines, cv2.COLOR_BGR2RGB)
        st.image(streamlines_rgb, use_container_width=True)

        # Download button
        img_pil = Image.fromarray(streamlines_rgb)
        import io
        buf = io.BytesIO()
        img_pil.save(buf, format='PNG')
        st.download_button(
            label="Download Streamlines Plot",
            data=buf.getvalue(),
            file_name="lspiv_streamlines_plot.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error rendering streamlines plot: {e}")


def render_raw_data_download(results):
    """
    Provide download options for raw velocity data.

    Args:
        results: LSPIVResults object
    """
    st.write("### Download Raw Data")

    col1, col2 = st.columns(2)

    with col1:
        # Download mean velocity fields as NPZ
        import io
        buf = io.BytesIO()
        np.savez(
            buf,
            u_mean=results.u_mean,
            v_mean=results.v_mean,
            speed=results.speed
        )
        st.download_button(
            label="Download Velocity Fields (NPZ)",
            data=buf.getvalue(),
            file_name="lspiv_velocity_fields.npz",
            mime="application/octet-stream"
        )

    with col2:
        # Download statistics as JSON
        import json
        stats_json = json.dumps(results.to_dict(), indent=2)
        st.download_button(
            label="Download Statistics (JSON)",
            data=stats_json,
            file_name="lspiv_statistics.json",
            mime="application/json"
        )
