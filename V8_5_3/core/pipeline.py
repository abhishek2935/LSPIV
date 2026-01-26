"""
Unified LSPIV Processing Pipeline.

Encapsulates the entire LSPIV workflow into a single class for easy integration
with web applications and other interfaces.
"""

import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_io.video_loader import read_first_frame, frame_generator
from image_processing.transforms import get_perspective_transform
from image_processing.frame_warper import FrameWarper
from piv.piv_core import compute_piv, streaming_piv
from piv.scaling import meters_per_pixel_from_width, scale_velocity
from visualization.plotter import (
    extract_streamlines,
    overlay_streamlines_vector,
    create_colorbar_image,
    overlay_colorbar,
    annotate_colorbar
)


@dataclass
class LSPIVResults:
    """Container for LSPIV processing results."""

    # Velocity fields
    u_mean: np.ndarray = None  # Time-averaged u velocity (m/s)
    v_mean: np.ndarray = None  # Time-averaged v velocity (m/s)
    speed: np.ndarray = None   # Velocity magnitude (m/s)

    # Raw data
    velocity_fields: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    # Statistics
    mean_velocity: float = 0.0
    max_velocity: float = 0.0
    min_velocity: float = 0.0
    valid_vector_fraction: float = 0.0
    dominant_direction: float = 0.0  # degrees

    # Processing info
    frames_processed: int = 0
    processing_time: float = 0.0
    meters_per_pixel: float = 0.0

    # Visualization data
    first_frame: np.ndarray = None
    roi_frame: np.ndarray = None
    streamlines: List = field(default_factory=list)
    transform_matrix: np.ndarray = None
    roi_shape: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'mean_velocity': float(self.mean_velocity),
            'max_velocity': float(self.max_velocity),
            'min_velocity': float(self.min_velocity),
            'valid_vector_fraction': float(self.valid_vector_fraction),
            'dominant_direction': float(self.dominant_direction),
            'frames_processed': self.frames_processed,
            'processing_time': float(self.processing_time),
            'meters_per_pixel': float(self.meters_per_pixel)
        }


class LSPIVPipeline:
    """
    Unified LSPIV processing pipeline.

    Encapsulates video loading, ROI selection, PIV computation, and visualization
    into a single easy-to-use interface.
    """

    def __init__(self):
        self.video_path: Optional[str] = None
        self.first_frame: Optional[np.ndarray] = None
        self.roi_points: Optional[np.ndarray] = None
        self.transform_matrix: Optional[np.ndarray] = None
        self.warper: Optional[FrameWarper] = None
        self.warp_size: Optional[Tuple[int, int]] = None

        # Parameters with defaults
        self.fps: float = 30.0
        self.roi_width_meters: float = 10.0
        self.win_size: int = 32
        self.overlap: int = 16
        self.max_frames: int = 50
        self.streamline_density: float = 1.5

        # Processing state
        self._is_loaded = False
        self._roi_set = False

    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load video and extract first frame.

        Args:
            video_path: Path to the video file

        Returns:
            np.ndarray: First frame (BGR)

        Raises:
            RuntimeError: If video cannot be read
        """
        self.video_path = video_path
        self.first_frame = read_first_frame(video_path)

        if self.first_frame is None:
            raise RuntimeError(f"Failed to read video: {video_path}")

        self._is_loaded = True
        return self.first_frame

    def set_roi(self, points: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Set ROI from 4 corner points.

        Args:
            points: numpy array of shape (4, 2) with ROI corner coordinates
                    Points should be in order: top-left, top-right, bottom-right, bottom-left

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Transform matrix and warped size

        Raises:
            ValueError: If points are invalid
            RuntimeError: If video not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Video must be loaded before setting ROI")

        points = np.array(points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("ROI must have exactly 4 points of shape (4, 2)")

        self.roi_points = points
        self.transform_matrix, self.warp_size = get_perspective_transform(points)
        self.warper = FrameWarper(self.transform_matrix, self.warp_size)

        self._roi_set = True
        return self.transform_matrix, self.warp_size

    def set_parameters(
        self,
        fps: Optional[float] = None,
        roi_width_meters: Optional[float] = None,
        win_size: Optional[int] = None,
        overlap: Optional[int] = None,
        max_frames: Optional[int] = None,
        streamline_density: Optional[float] = None
    ):
        """
        Set processing parameters.

        Args:
            fps: Video frame rate
            roi_width_meters: Physical width of ROI in meters
            win_size: PIV interrogation window size (pixels)
            overlap: Window overlap (pixels)
            max_frames: Maximum frames to process
            streamline_density: Density for streamline visualization
        """
        if fps is not None:
            self.fps = fps
        if roi_width_meters is not None:
            self.roi_width_meters = roi_width_meters
        if win_size is not None:
            self.win_size = win_size
        if overlap is not None:
            self.overlap = overlap
        if max_frames is not None:
            self.max_frames = max_frames
        if streamline_density is not None:
            self.streamline_density = streamline_density

    def process(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        use_streaming: bool = True
    ) -> LSPIVResults:
        """
        Run the full LSPIV processing pipeline.

        Args:
            progress_callback: Optional callback(current, total, status_text)
            use_streaming: Use memory-efficient streaming mode

        Returns:
            LSPIVResults: Processing results with velocity fields and statistics

        Raises:
            RuntimeError: If video or ROI not set
        """
        if not self._is_loaded:
            raise RuntimeError("Video must be loaded before processing")
        if not self._roi_set:
            raise RuntimeError("ROI must be set before processing")

        start_time = time.time()
        results = LSPIVResults()
        results.first_frame = self.first_frame.copy()
        results.transform_matrix = self.transform_matrix
        results.meters_per_pixel = self.roi_width_meters / self.warp_size[0]

        dt = 1.0 / self.fps

        def update_progress(current, total):
            if progress_callback:
                progress_callback(current, total, f"Processing frame pair {current}/{total}")

        if use_streaming:
            # Memory-efficient streaming processing
            velocity_fields_px = []
            gen = frame_generator(self.video_path)

            for u_px, v_px in streaming_piv(
                gen,
                self.warper,
                win_size=self.win_size,
                overlap=self.overlap,
                max_frames=self.max_frames,
                progress_callback=update_progress,
                verbose=False
            ):
                # Scale to m/s
                u, v = scale_velocity(u_px, v_px, results.meters_per_pixel, dt)
                results.velocity_fields.append((u, v))
        else:
            # Load all frames first (higher memory usage)
            if progress_callback:
                progress_callback(0, self.max_frames, "Loading frames...")

            roi_frames = []
            for i, frame in enumerate(frame_generator(self.video_path)):
                roi_frames.append(self.warper.warp(frame))
                if self.max_frames and len(roi_frames) >= self.max_frames:
                    break

            # Process frame pairs
            for i in range(len(roi_frames) - 1):
                update_progress(i + 1, len(roi_frames) - 1)

                u_px, v_px = compute_piv(
                    roi_frames[i],
                    roi_frames[i + 1],
                    win_size=self.win_size,
                    overlap=self.overlap,
                    dt=1
                )

                u, v = scale_velocity(u_px, v_px, results.meters_per_pixel, dt)
                results.velocity_fields.append((u, v))

        results.frames_processed = len(results.velocity_fields)

        if results.frames_processed == 0:
            results.processing_time = time.time() - start_time
            return results

        # Time averaging
        if progress_callback:
            progress_callback(
                results.frames_processed,
                results.frames_processed,
                "Computing time-averaged velocity..."
            )

        u_stack = np.array([uv[0] for uv in results.velocity_fields])
        v_stack = np.array([uv[1] for uv in results.velocity_fields])

        results.u_mean = np.nanmean(u_stack, axis=0)
        results.v_mean = np.nanmean(v_stack, axis=0)
        results.speed = np.sqrt(results.u_mean**2 + results.v_mean**2)

        # Get a sample ROI frame for visualization
        gen = frame_generator(self.video_path)
        first_warped = self.warper.warp(next(gen))
        results.roi_frame = first_warped
        results.roi_shape = first_warped.shape[:2]

        # Compute statistics
        results.mean_velocity = float(np.nanmean(results.speed))
        results.max_velocity = float(np.nanmax(results.speed))
        results.min_velocity = float(np.nanmin(results.speed))

        # Valid vector fraction
        valid_per_field = np.sum(~np.isnan(u_stack), axis=0)
        results.valid_vector_fraction = float(
            np.nanmean(valid_per_field) / u_stack.shape[0]
        )

        # Dominant flow direction (angle in degrees)
        mean_u = np.nanmean(results.u_mean)
        mean_v = np.nanmean(results.v_mean)
        results.dominant_direction = float(np.degrees(np.arctan2(mean_v, mean_u)))

        # Extract streamlines for visualization
        if progress_callback:
            progress_callback(
                results.frames_processed,
                results.frames_processed,
                "Extracting streamlines..."
            )

        results.streamlines = extract_streamlines(
            results.u_mean,
            results.v_mean,
            density=self.streamline_density
        )

        results.processing_time = time.time() - start_time

        return results

    def render_streamlines_overlay(
        self,
        results: LSPIVResults,
        thickness: int = 1,
        arrows_per_line: int = 3
    ) -> np.ndarray:
        """
        Render streamlines overlaid on the original frame.

        Args:
            results: LSPIVResults from process()
            thickness: Line thickness
            arrows_per_line: Number of arrow markers per streamline

        Returns:
            np.ndarray: BGR image with streamlines overlay
        """
        if results.first_frame is None:
            raise ValueError("No first frame in results")

        vmin = results.min_velocity
        vmax = results.max_velocity

        overlay = overlay_streamlines_vector(
            results.first_frame.copy(),
            results.streamlines,
            results.transform_matrix,
            roi_shape=results.roi_shape,
            piv_shape=results.u_mean.shape,
            speed_range=(vmin, vmax),
            thickness=thickness,
            arrows_per_line=arrows_per_line
        )

        # Add colorbar
        colorbar = create_colorbar_image(
            height=300,
            width=40,
            vmin=vmin,
            vmax=vmax,
            cmap=cv2.COLORMAP_JET
        )

        overlay, cb_x, cb_y, cb_h = overlay_colorbar(
            overlay,
            colorbar,
            margin=100
        )

        annotate_colorbar(
            overlay,
            cb_x + colorbar.shape[1],
            cb_y,
            cb_h,
            vmin,
            vmax
        )

        return overlay

    def render_speed_heatmap(self, results: LSPIVResults) -> np.ndarray:
        """
        Render velocity magnitude heatmap.

        Args:
            results: LSPIVResults from process()

        Returns:
            np.ndarray: BGR heatmap image
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        speed = results.speed

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(speed, cmap='jet')
        plt.colorbar(im, ax=ax, label='Speed (m/s)')
        ax.set_title('Velocity Magnitude (m/s)')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        ax.invert_yaxis()
        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        # RGBA to BGR
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    def render_quiver(self, results: LSPIVResults, step: int = 3) -> np.ndarray:
        """
        Render velocity vectors as quiver plot.

        Args:
            results: LSPIVResults from process()
            step: Subsampling step for vectors

        Returns:
            np.ndarray: BGR quiver plot image
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        u, v = results.u_mean, results.v_mean
        speed = results.speed
        mean_speed = np.nanmean(speed)
        scale = 1 / mean_speed if mean_speed > 0 else 1

        y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.quiver(
            x[::step, ::step], y[::step, ::step],
            u[::step, ::step], v[::step, ::step],
            scale=scale
        )
        ax.invert_yaxis()
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        ax.set_title('Velocity Field (Quiver)')
        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    def render_streamlines_plot(self, results: LSPIVResults) -> np.ndarray:
        """
        Render matplotlib streamlines plot.

        Args:
            results: LSPIVResults from process()

        Returns:
            np.ndarray: BGR streamlines plot image
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        u, v = results.u_mean, results.v_mean
        speed = results.speed

        y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]

        fig, ax = plt.subplots(figsize=(10, 6))
        strm = ax.streamplot(
            x, y, u, v,
            color=speed, cmap='jet',
            density=self.streamline_density, arrowsize=1.5
        )
        plt.colorbar(strm.lines, ax=ax, label='Speed (m/s)')
        ax.invert_yaxis()
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        ax.set_title('Streamlines')
        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
