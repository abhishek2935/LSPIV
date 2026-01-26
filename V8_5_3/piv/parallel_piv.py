"""
Parallel PIV processing module using ThreadPoolExecutor.

Provides parallel frame pair processing for improved performance on multi-core systems.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Callable, Optional
from .piv_core import compute_piv


def process_frame_pair(
    args: Tuple[int, np.ndarray, np.ndarray, int, int]
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Process a single frame pair for PIV.

    Args:
        args: Tuple of (index, frame_a, frame_b, win_size, overlap)

    Returns:
        Tuple[int, np.ndarray, np.ndarray]: (index, u, v)
    """
    idx, frame_a, frame_b, win_size, overlap = args
    u, v = compute_piv(frame_a, frame_b, win_size=win_size, overlap=overlap, dt=1)
    return idx, u, v


def parallel_piv(
    frames: List[np.ndarray],
    win_size: int = 32,
    overlap: int = 16,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Process multiple frame pairs in parallel using ThreadPoolExecutor.

    Args:
        frames: List of warped frames to process
        win_size: PIV interrogation window size
        overlap: Window overlap in pixels
        max_workers: Maximum number of worker threads (None for default)
        progress_callback: Optional callback(completed, total) for progress updates

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: List of (u, v) velocity fields in order
    """
    n_pairs = len(frames) - 1
    if n_pairs <= 0:
        return []

    # Prepare arguments for each frame pair
    args_list = [
        (i, frames[i], frames[i + 1], win_size, overlap)
        for i in range(n_pairs)
    ]

    # Results storage (indexed for proper ordering)
    results = [None] * n_pairs
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_frame_pair, args): args[0]
            for args in args_list
        }

        # Collect results as they complete
        for future in as_completed(futures):
            idx, u, v = future.result()
            results[idx] = (u, v)
            completed += 1

            if progress_callback:
                progress_callback(completed, n_pairs)

    return results


def batch_parallel_piv(
    frame_generator,
    warper,
    win_size: int = 32,
    overlap: int = 16,
    batch_size: int = 10,
    max_frames: Optional[int] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Process frames in batches with parallel PIV computation.

    Balances memory usage with parallelization by loading frames in batches.

    Args:
        frame_generator: Generator yielding video frames
        warper: FrameWarper instance for perspective transformation
        win_size: PIV interrogation window size
        overlap: Window overlap in pixels
        batch_size: Number of frames to load per batch
        max_frames: Maximum total frames to process (None for all)
        max_workers: Maximum number of worker threads (None for default)
        progress_callback: Optional callback(completed, total) for progress updates

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: All (u, v) velocity fields in order
    """
    all_results = []
    batch = []
    frame_count = 0
    total_pairs_processed = 0

    # Estimate total pairs for progress reporting
    estimated_total = max_frames - 1 if max_frames else None

    for frame in frame_generator:
        warped = warper.warp(frame)
        batch.append(warped)
        frame_count += 1

        # Process batch when full
        if len(batch) >= batch_size + 1:  # +1 for overlap between batches
            # Process this batch
            batch_results = parallel_piv(
                batch,
                win_size=win_size,
                overlap=overlap,
                max_workers=max_workers
            )

            all_results.extend(batch_results)
            total_pairs_processed += len(batch_results)

            if progress_callback and estimated_total:
                progress_callback(total_pairs_processed, estimated_total)

            # Keep last frame for next batch overlap
            batch = [batch[-1]]

        # Check max frames limit
        if max_frames and frame_count >= max_frames:
            break

    # Process remaining frames in batch
    if len(batch) > 1:
        batch_results = parallel_piv(
            batch,
            win_size=win_size,
            overlap=overlap,
            max_workers=max_workers
        )
        all_results.extend(batch_results)
        total_pairs_processed += len(batch_results)

        if progress_callback:
            progress_callback(total_pairs_processed, total_pairs_processed)

    return all_results
