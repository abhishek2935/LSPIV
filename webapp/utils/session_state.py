"""
Streamlit session state management utilities.

Provides centralized management of application state across Streamlit reruns.
"""

import streamlit as st
from typing import Any, Optional


# Default state values
DEFAULT_STATE = {
    # Video state
    'video_path': None,
    'video_uploaded': False,
    'first_frame': None,

    # ROI state
    'roi_points': None,
    'roi_set': False,

    # Parameters
    'fps': 30.0,
    'roi_width_meters': 10.0,
    'win_size': 32,
    'overlap': 16,
    'max_frames': 50,
    'streamline_density': 1.5,

    # Processing state
    'processing': False,
    'processed': False,
    'results': None,

    # Pipeline instance
    'pipeline': None,
}


def init_session_state():
    """Initialize session state with default values if not already set."""
    for key, default_value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default: Any = None) -> Any:
    """
    Get a value from session state.

    Args:
        key: State key to retrieve
        default: Default value if key doesn't exist

    Returns:
        State value or default
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """
    Set a value in session state.

    Args:
        key: State key to set
        value: Value to store
    """
    st.session_state[key] = value


def reset_state(keys: Optional[list] = None):
    """
    Reset session state values to defaults.

    Args:
        keys: List of specific keys to reset. If None, resets all state.
    """
    if keys is None:
        keys = list(DEFAULT_STATE.keys())

    for key in keys:
        if key in DEFAULT_STATE:
            st.session_state[key] = DEFAULT_STATE[key]


def reset_processing_state():
    """Reset only processing-related state (keeps video and ROI)."""
    reset_state(['processing', 'processed', 'results'])


def reset_roi_state():
    """Reset ROI and processing state (keeps video)."""
    reset_state(['roi_points', 'roi_set', 'processing', 'processed', 'results'])


def reset_all():
    """Reset all session state to defaults."""
    reset_state()
