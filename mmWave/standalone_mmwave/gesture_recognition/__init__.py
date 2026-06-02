"""
Gesture recognition, range peak picking, and Max OSC publishing.

Live and replay both use :mod:`gesture_recognition.publisher` and
:mod:`gesture_recognition.status` so terminal output and OSC addresses match.
"""

from gesture_recognition.gesture import (
    GESTURE_NONE,
    GESTURE_PULL,
    GESTURE_PUSH,
    GESTURE_PUSH_PULL,
    GESTURE_SINGLE,
    classify_gesture_from_rd,
    classify_gesture_from_velocity,
)
from gesture_recognition.osc import OscAddresses, OscPublisher, OscSender
from gesture_recognition.peaks import (
    RangePeakDetectionConfig,
    detect_peaks_volume,
    detect_range_peaks_frame,
    osc_targets_from_profile,
    peaks_from_profile,
)
from gesture_recognition.mode import (
    active_gesture_mode,
    describe_gesture_mode,
    make_gesture_processor,
    make_osc_gesture_limiter,
    simulate_gesture_volume,
)
from gesture_recognition.publisher import publish_radar_frame
from gesture_recognition.rd_peak import (
    RdKalmanConfig,
    RdKalmanProcessor,
    RdPeakConfig,
    RdPeakProcessor,
    RdPeakTarget,
)
from gesture_recognition.simple import SimpleGestureConfig, SimpleGestureProcessor
from gesture_recognition.status import StreamFrameResult, format_status_line
from gesture_recognition.tracker import (
    RangePeakTracker,
    TrackedTarget,
    TrackingConfig,
    TrackingVolume,
    gesture_from_tracks,
    simulate_tracking_volume,
    velocity_from_tracks,
)

__all__ = [
    "GESTURE_NONE",
    "GESTURE_PULL",
    "GESTURE_PUSH",
    "GESTURE_PUSH_PULL",
    "GESTURE_SINGLE",
    "OscAddresses",
    "OscPublisher",
    "OscSender",
    "RangePeakDetectionConfig",
    "RangePeakTracker",
    "StreamFrameResult",
    "TrackedTarget",
    "TrackingConfig",
    "TrackingVolume",
    "classify_gesture_from_rd",
    "classify_gesture_from_velocity",
    "detect_peaks_volume",
    "detect_range_peaks_frame",
    "format_status_line",
    "gesture_from_tracks",
    "osc_targets_from_profile",
    "peaks_from_profile",
    "RdPeakConfig",
    "RdPeakProcessor",
    "RdPeakTarget",
    "RdKalmanConfig",
    "RdKalmanProcessor",
    "SimpleGestureConfig",
    "SimpleGestureProcessor",
    "active_gesture_mode",
    "describe_gesture_mode",
    "make_gesture_processor",
    "make_osc_gesture_limiter",
    "simulate_gesture_volume",
    "publish_radar_frame",
    "velocity_from_tracks",
]
