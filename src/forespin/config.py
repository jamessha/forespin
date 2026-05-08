from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Thresholds:
    min_ball_confidence: float = 0.18
    min_player_confidence: float = 0.18
    min_court_confidence: float = 0.35
    hit_min_speed_px_s: float = 140.0
    hit_min_acceleration_px_s2: float = 320.0
    hit_reversal_dot_threshold: float = 0.35
    tracked_contact_distance_court: float = 0.17
    bounce_angle_change_deg: float = 16.0
    bounce_speed_drop_ratio: float = 0.82
    bounce_suppress_frames_after_hit: int = 10
    bounce_floor_window_frames: int = 12
    bounce_min_vertical_prominence_px: float = 6.0
    bounce_court_margin: float = 0.25
    bounce_min_separation_frames: int = 14
    bounce_hit_overlap_min_vertical_prominence_px: float = 40.0
    bounce_hit_overlap_max_speed_ratio: float = 2.0
    fallback_bounce_window_frames: int = 10
    hit_suppress_window_frames: int = 5
    dead_ball_gap_frames: int = 75
    baseline_zone_margin: float = 0.18
    net_zone_margin: float = 0.10
    service_line_near_y: float = 0.769
    service_line_far_y: float = 0.231
    refresh_court_every_n_frames: int = 45
    court_calibration_sample_frames: int = 5
    smoothing_window: int = 3
    interpolate_ball_gaps_up_to_frames: int = 5
    ball_track_segment_gap_frames: int = 30
    ball_outlier_max_speed_px_s: float = 3500.0
    ball_outlier_max_acceleration_px_s2: float = 90000.0
    ball_outlier_run_max_frames: int = 1
    ball_static_segment_min_frames: int = 6
    ball_static_segment_max_displacement_px: float = 25.0
    ball_out_of_play_court_margin: float = 0.35
    ball_out_of_play_confirm_frames: int = 4
    min_ball_visibility_ratio: float = 0.20
    min_player_visibility_ratio: float = 0.25
    min_average_court_confidence: float = 0.40


@dataclass(slots=True)
class AnalysisOptions:
    render_overlay: bool = True
    reject_low_quality: bool = True
    persist_json: bool = True
    persist_overlay: bool = True
    remove_net_for_court_calibration: bool = False
    net_removal_model: str = "gpt-image-2"
    use_court_calibration_cache: bool = True
    use_tracking_trace_cache: bool = True
    thresholds: Thresholds = field(default_factory=Thresholds)
