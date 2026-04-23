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
    bounce_suppress_frames_after_hit: int = 4
    fallback_bounce_window_frames: int = 10
    hit_suppress_window_frames: int = 5
    dead_ball_gap_frames: int = 75
    baseline_zone_margin: float = 0.18
    net_zone_margin: float = 0.10
    service_line_near_y: float = 0.769
    service_line_far_y: float = 0.231
    refresh_court_every_n_frames: int = 45
    smoothing_window: int = 3
    interpolate_ball_gaps_up_to_frames: int = 5
    min_ball_visibility_ratio: float = 0.20
    min_player_visibility_ratio: float = 0.25
    min_average_court_confidence: float = 0.40


@dataclass(slots=True)
class AnalysisOptions:
    render_overlay: bool = True
    reject_low_quality: bool = True
    persist_json: bool = True
    persist_overlay: bool = True
    thresholds: Thresholds = field(default_factory=Thresholds)

