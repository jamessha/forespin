from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TrackedPlayerSide(StrEnum):
    NEAR = "near"
    FAR = "far"


class Handedness(StrEnum):
    RIGHT = "right"
    LEFT = "left"


class PlayerActor(StrEnum):
    TRACKED = "tracked"
    OPPONENT = "opponent"
    UNKNOWN = "unknown"


class ShotType(StrEnum):
    SERVE = "serve"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    VOLLEY = "volley"
    UNKNOWN = "unknown"


class ShotOutcome(StrEnum):
    IN = "in"
    OUT = "out"
    UNKNOWN = "unknown"


class PointTermination(StrEnum):
    TRACKED_ERROR = "tracked_error"
    OPPONENT_WINNER = "opponent_winner"
    OPPONENT_ERROR = "opponent_error"
    TRACKED_WINNER = "tracked_winner"
    DEAD_BALL = "dead_ball"
    UNKNOWN = "unknown"


class AnalysisStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(slots=True)
class Point2D:
    x: float
    y: float


@dataclass(slots=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point2D:
        return Point2D((self.x1 + self.x2) / 2.0, self.y2)


@dataclass(slots=True)
class InputConfig:
    video_path: str
    tracked_player_side: TrackedPlayerSide
    handedness: Handedness
    output_dir: str | None = None
    tracknet_weights: str | None = None
    player_pose_weights: str | None = None
    court_weights: str | None = None


@dataclass(slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


@dataclass(slots=True)
class CourtCalibration:
    corners_px: list[Point2D]
    homography: list[list[float]]
    confidence: float
    source: str = "auto"


@dataclass(slots=True)
class BallTrack:
    position_px: Point2D | None
    confidence: float


@dataclass(slots=True)
class PlayerTrack:
    bbox_px: BBox | None
    feet_px: Point2D | None
    torso_px: Point2D | None
    confidence: float


@dataclass(slots=True)
class FrameObservation:
    frame_index: int
    timestamp_s: float
    ball_px: Point2D | None = None
    ball_court: Point2D | None = None
    ball_velocity_px_s: Point2D | None = None
    ball_confidence: float = 0.0
    tracked_player_bbox_px: BBox | None = None
    tracked_player_feet_px: Point2D | None = None
    tracked_player_feet_court: Point2D | None = None
    tracked_player_torso_px: Point2D | None = None
    tracked_player_torso_court: Point2D | None = None
    tracked_player_confidence: float = 0.0
    court_confidence: float = 0.0


@dataclass(slots=True)
class BounceEvent:
    frame_index: int
    timestamp_s: float
    ball_court: Point2D | None
    in_bounds: bool
    confidence: float
    inferred_from_fallback: bool = False


@dataclass(slots=True)
class HitEvent:
    frame_index: int
    timestamp_s: float
    actor: PlayerActor
    shot_type: ShotType
    ball_px: Point2D | None
    ball_court: Point2D | None
    player_court: Point2D | None
    confidence: float
    is_serve: bool = False
    is_volley: bool = False
    result: ShotOutcome = ShotOutcome.UNKNOWN
    result_reason: str | None = None


@dataclass(slots=True)
class PointEvent:
    point_index: int
    start_frame: int
    end_frame: int
    hit_events: list[HitEvent] = field(default_factory=list)
    bounce_events: list[BounceEvent] = field(default_factory=list)
    tracked_player_lost: bool = False
    winner: PlayerActor = PlayerActor.UNKNOWN
    terminal_reason: PointTermination = PointTermination.UNKNOWN
    tracked_player_final_location: Point2D | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class ShotMetric:
    attempts: int = 0
    in_count: int = 0
    in_rate: float = 0.0


@dataclass(slots=True)
class MetricBucket:
    count: int = 0
    percentage: float = 0.0


@dataclass(slots=True)
class MetricsReport:
    shot_stats: dict[str, ShotMetric] = field(default_factory=dict)
    lost_point_depth: dict[str, MetricBucket] = field(default_factory=dict)
    lost_point_lateral: dict[str, MetricBucket] = field(default_factory=dict)
    lost_point_count: int = 0
    ignored_tracked_shots: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QualityAssessment:
    accepted: bool
    warnings: list[str]
    ball_visibility_ratio: float
    player_visibility_ratio: float
    average_court_confidence: float


@dataclass(slots=True)
class AnalysisArtifacts:
    json_path: str | None = None
    overlay_path: str | None = None


@dataclass(slots=True)
class AnalysisResult:
    config: InputConfig
    metadata: VideoMetadata
    court: CourtCalibration
    observations: list[FrameObservation]
    hits: list[HitEvent]
    bounces: list[BounceEvent]
    points: list[PointEvent]
    metrics: MetricsReport
    quality: QualityAssessment
    status: AnalysisStatus
    warnings: list[str] = field(default_factory=list)
    artifacts: AnalysisArtifacts = field(default_factory=AnalysisArtifacts)
