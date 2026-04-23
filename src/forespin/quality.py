from __future__ import annotations

from forespin.config import Thresholds
from forespin.domain import FrameObservation, QualityAssessment
from forespin.geometry import safe_rate


def assess_quality(observations: list[FrameObservation], thresholds: Thresholds) -> QualityAssessment:
    if not observations:
        return QualityAssessment(
            accepted=False,
            warnings=["No frames were processed."],
            ball_visibility_ratio=0.0,
            player_visibility_ratio=0.0,
            average_court_confidence=0.0,
        )

    ball_visible = sum(1 for observation in observations if observation.ball_px is not None and observation.ball_confidence >= thresholds.min_ball_confidence)
    player_visible = sum(
        1
        for observation in observations
        if observation.tracked_player_feet_px is not None and observation.tracked_player_confidence >= thresholds.min_player_confidence
    )
    average_court_confidence = sum(observation.court_confidence for observation in observations) / len(observations)
    ball_ratio = safe_rate(ball_visible, len(observations))
    player_ratio = safe_rate(player_visible, len(observations))

    warnings: list[str] = []
    accepted = True
    if ball_ratio < thresholds.min_ball_visibility_ratio:
        warnings.append("Ball visibility is too low for reliable shot and bounce detection.")
        accepted = False
    if player_ratio < thresholds.min_player_visibility_ratio:
        warnings.append("Tracked player visibility is too low for reliable position metrics.")
        accepted = False
    if average_court_confidence < thresholds.min_average_court_confidence:
        warnings.append("Court calibration confidence is too low for a trustworthy homography.")
        accepted = False

    return QualityAssessment(
        accepted=accepted,
        warnings=warnings,
        ball_visibility_ratio=ball_ratio,
        player_visibility_ratio=player_ratio,
        average_court_confidence=average_court_confidence,
    )

