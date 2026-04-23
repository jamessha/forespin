from __future__ import annotations

from pathlib import Path

from forespin.config import AnalysisOptions
from forespin.domain import AnalysisArtifacts, AnalysisResult, AnalysisStatus, InputConfig
from forespin.events import annotate_hit_outcomes, annotate_shot_types, detect_bounces, detect_hits
from forespin.metrics import compute_metrics
from forespin.overlay import render_overlay_video
from forespin.points import assemble_points
from forespin.quality import assess_quality
from forespin.serialization import write_json
from forespin.tracking import build_observations


class TennisAnalyzer:
    def __init__(self, options: AnalysisOptions | None = None) -> None:
        self.options = options or AnalysisOptions()

    def analyze(self, input_config: InputConfig) -> AnalysisResult:
        metadata, observations, court = build_observations(input_config, self.options)
        hits = detect_hits(observations, input_config, self.options.thresholds)
        bounces = detect_bounces(observations, hits, input_config, self.options.thresholds)
        hits = annotate_shot_types(hits, observations, bounces, input_config, self.options.thresholds)
        hits = annotate_hit_outcomes(hits, bounces, input_config)
        points = assemble_points(observations, hits, bounces, input_config, self.options.thresholds)
        metrics = compute_metrics(hits, points, input_config, self.options.thresholds)
        quality = assess_quality(observations, self.options.thresholds)
        status = AnalysisStatus.ACCEPTED if quality.accepted else AnalysisStatus.REJECTED

        result = AnalysisResult(
            config=input_config,
            metadata=metadata,
            court=court,
            observations=observations,
            hits=hits,
            bounces=bounces,
            points=points,
            metrics=metrics,
            quality=quality,
            status=status,
            warnings=[*quality.warnings, *metrics.warnings],
            artifacts=AnalysisArtifacts(),
        )
        if self.options.reject_low_quality and not quality.accepted:
            return result
        return result

    def analyze_and_persist(self, input_config: InputConfig) -> AnalysisResult:
        result = self.analyze(input_config)
        output_dir = Path(input_config.output_dir or "outputs") / Path(input_config.video_path).stem
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.options.persist_json:
            result.artifacts.json_path = str(write_json(output_dir / "timeline.json", result))
        if self.options.render_overlay and self.options.persist_overlay and result.status == AnalysisStatus.ACCEPTED:
            result.artifacts.overlay_path = str(render_overlay_video(result, output_dir / "overlay.mp4"))
        return result


def analyze_video(input_config: InputConfig, options: AnalysisOptions | None = None) -> AnalysisResult:
    analyzer = TennisAnalyzer(options=options)
    return analyzer.analyze_and_persist(input_config)
