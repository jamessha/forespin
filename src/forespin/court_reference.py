from __future__ import annotations

from forespin.domain import Point2D


class CourtReference:
    """Reference singles court geometry used by the learned calibrator."""

    def __init__(self) -> None:
        self.baseline_top = ((286.0, 561.0), (1379.0, 561.0))
        self.baseline_bottom = ((286.0, 2935.0), (1379.0, 2935.0))
        self.net = ((286.0, 1748.0), (1379.0, 1748.0))
        self.left_court_line = ((286.0, 561.0), (286.0, 2935.0))
        self.right_court_line = ((1379.0, 561.0), (1379.0, 2935.0))
        self.left_inner_line = ((423.0, 561.0), (423.0, 2935.0))
        self.right_inner_line = ((1242.0, 561.0), (1242.0, 2935.0))
        self.middle_line = ((832.0, 1110.0), (832.0, 2386.0))
        self.top_inner_line = ((423.0, 1110.0), (1242.0, 1110.0))
        self.bottom_inner_line = ((423.0, 2386.0), (1242.0, 2386.0))
        self.top_extra_part = (832.5, 580.0)
        self.bottom_extra_part = (832.5, 2910.0)

        self.key_points: list[tuple[float, float]] = [
            *self.baseline_top,
            *self.baseline_bottom,
            *self.left_inner_line,
            *self.right_inner_line,
            *self.top_inner_line,
            *self.bottom_inner_line,
            *self.middle_line,
        ]
        self.border_points: list[tuple[float, float]] = [
            *self.baseline_top,
            *self.baseline_bottom[::-1],
        ]
        self.court_configurations: dict[int, list[tuple[float, float]]] = {
            1: [*self.baseline_top, *self.baseline_bottom],
            2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1], self.right_inner_line[1]],
            3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1], self.right_court_line[1]],
            4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1], self.right_inner_line[1]],
            5: [*self.top_inner_line, *self.bottom_inner_line],
            6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
            7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
            8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1], self.right_court_line[1]],
            9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1], self.left_inner_line[1]],
            10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0], self.middle_line[1]],
            11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1], self.bottom_inner_line[1]],
            12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
        }

    @property
    def left_x(self) -> float:
        return self.left_court_line[0][0]

    @property
    def right_x(self) -> float:
        return self.right_court_line[0][0]

    @property
    def top_y(self) -> float:
        return self.baseline_top[0][1]

    @property
    def bottom_y(self) -> float:
        return self.baseline_bottom[0][1]

    @property
    def width(self) -> float:
        return self.right_x - self.left_x

    @property
    def height(self) -> float:
        return self.bottom_y - self.top_y

    def normalized_point(self, point: tuple[float, float]) -> Point2D:
        return Point2D(
            x=(point[0] - self.left_x) / self.width,
            y=(point[1] - self.top_y) / self.height,
        )

    def normalized_key_points(self) -> list[Point2D]:
        return [self.normalized_point(point) for point in self.key_points]

    def normalized_border_points(self) -> list[Point2D]:
        return [self.normalized_point(point) for point in self.border_points]

    def configuration_indices(self) -> list[tuple[int, int, int, int]]:
        return [
            tuple(self.key_points.index(point) for point in configuration)
            for configuration in self.court_configurations.values()
        ]
