from std.testing import assert_equal, assert_true

from geometry import Vec3
from p101_io import load_tumor_world_points, load_water_depths_for_points


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var points = load_tumor_world_points()
    var depths = load_water_depths_for_points(280.0, points)
    assert_equal(len(depths), len(points))

    # Native lookup from the TRiP-generated P101 WET cube.
    var debug_points = List[Vec3]()
    debug_points.append(Vec3(210.4573, 333.5089, 136.5))
    var debug_depths = load_water_depths_for_points(280.0, debug_points)
    assert_equal(len(debug_depths), 1)
    assert_close(debug_depths[0], 114.71653747558594, 1.0e-6)

    debug_points = List[Vec3]()
    debug_points.append(Vec3(229.9893, 337.4153, 187.5))
    debug_depths = load_water_depths_for_points(325.0, debug_points)
    assert_equal(len(debug_depths), 1)
    assert_close(debug_depths[0], 134.62889099121094, 1.0e-6)
