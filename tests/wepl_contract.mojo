from std.testing import assert_equal

from p101_io import load_approx_water_depths, load_tumor_world_points


def main() raises:
    var points = load_tumor_world_points()
    var depths_280 = load_approx_water_depths(280.0)
    var depths_325 = load_approx_water_depths(325.0)

    assert_equal(len(depths_280), len(points))
    assert_equal(len(depths_325), len(points))
