from std.testing import assert_equal, assert_true

from geometry import Vec3
from phys_dose import physical_field_replay
from rst import RSTPlan, RasterSpot


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var plan = RSTPlan("reference/p101_simple_onefield_20260511_142537/simple_field1.rst")
    var spots = List[RasterSpot]()
    for i in range(len(plan.field)):
        if plan.field[i].energy == 230.98 and plan.field[i].x == 16.0 and plan.field[i].y == 8.0:
            spots.append(plan.field[i].copy())
            break

    assert_equal(len(spots), 1)
    spots[0].f2_max = 70.56

    var points = List[Vec3]()
    points.append(Vec3(227.05949798226357, 313.97689720988274, 154.5))
    var h2o = List[Float64]()
    h2o.append(106.25414616218718)

    var dose = physical_field_replay(points, spots, h2o, 280.0)
    assert_close(dose[0], 0.0, 0.0)
