from std.testing import assert_equal, assert_true

from geometry import Mat4, Vec3, patient_to_gantry_matrix, transform_point


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var translate = Mat4(
        1.0, 0.0, 0.0, 10.0,
        0.0, 1.0, 0.0, 20.0,
        0.0, 0.0, 1.0, 30.0,
        0.0, 0.0, 0.0, 1.0,
    )
    var scale = Mat4(
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    var combined = translate * scale
    var point = transform_point(combined, Vec3(1.0, 2.0, 3.0))

    assert_equal(point.x, 12.0)
    assert_equal(point.y, 26.0)
    assert_equal(point.z, 42.0)

    var p101 = patient_to_gantry_matrix(Vec3(220.223, 317.883, 162.0), 280.0, 90.0)
    assert_close(p101[0, 0], 0.0, 1e-6)
    assert_close(p101[0, 1], 0.0, 1e-6)
    assert_close(p101[0, 2], -1.0, 1e-6)
    assert_close(p101[0, 3], 162.0, 1e-6)
    assert_close(p101[1, 0], 0.984807753, 1e-6)
    assert_close(p101[1, 1], -0.173648178, 1e-6)
    assert_close(p101[1, 2], 0.0, 1e-6)
    assert_close(p101[1, 3], -161.677656, 5e-4)
    assert_close(p101[2, 0], -0.173648178, 1e-6)
    assert_close(p101[2, 1], -0.984807753, 1e-6)
    assert_close(p101[2, 2], 0.0, 1e-6)
    assert_close(p101[2, 3], 351.295219, 5e-4)
