from std.testing import assert_true

from case_model import load_native_case, native_case_from_exec_path, trip_target_window_center
from geometry import Vec3, patient_to_gantry_matrix, transform_point


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var loaded = load_native_case(native_case)
    var isocenter = trip_target_window_center(loaded.target)
    assert_close(isocenter.x, 220.22309979605674, 0.0)
    assert_close(isocenter.y, 317.883499761343, 0.0)
    assert_close(isocenter.z, 162.0, 0.0)

    var point = Vec3(209.48069813847542, 329.60249707102776, 127.5)
    var matrix = patient_to_gantry_matrix(isocenter, native_case.fields[0].gantry_degrees, native_case.fields[0].couch_degrees)
    var gantry = transform_point(matrix, point)
    assert_close(gantry.x, 34.5, 0.0)
    assert_close(gantry.y, -12.614182965267673, 2.0e-14)
    assert_close(gantry.z, -9.6755609365015403, 6.0e-14)
