from std.testing import assert_equal, assert_true, TestSuite

from case_model import native_case_from_exec_path
from energy_layers import build_energy_layers
from field_geometry import Bounds3f
from sis import read_sis_table


def assert_close(actual: Float64, expected: Float64, tol: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tol)


def test_p101_trip_setup_energy_layers() raises:
    var nc = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var sis = read_sis_table(nc.sis_path)

    var f1 = nc.fields[0].copy()
    var w1 = Bounds3f(-30.0, 30.0, -28.184192540053573, 25.471298957473522, 80.80056500358766, 153.94568270293192)
    var layers1 = build_energy_layers(f1, w1, sis)
    assert_equal(len(layers1), 25)
    assert_close(layers1[0].energy_mev_u, 197.58, 1.0e-9)
    assert_close(layers1[len(layers1) - 1].energy_mev_u, 282.67, 1.0e-9)

    var f2 = nc.fields[1].copy()
    var w2 = Bounds3f(-30.0, 30.0, -31.939987361033843, 24.177214182140233, 103.6683430429139, 172.61631250336094)
    var layers2 = build_energy_layers(f2, w2, sis)
    assert_equal(len(layers2), 23)
    assert_close(layers2[0].energy_mev_u, 228.53, 1.0e-9)
    assert_close(layers2[len(layers2) - 1].energy_mev_u, 303.14, 1.0e-9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
