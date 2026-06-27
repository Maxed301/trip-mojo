from std.testing import assert_equal, assert_true

from case_model import native_case_from_exec_path


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")

    assert_equal(native_case.ct_path, "/home/max/Projects/TRIP_DATA/P101/CT/P101_00.nrrd")
    assert_equal(native_case.voi_binfo_path, "/home/max/Projects/TRIP_DATA/P101/VOI/3D/P101.binfo")
    assert_equal(native_case.hlut_path, "/home/max/Projects/TRIP_DATA/Basedata/GSI/19990211.hlut")
    assert_equal(native_case.ddd_path_pattern, "/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/DDD/*ddd")
    assert_equal(native_case.spc_path_pattern, "/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/SPC/*spc")
    assert_equal(native_case.dedx_path, "/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/20040607.dedx")

    assert_equal(native_case.target_tissue, "chordom02")
    assert_equal(native_case.residual_tissue, "hirn02")
    assert_equal(native_case.prescription_dose_gy, 3.0)
    assert_equal(native_case.scancap_bolus_mm_h2o, 40.0)
    assert_equal(native_case.off_h2o_mm, 1.709)
    assert_equal(native_case.scancap_x_limit_mm, 90.0)
    assert_equal(native_case.scancap_y_limit_mm, 90.0)
    assert_equal(native_case.scancap_scanner_x_mm, 8832.0)
    assert_equal(native_case.scancap_scanner_y_mm, 7806.0)
    assert_equal(native_case.scancap_min_particles, 5000.0)
    assert_equal(native_case.target_voi_name, "Tumor")
    assert_close(native_case.target_isocenter.x, 220.2231, 1.0e-12)
    assert_close(native_case.target_isocenter.y, 317.8835, 1.0e-12)
    assert_close(native_case.target_isocenter.z, 162.0, 1.0e-12)

    assert_equal(len(native_case.fields), 2)
    assert_equal(native_case.fields[0].number, 1)
    assert_equal(native_case.fields[0].gantry_degrees, 280.0)
    assert_equal(native_case.fields[0].couch_degrees, 90.0)
    assert_equal(native_case.fields[0].raster_x_mm, 2.0)
    assert_equal(native_case.fields[0].raster_y_mm, 2.0)
    assert_equal(native_case.fields[0].contour_extension, 1.2)
    assert_equal(native_case.fields[0].z_step_mm, 3.0)
    assert_equal(native_case.fields[0].distal_extension_mm, 3.0)
    assert_equal(native_case.fields[0].robust_range_mm_h2o, 3.5)
    assert_equal(native_case.fields[0].robust_position_mm, 3.0)
    assert_equal(native_case.fields[0].target_voi, "Tumor")

    assert_equal(native_case.fields[1].number, 2)
    assert_equal(native_case.fields[1].gantry_degrees, 325.0)
    assert_equal(native_case.fields[1].target_voi, "Tumor")

    assert_equal(native_case.optimization.enabled, True)
    assert_equal(native_case.optimization.biological, True)
    assert_equal(native_case.optimization.optalg, "fdcb")
    assert_equal(native_case.optimization.dosealg, "msdb")
