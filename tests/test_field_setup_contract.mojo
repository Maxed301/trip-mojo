from std.testing import assert_equal, assert_true

from case_model import load_native_case, native_case_from_exec_path
from ct_ray import ct_grid_intersection_table, ray_water_depth_mm, vintpol1d
from field_geometry import (
    clip_param_axis,
    gantry_to_patient_from_matrix,
    point_inside_target_mask,
    target_voi_bev_binary_wdw_bounds,
    with_distal_extension,
)
from geometry import Vec3
from raster_grid import raster_grid_from_window


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var loaded = load_native_case(native_case)

    var f1 = native_case.fields[0].copy()
    var w1 = target_voi_bev_binary_wdw_bounds(f1, loaded.target, native_case.target_isocenter)
    var setup1 = with_distal_extension(w1, f1)
    var grid1 = raster_grid_from_window(setup1, f1.raster_x_mm, f1.raster_y_mm)
    assert_equal(len(grid1.points), 648)
    assert_close(grid1.points[0].x, -26.0, 1.0e-12)
    assert_close(grid1.points[0].y, -24.0, 1.0e-12)
    assert_close(grid1.points[len(grid1.points) - 1].x, -26.0, 1.0e-12)
    assert_close(grid1.points[len(grid1.points) - 1].y, 22.0, 1.0e-12)

    var p_min = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(10.0, 4.0, 26.5492595922924))
    assert_true(point_inside_target_mask(loaded.target, p_min))
    assert_close(p_min.x, 219.552100261498, 1.0e-6)
    assert_close(p_min.y, 291.042990367452, 1.0e-6)
    assert_close(p_min.z, 152.0, 1.0e-12)

    var origin0 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(0.0, 0.0, 0.0))
    var origin1 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(0.0, 0.0, 1.0))
    var direction = Vec3(origin1.x - origin0.x, origin1.y - origin0.y, origin1.z - origin0.z)
    var h2o_min = ray_water_depth_mm(loaded.ct, loaded.hlut, p_min, direction) + native_case.scancap_bolus_mm_h2o + native_case.off_h2o_mm
    assert_close(h2o_min, 85.3272018863558, 1.0e-4)

    var ray0 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(-26.0, -24.0, 0.0))
    var ray1 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(-26.0, -24.0, 1.0))
    var ray_dir = Vec3(ray0.x - ray1.x, ray0.y - ray1.y, ray0.z - ray1.z)
    var table = ct_grid_intersection_table(loaded.ct, loaded.hlut, ray0, ray_dir, native_case.scancap_bolus_mm_h2o + native_case.off_h2o_mm, 1.0)
    assert_equal(len(table.z), 603)
    assert_close(table.z[0], 327.019212674047, 1.0e-5)
    assert_close(table.z[len(table.z) - 1], -180.713584947869, 1.0e-5)
    assert_close(table.h2o[0], 41.709, 1.0e-12)
    assert_close(table.h2o[len(table.h2o) - 1], 171.167517424202, 1.0e-4)

    var min_ray0 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(10.0, 4.0, 0.0))
    var min_ray1 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(10.0, 4.0, 1.0))
    var min_ray_dir = Vec3(min_ray0.x - min_ray1.x, min_ray0.y - min_ray1.y, min_ray0.z - min_ray1.z)
    var min_table = ct_grid_intersection_table(loaded.ct, loaded.hlut, min_ray0, min_ray_dir, native_case.scancap_bolus_mm_h2o + native_case.off_h2o_mm, 1.0)
    assert_close(vintpol1d(26.5492595922924, min_table.z, min_table.h2o), 85.3272018863558, 1.0e-4)

    var p_out = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(-10.0, 0.0, 26.206050028606917))
    assert_true(not point_inside_target_mask(loaded.target, p_out))

    var z_min = -1.0e100
    var z_max = 1.0e100
    var p0 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(-26.0, -24.0, 0.0))
    var p1 = gantry_to_patient_from_matrix(f1, native_case.target_isocenter, Vec3(-26.0, -24.0, 1.0))
    var dz = Vec3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
    assert_true(clip_param_axis(p0.x, dz.x, loaded.ct.header.origin.x, loaded.ct.header.origin.x + Float64(loaded.ct.header.size_i) * loaded.ct.header.directions[0, 0], z_min, z_max))
    assert_true(clip_param_axis(p0.y, dz.y, loaded.ct.header.origin.y, loaded.ct.header.origin.y + Float64(loaded.ct.header.size_j) * loaded.ct.header.directions[1, 1], z_min, z_max))
    assert_true(clip_param_axis(p0.z, dz.z, loaded.ct.header.origin.z, loaded.ct.header.origin.z + Float64(loaded.ct.header.size_k) * loaded.ct.header.directions[2, 2], z_min, z_max))
    assert_close(z_min, -180.713584947869, 1.0e-5)
    assert_close(z_max, 327.019212674047, 1.0e-6)

    var f2 = native_case.fields[1].copy()
    var w2 = target_voi_bev_binary_wdw_bounds(f2, loaded.target, native_case.target_isocenter)
    var setup2 = with_distal_extension(w2, f2)
    var grid2 = raster_grid_from_window(setup2, f2.raster_x_mm, f2.raster_y_mm)
    assert_equal(len(grid2.points), 675)
    assert_close(grid2.points[0].x, -26.0, 1.0e-12)
    assert_close(grid2.points[0].y, -28.0, 1.0e-12)
    assert_close(grid2.points[len(grid2.points) - 1].x, 26.0, 1.0e-12)
    assert_close(grid2.points[len(grid2.points) - 1].y, 20.0, 1.0e-12)
