from std.testing import assert_equal, assert_true

from case_model import load_native_case, native_case_from_exec_path, trip_target_window_center
from energy_layers import build_energy_layers
from field_activation import build_native_optimization_field_state
from field_geometry import target_voi_bev_h2o_window
from geometry import Vec3
from opt_voxels import build_p101_target_avoidance_opt_voxels
from physical_sparse_matrix import (
    build_binned_physical_sparse_dose_matrix,
    build_physical_sparse_dose_matrix,
    field_h2o_depths_for_points,
    opt_voxel_patient_centers,
    optimization_state_to_raster_spots,
)
from raster_grid import raster_grid_from_window
from rst import RasterSpot
from sis import read_sis_table


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def first_points(values: List[Vec3], count: Int) -> List[Vec3]:
    var out = List[Vec3]()
    var n = count
    if len(values) < n:
        n = len(values)
    for i in range(n):
        out.append(values[i].copy())
    return out^


def first_depths(values: List[Float64], count: Int) -> List[Float64]:
    var out = List[Float64]()
    var n = count
    if len(values) < n:
        n = len(values)
    for i in range(n):
        out.append(values[i].copy())
    return out^


def first_spots(values: List[RasterSpot], count: Int) -> List[RasterSpot]:
    var out = List[RasterSpot]()
    var n = count
    if len(values) < n:
        n = len(values)
    for i in range(n):
        out.append(values[i].copy())
    return out^


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var loaded = load_native_case(native_case)
    var opt_voxels = build_p101_target_avoidance_opt_voxels(loaded.target, 5.0, 10.0, 2.0)
    assert_equal(len(opt_voxels.voxels), 37737)

    var field = native_case.fields[0].copy()
    var sis = read_sis_table(native_case.sis_path)
    var window = target_voi_bev_h2o_window(field, loaded.target, native_case.target_isocenter, loaded.ct, loaded.hlut, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm)
    var grid = raster_grid_from_window(window, field.raster_x_mm, field.raster_y_mm)
    var layers = build_energy_layers(field, window, sis)
    var state = build_native_optimization_field_state(0, field, grid, layers, loaded.target, native_case.target_isocenter, loaded.ct, loaded.hlut, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm, native_case.scancap_min_particles)
    assert_equal(state.active_spots(), 11644)

    var all_points = opt_voxel_patient_centers(opt_voxels, loaded.ct)
    var physical_isocenter = trip_target_window_center(loaded.target)
    var all_depths = field_h2o_depths_for_points(field, all_points, loaded.ct, loaded.hlut, physical_isocenter, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm)
    var all_spots = optimization_state_to_raster_spots(state, True)
    var points = first_points(all_points, 32)
    var depths = first_depths(all_depths, 32)
    var spots = first_spots(all_spots, 96)

    var binned = build_binned_physical_sparse_dose_matrix(points, spots, depths, field.gantry_degrees, physical_isocenter)
    var direct = build_physical_sparse_dose_matrix(points, spots, depths, field.gantry_degrees, physical_isocenter)
    assert_equal(binned.voxel_count, direct.voxel_count)
    assert_equal(binned.spot_count, direct.spot_count)
    assert_equal(len(binned.entries), len(direct.entries))
    for i in range(len(direct.entries)):
        assert_equal(binned.entries[i].voxel, direct.entries[i].voxel)
        assert_equal(binned.entries[i].spot, direct.entries[i].spot)
        assert_close(binned.entries[i].value, direct.entries[i].value, 0.0)
