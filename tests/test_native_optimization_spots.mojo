from std.testing import assert_equal, assert_true

from case_model import load_native_case, native_case_from_exec_path
from energy_layers import build_energy_layers
from field_activation import build_native_optimization_field_state
from field_geometry import target_voi_bev_h2o_window
from physical_sparse_matrix import optimization_state_to_raster_spots
from raster_grid import raster_grid_from_window
from sis import read_sis_table


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var loaded = load_native_case(native_case)
    var sis = read_sis_table(native_case.sis_path)
    var expected_counts = [11644, 11231]
    var expected_first_energy = [280.48, 303.14]
    var expected_last_energy = [197.58, 232.2]
    for i in range(len(native_case.fields)):
        var window = target_voi_bev_h2o_window(native_case.fields[i], loaded.target, native_case.target_isocenter, loaded.ct, loaded.hlut, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm)
        var grid = raster_grid_from_window(window, native_case.fields[i].raster_x_mm, native_case.fields[i].raster_y_mm)
        var layers = build_energy_layers(native_case.fields[i], window, sis)
        var state = build_native_optimization_field_state(i, native_case.fields[i], grid, layers, loaded.target, native_case.target_isocenter, loaded.ct, loaded.hlut, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm, native_case.scancap_min_particles)
        assert_equal(state.active_spots(), expected_counts[i])
        assert_close(state.spots[0].energy_mev_u, expected_first_energy[i], 1.0e-12)
        assert_close(state.spots[len(state.spots) - 1].energy_mev_u, expected_last_energy[i], 1.0e-12)
        var raster = optimization_state_to_raster_spots(state, True)
        assert_equal(len(raster), expected_counts[i])
        assert_close(raster[0].particles, 1.0, 0.0)
        assert_close(raster[0].energy, expected_first_energy[i], 1.0e-12)
