from std.testing import assert_equal

from case_model import native_case_from_exec_path
from field_geometry import target_voi_bev_raster_window
from optimization_model import FieldRasterMarkCounts, OptimizationFieldSpec
from optimizer_state import apply_min_particles, scale_particles, seed_uniform_optimization_field_state
from raster_grid import raster_grid_from_window
from rst import RSTPlan
from voi import load_voi_set_from_binfo


def make_energy_list(count: Int) -> List[Float64]:
    var values = List[Float64]()
    for i in range(count):
        values.append(300.0 - Float64(i) * 3.0)
    return values^


def make_range_list(count: Int) -> List[Float64]:
    var values = List[Float64]()
    for i in range(count):
        values.append(160.0 - Float64(i) * 3.0)
    return values^


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var vois = load_voi_set_from_binfo(native_case.voi_binfo_path)
    var target = vois.find(native_case.target_voi_name)
    var field1_window = target_voi_bev_raster_window(native_case.fields[0], target, native_case.target_isocenter)
    var field1_grid = raster_grid_from_window(field1_window, native_case.fields[0].raster_x_mm, native_case.fields[0].raster_y_mm)
    var written = len(RSTPlan("reference/trip4d_p101_cpu_20260511_140457/StaticP101_2110_field1_iGTV_R.rst").field)
    var spec = OptimizationFieldSpec(native_case.fields[0].copy(), FieldRasterMarkCounts(209250, 3588, 1997, 6054), written)

    var state = seed_uniform_optimization_field_state(0, spec, field1_grid, make_energy_list(25), make_range_list(25), 5000.0)
    assert_equal(len(state.spots), 11639)
    assert_equal(state.active_spots(), 11639)
    assert_equal(state.total_particles(), 58195000.0)
    assert_equal(state.spots[0].x, -32.0)
    assert_equal(state.spots[0].y, -32.0)
    assert_equal(state.spots[0].energy_index, 0)
    assert_equal(state.spots[1054].energy_index, 1)

    scale_particles(state, 2.0)
    assert_equal(state.total_particles(), 116390000.0)
    apply_min_particles(state, 10001.0)
    assert_equal(state.active_spots(), 0)
    assert_equal(state.total_particles(), 0.0)
