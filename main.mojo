from exec_parser import parse_exec_file, print_exec_plan_summary
from case_model import load_native_case, native_case_from_exec
from energy_layers import build_energy_layers
from field_activation import build_native_optimization_field_state, native_field_mark_mask
from field_geometry import target_voi_bev_h2o_window
from raster_grid import raster_grid_from_window
from sis import read_sis_table
from std.sys import argv

comptime DEFAULT_EXEC_PATH = "reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec"


def exec_path_from_args() -> String:
    var args = argv()
    if len(args) > 1:
        return args[1]
    return DEFAULT_EXEC_PATH


def main() raises:
    var plan = parse_exec_file(exec_path_from_args())
    print_exec_plan_summary(plan)
    var native_case = native_case_from_exec(plan)
    var loaded = load_native_case(native_case)
    var sis = read_sis_table(native_case.sis_path)
    var total_candidates = 0
    for i in range(len(native_case.fields)):
        var window = target_voi_bev_h2o_window(native_case.fields[i], loaded.target, native_case.target_isocenter, loaded.ct, loaded.hlut, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm)
        var grid = raster_grid_from_window(window, native_case.fields[i].raster_x_mm, native_case.fields[i].raster_y_mm)
        var layers = build_energy_layers(native_case.fields[i], window, sis)
        var candidates = len(grid.points) * len(layers)
        var marks = native_field_mark_mask(native_case.fields[i], grid, layers, loaded.target, native_case.target_isocenter, loaded.ct, loaded.hlut, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm)
        var state = build_native_optimization_field_state(i, native_case.fields[i], grid, layers, loaded.target, native_case.target_isocenter, loaded.ct, loaded.hlut, native_case.scancap_bolus_mm_h2o, native_case.off_h2o_mm, native_case.scancap_min_particles)
        total_candidates += candidates
        print("native field", native_case.fields[i].number, "window", window.min_x, window.max_x, window.min_y, window.max_y, window.min_z, window.max_z)
        print("native field", native_case.fields[i].number, "candidate raster points per energy", len(grid.points), "energies", len(layers), "total", candidates)
        print("native field", native_case.fields[i].number, "marked target", marks.counts.target_inside_points, "robust", marks.counts.robust_scenario_points, "extension", marks.counts.extension_points, "active", marks.counts.optimization_point_count())
        print("native field", native_case.fields[i].number, "optimizer spots", state.active_spots(), "first energy", state.spots[0].energy_mev_u, "last energy", state.spots[len(state.spots) - 1].energy_mev_u)
    print("native candidate raster points total:", total_candidates)
