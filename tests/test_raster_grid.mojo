from std.testing import assert_equal

from case_model import native_case_from_exec_path
from field_geometry import target_voi_bev_raster_window
from raster_grid import raster_axis_values, raster_grid_from_window
from voi import load_voi_set_from_binfo


def main() raises:
    var axis = raster_axis_values(-3.1, 4.1, 2.0)
    assert_equal(len(axis), 4)
    assert_equal(axis[0], -2.0)
    assert_equal(axis[1], 0.0)
    assert_equal(axis[2], 2.0)
    assert_equal(axis[3], 4.0)

    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var vois = load_voi_set_from_binfo(native_case.voi_binfo_path)
    var target = vois.find(native_case.target_voi_name)

    var field1_window = target_voi_bev_raster_window(native_case.fields[0], target, native_case.target_isocenter)
    var field1_grid = raster_grid_from_window(field1_window, native_case.fields[0].raster_x_mm, native_case.fields[0].raster_y_mm)
    assert_equal(len(field1_grid.x_values), 34)
    assert_equal(len(field1_grid.y_values), 31)
    assert_equal(len(field1_grid.points), 1054)
    assert_equal(field1_grid.x_values[0], -32.0)
    assert_equal(field1_grid.x_values[len(field1_grid.x_values) - 1], 34.0)
    assert_equal(field1_grid.y_values[0], -32.0)
    assert_equal(field1_grid.y_values[len(field1_grid.y_values) - 1], 28.0)
    assert_equal(field1_grid.points[0].x, -32.0)
    assert_equal(field1_grid.points[0].y, -32.0)
    assert_equal(field1_grid.points[len(field1_grid.x_values)].x, 34.0)
    assert_equal(field1_grid.points[len(field1_grid.x_values)].y, -30.0)

    var field2_window = target_voi_bev_raster_window(native_case.fields[1], target, native_case.target_isocenter)
    var field2_grid = raster_grid_from_window(field2_window, native_case.fields[1].raster_x_mm, native_case.fields[1].raster_y_mm)
    assert_equal(len(field2_grid.x_values), 34)
    assert_equal(len(field2_grid.y_values), 33)
    assert_equal(len(field2_grid.points), 1122)
    assert_equal(field2_grid.x_values[0], -32.0)
    assert_equal(field2_grid.x_values[len(field2_grid.x_values) - 1], 34.0)
    assert_equal(field2_grid.y_values[0], -36.0)
    assert_equal(field2_grid.y_values[len(field2_grid.y_values) - 1], 28.0)
