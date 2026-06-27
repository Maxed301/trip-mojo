from std.testing import assert_true

from case_model import native_case_from_exec_path
from field_geometry import contour_extension_mm, target_voi_bev_binary_wdw_bounds, target_voi_bev_corner_bounds, target_voi_bev_raster_window
from voi import load_voi_set_from_binfo


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var vois = load_voi_set_from_binfo(native_case.voi_binfo_path)
    var target = vois.find(native_case.target_voi_name)

    var field1_bounds = target_voi_bev_corner_bounds(native_case.fields[0], target, native_case.target_isocenter)
    assert_close(field1_bounds.min_x, -25.50000000000003, 1.0e-12)
    assert_close(field1_bounds.max_x, 28.5, 1.0e-12)
    assert_close(field1_bounds.min_y, -25.795443311967546, 1.0e-12)
    assert_close(field1_bounds.max_y, 22.29037128810532, 1.0e-12)

    var field2_bounds = target_voi_bev_corner_bounds(native_case.fields[1], target, native_case.target_isocenter)
    assert_close(field2_bounds.min_x, -25.50000000000003, 1.0e-12)
    assert_close(field2_bounds.max_x, 28.5, 1.0e-12)
    assert_close(field2_bounds.min_y, -29.922771406551163, 1.0e-12)
    assert_close(field2_bounds.max_y, 22.399827366369806, 1.0e-12)

    var field1_binary_bounds = target_voi_bev_binary_wdw_bounds(native_case.fields[0], target, native_case.target_isocenter)
    assert_close(field1_binary_bounds.min_x, -27.00000000000003, 1.0e-12)
    assert_close(field1_binary_bounds.max_x, 27.0, 1.0e-12)
    assert_close(field1_binary_bounds.min_y, -25.229769281016928, 1.0e-12)
    assert_close(field1_binary_bounds.max_y, 22.516875698436905, 1.0e-12)

    var field2_binary_bounds = target_voi_bev_binary_wdw_bounds(native_case.fields[1], target, native_case.target_isocenter)
    assert_close(field2_binary_bounds.min_x, -27.00000000000003, 1.0e-12)
    assert_close(field2_binary_bounds.max_x, 27.0, 1.0e-12)
    assert_close(field2_binary_bounds.min_y, -29.482531228166863, 1.0e-12)
    assert_close(field2_binary_bounds.max_y, 21.71975804927328, 1.0e-12)

    assert_close(contour_extension_mm(native_case.fields[0]), 7.2, 1.0e-12)
    assert_close(contour_extension_mm(native_case.fields[1]), 7.2, 1.0e-12)

    var field1_window = target_voi_bev_raster_window(native_case.fields[0], target, native_case.target_isocenter)
    assert_close(field1_window.min_x, -32.70000000000003, 1.0e-12)
    assert_close(field1_window.max_x, 35.7, 1.0e-12)
    assert_close(field1_window.min_y, -32.99544331196755, 1.0e-12)
    assert_close(field1_window.max_y, 29.490371288105318, 1.0e-12)
    assert_close(field1_window.min_z, -28.628275940442336, 1.0e-12)
    assert_close(field1_window.max_z, 31.11662694864998, 1.0e-12)

    var field2_window = target_voi_bev_raster_window(native_case.fields[1], target, native_case.target_isocenter)
    assert_close(field2_window.min_x, -32.70000000000003, 1.0e-12)
    assert_close(field2_window.max_x, 35.7, 1.0e-12)
    assert_close(field2_window.min_y, -37.122771406551166, 1.0e-12)
    assert_close(field2_window.max_y, 29.599827366369805, 1.0e-12)
    assert_close(field2_window.min_z, -30.44265093769502, 1.0e-12)
    assert_close(field2_window.max_z, 26.843455354532978, 1.0e-12)
