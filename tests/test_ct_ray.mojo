from std.testing import assert_true

from ct_ray import ray_water_depth_mm, read_ct_volume
from geometry import Vec3, gantry_to_patient_point
from hlut import read_hlut
from p101_io import load_tumor_world_points, load_water_depths_for_points
from phys_dose import p101_isocenter


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var ct = read_ct_volume("/home/max/Projects/beamline_transport/CT/P101_00.nrrd")
    var hlut = read_hlut("/home/max/Projects/TRIP_DATA/Basedata/GSI/19990211.hlut")
    var points = load_tumor_world_points()
    var wet = load_water_depths_for_points(280.0, points)
    var iso = p101_isocenter()
    var p0 = gantry_to_patient_point(iso, 280.0, 90.0, Vec3(0.0, 0.0, 0.0))
    var p1 = gantry_to_patient_point(iso, 280.0, 90.0, Vec3(0.0, 0.0, 1.0))
    var dir = Vec3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
    var back = Vec3(-dir.x, -dir.y, -dir.z)

    var d0 = ray_water_depth_mm(ct, hlut, points[0], back)
    assert_close(d0 + 41.0815, wet[0], 1.0)
