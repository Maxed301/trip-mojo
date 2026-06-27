from std.testing import assert_equal, assert_true

from geometry import Vec3, gantry_to_patient_point
from physical_sparse_matrix import build_physical_sparse_dose_matrix
from phys_dose import compute_physical_field_dose
from rst import RasterSpot
from sparse_optimizer import compute_sparse_dose, optimize_sparse_dose_cg


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var isocenter = Vec3(0.0, 0.0, 0.0)
    var points = List[Vec3]()
    points.append(gantry_to_patient_point(isocenter, 280.0, 90.0, Vec3(0.0, 0.0, 100.0)))
    points.append(gantry_to_patient_point(isocenter, 280.0, 90.0, Vec3(2.0, 0.0, 100.0)))
    var h2o = [100.0, 100.0]
    var spots = List[RasterSpot]()
    spots.append(RasterSpot(0.0, 0.0, 280.0, 5000.0, 7.5, 135.0, 0.0, True, 0))
    spots.append(RasterSpot(2.0, 0.0, 280.0, 3000.0, 7.5, 135.0, 0.0, True, 1))

    var matrix = build_physical_sparse_dose_matrix(points, spots, h2o, 280.0, isocenter)
    assert_equal(matrix.voxel_count, 2)
    assert_equal(matrix.spot_count, 2)
    assert_true(len(matrix.entries) > 0)

    var particles = [5000.0, 3000.0]
    var sparse_dose = compute_sparse_dose(matrix, particles)
    var direct_dose = compute_physical_field_dose(points, spots, h2o, 280.0, isocenter)
    assert_close(sparse_dose[0], direct_dose[0], 1.0e-14)
    assert_close(sparse_dose[1], direct_dose[1], 1.0e-14)

    var initial = [0.0, 0.0]
    var optimized = optimize_sparse_dose_cg(matrix, sparse_dose, initial, 10)
    assert_close(optimized.dose[0], sparse_dose[0], 1.0e-10)
    assert_close(optimized.dose[1], sparse_dose[1], 1.0e-10)
