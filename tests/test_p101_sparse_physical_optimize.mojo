from std.testing import assert_true

from case_model import native_case_from_exec_path
from geometry import Vec3, gantry_to_patient_point
from physical_sparse_matrix import build_physical_sparse_dose_matrix
from rst import RSTPlan, RasterSpot
from sparse_optimizer import compute_sparse_dose, optimize_sparse_dose_cg, sparse_relative_rmse


def points_at_spot_centers(spots: List[RasterSpot], count: Int, native_case_isocenter: Vec3, gantry: Float64) raises -> List[Vec3]:
    var points = List[Vec3]()
    var n = count
    if len(spots) < n:
        n = len(spots)
    for i in range(n):
        points.append(gantry_to_patient_point(native_case_isocenter, gantry, 90.0, Vec3(spots[i].x, spots[i].y, 100.0)))
    return points^


def first_spots(count: Int) raises -> List[RasterSpot]:
    var plan = RSTPlan("reference/trip4d_p101_cpu_20260511_140457/StaticP101_2110_field1_iGTV_R.rst")
    var spots = List[RasterSpot]()
    var n = count
    if len(plan.field) < n:
        n = len(plan.field)
    for i in range(n):
        spots.append(plan.field[i].copy())
    return spots^


def constant_depths(count: Int, value: Float64) -> List[Float64]:
    var depths = List[Float64]()
    for _ in range(count):
        depths.append(value)
    return depths^



def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var spots = first_spots(64)
    var points = points_at_spot_centers(spots, 16, native_case.target_isocenter, native_case.fields[0].gantry_degrees)
    var depths = constant_depths(len(points), 120.0)
    var matrix = build_physical_sparse_dose_matrix(points, spots, depths, native_case.fields[0].gantry_degrees, native_case.target_isocenter, 0.0)
    assert_true(len(matrix.entries) > 0)

    var seed_particles = List[Float64]()
    seed_particles.resize(len(spots), 0.0)
    for i in range(len(spots)):
        seed_particles[i] = spots[i].particles
    var target = compute_sparse_dose(matrix, seed_particles)
    var initial = List[Float64]()
    initial.resize(len(spots), 0.0)
    var start_dose = compute_sparse_dose(matrix, initial)
    var start_rmse = sparse_relative_rmse(start_dose, target)
    var optimized = optimize_sparse_dose_cg(matrix, target, initial, 64, 1.0e-12)
    assert_true(optimized.relative_rmse < start_rmse)
    assert_true(optimized.relative_rmse < 1.0e-8)
