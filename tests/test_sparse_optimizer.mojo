from std.testing import assert_equal, assert_true

from sparse_optimizer import SparseDoseEntry, SparseDoseMatrix, compute_sparse_dose, optimize_sparse_dose, optimize_sparse_dose_cg, sparse_relative_rmse


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var entries = List[SparseDoseEntry]()
    entries.append(SparseDoseEntry(0, 0, 1.0))
    entries.append(SparseDoseEntry(1, 1, 1.0))
    entries.append(SparseDoseEntry(2, 0, 0.5))
    entries.append(SparseDoseEntry(2, 1, 0.5))
    var matrix = SparseDoseMatrix(3, 2, entries^)
    var target = [3.0, 5.0, 4.0]
    var initial = [0.0, 0.0]
    var start = compute_sparse_dose(matrix, initial)
    var start_rmse = sparse_relative_rmse(start, target)
    var result = optimize_sparse_dose(matrix, target, initial, 250, 0.1)
    assert_true(result.relative_rmse < start_rmse)
    assert_close(result.particles[0], 3.0, 1.0e-7)
    assert_close(result.particles[1], 5.0, 1.0e-7)
    assert_close(result.dose[0], 3.0, 1.0e-7)
    assert_close(result.dose[1], 5.0, 1.0e-7)
    assert_close(result.dose[2], 4.0, 1.0e-7)
    assert_equal(result.iterations, 250)

    var constrained = optimize_sparse_dose(matrix, target, initial, 5, 0.1, 4.0)
    assert_equal(constrained.particles[0], 0.0)
    assert_equal(constrained.particles[1], 0.0)

    var cg = optimize_sparse_dose_cg(matrix, target, initial, 10)
    assert_close(cg.particles[0], 3.0, 1.0e-10)
    assert_close(cg.particles[1], 5.0, 1.0e-10)
    assert_close(cg.dose[0], 3.0, 1.0e-10)
    assert_close(cg.dose[1], 5.0, 1.0e-10)
    assert_close(cg.dose[2], 4.0, 1.0e-10)
    assert_true(cg.iterations <= 2)
