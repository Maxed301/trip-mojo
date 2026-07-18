from std.sys import has_accelerator
from std.testing import assert_equal, assert_true

from device_backend import partition_voxels
from optimizer import (
    MultiDeviceEvaluator,
    optimize_on_cpu,
    optimize_on_devices,
)
from optimization_problem import evaluate_physical_objective
from test_optimization_problem import build_flattened_4d_robust_problem


def assert_close(actual: Float64, expected: Float64) raises:
    var scale = max(abs(expected), 1.0)
    assert_true(abs(actual - expected) <= scale * 1.0e-12)


def main() raises:
    comptime assert has_accelerator(), "accelerator test requires GPUs"
    comptime if has_accelerator():
        var problem = build_flattened_4d_robust_problem()
        var shards = partition_voxels(problem, 3)
        assert_equal(len(shards), 3)
        assert_equal(
            shards[0].voxel_count
            + shards[1].voxel_count
            + shards[2].voxel_count,
            len(problem.voxels),
        )
        var expected = evaluate_physical_objective(problem, problem.particles)
        var evaluator = MultiDeviceEvaluator(problem, 3)
        var actual = evaluator.evaluate(problem, problem.particles)
        assert_close(actual.chi2, expected.chi2)
        for point in range(len(expected.gradient)):
            assert_close(actual.gradient[point], expected.gradient[point])

        problem.settings.max_iterations = UInt32(3)
        problem.settings.grace_iterations = UInt32(10)
        problem.settings.epsilon = 0.0
        var expected_opt = optimize_on_cpu(problem)
        var actual_opt = optimize_on_devices(problem, 3)
        assert_equal(actual_opt.iterations, expected_opt.iterations)
        assert_equal(actual_opt.stop_reason, expected_opt.stop_reason)
        for point in range(len(expected_opt.particles)):
            assert_close(
                actual_opt.particles[point], expected_opt.particles[point]
            )
        print("test_three_devices: PASS")
