from std.sys import has_accelerator
from std.testing import assert_equal, assert_true

from device_backend import partition_voxels
from optimizer import (
    MultiDeviceEvaluator,
    optimize_on_cpu,
    optimize_on_devices,
)
from optimization_problem import (
    evaluate_physical_objective,
)
from tests.test_optimization_problem import build_flattened_4d_robust_problem


def assert_close(actual: Float64, expected: Float64) raises:
    var scale = max(abs(expected), 1.0)
    assert_true(abs(actual - expected) <= scale * 1.0e-12)


def main() raises:
    comptime assert has_accelerator(), "accelerator test requires GPUs"
    comptime if has_accelerator():
        var problem = build_flattened_4d_robust_problem()
        var shards = partition_voxels(problem, 2)
        assert_equal(shards[0].voxel_offset, 0)
        assert_true(shards[0].voxel_count > 0)
        assert_equal(
            shards[0].voxel_count + shards[1].voxel_count,
            len(problem.voxels),
        )
        var invalid = build_flattened_4d_robust_problem()
        invalid.slices[1].coefficient_offset = UInt64(0)
        var rejected = False
        try:
            _ = partition_voxels(invalid, 2)
        except:
            rejected = True
        assert_true(rejected)

        problem.settings.include_dmax = True
        for voxel in range(shards[0].voxel_count):
            problem.voxels[voxel].prescribed_dose = -abs(
                problem.voxels[voxel].prescribed_dose
            )

        var expected = evaluate_physical_objective(problem, problem.particles)
        var evaluator = MultiDeviceEvaluator(problem, 2)
        var actual = evaluator.evaluate(problem, problem.particles)
        assert_close(actual.chi2, expected.chi2)
        assert_close(actual.weighted_dose2, expected.dose_p_weighted_avg2)
        for point in range(len(expected.gradient)):
            assert_close(actual.gradient[point], expected.gradient[point])

        problem.settings.max_iterations = UInt32(4)
        problem.settings.grace_iterations = UInt32(10)
        problem.settings.epsilon = 0.0
        var expected_opt = optimize_on_cpu(problem)
        var actual_opt = optimize_on_devices(problem, 2)
        assert_equal(actual_opt.iterations, expected_opt.iterations)
        assert_equal(actual_opt.stop_reason, expected_opt.stop_reason)
        assert_close(actual_opt.chi2, expected_opt.chi2)
        for point in range(len(expected_opt.particles)):
            assert_close(
                actual_opt.particles[point], expected_opt.particles[point]
            )
        print("test_two_devices: PASS")
