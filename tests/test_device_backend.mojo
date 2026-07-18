from std.sys import has_accelerator
from std.testing import assert_equal, assert_true

from device_backend import DeviceWorkspace
from cpu_backend import (
    evaluate_biological_objective,
    compute_biological_exact_step,
    compute_physical_exact_step,
    packed_slice_dot,
    packed_zero_gradient_direction,
)
from optimizer import (
    optimize_on_cpu,
    optimize_on_device,
)
from tests.support.reference_objective import (
    BiologicalMatrixEntry,
    BiologicalDoseMatrix,
    BiologicalScenarioSet,
    BioLQParams,
    VoxelObjective,
)
from tests.support.problem_factory import pack_biological_problem
from optimization_problem import (
    evaluate_physical_objective,
)
from optimization_problem import MinimumParticlePolicy, OptimizerSettings
from tests.test_cpu_backend import build_biological_problem
from tests.test_optimization_problem import (
    build_flattened_4d_robust_problem,
    build_multifield_problem,
    build_problem,
)


def assert_float64_device_equal(actual: Float64, expected: Float64) raises:
    var scale = abs(expected)
    if scale < 1.0:
        scale = 1.0
    assert_true(abs(actual - expected) <= scale * 8.0 * 2.220446049250313e-16)


def assert_reduction_close(
    actual: Float64, expected: Float64, term_count: Int
) raises:
    var scale = abs(expected)
    if scale < 1.0:
        scale = 1.0
    assert_true(
        abs(actual - expected)
        <= scale * Float64(term_count) * 2.220446049250313e-16
    )


def test_accelerator_slice_dots_match_cpu() raises:
    var problem = build_problem()
    var accelerator = DeviceWorkspace(problem)
    var actual = accelerator.slice_dots(problem.particles)
    assert_equal(len(actual), len(problem.slices))
    for i in range(len(actual)):
        assert_float64_device_equal(
            actual[i],
            packed_slice_dot(problem, problem.slices[i], problem.particles),
        )
    var second_particles = [25.0, 75.0]
    actual = accelerator.slice_dots(second_particles)
    for i in range(len(actual)):
        assert_float64_device_equal(
            actual[i],
            packed_slice_dot(problem, problem.slices[i], second_particles),
        )

    var moments = accelerator.moments(problem.particles)
    for scenario_index in range(len(problem.voxel_scenarios)):
        var state = problem.scenario_states[scenario_index].copy()
        var expected = [
            state.dose_minor,
            state.alpha_minor,
            state.sqrt_beta_minor,
            state.let_mix_minor,
            0.0,
        ]
        var scenario = problem.voxel_scenarios[scenario_index].copy()
        for local_slice in range(Int(scenario.slice_count)):
            var packed_slice = problem.slices[
                Int(scenario.slice_offset) + local_slice
            ].copy()
            var dot = packed_slice_dot(problem, packed_slice, problem.particles)
            expected[0] += dot * Float64(packed_slice.dose_coefficient)
            expected[1] += dot * Float64(packed_slice.alpha_coefficient)
            expected[2] += dot * Float64(packed_slice.sqrt_beta_coefficient)
            expected[3] += dot * Float64(packed_slice.let_mix_coefficient)
            expected[4] += dot * Float64(packed_slice.let_bar_coefficient)
        for moment in range(5):
            assert_equal(moments[scenario_index * 5 + moment], expected[moment])


def test_accelerator_zero_gradient_matches_cpu() raises:
    var problem = build_multifield_problem()
    var expected = packed_zero_gradient_direction(problem)
    var accelerator = DeviceWorkspace(problem)
    var actual = accelerator.zero_gradient_direction()
    assert_equal(len(actual), len(expected))
    for point in range(len(expected)):
        assert_float64_device_equal(actual[point], expected[point])


def test_accelerator_biological_evaluation_matches_cpu() raises:
    var problem = build_biological_problem(0.02)
    problem.settings.include_dmax = True
    problem.voxels[0].prescribed_let = 6.0
    problem.voxels[0].let_weight = 0.5
    problem.scenario_states[0].let_mix_minor = 1.0e7
    problem.scenario_states[1].let_mix_minor = 1.0e7
    var expected = evaluate_biological_objective(problem, problem.particles)
    var accelerator = DeviceWorkspace(problem)
    var actual = accelerator.evaluation(problem.particles)
    assert_float64_device_equal(
        actual.weighted_dose2(), expected.dose_p_weighted_avg2
    )
    for voxel in range(len(problem.voxels)):
        assert_float64_device_equal(
            actual.dose_min[voxel], expected.dose_min[voxel]
        )
        assert_float64_device_equal(
            actual.dose_max[voxel], expected.dose_max[voxel]
        )
        assert_equal(actual.min_scenario[voxel], expected.min_scenario[voxel])
        assert_equal(actual.max_scenario[voxel], expected.max_scenario[voxel])
    for scenario in range(len(expected.states)):
        var state = expected.states[scenario].copy()
        var offset = scenario * 4
        assert_float64_device_equal(actual.scenario_bio[offset], state.dose)
        assert_float64_device_equal(
            actual.scenario_bio[offset + 1], state.let_bar
        )
        assert_float64_device_equal(
            actual.scenario_bio[offset + 2], state.dose_phs
        )
        assert_float64_device_equal(
            actual.scenario_bio[offset + 3], state.gradient_denominator
        )
    assert_equal(len(actual.gradient), len(expected.gradient))
    for point in range(len(expected.gradient)):
        assert_float64_device_equal(
            actual.gradient[point], expected.gradient[point]
        )
    assert_float64_device_equal(
        accelerator.exact_step(expected.gradient),
        compute_biological_exact_step(problem, expected, expected.gradient),
    )
    assert_float64_device_equal(actual.chi2(), expected.chi2)
    assert_float64_device_equal(actual.gradient_norm, expected.gradient_norm)

    var zero_particles = [0.0]
    expected = evaluate_biological_objective(problem, zero_particles)
    actual = accelerator.evaluation(zero_particles)
    for point in range(len(expected.gradient)):
        assert_equal(actual.gradient[point], expected.gradient[point])

    problem.voxels[0].prescribed_dose = 0.0
    problem.voxels[0].initial_min_scenario = Int32(1)
    problem.voxels[0].initial_max_scenario = Int32(0)
    expected = evaluate_biological_objective(problem, problem.particles)
    accelerator = DeviceWorkspace(problem)
    actual = accelerator.evaluation(problem.particles)
    assert_equal(actual.dose_min[0], 0.0)
    assert_equal(actual.dose_max[0], 0.0)
    assert_equal(actual.min_scenario[0], expected.min_scenario[0])
    assert_equal(actual.max_scenario[0], expected.max_scenario[0])

    problem = build_biological_problem(0.02)
    problem.settings.include_dmax = True
    problem.point_active[0] = UInt8(0)
    expected = evaluate_biological_objective(problem, problem.particles)
    accelerator = DeviceWorkspace(problem)
    actual = accelerator.evaluation(problem.particles)
    assert_equal(actual.gradient[0], expected.gradient[0])


def test_accelerator_physical_evaluation_matches_cpu() raises:
    var problem = build_problem()
    problem.settings.include_dmax = True
    problem.voxels[0].maximum_dose_weight = 0.4
    problem.voxels[0].initial_dose = 0.25
    problem.voxels[0].prescribed_let = 1.0
    problem.voxels[0].let_weight = 100.0
    problem.scenario_states[0].dose_minor = 1.0e6
    problem.scenario_states[1].dose_minor = 2.0e6
    var expected = evaluate_physical_objective(problem, problem.particles)
    var accelerator = DeviceWorkspace(problem)
    var actual = accelerator.evaluation(problem.particles)
    for scenario in range(len(expected.dose_by_voxel_scenario)):
        assert_float64_device_equal(
            actual.scenario_bio[scenario * 4],
            expected.dose_by_voxel_scenario[scenario],
        )
    for voxel in range(len(problem.voxels)):
        assert_float64_device_equal(
            actual.dose_min[voxel], expected.dose_min[voxel]
        )
        assert_float64_device_equal(
            actual.dose_max[voxel], expected.dose_max[voxel]
        )
        assert_equal(actual.min_scenario[voxel], expected.min_scenario[voxel])
        assert_equal(actual.max_scenario[voxel], expected.max_scenario[voxel])
    for point in range(len(expected.gradient)):
        assert_float64_device_equal(
            actual.gradient[point], expected.gradient[point]
        )
    assert_float64_device_equal(actual.chi2(), expected.chi2)
    assert_float64_device_equal(
        actual.weighted_dose2(), expected.dose_p_weighted_avg2
    )
    assert_float64_device_equal(actual.gradient_norm, expected.gradient_norm)
    assert_float64_device_equal(
        accelerator.exact_step(expected.gradient),
        compute_physical_exact_step(problem, expected, expected.gradient),
    )


def test_accelerator_multifield_matches_cpu() raises:
    var problem = build_multifield_problem()
    var expected = evaluate_physical_objective(problem, problem.particles)
    var accelerator = DeviceWorkspace(problem)
    assert_equal(accelerator.active_slice_capacity, 3)
    var actual = accelerator.evaluation(problem.particles)
    assert_float64_device_equal(actual.dose_min[0], expected.dose_min[0])
    assert_float64_device_equal(actual.dose_max[0], expected.dose_max[0])
    assert_equal(actual.min_scenario[0], expected.min_scenario[0])
    assert_equal(actual.max_scenario[0], expected.max_scenario[0])
    for point in range(len(expected.gradient)):
        assert_float64_device_equal(
            actual.gradient[point], expected.gradient[point]
        )

    problem.settings.max_iterations = UInt32(4)
    problem.settings.grace_iterations = UInt32(10)
    problem.settings.epsilon = 0.0
    problem.settings.configured_step_factor = 1.0
    var expected_optimization = optimize_on_cpu(problem)
    var actual_optimization = optimize_on_device(problem)
    assert_equal(
        actual_optimization.iterations, expected_optimization.iterations
    )
    assert_equal(
        actual_optimization.stop_reason, expected_optimization.stop_reason
    )
    assert_float64_device_equal(
        actual_optimization.chi2, expected_optimization.chi2
    )
    for point in range(len(expected_optimization.particles)):
        assert_float64_device_equal(
            actual_optimization.particles[point],
            expected_optimization.particles[point],
        )


def test_accelerator_flattened_4d_robust_matches_cpu() raises:
    var problem = build_flattened_4d_robust_problem()
    var expected = evaluate_physical_objective(problem, problem.particles)
    var accelerator = DeviceWorkspace(problem)
    var actual = accelerator.evaluation(problem.particles)
    for voxel in range(len(problem.voxels)):
        assert_float64_device_equal(
            actual.dose_min[voxel], expected.dose_min[voxel]
        )
        assert_float64_device_equal(
            actual.dose_max[voxel], expected.dose_max[voxel]
        )
        assert_equal(actual.min_scenario[voxel], expected.min_scenario[voxel])
        assert_equal(actual.max_scenario[voxel], expected.max_scenario[voxel])
    for point in range(len(expected.gradient)):
        assert_float64_device_equal(
            actual.gradient[point], expected.gradient[point]
        )


def test_accelerator_preserves_nonmonotonic_slice_ranges() raises:
    var problem = build_problem()
    var first_offset = problem.slices[0].coefficient_offset
    problem.slices[0].coefficient_offset = problem.slices[1].coefficient_offset
    problem.slices[1].coefficient_offset = first_offset
    problem.validate()
    var accelerator = DeviceWorkspace(problem)
    var dots = accelerator.slice_dots(problem.particles)
    for i in range(len(problem.slices)):
        assert_float64_device_equal(
            dots[i],
            packed_slice_dot(problem, problem.slices[i], problem.particles),
        )
    var expected = evaluate_physical_objective(problem, problem.particles)
    var actual = accelerator.evaluation(problem.particles)
    assert_float64_device_equal(actual.chi2(), expected.chi2)
    for point in range(len(expected.gradient)):
        assert_float64_device_equal(
            actual.gradient[point], expected.gradient[point]
        )


def check_contiguous_uint16_indices(point_count: Int, local_point: Int) raises:
    var problem = build_problem()
    problem.field_slices[0].point_count = UInt32(point_count)
    problem.particles.resize(point_count, 0.0)
    problem.initial_direction.resize(point_count, 0.0)
    problem.initial_gradient.resize(point_count, 0.0)
    problem.point_active.resize(point_count, UInt8(1))
    problem.particles[local_point] = 17.0
    problem.coefficient_point_indices[2] = UInt16(local_point)
    problem.validate()
    var expected = evaluate_physical_objective(problem, problem.particles)
    var accelerator = DeviceWorkspace(problem)
    var dots = accelerator.slice_dots(problem.particles)
    for i in range(len(problem.slices)):
        assert_float64_device_equal(
            dots[i],
            packed_slice_dot(problem, problem.slices[i], problem.particles),
        )
    var actual = accelerator.evaluation(problem.particles)
    assert_float64_device_equal(actual.chi2(), expected.chi2)
    for point in range(point_count):
        assert_float64_device_equal(
            actual.gradient[point], expected.gradient[point]
        )


def test_accelerator_consumes_contiguous_uint16_indices() raises:
    check_contiguous_uint16_indices(300, 299)
    check_contiguous_uint16_indices(4097, 4096)


def test_accelerator_full_iterations_match_cpu() raises:
    var problem = build_biological_problem(0.02)
    problem.settings.include_dmax = True
    problem.settings.max_iterations = UInt32(4)
    problem.settings.grace_iterations = UInt32(10)
    problem.settings.epsilon = 0.0
    problem.settings.configured_step_factor = 1.0
    var expected = optimize_on_cpu(problem)
    var actual = optimize_on_device(problem)
    assert_equal(actual.iterations, expected.iterations)
    assert_equal(actual.stop_reason, expected.stop_reason)
    assert_equal(actual.backtracks, expected.backtracks)
    assert_float64_device_equal(actual.chi2, expected.chi2)
    for point in range(len(expected.particles)):
        assert_float64_device_equal(
            actual.particles[point], expected.particles[point]
        )


def test_accelerator_physical_iterations_match_cpu() raises:
    var problem = build_problem()
    problem.settings.include_dmax = True
    problem.voxels[0].maximum_dose_weight = 0.4
    problem.settings.max_iterations = UInt32(1)
    problem.settings.grace_iterations = UInt32(10)
    problem.settings.epsilon = 0.0
    problem.settings.configured_step_factor = 1.0
    var expected = optimize_on_cpu(problem)
    var actual = optimize_on_device(problem)
    assert_equal(actual.iterations, expected.iterations)
    assert_equal(actual.stop_reason, expected.stop_reason)
    assert_equal(actual.backtracks, expected.backtracks)
    assert_float64_device_equal(actual.chi2, expected.chi2)
    for point in range(len(expected.particles)):
        assert_float64_device_equal(
            actual.particles[point], expected.particles[point]
        )
        assert_float64_device_equal(
            actual.gradient[point], expected.gradient[point]
        )


def test_accelerator_metric_reduction_crosses_blocks() raises:
    comptime voxel_count = 530
    var low = List[BiologicalMatrixEntry]()
    var high = List[BiologicalMatrixEntry]()
    var objectives = List[VoxelObjective]()
    var lq = List[BioLQParams]()
    for voxel in range(voxel_count):
        low.append(
            BiologicalMatrixEntry(
                voxel, 0, 1.0, 8.0e7, 8.0e6, 0.0, 8.0e7, 4.0e8
            )
        )
        high.append(
            BiologicalMatrixEntry(
                voxel, 0, 1.0, 1.0e8, 1.0e7, 0.0, 1.0e8, 7.0e8
            )
        )
        objectives.append(VoxelObjective(20.0, 1.0, 1.0, 0.0, 0.05))
        lq.append(BioLQParams(0.1, 0.02, 30.0))
    var matrices = [
        BiologicalDoseMatrix(voxel_count, 1, low^),
        BiologicalDoseMatrix(voxel_count, 1, high^),
    ]
    var problem = pack_biological_problem(
        BiologicalScenarioSet(matrices^),
        objectives^,
        lq^,
        [10.0],
        OptimizerSettings.reference_defaults(),
        MinimumParticlePolicy.disabled(),
    )
    var accelerator = DeviceWorkspace(problem)
    var actual = accelerator.evaluation(problem.particles)
    var host_chi2 = 0.0
    var host_weighted = 0.0
    for voxel in range(voxel_count):
        host_chi2 += actual.chi2_per_voxel[voxel]
        host_weighted += actual.weighted_per_voxel[voxel]
    assert_reduction_close(actual.chi2(), host_chi2, voxel_count)
    assert_reduction_close(actual.weighted_dose2(), host_weighted, voxel_count)


def main() raises:
    comptime assert has_accelerator(), "accelerator test requires a GPU"
    comptime if has_accelerator():
        test_accelerator_slice_dots_match_cpu()
        test_accelerator_zero_gradient_matches_cpu()
        test_accelerator_biological_evaluation_matches_cpu()
        test_accelerator_physical_evaluation_matches_cpu()
        test_accelerator_multifield_matches_cpu()
        test_accelerator_flattened_4d_robust_matches_cpu()
        test_accelerator_preserves_nonmonotonic_slice_ranges()
        test_accelerator_consumes_contiguous_uint16_indices()
        test_accelerator_full_iterations_match_cpu()
        test_accelerator_physical_iterations_match_cpu()
        test_accelerator_metric_reduction_crosses_blocks()
        print("test_device_backend: PASS")
