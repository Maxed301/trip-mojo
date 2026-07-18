from std.testing import assert_equal, assert_raises, assert_true

from cpu_backend import compute_physical_exact_step, packed_directional_dose
from robust_objective import PhysicalScenarioSet, VoxelObjective
from problem_packing import (
    NativeFieldSlice,
    pack_physical_problem,
    pack_physical_problem_with_fields,
)
from optimization_problem import (
    MinimumParticlePolicy,
    OptimizationProblem,
    OptimizerSettings,
    MINIMUM_PARTICLE_COMPLEX_HOST_RNG,
    MEV_TO_GY,
    evaluate_physical_objective,
    compute_forward_dose,
)
from sparse_optimizer import SparseDoseEntry, SparseDoseMatrix


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var difference = actual - expected
    if difference < 0.0:
        difference = -difference
    if difference > tolerance:
        print("assert_close", actual, expected, difference, tolerance)
    assert_true(difference <= tolerance)


def packed_physical_value(value: Float64) -> Float64:
    return value * (1.0 / MEV_TO_GY) * MEV_TO_GY


def build_problem() raises -> OptimizationProblem:
    var low_entries = [
        SparseDoseEntry(0, 0, 0.008),
        SparseDoseEntry(0, 1, 0.015),
        SparseDoseEntry(1, 0, 0.025),
        SparseDoseEntry(1, 1, 0.008),
    ]
    var high_entries = [
        SparseDoseEntry(0, 0, 0.012),
        SparseDoseEntry(0, 1, 0.025),
        SparseDoseEntry(1, 0, 0.035),
        SparseDoseEntry(1, 1, 0.012),
    ]
    var matrices = [
        SparseDoseMatrix(2, 2, low_entries^),
        SparseDoseMatrix(2, 2, high_entries^),
    ]
    var scenarios = PhysicalScenarioSet(matrices^)
    var objectives = [
        VoxelObjective(3.0, 1.0, 1.0, 0.0, 0.05),
        VoxelObjective(-2.0, 0.3, 1.0, 0.0, 0.05),
    ]
    var particles = [100.0, 50.0]
    return pack_physical_problem(
        scenarios,
        objectives,
        particles,
        OptimizerSettings.reference_defaults(),
        MinimumParticlePolicy.complex_host_rng(UInt64(101)),
    )


def build_multifield_problem() raises -> OptimizationProblem:
    var low_entries = [
        SparseDoseEntry(0, 4, 0.005),
        SparseDoseEntry(0, 0, 0.010),
        SparseDoseEntry(0, 2, 0.020),
        SparseDoseEntry(0, 3, 0.030),
        SparseDoseEntry(0, 1, 0.040),
    ]
    var high_entries = [
        SparseDoseEntry(0, 4, 0.010),
        SparseDoseEntry(0, 0, 0.020),
        SparseDoseEntry(0, 2, 0.040),
        SparseDoseEntry(0, 3, 0.060),
        SparseDoseEntry(0, 1, 0.080),
    ]
    var matrices = [
        SparseDoseMatrix(1, 5, low_entries^),
        SparseDoseMatrix(1, 5, high_entries^),
    ]
    var fields = [
        NativeFieldSlice(0, 0, 0, 2, 2, 1.0),
        NativeFieldSlice(0, 1, 2, 1, 1, 2.0),
        NativeFieldSlice(1, 0, 3, 2, 2, 3.0),
    ]
    var objectives = [VoxelObjective(10.0, 1.0, 1.0, 0.0, 0.05)]
    return pack_physical_problem_with_fields(
        PhysicalScenarioSet(matrices^),
        objectives,
        fields,
        [10.0, 20.0, 30.0, 40.0, 50.0],
        OptimizerSettings.reference_defaults(),
        MinimumParticlePolicy.disabled(),
    )


def test_layout() raises:
    var problem = build_problem()
    assert_equal(
        problem.minimum_particle_policy.kind, MINIMUM_PARTICLE_COMPLEX_HOST_RNG
    )
    assert_equal(problem.minimum_particle_policy.seed, UInt64(101))
    assert_equal(len(problem.field_slices), 1)
    assert_equal(problem.field_slices[0].point_offset, UInt64(0))
    assert_equal(problem.field_slices[0].point_count, UInt32(2))
    assert_equal(len(problem.voxel_scenarios), 4)
    assert_equal(len(problem.slices), 4)
    assert_equal(problem.voxels[1].scenario_offset, UInt64(2))
    assert_equal(problem.voxel_scenarios[2].slice_offset, UInt64(2))
    assert_equal(problem.slices[0].coefficient_offset, UInt64(0))
    assert_equal(problem.slices[1].coefficient_offset, UInt64(2))
    assert_equal(len(problem.coefficients), 8)
    assert_equal(problem.coefficients[0], 0.008)
    assert_true(problem.coefficients[0] != Float64(Float32(0.008)))
    assert_equal(problem.slices[0].dose_coefficient, 1.0 / MEV_TO_GY)


def test_multifield_packing_layout_and_math() raises:
    var problem = build_multifield_problem()
    assert_equal(problem.field_count, UInt32(2))
    assert_equal(len(problem.field_slices), 3)
    assert_equal(problem.field_slices[1].field_index, UInt32(0))
    assert_equal(problem.field_slices[1].beam_index, UInt32(1))
    assert_equal(problem.field_slices[1].point_offset, UInt64(2))
    assert_equal(problem.field_slices[2].field_index, UInt32(1))
    assert_equal(problem.field_slices[2].minimum_particles, 3.0)
    assert_equal(len(problem.slices), 6)
    assert_equal(problem.voxel_scenarios[0].slice_count, UInt32(3))
    assert_equal(problem.voxel_scenarios[1].slice_offset, UInt64(3))
    assert_equal(problem.slices[0].coefficient_count, UInt32(2))
    assert_equal(problem.slices[1].coefficient_count, UInt32(1))
    assert_equal(problem.slices[2].coefficient_count, UInt32(2))
    assert_equal(problem.coefficient_point_indices[0], UInt16(0))
    assert_equal(problem.coefficient_point_indices[1], UInt16(1))
    assert_equal(problem.coefficient_point_indices[3], UInt16(1))
    assert_equal(problem.coefficient_point_indices[4], UInt16(0))

    var evaluation = evaluate_physical_objective(problem, problem.particles)
    var low = (
        packed_physical_value(0.010) * 10.0
        + packed_physical_value(0.040) * 20.0
        + packed_physical_value(0.020) * 30.0
        + packed_physical_value(0.005) * 50.0
        + packed_physical_value(0.030) * 40.0
    )
    var high = low * 2.0
    assert_close(evaluation.dose_min[0], low, 1.0e-14)
    assert_close(evaluation.dose_max[0], high, 1.0e-14)
    assert_equal(evaluation.min_scenario[0], Int32(0))
    assert_equal(evaluation.max_scenario[0], Int32(1))
    var factor = (10.0 - low) * 2.0
    var slice_factor = factor * (1.0 / MEV_TO_GY) * MEV_TO_GY
    assert_close(
        evaluation.gradient[0],
        slice_factor * 0.010,
        1.0e-14,
    )
    assert_close(
        evaluation.gradient[4],
        slice_factor * 0.005,
        1.0e-14,
    )


def test_validation_rejects_bad_ranges() raises:
    var problem = build_problem()
    problem.slices[0].coefficient_count = UInt32(100)
    with assert_raises():
        problem.validate()

    problem = build_problem()
    problem.host_rng_state.append(UInt32(0))
    with assert_raises():
        problem.validate()

    problem = build_problem()
    problem.voxel_scenarios[1].slice_offset = UInt64(len(problem.slices))
    with assert_raises():
        problem.validate()

    problem = build_problem()
    problem.slices[1].coefficient_offset = UInt64(len(problem.coefficients))
    with assert_raises():
        problem.validate()

    problem = build_problem()
    problem.voxels[0].prescribed_dose = 0.0
    problem.voxels[0].initial_min_scenario = Int32(2)
    with assert_raises():
        problem.validate()

    problem = build_problem()
    problem.field_slices[0].point_count = UInt32(UInt16.MAX) + UInt32(2)
    with assert_raises():
        problem.validate()

    problem = build_problem()
    problem.field_count = UInt32(2)
    with assert_raises():
        problem.validate()

    problem = build_problem()
    problem.field_slices[0].minimum_particles = 1.0
    with assert_raises():
        problem.validate()
    problem.field_slices[0].raster_stride = UInt32(1)
    problem.validate()


def test_native_field_layout_validation() raises:
    var entries = [SparseDoseEntry(0, 0, 0.1)]
    var matrices = [SparseDoseMatrix(1, 2, entries^)]
    var scenarios = PhysicalScenarioSet(matrices^)
    var objectives = [VoxelObjective(1.0, 1.0, 1.0, 0.0, 0.05)]
    var fields = [
        NativeFieldSlice(0, 0, 0, 1, 1, 0.0),
        NativeFieldSlice(2, 0, 1, 1, 1, 0.0),
    ]
    with assert_raises():
        _ = pack_physical_problem_with_fields(
            scenarios,
            objectives,
            fields,
            [1.0, 1.0],
            OptimizerSettings.reference_defaults(),
            MinimumParticlePolicy.disabled(),
        )


def test_nonmonotonic_coefficient_ranges() raises:
    var problem = build_problem()
    var first = problem.slices[0].coefficient_offset
    problem.slices[0].coefficient_offset = problem.slices[1].coefficient_offset
    problem.slices[1].coefficient_offset = first
    problem.validate()


def test_sparse_forward_and_robust_selection() raises:
    var problem = build_problem()
    var dose = compute_forward_dose(problem, problem.particles)
    assert_close(
        dose[0],
        packed_physical_value(0.008) * 100.0
        + packed_physical_value(0.015) * 50.0,
        1.0e-14,
    )
    assert_close(
        dose[1],
        packed_physical_value(0.012) * 100.0
        + packed_physical_value(0.025) * 50.0,
        1.0e-14,
    )
    var evaluation = evaluate_physical_objective(problem, problem.particles)
    assert_equal(evaluation.min_scenario[0], Int32(0))
    assert_equal(evaluation.max_scenario[0], Int32(1))
    assert_equal(evaluation.min_scenario[1], Int32(0))
    assert_equal(evaluation.max_scenario[1], Int32(1))

    problem.voxels[0].prescribed_dose = 0.0
    problem.voxels[0].initial_min_scenario = Int32(1)
    problem.voxels[0].initial_max_scenario = Int32(0)
    evaluation = evaluate_physical_objective(problem, problem.particles)
    assert_equal(evaluation.dose_min[0], 0.0)
    assert_equal(evaluation.dose_max[0], 0.0)
    assert_equal(evaluation.min_scenario[0], Int32(1))
    assert_equal(evaluation.max_scenario[0], Int32(0))


def test_gradient_backprojection() raises:
    var problem = build_problem()
    var evaluation = evaluate_physical_objective(problem, problem.particles)
    var target_residual = 3.0 - evaluation.dose_min[0]
    var oar_residual = 2.0 - evaluation.dose_max[1]
    var target_factor = target_residual * 2.0
    var oar_factor = (oar_residual * 0.3) * (2.0 * 0.3)
    var dose_coefficient = 1.0 / MEV_TO_GY
    var target_scale = target_factor * dose_coefficient * MEV_TO_GY
    var oar_scale = oar_factor * dose_coefficient * MEV_TO_GY
    var expected0 = target_scale * 0.008 + oar_scale * 0.035
    var expected1 = target_scale * 0.015 + oar_scale * 0.012
    assert_close(evaluation.gradient[0], expected0, 1.0e-14)
    assert_close(evaluation.gradient[1], expected1, 1.0e-14)

    var zeroed = [0.0, problem.particles[1]]
    evaluation = evaluate_physical_objective(problem, zeroed)
    assert_equal(evaluation.gradient[0], 0.0)
    assert_true(evaluation.gradient[1] != 0.0)


def build_flattened_4d_robust_problem() raises -> OptimizationProblem:
    # Rows 0/1 are motion state 0 and rows 2/3 are motion state 1. The packed
    # optimizer intentionally sees four voxels; robust scenarios remain the
    # inner, voxel-major axis and select independently for every state row.
    var scenario0 = [
        SparseDoseEntry(0, 0, 0.10),
        SparseDoseEntry(1, 1, 0.20),
        SparseDoseEntry(2, 0, 0.30),
        SparseDoseEntry(3, 1, 0.40),
    ]
    var scenario1 = [
        SparseDoseEntry(0, 0, 0.20),
        SparseDoseEntry(1, 1, 0.10),
        SparseDoseEntry(2, 0, 0.10),
        SparseDoseEntry(3, 1, 0.50),
    ]
    var scenario2 = [
        SparseDoseEntry(0, 0, 0.15),
        SparseDoseEntry(1, 1, 0.30),
        SparseDoseEntry(2, 0, 0.40),
        SparseDoseEntry(3, 1, 0.20),
    ]
    var matrices = [
        SparseDoseMatrix(4, 2, scenario0^),
        SparseDoseMatrix(4, 2, scenario1^),
        SparseDoseMatrix(4, 2, scenario2^),
    ]
    var objectives = [
        VoxelObjective(4.0, 1.0, 1.0, 0.0, 0.05),
        VoxelObjective(4.0, 1.0, 1.0, 0.0, 0.05),
        VoxelObjective(4.0, 1.0, 1.0, 0.0, 0.05),
        VoxelObjective(4.0, 1.0, 1.0, 0.0, 0.05),
    ]
    return pack_physical_problem(
        PhysicalScenarioSet(matrices^),
        objectives,
        [10.0, 20.0],
        OptimizerSettings.reference_defaults(),
        MinimumParticlePolicy.disabled(),
    )


def test_flattened_4d_states_keep_robust_axis_independent() raises:
    var problem = build_flattened_4d_robust_problem()
    assert_equal(len(problem.voxels), 4)
    assert_equal(len(problem.voxel_scenarios), 12)
    assert_equal(problem.voxels[2].scenario_offset, UInt64(6))

    var evaluation = evaluate_physical_objective(problem, problem.particles)
    assert_equal(evaluation.min_scenario[0], Int32(0))
    assert_equal(evaluation.min_scenario[1], Int32(1))
    assert_equal(evaluation.min_scenario[2], Int32(1))
    assert_equal(evaluation.min_scenario[3], Int32(2))
    assert_equal(evaluation.max_scenario[0], Int32(1))
    assert_equal(evaluation.max_scenario[1], Int32(2))
    assert_equal(evaluation.max_scenario[2], Int32(2))
    assert_equal(evaluation.max_scenario[3], Int32(1))
    assert_true(evaluation.gradient[0] != 0.0)
    assert_true(evaluation.gradient[1] != 0.0)


def test_exact_directional_reduction() raises:
    var problem = build_problem()
    var evaluation = evaluate_physical_objective(problem, problem.particles)
    var step = compute_physical_exact_step(
        problem, evaluation, evaluation.gradient
    )
    var target_response = packed_directional_dose(
        problem,
        problem.voxels[0],
        Int(evaluation.min_scenario[0]),
        evaluation.gradient,
    )
    var oar_response = packed_directional_dose(
        problem,
        problem.voxels[1],
        Int(evaluation.max_scenario[1]),
        evaluation.gradient,
    )
    var target_residual = 3.0 - evaluation.dose_min[0]
    var oar_residual = 2.0 - evaluation.dose_max[1]
    var target_weighted = (
        target_response
        * problem.voxels[0].dose_weight
        / problem.voxels[0].dose_divisor
    )
    var oar_weighted = (
        oar_response
        * problem.voxels[1].dose_weight
        / problem.voxels[1].dose_divisor
    )
    var numerator = (
        target_residual
        * problem.voxels[0].dose_weight
        / problem.voxels[0].dose_divisor
        * target_weighted
    ) + (
        oar_residual
        * problem.voxels[1].dose_weight
        / problem.voxels[1].dose_divisor
        * oar_weighted
    )
    var denominator = (
        target_weighted * target_weighted + oar_weighted * oar_weighted
    )
    assert_close(step, numerator / denominator, 1.0e-14)


def main() raises:
    test_layout()
    test_multifield_packing_layout_and_math()
    test_validation_rejects_bad_ranges()
    test_native_field_layout_validation()
    test_nonmonotonic_coefficient_ranges()
    test_sparse_forward_and_robust_selection()
    test_gradient_backprojection()
    test_flattened_4d_states_keep_robust_axis_independent()
    test_exact_directional_reduction()
