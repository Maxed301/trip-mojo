from std.testing import assert_equal, assert_raises, assert_true

from fdcb_cpu import (
    evaluate_packed_biological_fdcb,
    fdcb_exact_step_biological,
)
from fdcb_optimizer import (
    BioFDCBEntry,
    BioFDCBMatrix,
    BioFDCBScenarioSet,
    BioLQParams,
    FDCBVoxelObjective,
    evaluate_robust_bio_fdcb,
)
from fdcb_optimize import optimize_packed_fdcb
from fdcb_packing import (
    FDCBNativeFieldSlice,
    pack_biological_fdcb_problem_v1,
    pack_biological_fdcb_problem_v1_with_fields,
)
from fdcb_problem import (
    FDCBMinimumParticlePolicyV1,
    FDCBProblemV1,
    FDCBSettingsV1,
    FDCB_MEV_TO_GY,
)


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    assert_true(abs(actual - expected) <= tolerance)


def build_biological_problem(beta: Float64 = 0.0) raises -> FDCBProblemV1:
    var low = [BioFDCBEntry(0, 0, 1.0, 8.0e7, 8.0e6, 0.0, 8.0e7, 4.0e8)]
    var high = [BioFDCBEntry(0, 0, 1.0, 1.0e8, 1.0e7, 0.0, 1.0e8, 7.0e8)]
    var matrices = [
        BioFDCBMatrix(1, 1, low^),
        BioFDCBMatrix(1, 1, high^),
    ]
    var scenarios = BioFDCBScenarioSet(matrices^)
    var objectives = [FDCBVoxelObjective(20.0, 1.0, 1.0, 0.25, 0.05)]
    var lq = [BioLQParams(0.1, beta, 30.0)]
    var particles = [10.0]
    return pack_biological_fdcb_problem_v1(
        scenarios,
        objectives,
        lq,
        particles,
        FDCBSettingsV1.reference_defaults(),
        FDCBMinimumParticlePolicyV1.disabled(),
    )


def test_biological_dose_gradient_and_selection() raises:
    var problem = build_biological_problem()
    var evaluation = evaluate_packed_biological_fdcb(problem, problem.particles)
    assert_equal(evaluation.min_scenario[0], Int32(0))
    assert_equal(evaluation.max_scenario[0], Int32(1))
    assert_close(evaluation.dose_min[0], 8.0e8 * FDCB_MEV_TO_GY, 1.0e-12)
    assert_close(evaluation.dose_max[0], 1.0e9 * FDCB_MEV_TO_GY, 1.0e-12)
    var residual = 20.0 - evaluation.dose_min[0]
    var expected_gradient = residual * 2.0 * (8.0e6 * FDCB_MEV_TO_GY / 0.1)
    assert_close(evaluation.gradient[0], expected_gradient, 1.0e-12)

    var zeroed = [0.0]
    var zeroed_evaluation = evaluate_packed_biological_fdcb(problem, zeroed)
    assert_equal(zeroed_evaluation.gradient[0], 0.0)

    var step = fdcb_exact_step_biological(
        problem, evaluation, evaluation.gradient
    )
    var response = evaluation.gradient[0] * 8.0e7 * FDCB_MEV_TO_GY
    assert_close(step, residual / response, 1.0e-14)

    problem.settings.max_iterations = UInt32(4)
    problem.settings.grace_iterations = UInt32(10)
    problem.settings.epsilon = 0.0
    var optimized = optimize_packed_fdcb(problem)
    assert_true(optimized.iterations > UInt32(0))
    assert_true(optimized.chi2 < evaluation.chi2)


def test_biological_validation_rejects_zero_rbe_model() raises:
    var problem = build_biological_problem()
    problem.voxels[0].rbe_alpha = 0.0
    problem.voxels[0].rbe_slope_max = 0.0
    with assert_raises():
        problem.validate()


def test_multifield_biological_packing_matches_native() raises:
    var low = [
        BioFDCBEntry(
            0, 2, 1.0, 67108864.0, 8388608.0, 0.0, 67108864.0, 134217728.0
        ),
        BioFDCBEntry(
            0, 0, 1.0, 33554432.0, 4194304.0, 0.0, 33554432.0, 67108864.0
        ),
        BioFDCBEntry(
            0, 1, 2.0, 33554432.0, 4194304.0, 0.0, 33554432.0, 67108864.0
        ),
    ]
    var high = [
        BioFDCBEntry(
            0, 2, 1.0, 134217728.0, 16777216.0, 0.0, 134217728.0, 268435456.0
        ),
        BioFDCBEntry(
            0, 0, 1.0, 67108864.0, 8388608.0, 0.0, 67108864.0, 134217728.0
        ),
        BioFDCBEntry(
            0, 1, 2.0, 67108864.0, 8388608.0, 0.0, 67108864.0, 134217728.0
        ),
    ]
    var matrices = [
        BioFDCBMatrix(1, 3, low^),
        BioFDCBMatrix(1, 3, high^),
    ]
    var scenarios = BioFDCBScenarioSet(matrices^)
    var objectives = [FDCBVoxelObjective(100.0, 1.0, 1.0, 0.0, 0.05)]
    var lq = [BioLQParams(0.1, 0.0, 100.0)]
    var particles = [4.0, 3.0, 2.0]
    var fields = [
        FDCBNativeFieldSlice(0, 0, 0, 2, 2, 1.0),
        FDCBNativeFieldSlice(1, 0, 2, 1, 1, 2.0),
    ]
    var native = evaluate_robust_bio_fdcb(scenarios, objectives, lq, particles)
    var problem = pack_biological_fdcb_problem_v1_with_fields(
        scenarios,
        objectives,
        lq,
        fields,
        particles,
        FDCBSettingsV1.reference_defaults(),
        FDCBMinimumParticlePolicyV1.disabled(),
    )
    var packed = evaluate_packed_biological_fdcb(problem, problem.particles)
    assert_equal(problem.field_count, UInt32(2))
    assert_equal(len(problem.field_slices), 2)
    assert_equal(len(problem.slices), 4)
    assert_equal(problem.voxel_scenarios[0].slice_count, UInt32(2))
    assert_equal(problem.slices[0].coefficient_count, UInt32(2))
    assert_equal(problem.slices[1].coefficient_count, UInt32(1))
    assert_equal(problem.coefficient_point_indices[0], UInt16(0))
    assert_equal(problem.coefficient_point_indices[1], UInt16(1))
    assert_equal(problem.coefficient_point_indices[2], UInt16(0))
    assert_equal(packed.min_scenario[0], Int32(native.min_scenario[0]))
    assert_equal(packed.max_scenario[0], Int32(native.max_scenario[0]))
    assert_close(packed.dose_min[0], native.dose_min[0], 1.0e-12)
    assert_close(packed.dose_max[0], native.dose_max[0], 1.0e-12)
    for point in range(len(particles)):
        assert_close(packed.gradient[point], native.gradient[point], 1.0e-12)

    scenarios.matrices[0].entries[2].ddd = 67108864.0
    with assert_raises():
        _ = pack_biological_fdcb_problem_v1_with_fields(
            scenarios,
            objectives,
            lq,
            fields,
            particles,
            FDCBSettingsV1.reference_defaults(),
            FDCBMinimumParticlePolicyV1.disabled(),
        )


def test_let_objective_uses_reference_activation_threshold() raises:
    var problem = build_biological_problem()
    problem.scenario_states[0].let_mix_minor = 1.0e7
    problem.scenario_states[1].let_mix_minor = 1.0e7
    var without_let = evaluate_packed_biological_fdcb(
        problem, problem.particles
    )
    problem.voxels[0].prescribed_let = 6.0
    problem.voxels[0].let_weight = 0.5
    var with_let = evaluate_packed_biological_fdcb(problem, problem.particles)
    assert_true(with_let.chi2 > without_let.chi2)
    assert_true(with_let.gradient[0] != without_let.gradient[0])
    problem.voxels[0].let_weight = 1.0e-17
    var below_epsilon = evaluate_packed_biological_fdcb(
        problem, problem.particles
    )
    assert_equal(below_epsilon.chi2, without_let.chi2)
    assert_equal(below_epsilon.gradient[0], without_let.gradient[0])


def main() raises:
    test_biological_dose_gradient_and_selection()
    test_biological_validation_rejects_zero_rbe_model()
    test_multifield_biological_packing_matches_native()
    test_let_objective_uses_reference_activation_threshold()
