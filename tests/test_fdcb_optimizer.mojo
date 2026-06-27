from std.testing import assert_equal, assert_true

from fdcb_optimizer import BioFDCBEntry, BioFDCBMatrix, BioFDCBScenarioSet, BioLQParams, FDCBScenarioSet, FDCBVoxelObjective, evaluate_robust_bio_fdcb, evaluate_robust_physical_fdcb, fdcb_exact_dmy_robust_physical, fdcb_fletcher_reeves_direction, fdcb_robust_physical_step, p101_fdcb_objectives_from_opt_voxels
from opt_voxels import OptVoxel, OptVoxelSet
from sparse_optimizer import SparseDoseEntry, SparseDoseMatrix


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def matrix(var entries: List[SparseDoseEntry]) -> SparseDoseMatrix:
    return SparseDoseMatrix(2, 2, entries^)


def main() raises:
    var nominal_entries = List[SparseDoseEntry]()
    nominal_entries.append(SparseDoseEntry(0, 0, 0.01))
    nominal_entries.append(SparseDoseEntry(0, 1, 0.02))
    nominal_entries.append(SparseDoseEntry(1, 0, 0.03))
    nominal_entries.append(SparseDoseEntry(1, 1, 0.01))
    var low_entries = List[SparseDoseEntry]()
    low_entries.append(SparseDoseEntry(0, 0, 0.008))
    low_entries.append(SparseDoseEntry(0, 1, 0.015))
    low_entries.append(SparseDoseEntry(1, 0, 0.025))
    low_entries.append(SparseDoseEntry(1, 1, 0.008))
    var high_entries = List[SparseDoseEntry]()
    high_entries.append(SparseDoseEntry(0, 0, 0.012))
    high_entries.append(SparseDoseEntry(0, 1, 0.025))
    high_entries.append(SparseDoseEntry(1, 0, 0.035))
    high_entries.append(SparseDoseEntry(1, 1, 0.012))

    var scenario_matrices = List[SparseDoseMatrix]()
    scenario_matrices.append(matrix(nominal_entries^))
    scenario_matrices.append(matrix(low_entries^))
    scenario_matrices.append(matrix(high_entries^))
    var scenarios = FDCBScenarioSet(scenario_matrices^)

    var objectives = List[FDCBVoxelObjective]()
    objectives.append(FDCBVoxelObjective(3.0, 1.0, 1.0, 0.0, 0.05))
    objectives.append(FDCBVoxelObjective(-2.0, 0.3, 1.0, 0.0, 0.05))
    var particles = [100.0, 50.0]

    var evaluation = evaluate_robust_physical_fdcb(scenarios, objectives, particles)
    assert_close(evaluation.dose_by_scenario[0][0], 2.0, 1.0e-12)
    assert_close(evaluation.dose_by_scenario[1][0], 1.55, 1.0e-12)
    assert_close(evaluation.dose_by_scenario[2][0], 2.45, 1.0e-12)
    assert_equal(evaluation.min_scenario[0], 1)
    assert_equal(evaluation.max_scenario[1], 2)
    assert_close(evaluation.chi2, (3.0 - 1.55) * (3.0 - 1.55) + (-0.3 * (2.0 - 4.1)) * (-0.3 * (2.0 - 4.1)), 1.0e-12)
    assert_close(evaluation.dose_p_weighted_avg2, 9.0 + 0.36, 1.0e-12)

    # Gradient follows TRPOptDosePhysFDCRobust: selected target min scenario
    # plus selected OAR max scenario. OAR underdose is ignored, overdose is not.
    var target_factor = (1.45 * 1.0) * 2.0
    var oar_factor = ((2.0 - 4.1) * 0.3) * (2.0 * 0.3)
    assert_close(evaluation.gradient[0], target_factor * 0.008 + oar_factor * 0.035, 1.0e-12)
    assert_close(evaluation.gradient[1], target_factor * 0.015 + oar_factor * 0.012, 1.0e-12)

    var dmy = fdcb_exact_dmy_robust_physical(scenarios, objectives, evaluation, evaluation.gradient)
    var target_r = evaluation.gradient[0] * 0.008 + evaluation.gradient[1] * 0.015
    var oar_r = evaluation.gradient[0] * 0.035 + evaluation.gradient[1] * 0.012
    var expected_dmy_num = 1.45 * target_r + (2.0 - 4.1) * 0.3 * (oar_r * 0.3)
    var expected_dmy_den = target_r * target_r + (oar_r * 0.3) * (oar_r * 0.3)
    assert_close(dmy, expected_dmy_num / expected_dmy_den, 1.0e-12)

    var step = fdcb_robust_physical_step(scenarios, objectives, particles, evaluation.gradient, 0.5, 0.0)
    assert_close(step.dmy, dmy, 1.0e-12)
    assert_close(step.particles[0], particles[0] + dmy * 0.5 * evaluation.gradient[0], 1.0e-12)
    assert_close(step.particles[1], particles[1] + dmy * 0.5 * evaluation.gradient[1], 1.0e-12)

    var prev_grad = [1.0, 2.0]
    var prev_dir = [0.5, 1.0]
    var grad = [2.0, 1.0]
    var direction = fdcb_fletcher_reeves_direction(grad, prev_grad, prev_dir)
    assert_close(direction[0], 2.0 + 0.5, 1.0e-12)
    assert_close(direction[1], 1.0 + 1.0, 1.0e-12)

    var voxels = List[OptVoxel]()
    voxels.append(OptVoxel(0, 0, 0, 1, True))
    voxels.append(OptVoxel(0, 0, 1, 2, False))
    var opt_voxels = OptVoxelSet(voxels^, 1, 1, 1, 1)
    var p101_objectives = p101_fdcb_objectives_from_opt_voxels(opt_voxels, 3.0, 1.0, 0.3, 0.9, True)
    assert_close(p101_objectives[0].dose_p, 3.0, 0.0)
    assert_close(p101_objectives[0].dose_w, 1.0, 0.0)
    assert_close(p101_objectives[0].dose_d, 1.0, 0.0)
    assert_close(p101_objectives[1].dose_p, -2.7, 1.0e-15)
    assert_close(p101_objectives[1].dose_w, 0.3, 0.0)
    assert_close(p101_objectives[1].dose_d, 1.0, 0.0)

    var bio_entries_low = List[BioFDCBEntry]()
    bio_entries_low.append(BioFDCBEntry(0, 0, 1.0, 8.0e7, 8.0e6, 0.0, 8.0e7, 0.0))
    var bio_entries_high = List[BioFDCBEntry]()
    bio_entries_high.append(BioFDCBEntry(0, 0, 1.0, 1.0e8, 1.0e7, 0.0, 1.0e8, 0.0))
    var bio_matrices = List[BioFDCBMatrix]()
    bio_matrices.append(BioFDCBMatrix(1, 1, bio_entries_low^))
    bio_matrices.append(BioFDCBMatrix(1, 1, bio_entries_high^))
    var bio_scenarios = BioFDCBScenarioSet(bio_matrices^)
    var bio_objectives = List[FDCBVoxelObjective]()
    bio_objectives.append(FDCBVoxelObjective(20.0, 1.0, 1.0, 0.0, 0.05))
    var lq = List[BioLQParams]()
    lq.append(BioLQParams(0.1, 0.0, 30.0))
    var bio_particles = [10.0]
    var bio_eval = evaluate_robust_bio_fdcb(bio_scenarios, bio_objectives, lq, bio_particles)
    assert_equal(bio_eval.min_scenario[0], 0)
    assert_equal(bio_eval.max_scenario[0], 1)
    assert_close(bio_eval.dose_min[0], 8.0e8 * 1.602189e-8, 1.0e-12)
    assert_close(bio_eval.dose_max[0], 1.0e9 * 1.602189e-8, 1.0e-12)
    assert_true(bio_eval.gradient[0] > 0.0)
