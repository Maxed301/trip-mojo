from std.sys import has_accelerator
from std.testing import assert_equal, assert_true

from fdcb_accelerator import FDCBAccelerator, FDCB_ACCELERATOR_MIXED32
from fdcb_cpu import evaluate_packed_biological_fdcb
from fdcb_problem import FDCB_PRECISION_MIXED32, evaluate_packed_physical_fdcb
from test_fdcb_cpu import build_biological_problem
from test_fdcb_problem import build_problem


def assert_mixed32_close(actual: Float64, expected: Float64) raises:
    var scale = abs(expected)
    if scale < 1.0:
        scale = 1.0
    assert_true(abs(actual - expected) <= scale * 64.0 * 1.1920929e-7)


def main() raises:
    comptime if has_accelerator() and FDCB_ACCELERATOR_MIXED32:
        var problem = build_biological_problem(0.02)
        var expected = evaluate_packed_biological_fdcb(
            problem, problem.particles
        )
        problem.settings.precision_mode = FDCB_PRECISION_MIXED32
        var accelerator = FDCBAccelerator(problem)
        var actual = accelerator.evaluation(problem.particles)
        assert_equal(actual.min_scenario[0], expected.min_scenario[0])
        assert_equal(actual.max_scenario[0], expected.max_scenario[0])
        assert_mixed32_close(actual.dose_min[0], expected.dose_min[0])
        assert_mixed32_close(actual.dose_max[0], expected.dose_max[0])
        assert_mixed32_close(actual.chi2(), expected.chi2)
        assert_mixed32_close(actual.gradient[0], expected.gradient[0])

        var physical = build_problem()
        physical.voxels[0].initial_dose = 0.25
        physical.voxels[0].prescribed_let = 1.0
        physical.voxels[0].let_weight = 100.0
        physical.scenario_states[0].dose_minor = 1.0e6
        physical.scenario_states[1].dose_minor = 2.0e6
        var expected_physical = evaluate_packed_physical_fdcb(
            physical, physical.particles
        )
        physical.settings.precision_mode = FDCB_PRECISION_MIXED32
        accelerator = FDCBAccelerator(physical)
        actual = accelerator.evaluation(physical.particles)
        assert_equal(actual.min_scenario[0], expected_physical.min_scenario[0])
        assert_equal(actual.max_scenario[0], expected_physical.max_scenario[0])
        assert_mixed32_close(actual.dose_min[0], expected_physical.dose_min[0])
        assert_mixed32_close(actual.dose_max[0], expected_physical.dose_max[0])
        assert_mixed32_close(actual.chi2(), expected_physical.chi2)
        assert_mixed32_close(actual.gradient[0], expected_physical.gradient[0])
