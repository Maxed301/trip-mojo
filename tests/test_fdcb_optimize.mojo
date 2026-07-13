from std.testing import assert_equal, assert_true

from fdcb_optimize import (
    FDCB_STOP_CHI_SQUARE_LIMIT,
    FDCB_STOP_MAX_ITERATIONS,
    evaluate_packed_iteration,
    fletcher_reeves_direction,
    initial_direction,
    optimize_packed_fdcb,
)
from fdcb_problem import (
    FDCBFieldSliceV1,
    FDCB_FLAG_INITIALIZE,
    evaluate_packed_physical_fdcb,
)
from test_fdcb_problem import build_problem


def test_full_iteration_reduces_chi2() raises:
    var problem = build_problem()
    problem.settings.max_iterations = UInt32(20)
    problem.settings.grace_iterations = UInt32(5)
    problem.settings.epsilon = 1.0e-12
    var initial = evaluate_packed_physical_fdcb(problem, problem.particles)
    var result = optimize_packed_fdcb(problem)
    assert_true(result.iterations > UInt32(0))
    assert_true(result.iterations <= problem.settings.max_iterations)
    assert_true(result.chi2 < initial.chi2)


def test_chi_square_limit_stops_without_changing_tolerance() raises:
    var problem = build_problem()
    problem.settings.max_iterations = UInt32(20)
    problem.settings.chi_square_limit = 100.0
    var result = optimize_packed_fdcb(problem)
    assert_equal(result.iterations, UInt32(1))
    assert_equal(result.stop_reason, FDCB_STOP_CHI_SQUARE_LIMIT)


def test_oversized_step_backtracks() raises:
    var problem = build_problem()
    problem.settings.max_iterations = UInt32(2)
    problem.settings.grace_iterations = UInt32(0)
    problem.settings.configured_step_factor = 100.0
    var initial = evaluate_packed_iteration(problem, problem.particles)
    var result = optimize_packed_fdcb(problem)
    assert_true(result.backtracks > UInt64(0))
    assert_true(result.chi2 < initial.chi2)


def test_max_iterations_are_not_capped() raises:
    var problem = build_problem()
    problem.settings.max_iterations = UInt32(7)
    problem.settings.grace_iterations = UInt32(100)
    problem.settings.epsilon = 0.0
    var result = optimize_packed_fdcb(problem)
    assert_equal(result.iterations, UInt32(7))
    assert_equal(result.stop_reason, FDCB_STOP_MAX_ITERATIONS)


def test_complex_minimum_particles_are_repeatable() raises:
    var problem = build_problem()
    problem.settings.max_iterations = UInt32(4)
    problem.settings.grace_iterations = UInt32(10)
    problem.settings.epsilon = 0.0
    problem.field_slices[0].minimum_particles = 120.0
    problem.field_slices[0].raster_stride = UInt32(2)
    var first = optimize_packed_fdcb(problem)
    var second = optimize_packed_fdcb(problem)
    assert_equal(first.particles[0], second.particles[0])
    assert_equal(first.particles[1], second.particles[1])
    assert_equal(first.random_draws, second.random_draws)
    assert_true(first.random_draws > UInt64(0))


def test_initialization_is_not_counted_as_trip_iteration() raises:
    var problem = build_problem()
    problem.particles[0] = 0.0
    problem.particles[1] = 0.0
    problem.settings.flags |= FDCB_FLAG_INITIALIZE
    problem.settings.max_iterations = UInt32(1)
    var initial = evaluate_packed_iteration(problem, problem.particles)
    assert_equal(initial.gradient_norm, 0.0)
    var result = optimize_packed_fdcb(problem)
    assert_equal(result.iterations, UInt32(1))
    assert_true(result.particles[0] != 0.0 or result.particles[1] != 0.0)
    assert_true(result.chi2 < initial.chi2)


def test_initialization_prefers_packed_host_direction() raises:
    var problem = build_problem()
    problem.settings.flags |= FDCB_FLAG_INITIALIZE
    problem.initial_direction[0] = 7.0
    problem.initial_direction[1] = 11.0
    var direction = initial_direction(problem, [3.0, 5.0])
    assert_equal(direction[0], 7.0)
    assert_equal(direction[1], 11.0)


def test_fletcher_reeves_preserves_trip_field_weighting() raises:
    var problem = build_problem()
    problem.field_count = UInt32(2)
    problem.field_slices = [
        FDCBFieldSliceV1(0, 0, 0, 1, 0, 0.0),
        FDCBFieldSliceV1(1, 0, 1, 1, 0, 0.0),
    ]
    var direction = fletcher_reeves_direction(
        problem, [2.0, 3.0], [1.0, 2.0], [1.0, 1.0]
    )
    var gamma = 17.0 / 6.0
    assert_equal(direction[0], 2.0 + gamma)
    assert_equal(direction[1], 3.0 + gamma)


def main() raises:
    test_full_iteration_reduces_chi2()
    test_max_iterations_are_not_capped()
    test_complex_minimum_particles_are_repeatable()
    test_initialization_is_not_counted_as_trip_iteration()
    test_initialization_prefers_packed_host_direction()
    test_fletcher_reeves_preserves_trip_field_weighting()
    test_chi_square_limit_stops_without_changing_tolerance()
    test_oversized_step_backtracks()
