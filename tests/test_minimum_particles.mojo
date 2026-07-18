from std.testing import assert_equal, assert_true

from minimum_particles import (
    HostRandomState,
    apply_final_minimum_particle_limit,
    apply_minimum_particle_policy,
    update_with_minimum_particle_policy,
)
from optimization_problem import MINIMUM_PARTICLE_COMPLEX_HOST_RNG
from test_optimization_problem import build_problem


def test_reference_rng_sequence() raises:
    var rng = HostRandomState(UInt64(1))
    assert_equal(rng.next(), UInt32(1804289383))
    assert_equal(rng.next(), UInt32(846930886))
    assert_equal(rng.next(), UInt32(1681692777))
    assert_equal(rng.next(), UInt32(1714636915))
    assert_equal(rng.next(), UInt32(1957747793))


def test_complex_policy_is_seeded_and_repeatable() raises:
    var problem = build_problem()
    problem.minimum_particle_policy.kind = MINIMUM_PARTICLE_COMPLEX_HOST_RNG
    problem.field_slices[0].minimum_particles = 100.0
    problem.field_slices[0].raster_stride = UInt32(2)
    var first = [25.0, 50.0]
    var second = first.copy()
    var direction = [1.0, -1.0]
    var rng1 = HostRandomState(problem.minimum_particle_policy.seed)
    var rng2 = HostRandomState(problem.minimum_particle_policy.seed)
    var result1 = apply_minimum_particle_policy(
        problem, first, direction, 1.0, 1.0, rng1
    )
    var result2 = apply_minimum_particle_policy(
        problem, second, direction, 1.0, 1.0, rng2
    )
    assert_equal(first[0], second[0])
    assert_equal(first[1], second[1])
    assert_equal(result1.deleted, result2.deleted)
    assert_equal(result1.random_draws, result2.random_draws)
    assert_true(result1.random_draws > UInt64(0))


def test_rng_state_restores_exact_position() raises:
    var source = HostRandomState(UInt64(101))
    for _ in range(17):
        _ = source.next()
    var restored = HostRandomState(
        source.state.copy(), UInt32(source.front), UInt32(source.rear)
    )
    for _ in range(20):
        assert_equal(restored.next(), source.next())


def test_update_preserves_trip_point_order() raises:
    var problem = build_problem()
    problem.field_slices[0].minimum_particles = 100.0
    problem.field_slices[0].raster_stride = UInt32(2)
    var zero_state = List[UInt32]()
    zero_state.resize(31, UInt32(0))
    var rng = HostRandomState(zero_state^, UInt32(3), UInt32(0))
    var update = update_with_minimum_particle_policy(
        problem, [50.0, 120.0], [0.0, -1.0], 0.0, 1.0, 1.0, rng
    )
    assert_equal(update.particles[0], 100.0)
    assert_equal(update.particles[1], 120.0)


def test_final_complex_limit_matches_trip_threshold() raises:
    var problem = build_problem()
    problem.minimum_particle_policy.kind = MINIMUM_PARTICLE_COMPLEX_HOST_RNG
    problem.field_slices[0].minimum_particles = 100.0
    problem.point_active = [UInt8(1), UInt8(1)]
    var particles = [79.0, 80.0]
    var result = apply_final_minimum_particle_limit(problem, particles)
    assert_equal(particles[0], 0.0)
    assert_equal(particles[1], 100.0)
    assert_equal(result.changed, UInt64(2))
    assert_equal(result.deleted, UInt64(1))


def main() raises:
    test_reference_rng_sequence()
    test_complex_policy_is_seeded_and_repeatable()
    test_rng_state_restores_exact_position()
    test_update_preserves_trip_point_order()
    test_final_complex_limit_matches_trip_threshold()
