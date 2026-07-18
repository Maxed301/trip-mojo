"""Deterministic host-side minimum-particle policy."""

from optimization_problem import (
    FieldSlice,
    OptimizationProblem,
    MINIMUM_PARTICLE_COMPLEX_HOST_RNG,
    MINIMUM_PARTICLE_DISABLED,
)
from reference_math import reference_exp


@fieldwise_init
struct HostRandomState(Copyable, Movable):
    """The seeded 31-word generator used by the reference libc rand()."""

    var state: List[UInt32]
    var front: Int
    var rear: Int

    def __init__(out self, seed: UInt64):
        self.state = List[UInt32]()
        self.state.resize(31, UInt32(0))
        var word = seed & UInt64(0xFFFFFFFF)
        if word == UInt64(0):
            word = UInt64(1)
        self.state[0] = UInt32(word)
        for i in range(1, 31):
            word = (UInt64(16807) * word) % UInt64(2147483647)
            self.state[i] = UInt32(word)
        self.front = 3
        self.rear = 0
        for _ in range(310):
            _ = self.next()

    def __init__(
        out self, var state: List[UInt32], front: UInt32, rear: UInt32
    ) raises:
        if len(state) != 31 or front >= UInt32(31) or rear >= UInt32(31):
            raise Error("invalid restored optimizer host RNG state")
        self.state = state^
        self.front = Int(front)
        self.rear = Int(rear)

    def next(mut self) -> UInt32:
        var value = UInt32(
            (UInt64(self.state[self.front]) + UInt64(self.state[self.rear]))
            & UInt64(0xFFFFFFFF)
        )
        self.state[self.front] = value
        self.front += 1
        self.rear += 1
        if self.front == 31:
            self.front = 0
        if self.rear == 31:
            self.rear = 0
        return (value >> 1) & UInt32(0x7FFFFFFF)

    def less_than(mut self, probability: Float64) -> Bool:
        return Float64(self.next()) < 2147483647.0 * probability


def make_host_rng(problem: OptimizationProblem) raises -> HostRandomState:
    if len(problem.host_rng_state) == 31:
        return HostRandomState(
            problem.host_rng_state.copy(),
            problem.minimum_particle_policy.rng_front,
            problem.minimum_particle_policy.rng_rear,
        )
    return HostRandomState(problem.minimum_particle_policy.seed)


@fieldwise_init
struct MinimumParticleUpdate(Copyable, Movable):
    var deleted: UInt64
    var random_draws: UInt64


@fieldwise_init
struct ParticleUpdate(Copyable, Movable):
    var particles: List[Float64]
    var minimum_particles: MinimumParticleUpdate


@fieldwise_init
struct FinalMinimumParticleUpdate(Copyable, Movable):
    var changed: UInt64
    var deleted: UInt64


def apply_final_minimum_particle_limit(
    problem: OptimizationProblem, mut particles: List[Float64]
) raises -> FinalMinimumParticleUpdate:
    """Apply TRiP's non-random post-optimization complexminp clamp."""
    if len(particles) != len(problem.particles):
        raise Error("minimum-particle vector length mismatch")
    if (
        problem.minimum_particle_policy.kind
        != MINIMUM_PARTICLE_COMPLEX_HOST_RNG
    ):
        return FinalMinimumParticleUpdate(UInt64(0), UInt64(0))
    var changed = UInt64(0)
    var deleted = UInt64(0)
    for field_slice in problem.field_slices:
        var base = Int(field_slice.point_offset)
        for local_point in range(Int(field_slice.point_count)):
            var point = base + local_point
            var value = particles[point]
            if (
                problem.point_active[point] == UInt8(0)
                or value == 0.0
                or value >= field_slice.minimum_particles
            ):
                continue
            changed += UInt64(1)
            if value < 0.8 * field_slice.minimum_particles:
                particles[point] = 0.0
                deleted += UInt64(1)
            else:
                particles[point] = field_slice.minimum_particles
    return FinalMinimumParticleUpdate(changed, deleted)


def apply_minimum_particle_policy(
    problem: OptimizationProblem,
    mut particles: List[Float64],
    direction: List[Float64],
    gradient_limit: Float64,
    iteration_probability: Float64,
    mut rng: HostRandomState,
) raises -> MinimumParticleUpdate:
    if len(particles) != len(problem.particles) or len(direction) != len(
        particles
    ):
        raise Error("minimum-particle vector length mismatch")
    var deleted = UInt64(0)
    var draws = UInt64(0)
    for slice_index in range(len(problem.field_slices)):
        var field_slice = problem.field_slices[slice_index].copy()
        var base = Int(field_slice.point_offset)
        for local_point in range(Int(field_slice.point_count)):
            var point = base + local_point
            if (
                problem.point_active[point] == UInt8(0)
                or particles[point] >= field_slice.minimum_particles
            ):
                continue
            var result = handle_minimum_particle(
                problem,
                field_slice,
                local_point,
                direction,
                gradient_limit,
                iteration_probability,
                particles,
                rng,
            )
            deleted += result.deleted
            draws += result.random_draws
    return MinimumParticleUpdate(deleted, draws)


def update_with_minimum_particle_policy(
    problem: OptimizationProblem,
    baseline: List[Float64],
    direction: List[Float64],
    step: Float64,
    gradient_limit: Float64,
    iteration_probability: Float64,
    mut rng: HostRandomState,
) raises -> ParticleUpdate:
    var particles = baseline.copy()
    var deleted = UInt64(0)
    var draws = UInt64(0)
    for field_slice in problem.field_slices:
        var base = Int(field_slice.point_offset)
        for local_point in range(Int(field_slice.point_count)):
            var point = base + local_point
            if problem.point_active[point] == UInt8(0):
                continue
            particles[point] = baseline[point] + step * direction[point]
            if particles[point] >= field_slice.minimum_particles:
                continue
            var result = handle_minimum_particle(
                problem,
                field_slice,
                local_point,
                direction,
                gradient_limit,
                iteration_probability,
                particles,
                rng,
            )
            deleted += result.deleted
            draws += result.random_draws
    return ParticleUpdate(particles^, MinimumParticleUpdate(deleted, draws))


def handle_minimum_particle(
    problem: OptimizationProblem,
    field_slice: FieldSlice,
    local_point: Int,
    direction: List[Float64],
    gradient_limit: Float64,
    iteration_probability: Float64,
    mut particles: List[Float64],
    mut rng: HostRandomState,
) -> MinimumParticleUpdate:
    var point = Int(field_slice.point_offset) + local_point
    if problem.minimum_particle_policy.kind == MINIMUM_PARTICLE_DISABLED:
        return MinimumParticleUpdate(UInt64(0), UInt64(0))
    if (
        problem.minimum_particle_policy.kind
        != MINIMUM_PARTICLE_COMPLEX_HOST_RNG
    ) or particles[point] < 0.0:
        particles[point] = 0.0
        return MinimumParticleUpdate(UInt64(1), UInt64(0))
    if not rng.less_than(iteration_probability):
        return MinimumParticleUpdate(UInt64(0), UInt64(1))
    var previous = particles[point]
    var particle_probability = 1.0 / (
        1.0 + reference_exp(-(previous / field_slice.minimum_particles - 0.5))
    )
    var gradient_probability = 0.5
    if abs(gradient_limit) > 1.0e-30:
        gradient_probability = 1.0 / (
            1.0 + reference_exp(-0.25 * direction[point] / gradient_limit)
        )
    var was_deleted = not rng.less_than(
        particle_probability * gradient_probability
    )
    if was_deleted:
        particles[point] = 0.0
    else:
        particles[point] = field_slice.minimum_particles
    redistribute_minimum_particle_delta(
        problem,
        field_slice,
        local_point,
        previous,
        was_deleted,
        direction,
        particles,
    )
    var deleted = UInt64(0)
    if was_deleted:
        deleted = UInt64(1)
    return MinimumParticleUpdate(deleted, UInt64(2))


def redistribute_minimum_particle_delta(
    problem: OptimizationProblem,
    field_slice: FieldSlice,
    local_point: Int,
    previous: Float64,
    was_deleted: Bool,
    direction: List[Float64],
    mut particles: List[Float64],
):
    var stride = Int(field_slice.raster_stride)
    var neighbors = [
        local_point - 1,
        local_point + 1,
        local_point - stride,
        local_point + stride,
    ]
    var selected = -1
    var selected_gradient = 0.0
    var base = Int(field_slice.point_offset)
    for neighbor in neighbors:
        if neighbor < 0 or neighbor >= Int(field_slice.point_count):
            continue
        var point = base + neighbor
        if problem.point_active[point] == UInt8(0):
            continue
        if (was_deleted and direction[point] > selected_gradient) or (
            not was_deleted and direction[point] < selected_gradient
        ):
            selected = neighbor
            selected_gradient = direction[point]
    if selected >= 0:
        var point = base + local_point
        var neighbor = base + selected
        particles[neighbor] += previous - particles[point]
        if particles[neighbor] < 0.0:
            particles[neighbor] = 0.0
