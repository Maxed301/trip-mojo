"""Full host iteration controller shared by packed FDCB backends."""

from std.math import sqrt

from fdcb_accelerator import FDCBAccelerator
from fdcb_cpu import (
    evaluate_validated_packed_biological_fdcb,
    fdcb_exact_step,
    packed_zero_gradient_direction,
)
from fdcb_min_particles import (
    FDCBHostRNG,
    FDCBMinimumParticleResult,
    apply_final_minimum_particle_limit,
    make_host_rng,
    update_with_minimum_particle_policy,
)
from fdcb_problem import (
    FDCBProblemV1,
    FDCB_FLAG_BIOLOGICAL,
    FDCB_FLAG_DEVICE_BOOTSTRAP,
    FDCB_FLAG_INITIALIZE,
    evaluate_validated_packed_physical_fdcb,
)
from reference_math import reference_exp


comptime FDCB_STOP_MAX_ITERATIONS = UInt32(1)
comptime FDCB_STOP_CHI2_NOT_DECREASING = UInt32(2)
comptime FDCB_STOP_CHI_SQUARE_LIMIT = UInt32(3)
comptime FDCB_STOP_EPSILON = UInt32(4)
comptime FDCB_STOP_VECTOR_CHANGE = UInt32(5)


@fieldwise_init
struct FDCBIterationEvaluation(Copyable, Movable):
    var dose_min: List[Float64]
    var dose_max: List[Float64]
    var min_scenario: List[Int32]
    var max_scenario: List[Int32]
    var gradient: List[Float64]
    var chi2: Float64
    var weighted_dose2: Float64
    var gradient_norm: Float64

    def residual_percent(self) -> Float64:
        if self.weighted_dose2 <= 0.0:
            return 0.0
        return sqrt(self.chi2 / self.weighted_dose2) * 100.0


@fieldwise_init
struct FDCBOptimizationResult(Copyable, Movable):
    var particles: List[Float64]
    var gradient: List[Float64]
    var direction: List[Float64]
    var chi2: Float64
    var residual_percent: Float64
    var relative_chi2_change: Float64
    var step_factor: Float64
    var exact_step: Float64
    var iterations: UInt32
    var backtracks: UInt64
    var minimum_particle_deleted: UInt64
    var random_draws: UInt64
    var stop_reason: UInt32


@fieldwise_init
struct FDCBCandidate(Copyable, Movable):
    var particles: List[Float64]
    var minimum_particles: FDCBMinimumParticleResult


def evaluate_packed_iteration(
    problem: FDCBProblemV1, particles: List[Float64]
) raises -> FDCBIterationEvaluation:
    problem.validate()
    return evaluate_validated_packed_iteration(problem, particles)


def evaluate_validated_packed_iteration(
    problem: FDCBProblemV1, particles: List[Float64]
) raises -> FDCBIterationEvaluation:
    if (problem.settings.flags & FDCB_FLAG_BIOLOGICAL) != UInt32(0):
        var evaluation = evaluate_validated_packed_biological_fdcb(
            problem, particles
        )
        return FDCBIterationEvaluation(
            evaluation.dose_min.copy(),
            evaluation.dose_max.copy(),
            evaluation.min_scenario.copy(),
            evaluation.max_scenario.copy(),
            evaluation.gradient.copy(),
            evaluation.chi2,
            evaluation.dose_p_weighted_avg2,
            evaluation.gradient_norm,
        )
    var evaluation = evaluate_validated_packed_physical_fdcb(problem, particles)
    return FDCBIterationEvaluation(
        evaluation.dose_min.copy(),
        evaluation.dose_max.copy(),
        evaluation.min_scenario.copy(),
        evaluation.max_scenario.copy(),
        evaluation.gradient.copy(),
        evaluation.chi2,
        evaluation.dose_p_weighted_avg2,
        evaluation.gradient_norm,
    )


trait FDCBEvaluator:
    def initial_direction(
        mut self, problem: FDCBProblemV1, gradient: List[Float64]
    ) raises -> List[Float64]:
        ...

    def evaluate(
        mut self, problem: FDCBProblemV1, particles: List[Float64]
    ) raises -> FDCBIterationEvaluation:
        ...

    def exact_step(
        mut self,
        problem: FDCBProblemV1,
        evaluation: FDCBIterationEvaluation,
        direction: List[Float64],
    ) raises -> Float64:
        ...


struct FDCBCPUEvaluator(FDCBEvaluator):
    def __init__(out self):
        pass

    def evaluate(
        mut self, problem: FDCBProblemV1, particles: List[Float64]
    ) raises -> FDCBIterationEvaluation:
        return evaluate_validated_packed_iteration(problem, particles)

    def initial_direction(
        mut self, problem: FDCBProblemV1, gradient: List[Float64]
    ) raises -> List[Float64]:
        return initial_direction(problem, gradient)

    def exact_step(
        mut self,
        problem: FDCBProblemV1,
        evaluation: FDCBIterationEvaluation,
        direction: List[Float64],
    ) raises -> Float64:
        return fdcb_exact_step(
            problem,
            evaluation.dose_min,
            evaluation.dose_max,
            evaluation.min_scenario,
            evaluation.max_scenario,
            direction,
        )


struct FDCBDeviceEvaluator(FDCBEvaluator, Movable):
    var accelerator: FDCBAccelerator

    def __init__(out self, problem: FDCBProblemV1) raises:
        self.accelerator = FDCBAccelerator(problem)

    def __init__[
        origin: Origin
    ](
        out self,
        problem: FDCBProblemV1,
        matrix_storage: OpaquePointer[origin],
        coefficient_count: Int,
    ) raises:
        self.accelerator = FDCBAccelerator(
            problem, matrix_storage, coefficient_count
        )

    def evaluate(
        mut self, problem: FDCBProblemV1, particles: List[Float64]
    ) raises -> FDCBIterationEvaluation:
        _ = problem
        var result = self.accelerator.evaluation(particles, False)
        return FDCBIterationEvaluation(
            result.dose_min.copy(),
            result.dose_max.copy(),
            result.min_scenario.copy(),
            result.max_scenario.copy(),
            result.gradient.copy(),
            result.chi2(),
            result.weighted_dose2(),
            result.gradient_norm,
        )

    def initial_direction(
        mut self, problem: FDCBProblemV1, gradient: List[Float64]
    ) raises -> List[Float64]:
        if (problem.settings.flags & FDCB_FLAG_DEVICE_BOOTSTRAP) != UInt32(0):
            return self.accelerator.zero_gradient_direction()
        return initial_direction(problem, gradient)

    def exact_step(
        mut self,
        problem: FDCBProblemV1,
        evaluation: FDCBIterationEvaluation,
        direction: List[Float64],
    ) raises -> Float64:
        _ = problem
        _ = evaluation
        return self.accelerator.exact_step(direction)


def optimize_packed_fdcb(
    problem: FDCBProblemV1,
) raises -> FDCBOptimizationResult:
    problem.validate()
    var evaluator = FDCBCPUEvaluator()
    return optimize_packed_fdcb_with_evaluator(problem, evaluator)


def optimize_packed_fdcb_accelerator(
    problem: FDCBProblemV1,
) raises -> FDCBOptimizationResult:
    var evaluator = FDCBDeviceEvaluator(problem)
    return optimize_packed_fdcb_with_evaluator(problem, evaluator)


def optimize_packed_fdcb_accelerator_matrix[
    origin: Origin
](
    problem: FDCBProblemV1,
    matrix_storage: OpaquePointer[origin],
    coefficient_count: Int,
) raises -> FDCBOptimizationResult:
    var evaluator = FDCBDeviceEvaluator(
        problem, matrix_storage, coefficient_count
    )
    return optimize_packed_fdcb_with_evaluator(problem, evaluator)


def optimize_packed_fdcb_with_evaluator[
    Evaluator: FDCBEvaluator
](
    problem: FDCBProblemV1, mut evaluator: Evaluator
) raises -> FDCBOptimizationResult:
    if problem.settings.max_iterations == UInt32(0):
        raise Error("packed FDCB optimizer requires max_iterations > 0")
    var particles = problem.particles.copy()
    var evaluation = evaluator.evaluate(problem, particles)
    var direction = evaluator.initial_direction(problem, evaluation.gradient)
    var active_gradient_norm = evaluation.gradient_norm
    if (problem.settings.flags & FDCB_FLAG_INITIALIZE) != UInt32(0):
        active_gradient_norm = vector_norm(direction)
    var rng = make_host_rng(problem)
    var total_iteration = problem.settings.total_iterations
    var iterations = UInt32(0)
    var backtracks = UInt64(0)
    var deleted = UInt64(0)
    var random_draws = UInt64(0)
    var relative_change = 0.0
    var last_factor = 0.0
    var last_exact_step = 0.0
    var stop_reason = FDCB_STOP_MAX_ITERATIONS

    var initialize = (problem.settings.flags & FDCB_FLAG_INITIALIZE) != UInt32(
        0
    )
    var update_count = Int(problem.settings.max_iterations)
    if initialize:
        update_count += 1
    for update_index in range(update_count):
        var initialization_update = initialize and update_index == 0
        var iteration = update_index
        if initialize and update_index > 0:
            iteration -= 1
        var exact_step = evaluator.exact_step(problem, evaluation, direction)
        last_exact_step = exact_step
        var active_count = count_active_points(problem)
        var gradient_limit = 0.0
        var iteration_probability = 0.0
        if active_count > 0:
            gradient_limit = active_gradient_norm / sqrt(Float64(active_count))
            iteration_probability = 1.0 / (
                1.0
                + reference_exp(
                    -20.0 * (Float64(total_iteration) / 100.0 - 0.2)
                )
            )
        total_iteration += UInt32(1)

        var factor = default_step_factor(problem)
        if (problem.settings.flags & FDCB_FLAG_BIOLOGICAL) != UInt32(
            0
        ) and problem.settings.configured_step_factor <= 0.0:
            var interval = problem.settings.step_factor_iterations
            if interval == UInt32(0):
                interval = problem.settings.max_iterations
            if iteration % Int(interval) == 2:
                factor = select_biological_step_factor(
                    problem,
                    evaluator,
                    particles,
                    direction,
                    exact_step,
                    gradient_limit,
                    iteration_probability,
                    rng,
                    deleted,
                    random_draws,
                )

        var accepted = False
        var candidate_particles = particles.copy()
        var candidate_evaluation = evaluation.copy()
        var vector_change = 0.0
        while not accepted:
            var candidate = update_candidate(
                problem,
                particles,
                direction,
                exact_step,
                factor,
                gradient_limit,
                iteration_probability,
                rng,
            )
            deleted += candidate.minimum_particles.deleted
            random_draws += candidate.minimum_particles.random_draws
            candidate_particles = candidate.particles.copy()
            candidate_evaluation = evaluator.evaluate(
                problem, candidate_particles
            )
            var chi2_change = evaluation.chi2 - candidate_evaluation.chi2
            if chi2_change < 0.0:
                factor *= 0.5
            if (
                chi2_change < 0.0
                and factor > 1.0e-2
                and iteration >= Int(problem.settings.grace_iterations) - 1
            ):
                backtracks += UInt64(1)
                continue
            accepted = True
            vector_change = rms_vector_change(
                problem, particles, candidate_particles
            )

        var chi2_change = evaluation.chi2 - candidate_evaluation.chi2
        if evaluation.chi2 != 0.0:
            relative_change = chi2_change / evaluation.chi2
        else:
            relative_change = 0.0
        last_factor = factor
        if (
            not initialization_update
            and iteration >= Int(problem.settings.grace_iterations)
            and relative_change <= 0.0
        ):
            stop_reason = FDCB_STOP_CHI2_NOT_DECREASING
            break

        particles = candidate_particles^
        var old_gradient = evaluation.gradient.copy()
        if initialization_update:
            old_gradient = direction.copy()
        evaluation = candidate_evaluation^
        active_gradient_norm = evaluation.gradient_norm
        if initialization_update:
            if problem.settings.max_iterations > UInt32(1):
                direction = fletcher_reeves_direction(
                    problem,
                    evaluation.gradient,
                    old_gradient,
                    direction,
                )
            continue
        iterations += UInt32(1)
        if (
            problem.settings.chi_square_limit > 0.0
            and evaluation.residual_percent()
            <= problem.settings.chi_square_limit
        ):
            stop_reason = FDCB_STOP_CHI_SQUARE_LIMIT
            break
        if (
            iteration >= Int(problem.settings.grace_iterations) - 1
            and relative_change < problem.settings.epsilon
        ):
            stop_reason = FDCB_STOP_EPSILON
            break
        if (
            iteration >= Int(problem.settings.grace_iterations) - 1
            and iteration > 0
            and vector_change < problem.settings.epsilon * 1.0e-3
        ):
            stop_reason = FDCB_STOP_VECTOR_CHANGE
            break
        if iteration == Int(problem.settings.max_iterations) - 1:
            stop_reason = FDCB_STOP_MAX_ITERATIONS
            break
        direction = fletcher_reeves_direction(
            problem, evaluation.gradient, old_gradient, direction
        )

    var final_minimum = apply_final_minimum_particle_limit(problem, particles)
    if final_minimum.changed > UInt64(0):
        deleted += final_minimum.deleted
        evaluation = evaluator.evaluate(problem, particles)

    return FDCBOptimizationResult(
        particles^,
        evaluation.gradient.copy(),
        direction^,
        evaluation.chi2,
        evaluation.residual_percent(),
        relative_change,
        last_factor,
        last_exact_step,
        iterations,
        backtracks,
        deleted,
        random_draws,
        stop_reason,
    )


def initial_direction(
    problem: FDCBProblemV1, gradient: List[Float64]
) raises -> List[Float64]:
    for value in problem.initial_direction:
        if value != 0.0:
            return problem.initial_direction.copy()
    if (problem.settings.flags & FDCB_FLAG_INITIALIZE) != UInt32(0):
        return packed_zero_gradient_direction(problem)
    return gradient.copy()


def vector_norm(values: List[Float64]) -> Float64:
    var total = 0.0
    for value in values:
        total += value * value
    return sqrt(total)


def default_step_factor(problem: FDCBProblemV1) -> Float64:
    if problem.settings.configured_step_factor > 0.0:
        return problem.settings.configured_step_factor
    if (problem.settings.flags & FDCB_FLAG_BIOLOGICAL) != UInt32(0):
        return 1.0
    return 0.5


def select_biological_step_factor[
    Evaluator: FDCBEvaluator
](
    problem: FDCBProblemV1,
    mut evaluator: Evaluator,
    particles: List[Float64],
    direction: List[Float64],
    exact_step: Float64,
    gradient_limit: Float64,
    iteration_probability: Float64,
    mut rng: FDCBHostRNG,
    mut deleted: UInt64,
    mut random_draws: UInt64,
) raises -> Float64:
    var factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    var best_factor = factors[0]
    var best_chi2 = Float64.MAX
    for factor in factors:
        var candidate = update_candidate(
            problem,
            particles,
            direction,
            exact_step,
            factor,
            gradient_limit,
            iteration_probability,
            rng,
        )
        deleted += candidate.minimum_particles.deleted
        random_draws += candidate.minimum_particles.random_draws
        var trial = evaluator.evaluate(problem, candidate.particles)
        if trial.chi2 < best_chi2:
            best_chi2 = trial.chi2
            best_factor = factor
    return best_factor


def update_candidate(
    problem: FDCBProblemV1,
    particles: List[Float64],
    direction: List[Float64],
    exact_step: Float64,
    factor: Float64,
    gradient_limit: Float64,
    iteration_probability: Float64,
    mut rng: FDCBHostRNG,
) raises -> FDCBCandidate:
    var update = update_with_minimum_particle_policy(
        problem,
        particles,
        direction,
        exact_step * factor,
        gradient_limit,
        iteration_probability,
        rng,
    )
    return FDCBCandidate(
        update.particles.copy(), update.minimum_particles.copy()
    )


def fletcher_reeves_direction(
    problem: FDCBProblemV1,
    gradient: List[Float64],
    previous_gradient: List[Float64],
    previous_direction: List[Float64],
) -> List[Float64]:
    var numerator = 0.0
    var denominator = 0.0
    for field_slice in problem.field_slices:
        var weight = Float64(problem.field_count - field_slice.field_index)
        var base = Int(field_slice.point_offset)
        for local_point in range(Int(field_slice.point_count)):
            var point = base + local_point
            if problem.point_active[point] != UInt8(0):
                numerator += weight * gradient[point] * gradient[point]
                denominator += (
                    weight * previous_gradient[point] * previous_gradient[point]
                )
    var gamma = 0.0
    if denominator != 0.0:
        gamma = numerator / denominator
    var direction = List[Float64]()
    direction.resize(len(gradient), 0.0)
    for i in range(len(gradient)):
        if problem.point_active[i] != UInt8(0):
            direction[i] = gradient[i] + previous_direction[i] * gamma
    return direction^


def count_active_points(problem: FDCBProblemV1) -> Int:
    var count = 0
    for active in problem.point_active:
        if active != UInt8(0):
            count += 1
    return count


def rms_vector_change(
    problem: FDCBProblemV1,
    previous: List[Float64],
    current: List[Float64],
) -> Float64:
    var sum2 = 0.0
    var count = 0
    for i in range(len(previous)):
        if problem.point_active[i] != UInt8(0):
            var difference = previous[i] - current[i]
            sum2 += difference * difference
            count += 1
    if count == 0:
        return 0.0
    var denominator = count
    if problem.settings.active_point_count > UInt32(0):
        denominator = Int(problem.settings.active_point_count)
    return sqrt(sum2 / Float64(denominator))
