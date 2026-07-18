"""Host iteration controller shared by the CPU and device backends."""

from std.math import sqrt
from std.memory import ArcPointer, OpaquePointer
from std.sys.info import size_of
from std.time import perf_counter_ns

from device_backend import (
    DeviceWorkspace,
    partition_voxels,
)
from cpu_backend import (
    evaluate_biological_objective_validated,
    compute_exact_step,
    packed_zero_gradient_direction,
)
from minimum_particles import (
    HostRandomState,
    MinimumParticleUpdate,
    apply_final_minimum_particle_limit,
    make_host_rng,
    update_with_minimum_particle_policy,
)
from optimization_problem import (
    OptimizationProblem,
    ScenarioState,
    DoseMatrixSlice,
    RobustScenario,
    OptimizationVoxel,
    evaluate_physical_objective_validated,
)
from reference_math import reference_exp


comptime STOP_MAX_ITERATIONS = UInt32(1)
comptime STOP_CHI2_NOT_DECREASING = UInt32(2)
comptime STOP_CHI_SQUARE_LIMIT = UInt32(3)
comptime STOP_EPSILON = UInt32(4)
comptime STOP_VECTOR_CHANGE = UInt32(5)


@fieldwise_init
struct IterationEvaluation(Copyable, Movable):
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
struct OptimizationResult(Copyable, Movable):
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
struct StepCandidate(Copyable, Movable):
    var particles: List[Float64]
    var minimum_particles: MinimumParticleUpdate


def evaluate_objective(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> IterationEvaluation:
    problem.validate()
    return evaluate_objective_validated(problem, particles)


def evaluate_objective_validated(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> IterationEvaluation:
    if problem.settings.biological:
        var evaluation = evaluate_biological_objective_validated(
            problem, particles
        )
        return IterationEvaluation(
            evaluation.dose_min.copy(),
            evaluation.dose_max.copy(),
            evaluation.min_scenario.copy(),
            evaluation.max_scenario.copy(),
            evaluation.gradient.copy(),
            evaluation.chi2,
            evaluation.dose_p_weighted_avg2,
            evaluation.gradient_norm,
        )
    var evaluation = evaluate_physical_objective_validated(problem, particles)
    return IterationEvaluation(
        evaluation.dose_min.copy(),
        evaluation.dose_max.copy(),
        evaluation.min_scenario.copy(),
        evaluation.max_scenario.copy(),
        evaluation.gradient.copy(),
        evaluation.chi2,
        evaluation.dose_p_weighted_avg2,
        evaluation.gradient_norm,
    )


trait ObjectiveEvaluator:
    def initial_direction(
        mut self, problem: OptimizationProblem, gradient: List[Float64]
    ) raises -> List[Float64]:
        ...

    def evaluate(
        mut self, problem: OptimizationProblem, particles: List[Float64]
    ) raises -> IterationEvaluation:
        ...

    def exact_step(
        mut self,
        problem: OptimizationProblem,
        evaluation: IterationEvaluation,
        direction: List[Float64],
    ) raises -> Float64:
        ...


struct CpuEvaluator(ObjectiveEvaluator):
    def __init__(out self):
        pass

    def evaluate(
        mut self, problem: OptimizationProblem, particles: List[Float64]
    ) raises -> IterationEvaluation:
        return evaluate_objective_validated(problem, particles)

    def initial_direction(
        mut self, problem: OptimizationProblem, gradient: List[Float64]
    ) raises -> List[Float64]:
        return host_initial_direction(problem, gradient)

    def exact_step(
        mut self,
        problem: OptimizationProblem,
        evaluation: IterationEvaluation,
        direction: List[Float64],
    ) raises -> Float64:
        return compute_exact_step(
            problem,
            evaluation.dose_min,
            evaluation.dose_max,
            evaluation.min_scenario,
            evaluation.max_scenario,
            direction,
        )


struct DeviceEvaluator(Movable, ObjectiveEvaluator):
    var workspace: DeviceWorkspace
    var device_bootstrap: Bool

    def __init__(out self, problem: OptimizationProblem) raises:
        self.workspace = DeviceWorkspace(problem)
        self.device_bootstrap = False

    def __init__[
        origin: Origin
    ](
        out self,
        problem: OptimizationProblem,
        matrix_storage: OpaquePointer[origin],
        coefficient_count: Int,
    ) raises:
        self.workspace = DeviceWorkspace(
            problem, matrix_storage, coefficient_count
        )
        self.device_bootstrap = True

    def evaluate(
        mut self, problem: OptimizationProblem, particles: List[Float64]
    ) raises -> IterationEvaluation:
        _ = problem
        var result = self.workspace.evaluation(particles, False)
        return IterationEvaluation(
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
        mut self, problem: OptimizationProblem, gradient: List[Float64]
    ) raises -> List[Float64]:
        if self.device_bootstrap:
            return self.workspace.zero_gradient_direction()
        return host_initial_direction(problem, gradient)

    def exact_step(
        mut self,
        problem: OptimizationProblem,
        evaluation: IterationEvaluation,
        direction: List[Float64],
    ) raises -> Float64:
        _ = problem
        _ = evaluation
        return self.workspace.exact_step(direction)


struct MultiDeviceEvaluator(Movable, ObjectiveEvaluator):
    var workspaces: List[ArcPointer[DeviceWorkspace]]

    def __init__(
        out self, problem: OptimizationProblem, device_count: Int
    ) raises:
        problem.validate()
        var shards = partition_voxels(problem, device_count)
        var workspaces = List[ArcPointer[DeviceWorkspace]]()
        workspaces.reserve(device_count)
        for device in range(device_count):
            workspaces.append(
                ArcPointer(
                    DeviceWorkspace(
                        problem,
                        device,
                        shards[device].voxel_offset,
                        shards[device].voxel_count,
                        False,
                        True,
                    )
                )
            )
        for device in range(device_count):
            workspaces[device][].synchronize()
        self.workspaces = workspaces^

    def __init__[
        origin: Origin
    ](
        out self,
        problem: OptimizationProblem,
        matrix_storages: List[OpaquePointer[origin]],
        coefficient_counts: List[Int],
        voxel_offsets: List[Int],
        voxel_counts: List[Int],
    ) raises:
        var device_count = len(matrix_storages)
        if (
            device_count < 2
            or device_count > 3
            or len(coefficient_counts) != device_count
            or len(voxel_offsets) != device_count
            or len(voxel_counts) != device_count
        ):
            raise Error("invalid external multi-device dose matrix shards")
        var maximum_count = 0
        var expected_voxel = 0
        for device in range(device_count):
            if (
                coefficient_counts[device] < 1
                or voxel_offsets[device] != expected_voxel
                or voxel_counts[device] < 1
            ):
                raise Error("external dose matrix shards are not contiguous")
            expected_voxel += voxel_counts[device]
            if coefficient_counts[device] > maximum_count:
                maximum_count = coefficient_counts[device]
        if expected_voxel != len(problem.voxels):
            raise Error("external dose matrix shards do not cover all voxels")
        problem.validate(external_coefficient_count=UInt64(maximum_count))
        var workspaces = List[ArcPointer[DeviceWorkspace]]()
        workspaces.reserve(device_count)
        for device in range(device_count):
            workspaces.append(
                ArcPointer(
                    DeviceWorkspace(
                        problem,
                        matrix_storages[device],
                        coefficient_counts[device],
                        device,
                        voxel_offsets[device],
                        voxel_counts[device],
                        False,
                    )
                )
            )
        self.workspaces = workspaces^

    def evaluate(
        mut self, problem: OptimizationProblem, particles: List[Float64]
    ) raises -> IterationEvaluation:
        _ = problem
        for device in range(len(self.workspaces)):
            self.workspaces[device][].enqueue_evaluation_front(particles)
        for device in range(len(self.workspaces)):
            self.workspaces[device][].enqueue_evaluation_backprojection()
        var first = self.workspaces[0][].collect_evaluation(False)
        var gradient = first.gradient.copy()
        var chi2 = first.chi2()
        var weighted = first.weighted_dose2()
        for device in range(1, len(self.workspaces)):
            var partial = self.workspaces[device][].collect_evaluation(False)
            chi2 += partial.chi2()
            weighted += partial.weighted_dose2()
            for point in range(len(gradient)):
                gradient[point] += partial.gradient[point]
        var norm2 = 0.0
        for value in gradient:
            norm2 += value * value
        return IterationEvaluation(
            List[Float64](),
            List[Float64](),
            List[Int32](),
            List[Int32](),
            gradient^,
            chi2,
            weighted,
            sqrt(norm2),
        )

    def initial_direction(
        mut self, problem: OptimizationProblem, gradient: List[Float64]
    ) raises -> List[Float64]:
        _ = problem
        _ = gradient
        var direction = self.workspaces[0][].zero_gradient_direction()
        for device in range(1, len(self.workspaces)):
            var partial = self.workspaces[device][].zero_gradient_direction()
            for point in range(len(direction)):
                direction[point] += partial[point]
        return direction^

    def exact_step(
        mut self,
        problem: OptimizationProblem,
        evaluation: IterationEvaluation,
        direction: List[Float64],
    ) raises -> Float64:
        _ = problem
        _ = evaluation
        for device in range(len(self.workspaces)):
            self.workspaces[device][].enqueue_exact_step(direction)
        var numerator = 0.0
        var denominator = 0.0
        for device in range(len(self.workspaces)):
            var partial = self.workspaces[device][].collect_exact_step_terms()
            numerator += partial.numerator
            denominator += partial.denominator
        if denominator == 0.0:
            return 0.0
        return numerator / denominator


def optimize_on_cpu(
    mut problem: OptimizationProblem,
) raises -> OptimizationResult:
    problem.validate()
    var evaluator = CpuEvaluator()
    return run_optimization(problem, evaluator)


def optimize_on_device(
    mut problem: OptimizationProblem,
) raises -> OptimizationResult:
    var evaluator = DeviceEvaluator(problem)
    return run_optimization(problem, evaluator)


def optimize_on_devices(
    mut problem: OptimizationProblem, device_count: Int
) raises -> OptimizationResult:
    var start_ns = perf_counter_ns()
    var evaluator = MultiDeviceEvaluator(problem, device_count)
    var setup_ns = perf_counter_ns()
    var result = run_optimization(problem, evaluator)
    var done_ns = perf_counter_ns()
    print(
        "<TIME> Mojo optimizer multi setup: ",
        Float64(setup_ns - start_ns) * 1.0e-9,
        " sec iterations: ",
        Float64(done_ns - setup_ns) * 1.0e-9,
        " sec",
        sep="",
    )
    return result^


def optimize_with_matrix_shards[
    origin: Origin
](
    mut problem: OptimizationProblem,
    matrix_storages: List[OpaquePointer[origin]],
    coefficient_counts: List[Int],
    voxel_offsets: List[Int],
    voxel_counts: List[Int],
) raises -> OptimizationResult:
    var start_ns = perf_counter_ns()
    var evaluator = MultiDeviceEvaluator(
        problem,
        matrix_storages,
        coefficient_counts,
        voxel_offsets,
        voxel_counts,
    )
    var setup_ns = perf_counter_ns()
    var result = run_optimization(
        problem, evaluator, release_host_problem_arrays=True
    )
    var done_ns = perf_counter_ns()
    print(
        "<TIME> Mojo optimizer multi-direct setup: ",
        Float64(setup_ns - start_ns) * 1.0e-9,
        " sec iterations: ",
        Float64(done_ns - setup_ns) * 1.0e-9,
        " sec",
        sep="",
    )
    return result^


def optimize_with_matrix[
    origin: Origin
](
    mut problem: OptimizationProblem,
    matrix_storage: OpaquePointer[origin],
    coefficient_count: Int,
) raises -> OptimizationResult:
    var evaluator = DeviceEvaluator(problem, matrix_storage, coefficient_count)
    return run_optimization(
        problem, evaluator, release_host_problem_arrays=True
    )


def run_optimization[
    Evaluator: ObjectiveEvaluator
](
    mut problem: OptimizationProblem,
    mut evaluator: Evaluator,
    release_host_problem_arrays: Bool = False,
) raises -> OptimizationResult:
    if problem.settings.max_iterations == UInt32(0):
        raise Error("packed optimizer requires max_iterations > 0")
    var particles = problem.particles.copy()
    var evaluation = evaluator.evaluate(problem, particles)
    var direction = evaluator.initial_direction(problem, evaluation.gradient)
    if release_host_problem_arrays:
        var released_bytes = (
            len(problem.voxels) * size_of[OptimizationVoxel]()
            + len(problem.voxel_scenarios) * size_of[RobustScenario]()
            + len(problem.scenario_states) * size_of[ScenarioState]()
            + len(problem.slices) * size_of[DoseMatrixSlice]()
        )
        problem.voxels = List[OptimizationVoxel]()
        problem.voxel_scenarios = List[RobustScenario]()
        problem.scenario_states = List[ScenarioState]()
        problem.slices = List[DoseMatrixSlice]()
        print("<I> optimizer released-host-bytes=", released_bytes, sep="")
    var active_gradient_norm = vector_norm(direction)
    var rng = make_host_rng(problem)
    var total_iteration = problem.settings.total_iterations
    var iterations = UInt32(0)
    var backtracks = UInt64(0)
    var deleted = UInt64(0)
    var random_draws = UInt64(0)
    var relative_change = 0.0
    var last_factor = 0.0
    var last_exact_step = 0.0
    var stop_reason = STOP_MAX_ITERATIONS
    var update_count = Int(problem.settings.max_iterations) + 1
    for update_index in range(update_count):
        var initialization_update = update_index == 0
        var iteration = update_index - 1
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
        if (
            problem.settings.biological
            and problem.settings.configured_step_factor <= 0.0
        ):
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
            stop_reason = STOP_CHI2_NOT_DECREASING
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
            stop_reason = STOP_CHI_SQUARE_LIMIT
            break
        if (
            iteration >= Int(problem.settings.grace_iterations) - 1
            and relative_change < problem.settings.epsilon
        ):
            stop_reason = STOP_EPSILON
            break
        if (
            iteration >= Int(problem.settings.grace_iterations) - 1
            and iteration > 0
            and vector_change < problem.settings.epsilon * 1.0e-3
        ):
            stop_reason = STOP_VECTOR_CHANGE
            break
        if iteration == Int(problem.settings.max_iterations) - 1:
            stop_reason = STOP_MAX_ITERATIONS
            break
        direction = fletcher_reeves_direction(
            problem, evaluation.gradient, old_gradient, direction
        )

    var final_minimum = apply_final_minimum_particle_limit(problem, particles)
    if final_minimum.changed > UInt64(0):
        deleted += final_minimum.deleted
        evaluation = evaluator.evaluate(problem, particles)

    return OptimizationResult(
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


def host_initial_direction(
    problem: OptimizationProblem, gradient: List[Float64]
) raises -> List[Float64]:
    for value in problem.initial_direction:
        if value != 0.0:
            return problem.initial_direction.copy()
    return packed_zero_gradient_direction(problem)


def vector_norm(values: List[Float64]) -> Float64:
    var total = 0.0
    for value in values:
        total += value * value
    return sqrt(total)


def default_step_factor(problem: OptimizationProblem) -> Float64:
    if problem.settings.configured_step_factor > 0.0:
        return problem.settings.configured_step_factor
    if problem.settings.biological:
        return 1.0
    return 0.5


def select_biological_step_factor[
    Evaluator: ObjectiveEvaluator
](
    problem: OptimizationProblem,
    mut evaluator: Evaluator,
    particles: List[Float64],
    direction: List[Float64],
    exact_step: Float64,
    gradient_limit: Float64,
    iteration_probability: Float64,
    mut rng: HostRandomState,
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
    problem: OptimizationProblem,
    particles: List[Float64],
    direction: List[Float64],
    exact_step: Float64,
    factor: Float64,
    gradient_limit: Float64,
    iteration_probability: Float64,
    mut rng: HostRandomState,
) raises -> StepCandidate:
    var update = update_with_minimum_particle_policy(
        problem,
        particles,
        direction,
        exact_step * factor,
        gradient_limit,
        iteration_probability,
        rng,
    )
    return StepCandidate(
        update.particles.copy(), update.minimum_particles.copy()
    )


def fletcher_reeves_direction(
    problem: OptimizationProblem,
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


def count_active_points(problem: OptimizationProblem) -> Int:
    var count = 0
    for active in problem.point_active:
        if active != UInt8(0):
            count += 1
    return count


def rms_vector_change(
    problem: OptimizationProblem,
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
