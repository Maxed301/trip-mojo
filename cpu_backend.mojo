"""Reference CPU kernels over the packed optimization boundary."""

from std.algorithm import parallelize
from std.math import sqrt

from optimization_problem import (
    ObjectiveEvaluation,
    OptimizationProblem,
    DoseMatrixSlice,
    OptimizationVoxel,
    OPTIMIZER_FLAG_ROBUST_INCLUDE_DMAX,
    FLOAT64_EPSILON,
    MEV_TO_GY,
    abs_f64,
)


comptime CPU_THREADS = 12


def packed_zero_gradient_direction(
    problem: OptimizationProblem,
) raises -> List[Float64]:
    """Native fallback when no host-computed bootstrap was packed."""
    problem.validate()
    var gradient = List[Float64]()
    gradient.resize(len(problem.particles), 0.0)
    for voxel_index in range(len(problem.voxels)):
        var voxel = problem.voxels[voxel_index].copy()
        if voxel.prescribed_dose <= 0.0:
            continue
        var weight = voxel.dose_weight / voxel.dose_divisor
        var factor = voxel.prescribed_dose * weight * (2.0 * weight) * MEV_TO_GY
        var scenario = problem.voxel_scenarios[
            Int(voxel.scenario_offset)
        ].copy()
        for local_slice in range(Int(scenario.slice_count)):
            var packed_slice = problem.slices[
                Int(scenario.slice_offset) + local_slice
            ].copy()
            var field_slice = problem.field_slices[
                Int(packed_slice.field_slice_index)
            ].copy()
            var scale = factor * Float64(packed_slice.dose_coefficient)
            for local_entry in range(Int(packed_slice.coefficient_count)):
                var coefficient_index = (
                    Int(packed_slice.coefficient_offset) + local_entry
                )
                var point = Int(field_slice.point_offset) + Int(
                    problem.coefficient_point_indices[coefficient_index]
                )
                gradient[point] += scale * Float64(
                    problem.coefficients[coefficient_index]
                )
    return gradient^


@fieldwise_init
struct BiologicalState(Copyable, Movable):
    var dose: Float64
    var dose_phys: Float64
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64
    var dose_phs: Float64
    var gradient_denominator: Float64


@fieldwise_init
struct BiologicalEvaluation(Copyable, Movable):
    var states: List[BiologicalState]
    var dose_min: List[Float64]
    var dose_max: List[Float64]
    var min_scenario: List[Int32]
    var max_scenario: List[Int32]
    var gradient: List[Float64]
    var chi2: Float64
    var dose_p_weighted_avg2: Float64
    var gradient_norm: Float64


def packed_slice_dot(
    problem: OptimizationProblem,
    packed_slice: DoseMatrixSlice,
    values: List[Float64],
) -> Float64:
    var field_slice = problem.field_slices[
        Int(packed_slice.field_slice_index)
    ].copy()
    var total = 0.0
    for local_entry in range(Int(packed_slice.coefficient_count)):
        var coefficient_index = (
            Int(packed_slice.coefficient_offset) + local_entry
        )
        var point_index = Int(field_slice.point_offset) + Int(
            problem.coefficient_point_indices[coefficient_index]
        )
        total += values[point_index] * Float64(
            problem.coefficients[coefficient_index]
        )
    return total


def evaluate_biological_objective(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> BiologicalEvaluation:
    problem.validate()
    return evaluate_biological_objective_validated(problem, particles)


def evaluate_biological_objective_validated(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> BiologicalEvaluation:
    """Evaluate a problem already validated by the owning controller."""
    if len(particles) != len(problem.particles):
        raise Error(
            "particle vector length does not match packed optimization problem"
        )
    var states = packed_biological_states(problem, particles)
    var dose_min = List[Float64]()
    var dose_max = List[Float64]()
    var min_scenario = List[Int32]()
    var max_scenario = List[Int32]()
    var gradient = List[Float64]()
    dose_min.resize(len(problem.voxels), 0.0)
    dose_max.resize(len(problem.voxels), 0.0)
    min_scenario.resize(len(problem.voxels), Int32(0))
    max_scenario.resize(len(problem.voxels), Int32(0))
    gradient.resize(len(problem.particles), 0.0)
    var chi2 = 0.0
    var weighted = 0.0

    for voxel_index in range(len(problem.voxels)):
        var voxel = problem.voxels[voxel_index].copy()
        if voxel.prescribed_dose == 0.0:
            min_scenario[voxel_index] = voxel.initial_min_scenario
            max_scenario[voxel_index] = voxel.initial_max_scenario
            continue
        var base = Int(voxel.scenario_offset)
        var dmin = states[base].dose
        var dmax = states[base].dose
        var imin = 0
        var imax = 0
        for scenario in range(1, Int(problem.scenario_count)):
            var dose = states[base + scenario].dose
            if dose < dmin:
                dmin = dose
                imin = scenario
            if dose > dmax:
                dmax = dose
                imax = scenario
        dose_min[voxel_index] = dmin
        dose_max[voxel_index] = dmax
        min_scenario[voxel_index] = Int32(imin)
        max_scenario[voxel_index] = Int32(imax)

        var prescribed = abs_f64(voxel.prescribed_dose)
        var selected = imin
        if voxel.prescribed_dose < 0.0:
            selected = imax
        var selected_state = states[base + selected].copy()
        var dose_weight = voxel.dose_weight / voxel.dose_divisor
        var dose_residual = prescribed - selected_state.dose
        var let_residual = packed_let_residual(
            voxel.prescribed_dose,
            voxel.prescribed_let,
            voxel.let_weight,
            selected_state.let_bar,
        )
        if dose_residual < 0.0 or voxel.prescribed_dose > 0.0:
            chi2 += (dose_residual * dose_weight) * (
                dose_residual * dose_weight
            ) + let_residual * let_residual
        weighted += (dose_weight * voxel.prescribed_dose) * (
            dose_weight * voxel.prescribed_dose
        )
        if (
            problem.settings.flags & OPTIMIZER_FLAG_ROBUST_INCLUDE_DMAX
        ) != UInt32(0) and voxel.prescribed_dose > 0.0:
            var max_weight = voxel.maximum_dose_weight / voxel.dose_divisor
            var max_residual = prescribed - dmax
            chi2 += (max_residual * max_weight) * (max_residual * max_weight)
            weighted += (max_weight * voxel.prescribed_dose) * (
                max_weight * voxel.prescribed_dose
            )
    var point_count = len(problem.particles)
    var thread_gradients = List[Float64]()
    thread_gradients.resize(CPU_THREADS * point_count, 0.0)

    @parameter
    def scatter_worker(worker: Int):
        var start = worker * len(problem.voxels) // CPU_THREADS
        var end = (worker + 1) * len(problem.voxels) // CPU_THREADS
        var output_base = worker * point_count
        for voxel_index in range(start, end):
            var voxel = problem.voxels[voxel_index].copy()
            if voxel.prescribed_dose == 0.0:
                continue
            var selected = Int(min_scenario[voxel_index])
            if voxel.prescribed_dose < 0.0:
                selected = Int(max_scenario[voxel_index])
            var state_index = Int(voxel.scenario_offset) + selected
            scatter_packed_biological_gradient(
                problem,
                particles,
                voxel_index,
                selected,
                states[state_index],
                voxel.dose_weight,
                thread_gradients,
                False,
                output_base,
            )
            if (
                problem.settings.flags & OPTIMIZER_FLAG_ROBUST_INCLUDE_DMAX
            ) != UInt32(0) and voxel.prescribed_dose > 0.0:
                var maximum = Int(max_scenario[voxel_index])
                scatter_packed_biological_gradient(
                    problem,
                    particles,
                    voxel_index,
                    maximum,
                    states[Int(voxel.scenario_offset) + maximum],
                    voxel.maximum_dose_weight,
                    thread_gradients,
                    True,
                    output_base,
                )

    parallelize[scatter_worker](CPU_THREADS, CPU_THREADS)
    for worker in range(CPU_THREADS):
        var base = worker * point_count
        for point in range(point_count):
            gradient[point] += thread_gradients[base + point]

    var norm2 = 0.0
    for value in gradient:
        norm2 += value * value
    return BiologicalEvaluation(
        states^,
        dose_min^,
        dose_max^,
        min_scenario^,
        max_scenario^,
        gradient^,
        chi2,
        weighted,
        sqrt(norm2),
    )


def packed_biological_states(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> List[BiologicalState]:
    comptime assert CPU_THREADS > 0
    var states = List[BiologicalState]()
    states.resize(
        len(problem.voxel_scenarios),
        BiologicalState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    var particle_ptr = Span(particles).unsafe_ptr()
    var voxel_ptr = Span(problem.voxels).unsafe_ptr()
    var scenario_ptr = Span(problem.voxel_scenarios).unsafe_ptr()
    var state_ptr = Span(problem.scenario_states).unsafe_ptr()
    var slice_ptr = Span(problem.slices).unsafe_ptr()
    var field_slice_ptr = Span(problem.field_slices).unsafe_ptr()
    var index_ptr = Span(problem.coefficient_point_indices).unsafe_ptr()
    var coefficient_ptr = Span(problem.coefficients).unsafe_ptr()

    @parameter
    def compute_state(vs_index: Int):
        var voxel = voxel_ptr[vs_index // Int(problem.scenario_count)].copy()
        var base = state_ptr[vs_index].copy()
        var dose_phys = base.dose_minor
        var alpha = base.alpha_minor
        var sqrt_beta = base.sqrt_beta_minor
        var let_mix = base.let_mix_minor
        var let_bar_sum = 0.0
        var scenario = scenario_ptr[vs_index].copy()
        for local_slice in range(Int(scenario.slice_count)):
            var packed_slice = slice_ptr[
                Int(scenario.slice_offset) + local_slice
            ].copy()
            var field_slice = field_slice_ptr[
                Int(packed_slice.field_slice_index)
            ].copy()
            var dot = 0.0
            var coefficient_base = Int(packed_slice.coefficient_offset)
            var point_base = Int(field_slice.point_offset)
            for local_entry in range(Int(packed_slice.coefficient_count)):
                var coefficient_index = coefficient_base + local_entry
                var point_index = point_base + Int(index_ptr[coefficient_index])
                dot += particle_ptr[point_index] * Float64(
                    coefficient_ptr[coefficient_index]
                )
            dose_phys += dot * Float64(packed_slice.dose_coefficient)
            alpha += dot * Float64(packed_slice.alpha_coefficient)
            sqrt_beta += dot * Float64(packed_slice.sqrt_beta_coefficient)
            let_mix += dot * Float64(packed_slice.let_mix_coefficient)
            let_bar_sum += dot * Float64(packed_slice.let_bar_coefficient)

        var dose = voxel.initial_dose
        var let_bar = 0.0
        var dose_phs = let_mix * MEV_TO_GY
        var denominator = 0.0
        if dose_phys > 0.0 and let_mix > 0.0:
            let_bar = let_bar_sum / let_mix
            if dose_phs <= voxel.rbe_cut:
                var damage = MEV_TO_GY * (
                    sqrt_beta * sqrt_beta * MEV_TO_GY + alpha
                )
                if voxel.rbe_beta != 0.0:
                    denominator = sqrt(
                        damage * voxel.rbe_beta * 4.0
                        + voxel.rbe_alpha * voxel.rbe_alpha
                    )
                    dose += (
                        dose_phys
                        * MEV_TO_GY
                        * (
                            (denominator - voxel.rbe_alpha)
                            / (voxel.rbe_beta * 2.0)
                        )
                        / dose_phs
                    )
                else:
                    denominator = voxel.rbe_alpha
                    dose += (
                        dose_phys
                        * MEV_TO_GY
                        * (damage / voxel.rbe_alpha)
                        / dose_phs
                    )
            else:
                var cut_scale = voxel.rbe_cut / let_mix
                var damage = (
                    sqrt_beta * sqrt_beta * cut_scale + alpha
                ) * cut_scale + (dose_phs - voxel.rbe_cut) * voxel.rbe_slope_max
                var bio_dose = (
                    damage - voxel.rbe_damage_cut
                ) / voxel.rbe_slope_max + voxel.rbe_cut
                denominator = voxel.rbe_slope_max
                dose += dose_phys * MEV_TO_GY * bio_dose / dose_phs
        states[vs_index] = BiologicalState(
            dose,
            dose_phys,
            alpha,
            sqrt_beta,
            let_mix,
            let_bar,
            dose_phs,
            denominator,
        )

    parallelize[compute_state](len(states), CPU_THREADS)
    return states^


def scatter_packed_biological_gradient(
    problem: OptimizationProblem,
    particles: List[Float64],
    voxel_index: Int,
    scenario_index: Int,
    state: BiologicalState,
    pass_weight: Float64,
    mut gradient: List[Float64],
    maximum_pass: Bool = False,
    output_base: Int = 0,
):
    var voxel = problem.voxels[voxel_index].copy()
    var residual = abs_f64(voxel.prescribed_dose) - state.dose
    var divisor = voxel.dose_divisor
    if voxel.prescribed_dose < 0.0 or maximum_pass:
        if residual > 0.0:
            residual = 0.0
    elif residual < 0.0:
        if residual > voxel.prescribed_dose * -voxel.overdose_tolerance:
            residual = 0.0
        else:
            residual += voxel.prescribed_dose * voxel.overdose_tolerance
            divisor /= problem.settings.overdose_weight
    if residual == 0.0 or state.let_mix <= 0.0:
        return
    var scaled_weight = pass_weight / divisor
    var factor = residual * scaled_weight * (2.0 * scaled_weight)
    var let_residual = packed_let_residual(
        voxel.prescribed_dose,
        voxel.prescribed_let,
        voxel.let_weight,
        state.let_bar,
    )
    var scenario = problem.voxel_scenarios[
        Int(voxel.scenario_offset) + scenario_index
    ].copy()
    var particle_ptr = Span(particles).unsafe_ptr()
    var active_ptr = Span(problem.point_active).unsafe_ptr()
    var output_ptr = Span(gradient).unsafe_ptr()
    var slice_ptr = Span(problem.slices).unsafe_ptr()
    var field_slice_ptr = Span(problem.field_slices).unsafe_ptr()
    var index_ptr = Span(problem.coefficient_point_indices).unsafe_ptr()
    var coefficient_ptr = Span(problem.coefficients).unsafe_ptr()
    for local_slice in range(Int(scenario.slice_count)):
        var packed_slice = slice_ptr[
            Int(scenario.slice_offset) + local_slice
        ].copy()
        var let_prim = (
            Float64(packed_slice.let_bar_coefficient)
            - state.let_bar * Float64(packed_slice.let_mix_coefficient)
        ) / state.let_mix
        var grad_bio: Float64
        if state.dose_phs <= voxel.rbe_cut:
            grad_bio = (
                Float64(packed_slice.alpha_coefficient)
                + MEV_TO_GY
                * state.sqrt_beta
                * 2.0
                * Float64(packed_slice.sqrt_beta_coefficient)
            ) * MEV_TO_GY
        else:
            var cut_scale = voxel.rbe_cut / state.let_mix
            grad_bio = (
                Float64(packed_slice.alpha_coefficient)
                - Float64(packed_slice.let_mix_coefficient)
                * state.alpha
                / state.let_mix
                + (
                    Float64(packed_slice.sqrt_beta_coefficient)
                    - Float64(packed_slice.let_mix_coefficient)
                    * state.sqrt_beta
                    / state.let_mix
                )
                * state.sqrt_beta
                * 2.0
                * cut_scale
            ) * cut_scale + Float64(
                packed_slice.let_mix_coefficient
            ) * MEV_TO_GY * voxel.rbe_slope_max
        var scale = (
            factor * grad_bio / state.gradient_denominator
            + let_residual * let_prim
        )
        var field_slice = field_slice_ptr[
            Int(packed_slice.field_slice_index)
        ].copy()
        var coefficient_base = Int(packed_slice.coefficient_offset)
        var point_base = Int(field_slice.point_offset)
        for local_entry in range(Int(packed_slice.coefficient_count)):
            var coefficient_index = coefficient_base + local_entry
            var point_index = point_base + Int(index_ptr[coefficient_index])
            if (
                active_ptr[point_index] != UInt8(0)
                and particle_ptr[point_index] != 0.0
            ):
                output_ptr[output_base + point_index] += scale * Float64(
                    coefficient_ptr[coefficient_index]
                )


def packed_let_residual(
    prescribed_dose: Float64,
    prescribed_let: Float64,
    let_weight: Float64,
    actual_let: Float64,
) -> Float64:
    if let_weight <= FLOAT64_EPSILON or prescribed_let <= 0.0:
        return 0.0
    var residual = prescribed_let - actual_let
    if prescribed_dose > 0.0:
        if residual < 0.0:
            residual = 0.0
    elif residual > 0.0:
        residual = 0.0
    return residual * let_weight * prescribed_dose / prescribed_let


def scatter_slice(
    problem: OptimizationProblem,
    particles: List[Float64],
    packed_slice: DoseMatrixSlice,
    scale: Float64,
    mut output: List[Float64],
    output_base: Int = 0,
):
    var field_slice = problem.field_slices[
        Int(packed_slice.field_slice_index)
    ].copy()
    for local_entry in range(Int(packed_slice.coefficient_count)):
        var coefficient_index = (
            Int(packed_slice.coefficient_offset) + local_entry
        )
        var point_index = Int(field_slice.point_offset) + Int(
            problem.coefficient_point_indices[coefficient_index]
        )
        if (
            problem.point_active[point_index] != UInt8(0)
            and particles[point_index] != 0.0
        ):
            output[output_base + point_index] += scale * Float64(
                problem.coefficients[coefficient_index]
            )


def compute_physical_exact_step(
    problem: OptimizationProblem,
    evaluation: ObjectiveEvaluation,
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


def compute_biological_exact_step(
    problem: OptimizationProblem,
    evaluation: BiologicalEvaluation,
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


def compute_exact_step(
    problem: OptimizationProblem,
    dose_min: List[Float64],
    dose_max: List[Float64],
    min_scenario: List[Int32],
    max_scenario: List[Int32],
    direction: List[Float64],
) raises -> Float64:
    if len(direction) != len(problem.particles):
        raise Error(
            "direction length does not match packed optimization problem"
        )
    var numerators = List[Float64]()
    var denominators = List[Float64]()
    numerators.resize(CPU_THREADS, 0.0)
    denominators.resize(CPU_THREADS, 0.0)
    var direction_ptr = Span(direction).unsafe_ptr()
    var scenario_ptr = Span(problem.voxel_scenarios).unsafe_ptr()
    var slice_ptr = Span(problem.slices).unsafe_ptr()
    var field_slice_ptr = Span(problem.field_slices).unsafe_ptr()
    var index_ptr = Span(problem.coefficient_point_indices).unsafe_ptr()
    var coefficient_ptr = Span(problem.coefficients).unsafe_ptr()

    @parameter
    def exact_step_worker(worker: Int):
        var numerator = 0.0
        var denominator = 0.0
        var start = worker * len(problem.voxels) // CPU_THREADS
        var end = (worker + 1) * len(problem.voxels) // CPU_THREADS
        for voxel_index in range(start, end):
            var voxel = problem.voxels[voxel_index].copy()
            if voxel.prescribed_dose == 0.0:
                continue
            var rmin = 0.0
            var rmax = 0.0
            for response_pass in range(2):
                var scenario_index = Int(min_scenario[voxel_index])
                if response_pass != 0:
                    scenario_index = Int(max_scenario[voxel_index])
                var scenario = scenario_ptr[
                    Int(voxel.scenario_offset) + scenario_index
                ].copy()
                var response = 0.0
                for local_slice in range(Int(scenario.slice_count)):
                    var packed_slice = slice_ptr[
                        Int(scenario.slice_offset) + local_slice
                    ].copy()
                    var field_slice = field_slice_ptr[
                        Int(packed_slice.field_slice_index)
                    ].copy()
                    var dot = 0.0
                    var coefficient_base = Int(packed_slice.coefficient_offset)
                    var point_base = Int(field_slice.point_offset)
                    for local_entry in range(
                        Int(packed_slice.coefficient_count)
                    ):
                        var coefficient_index = coefficient_base + local_entry
                        var point_index = point_base + Int(
                            index_ptr[coefficient_index]
                        )
                        dot += direction_ptr[point_index] * Float64(
                            coefficient_ptr[coefficient_index]
                        )
                    response += dot * Float64(packed_slice.dose_coefficient)
                response *= MEV_TO_GY * problem.settings.fractions
                if response_pass == 0:
                    rmin = response
                else:
                    rmax = response
            var residual = (
                abs_f64(voxel.prescribed_dose) - dose_min[voxel_index]
            )
            var response = rmin
            if voxel.prescribed_dose < 0.0:
                residual = (
                    abs_f64(voxel.prescribed_dose) - dose_max[voxel_index]
                )
                response = rmax
                if residual > 0.0:
                    residual = 0.0
            var weighted_response = (
                response * voxel.dose_weight / voxel.dose_divisor
            )
            numerator += (
                residual
                * voxel.dose_weight
                / voxel.dose_divisor
                * weighted_response
            )
            denominator += weighted_response * weighted_response
            if (
                problem.settings.flags & OPTIMIZER_FLAG_ROBUST_INCLUDE_DMAX
            ) != UInt32(0) and voxel.prescribed_dose > 0.0:
                residual = (
                    abs_f64(voxel.prescribed_dose) - dose_max[voxel_index]
                )
                weighted_response = (
                    rmax * voxel.maximum_dose_weight / voxel.dose_divisor
                )
                numerator += (
                    residual
                    * voxel.maximum_dose_weight
                    / voxel.dose_divisor
                    * weighted_response
                )
                denominator += weighted_response * weighted_response
        numerators[worker] = numerator
        denominators[worker] = denominator

    parallelize[exact_step_worker](CPU_THREADS, CPU_THREADS)
    var numerator = 0.0
    var denominator = 0.0
    for worker in range(CPU_THREADS):
        numerator += numerators[worker]
        denominator += denominators[worker]
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def packed_directional_dose(
    problem: OptimizationProblem,
    voxel: OptimizationVoxel,
    scenario_index: Int,
    direction: List[Float64],
) -> Float64:
    var scenario = problem.voxel_scenarios[
        Int(voxel.scenario_offset) + scenario_index
    ].copy()
    var total = 0.0
    for local_slice in range(Int(scenario.slice_count)):
        var packed_slice = problem.slices[
            Int(scenario.slice_offset) + local_slice
        ].copy()
        total += packed_slice_dot(problem, packed_slice, direction) * Float64(
            packed_slice.dose_coefficient
        )
    return total * MEV_TO_GY * problem.settings.fractions
