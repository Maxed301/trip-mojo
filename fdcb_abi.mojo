"""Thin C ABI for the packed FDCB optimizer."""

from std.algorithm import parallelize
from std.memory import OpaquePointer, UnsafePointer
from std.sys import get_defined_bool, get_defined_int

from clinical_dose import ClinicalDoseOutputV1, clinical_dose_compute_abi_v1

from fdcb_optimize import (
    FDCBOptimizationResult,
    evaluate_packed_iteration,
    optimize_packed_fdcb,
    optimize_packed_fdcb_accelerator,
)
from fdcb_problem import (
    FDCBFieldSliceV1,
    FDCBMinimumParticlePolicyV1,
    FDCBProblemV1,
    FDCBScenarioStateV1,
    FDCBSettingsV1,
    FDCBSliceV1,
    FDCBVoxelScenarioV1,
    FDCBVoxelV1,
)


comptime FDCB_ABI_ACCELERATOR = get_defined_bool[
    "FDCB_ABI_ACCELERATOR", False
]()
comptime FDCB_ABI_COPY_THREADS = get_defined_int[
    "FDCB_CPU_THREADS", 12
]()
comptime FDCB_OPTIMIZER_FDCB = UInt32(1)
comptime FDCB_DOSE_MS = UInt32(1)
comptime FDCB_DOSE_MSDB = UInt32(2)
comptime FDCB_BIOLOGY_NONE = UInt32(0)
comptime FDCB_BIOLOGY_LOW_DOSE = UInt32(1)
comptime FDCB_FLAG_BIOLOGICAL = UInt32(2)


@export("trip_clinical_dose_compute_v1", ABI="C")
def trip_clinical_dose_compute_v1(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    output: UnsafePointer[ClinicalDoseOutputV1, MutExternalOrigin],
    output_count: UInt64,
) -> Int32:
    return clinical_dose_compute_abi_v1(
        problem_pointer, output, output_count
    )


@fieldwise_init
struct ABIFieldSlice(Copyable, Movable):
    var field_index: UInt32
    var beam_index: UInt32
    var point_offset: UInt64
    var point_count: UInt32
    var raster_stride: UInt32
    var minimum_particles: Float64


@fieldwise_init
struct ABIVoxel(Copyable, Movable):
    var prescribed_dose: Float64
    var dose_weight: Float64
    var dose_divisor: Float64
    var maximum_dose_weight: Float64
    var initial_dose: Float64
    var prescribed_let: Float64
    var let_weight: Float64
    var overdose_tolerance: Float64
    var rbe_cut: Float64
    var rbe_alpha: Float64
    var rbe_beta: Float64
    var rbe_slope_max: Float64
    var rbe_damage_cut: Float64
    var initial_min_scenario: Int32
    var initial_max_scenario: Int32
    var scenario_offset: UInt64


@fieldwise_init
struct ABIVoxelScenario(Copyable, Movable):
    var slice_offset: UInt64
    var slice_count: UInt32
    var reserved: UInt32


@fieldwise_init
struct ABISlice(Copyable, Movable):
    var field_slice_index: UInt32
    var reserved: UInt32
    var coefficient_offset: UInt64
    var coefficient_count: UInt32
    var dose_coefficient: Float64
    var alpha_coefficient: Float64
    var sqrt_beta_coefficient: Float64
    var let_mix_coefficient: Float64
    var let_bar_coefficient: Float64


@fieldwise_init
struct ABIScenarioState(Copyable, Movable):
    var dose_minor: Float64
    var alpha_minor: Float64
    var sqrt_beta_minor: Float64
    var let_mix_minor: Float64


@fieldwise_init
struct ABIProblemView(Copyable, Movable):
    var version: UInt32
    var flags: UInt32
    var precision_mode: UInt32
    var minimum_particle_policy: UInt32
    var optimizer_algorithm: UInt32
    var dose_algorithm: UInt32
    var biology_model: UInt32
    var max_threads: UInt32
    var reserved: UInt32
    var rng_seed: UInt64
    var rng_front: UInt32
    var rng_rear: UInt32
    var rng_state_count: UInt64
    var field_count: UInt32
    var scenario_count: UInt32
    var max_iterations: UInt32
    var grace_iterations: UInt32
    var avoidance_voxel_count: UInt32
    var active_point_count: UInt32
    var total_iterations: UInt32
    var step_factor_iterations: UInt32
    var fractions: Float64
    var current_chi2: Float64
    var dose_p_weighted_avg2: Float64
    var epsilon: Float64
    var chi_square_limit: Float64
    var overdose_weight: Float64
    var prescribed_dose: Float64
    var configured_step_factor: Float64
    var current_step: Float64
    var update_elapsed: Float64
    var field_slice_count: UInt64
    var point_count: UInt64
    var voxel_count: UInt64
    var voxel_scenario_count: UInt64
    var slice_count: UInt64
    var coefficient_count: UInt64
    var field_slices: UnsafePointer[ABIFieldSlice, MutExternalOrigin]
    var rng_state: UnsafePointer[UInt32, MutExternalOrigin]
    var particles: UnsafePointer[Float64, MutExternalOrigin]
    var initial_direction: UnsafePointer[Float64, MutExternalOrigin]
    var initial_gradient: UnsafePointer[Float64, MutExternalOrigin]
    var point_active: UnsafePointer[UInt8, MutExternalOrigin]
    var voxels: UnsafePointer[ABIVoxel, MutExternalOrigin]
    var voxel_scenarios: UnsafePointer[ABIVoxelScenario, MutExternalOrigin]
    var scenario_states: UnsafePointer[ABIScenarioState, MutExternalOrigin]
    var slices: UnsafePointer[ABISlice, MutExternalOrigin]
    var coefficient_point_indices: UnsafePointer[UInt16, MutExternalOrigin]
    var coefficients: UnsafePointer[Float64, MutExternalOrigin]


@fieldwise_init
struct ABIResult(Copyable, Movable):
    var chi2: Float64
    var residual_percent: Float64
    var relative_chi2_change: Float64
    var step_factor: Float64
    var exact_step: Float64
    var iterations: UInt64
    var backtracks: UInt64
    var minimum_particle_deleted: UInt64
    var random_draws: UInt64
    var stop_reason: UInt32
    var reserved: UInt32


@fieldwise_init
struct ABIEvaluationResult(Copyable, Movable):
    var chi2: Float64
    var dose_p_weighted_avg2: Float64
    var gradient_norm: Float64
    var residual_percent: Float64


@export("trip_fdcb_evaluate_v1", ABI="C")
def trip_fdcb_evaluate_v1(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    gradient_out: UnsafePointer[Float64, MutExternalOrigin],
    gradient_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    try:
        var view = problem_pointer.bitcast[ABIProblemView]()[].copy()
        if gradient_out_count != view.point_count:
            return Int32(-2)
        var problem = copy_problem(view)
        var result = evaluate_packed_iteration(problem, problem.particles)
        for i in range(len(result.gradient)):
            gradient_out[i] = result.gradient[i]
        var output = result_pointer.bitcast[ABIEvaluationResult]()
        output[] = ABIEvaluationResult(
            result.chi2,
            result.weighted_dose2,
            result.gradient_norm,
            result.residual_percent(),
        )
        return Int32(0)
    except:
        return Int32(-1)


@export("trip_fdcb_optimize_v1", ABI="C")
def trip_fdcb_optimize_v1(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    return optimize_v1[False](
        problem_pointer,
        particles_out,
        particles_out_count,
        result_pointer,
    )


@export("trip_fdcb_optimize_accelerator_v1", ABI="C")
def trip_fdcb_optimize_accelerator_v1(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if FDCB_ABI_ACCELERATOR:
        return optimize_v1[True](
            problem_pointer,
            particles_out,
            particles_out_count,
            result_pointer,
        )
    else:
        return Int32(-3)


def optimize_v1[
    accelerator: Bool
](
    problem_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    try:
        var view = problem_pointer.bitcast[ABIProblemView]()[].copy()
        if particles_out_count != view.point_count:
            return Int32(-2)
        var problem = copy_problem(view)
        var result: FDCBOptimizationResult
        comptime if accelerator:
            result = optimize_packed_fdcb_accelerator(problem)
        else:
            result = optimize_packed_fdcb(problem)
        for i in range(len(result.particles)):
            particles_out[i] = result.particles[i]
        var output = result_pointer.bitcast[ABIResult]()
        output[] = ABIResult(
            result.chi2,
            result.residual_percent,
            result.relative_chi2_change,
            result.step_factor,
            result.exact_step,
            UInt64(result.iterations),
            result.backtracks,
            result.minimum_particle_deleted,
            result.random_draws,
            result.stop_reason,
            UInt32(0),
        )
        return Int32(0)
    except:
        return Int32(-1)


def copy_problem(view: ABIProblemView) raises -> FDCBProblemV1:
    if view.optimizer_algorithm != FDCB_OPTIMIZER_FDCB:
        raise Error("FDCB ABI accepts only the FDCB optimizer")
    if view.dose_algorithm != FDCB_DOSE_MS and view.dose_algorithm != FDCB_DOSE_MSDB:
        raise Error("FDCB ABI accepts only ms and msdb sparse matrices")
    var biological = (view.flags & FDCB_FLAG_BIOLOGICAL) != UInt32(0)
    if biological and view.biology_model != FDCB_BIOLOGY_LOW_DOSE:
        raise Error("biological FDCB ABI requires low-dose biology")
    if not biological and view.biology_model != FDCB_BIOLOGY_NONE:
        raise Error("physical FDCB ABI must not request a biology model")
    if view.max_threads != UInt32(FDCB_ABI_COPY_THREADS):
        raise Error("FDCB ABI max_threads does not match the CPU backend build")
    if view.reserved != UInt32(0):
        raise Error("FDCB ABI reserved contract field must be zero")
    var field_slices = List[FDCBFieldSliceV1]()
    for i in range(Int(view.field_slice_count)):
        var value = view.field_slices[i].copy()
        field_slices.append(
            FDCBFieldSliceV1(
                value.field_index,
                value.beam_index,
                value.point_offset,
                value.point_count,
                value.raster_stride,
                value.minimum_particles,
            )
        )
    var particles = copy_f64(view.particles, view.point_count)
    var rng_state = List[UInt32]()
    for i in range(Int(view.rng_state_count)):
        rng_state.append(view.rng_state[i])
    var direction = copy_f64(view.initial_direction, view.point_count)
    var gradient = copy_f64(view.initial_gradient, view.point_count)
    var active = List[UInt8]()
    for i in range(Int(view.point_count)):
        active.append(view.point_active[i])
    var voxels = List[FDCBVoxelV1]()
    for i in range(Int(view.voxel_count)):
        var value = view.voxels[i].copy()
        voxels.append(
            FDCBVoxelV1(
                value.prescribed_dose,
                value.dose_weight,
                value.dose_divisor,
                value.maximum_dose_weight,
                value.initial_dose,
                value.prescribed_let,
                value.let_weight,
                value.overdose_tolerance,
                value.rbe_cut,
                value.rbe_alpha,
                value.rbe_beta,
                value.rbe_slope_max,
                value.rbe_damage_cut,
                value.initial_min_scenario,
                value.initial_max_scenario,
                value.scenario_offset,
            )
        )
    var voxel_scenarios = List[FDCBVoxelScenarioV1]()
    for i in range(Int(view.voxel_scenario_count)):
        var value = view.voxel_scenarios[i].copy()
        voxel_scenarios.append(
            FDCBVoxelScenarioV1(value.slice_offset, value.slice_count)
        )
    var states = List[FDCBScenarioStateV1]()
    for i in range(Int(view.voxel_scenario_count)):
        var value = view.scenario_states[i].copy()
        states.append(
            FDCBScenarioStateV1(
                value.dose_minor,
                value.alpha_minor,
                value.sqrt_beta_minor,
                value.let_mix_minor,
            )
        )
    var slices = List[FDCBSliceV1]()
    slices.resize(
        Int(view.slice_count),
        FDCBSliceV1(
            UInt32(0),
            UInt64(0),
            UInt32(0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )

    @parameter
    def copy_slice(i: Int):
        var value = view.slices[i].copy()
        slices[i] = FDCBSliceV1(
            value.field_slice_index,
            value.coefficient_offset,
            value.coefficient_count,
            value.dose_coefficient,
            value.alpha_coefficient,
            value.sqrt_beta_coefficient,
            value.let_mix_coefficient,
            value.let_bar_coefficient,
        )

    parallelize[copy_slice](len(slices), FDCB_ABI_COPY_THREADS)
    var indices = List[UInt16]()
    var coefficients = List[Float64]()
    indices.resize(Int(view.coefficient_count), UInt16(0))
    coefficients.resize(Int(view.coefficient_count), 0.0)

    @parameter
    def copy_coefficient(i: Int):
        indices[i] = view.coefficient_point_indices[i]
        coefficients[i] = view.coefficients[i]

    parallelize[
        copy_coefficient
    ](len(coefficients), FDCB_ABI_COPY_THREADS)
    var settings = FDCBSettingsV1(
        view.flags,
        view.precision_mode,
        view.max_iterations,
        view.grace_iterations,
        view.avoidance_voxel_count,
        view.active_point_count,
        view.total_iterations,
        view.step_factor_iterations,
        view.fractions,
        view.current_chi2,
        view.dose_p_weighted_avg2,
        view.epsilon,
        view.chi_square_limit,
        view.overdose_weight,
        view.prescribed_dose,
        view.configured_step_factor,
        view.current_step,
        view.update_elapsed,
    )
    var problem = FDCBProblemV1(
        view.version,
        settings^,
        FDCBMinimumParticlePolicyV1(
            view.minimum_particle_policy,
            view.rng_seed,
            view.rng_front,
            view.rng_rear,
        ),
        rng_state^,
        view.field_count,
        view.scenario_count,
        field_slices^,
        particles^,
        direction^,
        gradient^,
        active^,
        voxels^,
        voxel_scenarios^,
        states^,
        slices^,
        indices^,
        coefficients^,
    )
    return problem^


def copy_f64(
    pointer: UnsafePointer[Float64, MutExternalOrigin], count: UInt64
) -> List[Float64]:
    var values = List[Float64]()
    for i in range(Int(count)):
        values.append(pointer[i])
    return values^
