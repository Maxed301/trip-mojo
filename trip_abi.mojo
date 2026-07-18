"""Thin C ABI for the packed optimizer."""

from std.algorithm import parallelize
from std.memory import OpaquePointer, UnsafePointer, alloc
from std.sys import has_accelerator

from clinical_dose import (
    ClinicalDoseOutput,
    clinical_dose_compute_abi,
)
from clinical_dose_accelerator import clinical_dose_compute_accelerator_abi
from matrix_builder import (
    MatrixBuildResult,
    DeviceMatrix,
    build_device_matrix_from_abi,
    release_matrix_build_buffers_from_abi,
    destroy_device_matrix_from_abi,
)

from optimizer import (
    OptimizationResult,
    evaluate_objective,
    optimize_on_cpu,
    optimize_on_device,
    optimize_on_devices,
    optimize_with_matrix_shards,
    optimize_with_matrix,
)
from optimization_problem import (
    FieldSlice,
    MinimumParticlePolicy,
    OptimizationProblem,
    ScenarioState,
    OptimizerSettings,
    DoseMatrixSlice,
    RobustScenario,
    OptimizationVoxel,
)


comptime HAS_ACCELERATOR_ABI = has_accelerator()
comptime OPTIMIZER_ABI_COPY_THREADS = 12
comptime DOSE_ALGORITHM_MS = UInt32(1)
comptime DOSE_ALGORITHM_MSDB = UInt32(2)
comptime OPTIMIZATION_RESULT_FINAL_MIN_PARTICLES_APPLIED = UInt32(1)


@export("trip_optimizer_build_device_matrix", ABI="C")
def trip_optimizer_build_device_matrix(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    storage_out: UnsafePointer[
        OpaquePointer[MutExternalOrigin], MutExternalOrigin
    ],
    result_out: UnsafePointer[MatrixBuildResult, MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        return build_device_matrix_from_abi(
            problem_pointer, storage_out, result_out
        )
    else:
        return Int32(-3)


@export("trip_optimizer_destroy_device_matrix", ABI="C")
def trip_optimizer_destroy_device_matrix(
    storage_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        return destroy_device_matrix_from_abi(storage_pointer)
    else:
        return Int32(-3)


@export("trip_optimizer_release_matrix_build_buffers", ABI="C")
def trip_optimizer_release_matrix_build_buffers(
    storage_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        return release_matrix_build_buffers_from_abi(storage_pointer)
    else:
        return Int32(-3)


@export("trip_compute_clinical_dose", ABI="C")
def trip_compute_clinical_dose(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    output: UnsafePointer[ClinicalDoseOutput, MutExternalOrigin],
    output_count: UInt64,
) -> Int32:
    return clinical_dose_compute_abi(problem_pointer, output, output_count)


@export("trip_compute_clinical_dose_on_device", ABI="C")
def trip_compute_clinical_dose_on_device(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    output: UnsafePointer[ClinicalDoseOutput, MutExternalOrigin],
    output_count: UInt64,
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        return clinical_dose_compute_accelerator_abi(
            problem_pointer, output, output_count
        )
    else:
        return Int32(-3)


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
    var matrix_slice_index: UInt32
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
    var flags: UInt32
    var minimum_particle_policy: UInt32
    var dose_algorithm: UInt32
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
struct ABIWritableArrays(Copyable, Movable):
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
struct ABIProblemStorage(Movable):
    var problem: OptimizationProblem


@fieldwise_init
struct ABIMatrixProblemStorage(Movable):
    var problem: OptimizationProblem
    var matrix_storage: OpaquePointer[MutExternalOrigin]
    var coefficient_count: Int


@fieldwise_init
struct ABIMultiMatrixProblemStorage(Movable):
    var problem: OptimizationProblem
    var matrix_storages: List[OpaquePointer[MutExternalOrigin]]
    var coefficient_counts: List[Int]
    var voxel_offsets: List[Int]
    var voxel_counts: List[Int]


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
    var flags: UInt32


@fieldwise_init
struct ABIEvaluationResult(Copyable, Movable):
    var chi2: Float64
    var dose_p_weighted_avg2: Float64
    var gradient_norm: Float64
    var residual_percent: Float64


def writable_arrays(
    ref[MutExternalOrigin] problem: OptimizationProblem,
) -> ABIWritableArrays:
    return ABIWritableArrays(
        problem.field_slices.unsafe_ptr().bitcast[ABIFieldSlice](),
        problem.host_rng_state.unsafe_ptr(),
        problem.particles.unsafe_ptr(),
        problem.initial_direction.unsafe_ptr(),
        problem.initial_gradient.unsafe_ptr(),
        problem.point_active.unsafe_ptr(),
        problem.voxels.unsafe_ptr().bitcast[ABIVoxel](),
        problem.voxel_scenarios.unsafe_ptr().bitcast[ABIVoxelScenario](),
        problem.scenario_states.unsafe_ptr().bitcast[ABIScenarioState](),
        problem.slices.unsafe_ptr().bitcast[ABISlice](),
        problem.coefficient_point_indices.unsafe_ptr(),
        problem.coefficients.unsafe_ptr(),
    )


@export("trip_optimizer_create_problem", ABI="C")
def trip_optimizer_create_problem(
    template_pointer: OpaquePointer[MutExternalOrigin],
    storage_out: UnsafePointer[
        OpaquePointer[MutExternalOrigin], MutExternalOrigin
    ],
    arrays_out: UnsafePointer[ABIWritableArrays, MutExternalOrigin],
) -> Int32:
    try:
        var view = template_pointer.bitcast[ABIProblemView]()[].copy()
        validate_abi_contract(view)
        var problem = allocate_problem(view)
        var storage = alloc[ABIProblemStorage](1)
        storage.init_pointee_move(ABIProblemStorage(problem^))
        arrays_out[] = writable_arrays(storage[].problem)
        storage_out[] = storage.bitcast[NoneType]()
        return Int32(0)
    except error:
        print("optimizer storage create ABI error:", error)
        return Int32(-1)


@export("trip_optimizer_create_problem_with_matrix", ABI="C")
def trip_optimizer_create_problem_with_matrix(
    template_pointer: OpaquePointer[MutExternalOrigin],
    matrix_pointer: OpaquePointer[MutExternalOrigin],
    storage_out: UnsafePointer[
        OpaquePointer[MutExternalOrigin], MutExternalOrigin
    ],
    arrays_out: UnsafePointer[ABIWritableArrays, MutExternalOrigin],
) -> Int32:
    try:
        var view = template_pointer.bitcast[ABIProblemView]()[].copy()
        validate_abi_contract(view)
        var matrix = matrix_pointer.bitcast[DeviceMatrix]()
        if UInt64(matrix[].entry_count) != view.coefficient_count:
            raise Error("matrix and optimizer coefficient counts differ")
        var problem = allocate_problem(view, allocate_coefficients=False)
        var storage = alloc[ABIMatrixProblemStorage](1)
        storage.init_pointee_move(
            ABIMatrixProblemStorage(
                problem^, matrix_pointer, matrix[].entry_count
            )
        )
        arrays_out[] = writable_arrays(storage[].problem)
        storage_out[] = storage.bitcast[NoneType]()
        return Int32(0)
    except error:
        print("dose matrix problem storage create ABI error:", error)
        return Int32(-1)


@export("trip_optimizer_create_problem_with_matrix_shards", ABI="C")
def trip_optimizer_create_problem_with_matrix_shards(
    template_pointer: OpaquePointer[MutExternalOrigin],
    matrix_pointers: UnsafePointer[
        OpaquePointer[MutExternalOrigin], MutExternalOrigin
    ],
    matrix_entry_counts: UnsafePointer[UInt64, MutExternalOrigin],
    voxel_offsets_pointer: UnsafePointer[UInt64, MutExternalOrigin],
    voxel_counts_pointer: UnsafePointer[UInt64, MutExternalOrigin],
    device_count: UInt32,
    storage_out: UnsafePointer[
        OpaquePointer[MutExternalOrigin], MutExternalOrigin
    ],
    arrays_out: UnsafePointer[ABIWritableArrays, MutExternalOrigin],
) -> Int32:
    try:
        if device_count < UInt32(2) or device_count > UInt32(3):
            return Int32(-2)
        var view = template_pointer.bitcast[ABIProblemView]()[].copy()
        validate_abi_contract(view)
        var matrices = List[OpaquePointer[MutExternalOrigin]]()
        var counts = List[Int]()
        var voxel_offsets = List[Int]()
        var voxel_counts = List[Int]()
        var total_entries = UInt64(0)
        var expected_voxel = UInt64(0)
        for device in range(Int(device_count)):
            var matrix_pointer = matrix_pointers[device]
            var matrix = matrix_pointer.bitcast[DeviceMatrix]()
            var count = matrix_entry_counts[device]
            if (
                count == UInt64(0)
                or UInt64(matrix[].entry_count) != count
                or voxel_offsets_pointer[device] != expected_voxel
                or voxel_counts_pointer[device] == UInt64(0)
            ):
                raise Error("invalid external dose matrix shard")
            matrices.append(matrix_pointer)
            counts.append(Int(count))
            voxel_offsets.append(Int(expected_voxel))
            voxel_counts.append(Int(voxel_counts_pointer[device]))
            total_entries += count
            expected_voxel += voxel_counts_pointer[device]
        if total_entries != view.coefficient_count:
            raise Error("matrix and optimizer coefficient counts differ")
        if expected_voxel != view.voxel_count:
            raise Error("matrix shards do not cover optimizer voxels")
        var problem = allocate_problem(view, allocate_coefficients=False)
        var storage = alloc[ABIMultiMatrixProblemStorage](1)
        storage.init_pointee_move(
            ABIMultiMatrixProblemStorage(
                problem^,
                matrices^,
                counts^,
                voxel_offsets^,
                voxel_counts^,
            )
        )
        arrays_out[] = writable_arrays(storage[].problem)
        storage_out[] = storage.bitcast[NoneType]()
        return Int32(0)
    except error:
        print("optimizer multi-matrix problem storage create ABI error:", error)
        return Int32(-1)


@export("trip_optimizer_destroy_problem", ABI="C")
def trip_optimizer_destroy_problem(
    storage_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    var storage = storage_pointer.bitcast[ABIProblemStorage]()
    storage.destroy_pointee()
    storage.free()
    return Int32(0)


@export("trip_optimizer_destroy_problem_with_matrix", ABI="C")
def trip_optimizer_destroy_problem_with_matrix(
    storage_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    var storage = storage_pointer.bitcast[ABIMatrixProblemStorage]()
    _ = destroy_device_matrix_from_abi(storage[].matrix_storage)
    storage.destroy_pointee()
    storage.free()
    return Int32(0)


@export("trip_optimizer_destroy_problem_with_matrix_shards", ABI="C")
def trip_optimizer_destroy_problem_with_matrix_shards(
    storage_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    var storage = storage_pointer.bitcast[ABIMultiMatrixProblemStorage]()
    for matrix in storage[].matrix_storages:
        _ = destroy_device_matrix_from_abi(matrix)
    storage.destroy_pointee()
    storage.free()
    return Int32(0)


@export("trip_optimizer_optimize_problem", ABI="C")
def trip_optimizer_optimize_problem(
    storage_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    return optimize_owned_problem[False](
        storage_pointer,
        particles_out,
        particles_out_count,
        result_pointer,
    )


@export("trip_optimizer_optimize_problem_on_device", ABI="C")
def trip_optimizer_optimize_problem_on_device(
    storage_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        return optimize_owned_problem[True](
            storage_pointer,
            particles_out,
            particles_out_count,
            result_pointer,
        )
    else:
        return Int32(-3)


@export("trip_optimizer_optimize_problem_on_devices", ABI="C")
def trip_optimizer_optimize_problem_on_devices(
    storage_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    device_count: UInt32,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        try:
            if device_count < UInt32(2) or device_count > UInt32(3):
                return Int32(-2)
            var storage = storage_pointer.bitcast[ABIProblemStorage]()
            if particles_out_count != UInt64(len(storage[].problem.particles)):
                return Int32(-2)
            var result = optimize_on_devices(
                storage[].problem, Int(device_count)
            )
            return write_optimization_result(
                result, particles_out, result_pointer
            )
        except error:
            print("optimizer multi-device optimize ABI error:", error)
            return Int32(-1)
    else:
        return Int32(-3)


@export("trip_optimizer_optimize_matrix_problem", ABI="C")
def trip_optimizer_optimize_matrix_problem(
    storage_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        try:
            var storage = storage_pointer.bitcast[ABIMatrixProblemStorage]()
            if particles_out_count != UInt64(len(storage[].problem.particles)):
                return Int32(-2)
            var result = optimize_with_matrix(
                storage[].problem,
                storage[].matrix_storage,
                storage[].coefficient_count,
            )
            return write_optimization_result(
                result, particles_out, result_pointer
            )
        except error:
            print("dose matrix problem optimize ABI error:", error)
            return Int32(-1)
    else:
        return Int32(-3)


@export("trip_optimizer_optimize_matrix_problem_shards", ABI="C")
def trip_optimizer_optimize_matrix_problem_shards(
    storage_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        try:
            var storage = storage_pointer.bitcast[
                ABIMultiMatrixProblemStorage
            ]()
            if particles_out_count != UInt64(len(storage[].problem.particles)):
                return Int32(-2)
            var result = optimize_with_matrix_shards(
                storage[].problem,
                storage[].matrix_storages,
                storage[].coefficient_counts,
                storage[].voxel_offsets,
                storage[].voxel_counts,
            )
            return write_optimization_result(
                result, particles_out, result_pointer
            )
        except error:
            print("optimizer multi-matrix optimize ABI error:", error)
            return Int32(-1)
    else:
        return Int32(-3)


@export("trip_optimizer_evaluate_view", ABI="C")
def trip_optimizer_evaluate_view(
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
        return evaluate_problem(problem, gradient_out, result_pointer)
    except error:
        print("optimizer evaluate ABI error:", error)
        return Int32(-1)


@export("trip_optimizer_evaluate_problem", ABI="C")
def trip_optimizer_evaluate_problem(
    storage_pointer: OpaquePointer[MutExternalOrigin],
    gradient_out: UnsafePointer[Float64, MutExternalOrigin],
    gradient_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    try:
        var storage = storage_pointer.bitcast[ABIProblemStorage]()
        if gradient_out_count != UInt64(len(storage[].problem.particles)):
            return Int32(-2)
        return evaluate_problem(storage[].problem, gradient_out, result_pointer)
    except error:
        print("optimizer storage evaluate ABI error:", error)
        return Int32(-1)


def evaluate_problem(
    problem: OptimizationProblem,
    gradient_out: UnsafePointer[Float64, MutExternalOrigin],
    result_pointer: OpaquePointer[MutExternalOrigin],
) raises -> Int32:
    var result = evaluate_objective(problem, problem.particles)
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


@export("trip_optimizer_optimize_view", ABI="C")
def trip_optimizer_optimize_view(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    return optimize_view[False](
        problem_pointer,
        particles_out,
        particles_out_count,
        result_pointer,
    )


@export("trip_optimizer_optimize_view_on_device", ABI="C")
def trip_optimizer_optimize_view_on_device(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    comptime if HAS_ACCELERATOR_ABI:
        return optimize_view[True](
            problem_pointer,
            particles_out,
            particles_out_count,
            result_pointer,
        )
    else:
        return Int32(-3)


def optimize_view[
    use_device: Bool
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
        return run_selected_backend[use_device](
            problem, particles_out, result_pointer
        )
    except error:
        print("optimizer optimize ABI error:", error)
        return Int32(-1)


def optimize_owned_problem[
    use_device: Bool
](
    storage_pointer: OpaquePointer[MutExternalOrigin],
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    particles_out_count: UInt64,
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    try:
        var storage = storage_pointer.bitcast[ABIProblemStorage]()
        if particles_out_count != UInt64(len(storage[].problem.particles)):
            return Int32(-2)
        return run_selected_backend[use_device](
            storage[].problem, particles_out, result_pointer
        )
    except error:
        print("optimizer storage optimize ABI error:", error)
        return Int32(-1)


def run_selected_backend[
    use_device: Bool
](
    mut problem: OptimizationProblem,
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    result_pointer: OpaquePointer[MutExternalOrigin],
) raises -> Int32:
    var result: OptimizationResult
    comptime if use_device:
        result = optimize_on_device(problem)
    else:
        result = optimize_on_cpu(problem)
    return write_optimization_result(result, particles_out, result_pointer)


def write_optimization_result(
    result: OptimizationResult,
    particles_out: UnsafePointer[Float64, MutExternalOrigin],
    result_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
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
        OPTIMIZATION_RESULT_FINAL_MIN_PARTICLES_APPLIED,
    )
    return Int32(0)


def validate_abi_contract(view: ABIProblemView) raises:
    if (
        view.dose_algorithm != DOSE_ALGORITHM_MS
        and view.dose_algorithm != DOSE_ALGORITHM_MSDB
    ):
        raise Error("optimizer ABI accepts only ms and msdb sparse matrices")


def settings_from_view(view: ABIProblemView) -> OptimizerSettings:
    return OptimizerSettings(
        view.flags,
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


def allocate_problem(
    view: ABIProblemView, allocate_coefficients: Bool = True
) -> OptimizationProblem:
    var field_slices = List[FieldSlice](
        unsafe_uninit_length=Int(view.field_slice_count)
    )
    var rng_state = List[UInt32](unsafe_uninit_length=Int(view.rng_state_count))
    var particles = List[Float64](unsafe_uninit_length=Int(view.point_count))
    var direction = List[Float64](unsafe_uninit_length=Int(view.point_count))
    var gradient = List[Float64](unsafe_uninit_length=Int(view.point_count))
    var active = List[UInt8](unsafe_uninit_length=Int(view.point_count))
    var voxels = List[OptimizationVoxel](
        unsafe_uninit_length=Int(view.voxel_count)
    )
    var voxel_scenarios = List[RobustScenario](
        unsafe_uninit_length=Int(view.voxel_scenario_count)
    )
    var states = List[ScenarioState](
        unsafe_uninit_length=Int(view.voxel_scenario_count)
    )
    var slices = List[DoseMatrixSlice](
        unsafe_uninit_length=Int(view.slice_count)
    )
    var indices = List[UInt16]()
    var coefficients = List[Float64]()
    if allocate_coefficients:
        indices = List[UInt16](unsafe_uninit_length=Int(view.coefficient_count))
        coefficients = List[Float64](
            unsafe_uninit_length=Int(view.coefficient_count)
        )
    return OptimizationProblem(
        settings_from_view(view),
        MinimumParticlePolicy(
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


def copy_problem(view: ABIProblemView) raises -> OptimizationProblem:
    validate_abi_contract(view)
    var problem = allocate_problem(view)
    var field_slices = problem.field_slices.unsafe_ptr().bitcast[
        ABIFieldSlice
    ]()
    var voxels = problem.voxels.unsafe_ptr().bitcast[ABIVoxel]()
    var voxel_scenarios = problem.voxel_scenarios.unsafe_ptr().bitcast[
        ABIVoxelScenario
    ]()
    var states = problem.scenario_states.unsafe_ptr().bitcast[
        ABIScenarioState
    ]()
    var slices = problem.slices.unsafe_ptr().bitcast[ABISlice]()
    for i in range(Int(view.field_slice_count)):
        field_slices[i] = view.field_slices[i].copy()
    for i in range(Int(view.rng_state_count)):
        problem.host_rng_state[i] = view.rng_state[i]
    for i in range(Int(view.point_count)):
        problem.particles[i] = view.particles[i]
        problem.initial_direction[i] = view.initial_direction[i]
        problem.initial_gradient[i] = view.initial_gradient[i]
        problem.point_active[i] = view.point_active[i]
    for i in range(Int(view.voxel_count)):
        voxels[i] = view.voxels[i].copy()
    for i in range(Int(view.voxel_scenario_count)):
        voxel_scenarios[i] = view.voxel_scenarios[i].copy()
        states[i] = view.scenario_states[i].copy()

    @parameter
    def copy_slice(i: Int):
        slices[i] = view.slices[i].copy()

    parallelize[copy_slice](Int(view.slice_count), OPTIMIZER_ABI_COPY_THREADS)

    @parameter
    def copy_coefficient(i: Int):
        problem.coefficient_point_indices[i] = view.coefficient_point_indices[i]
        problem.coefficients[i] = view.coefficients[i]

    parallelize[copy_coefficient](
        Int(view.coefficient_count), OPTIMIZER_ABI_COPY_THREADS
    )
    return problem^
