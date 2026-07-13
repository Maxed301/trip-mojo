"""Shared accelerator primitives for packed FDCB backends."""

from layout import Idx, TensorLayout, TileTensor, row_major
from std.atomic import Atomic, Ordering
from std.gpu import WARP_SIZE, barrier, global_idx, lane_id, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.memory import AddressSpace
from std.math import sqrt
from std.memory import bitcast, stack_allocation
from std.sys import get_defined_bool

from fdcb_problem import (
    FDCBProblemV1,
    FDCB_FLAG_BIOLOGICAL,
    FDCB_FLAG_ROBUST_INCLUDE_DMAX,
    FDCB_FLOAT64_EPSILON,
    FDCB_MEV_TO_GY,
    FDCB_PRECISION_MIXED32,
)

comptime FDCB_ACCELERATOR_BLOCK_SIZE = 128
comptime FDCB_REDUCTION_BLOCK_SIZE = 256
comptime FDCB_ACCELERATOR_MIXED32 = get_defined_bool["FDCB_MIXED32", False]()
comptime FDCB_ACCELERATOR_DTYPE = (
    DType.float32 if FDCB_ACCELERATOR_MIXED32 else DType.float64
)
comptime FDCB_ACCELERATOR_MEV_TO_GY = Scalar[FDCB_ACCELERATOR_DTYPE](
    FDCB_MEV_TO_GY
)
comptime FDCB_ACCELERATOR_EPSILON = Scalar[FDCB_ACCELERATOR_DTYPE](
    FDCB_FLOAT64_EPSILON
)


def fdcb_sqrt64(value: Float64) -> Float64:
    """Portable Float64 Newton sqrt for accelerator targets."""
    if value <= 0.0:
        return 0.0
    var bits = bitcast[DType.uint64](value)
    var root = bitcast[DType.float64]((bits >> 1) + UInt64(0x1FF8000000000000))
    for _ in range(7):
        root = 0.5 * (root + value / root)
    var root_bits = bitcast[DType.uint64](root)
    if root * root > value:
        root_bits -= UInt64(1)
        root = bitcast[DType.float64](root_bits)
    var upper = bitcast[DType.float64](root_bits + UInt64(1))
    if value - root * root > upper * upper - value:
        return upper
    return root


def fdcb_accelerator_sqrt(
    value: Scalar[FDCB_ACCELERATOR_DTYPE],
) -> Scalar[FDCB_ACCELERATOR_DTYPE]:
    comptime if FDCB_ACCELERATOR_MIXED32:
        return sqrt(value)
    else:
        return Scalar[FDCB_ACCELERATOR_DTYPE](fdcb_sqrt64(Float64(value)))


@fieldwise_init
struct FDCBAcceleratorEvaluation(Copyable, Movable):
    var scenario_bio: List[Float64]
    var dose_min: List[Float64]
    var dose_max: List[Float64]
    var min_scenario: List[Int32]
    var max_scenario: List[Int32]
    var chi2_per_voxel: List[Float64]
    var weighted_per_voxel: List[Float64]
    var gradient: List[Float64]
    var total_chi2: Float64
    var total_weighted: Float64
    var gradient_norm: Float64

    def chi2(self) -> Float64:
        return self.total_chi2

    def weighted_dose2(self) -> Float64:
        return self.total_weighted


@fieldwise_init
struct FDCBAcceleratorMetrics(Copyable, Movable):
    var chi2: Float64
    var weighted_dose2: Float64
    var gradient_norm: Float64


def _copy_list_to_device[
    dtype: DType
](context: DeviceContext, values: List[Scalar[dtype]]) raises -> DeviceBuffer[
    dtype
]:
    """Stage one packed array at a time to bound setup host memory."""
    var count = len(values)
    var host = context.enqueue_create_host_buffer[dtype](count)
    var tensor = TileTensor(host, row_major(Idx(count)))
    comptime assert tensor.flat_rank == 1
    for i in range(count):
        tensor[i] = rebind[tensor.ElementType](values[i])
    var device = context.enqueue_create_buffer[dtype](count)
    context.enqueue_copy(device, host)
    context.synchronize()
    return device^


def _copy_coefficients_to_device(
    context: DeviceContext, values: List[Float64]
) raises -> DeviceBuffer[FDCB_ACCELERATOR_DTYPE]:
    """Store reference coefficients exactly; mixed32 narrows explicitly."""
    var count = len(values)
    var host = context.enqueue_create_host_buffer[FDCB_ACCELERATOR_DTYPE](count)
    var tensor = TileTensor(host, row_major(Idx(count)))
    comptime assert tensor.flat_rank == 1
    for i in range(count):
        tensor[i] = Scalar[FDCB_ACCELERATOR_DTYPE](values[i])
    var device = context.enqueue_create_buffer[FDCB_ACCELERATOR_DTYPE](count)
    context.enqueue_copy(device, host)
    context.synchronize()
    return device^


def _coefficient_index_bits(problem: FDCBProblemV1) -> Int:
    var maximum_points = 0
    for field_slice in problem.field_slices:
        if Int(field_slice.point_count) > maximum_points:
            maximum_points = Int(field_slice.point_count)
    if maximum_points <= 256:
        return 8
    if maximum_points <= 4096:
        return 12
    return 16


def _pack_coefficient_indices_to_device(
    context: DeviceContext, values: List[UInt16], bits: Int
) raises -> DeviceBuffer[DType.uint32]:
    """Losslessly compact slice-local UInt16 indices for resident GPU use."""
    var word_count = (len(values) * bits + 31) // 32
    var host = context.enqueue_create_host_buffer[DType.uint32](word_count)
    var tensor = TileTensor(host, row_major(Idx(word_count)))
    comptime assert tensor.flat_rank == 1
    var accumulator = UInt64(0)
    var used = 0
    var word = 0
    for value in values:
        accumulator |= UInt64(value) << UInt64(used)
        used += bits
        if used >= 32:
            tensor[word] = UInt32(accumulator & UInt64(0xFFFFFFFF))
            word += 1
            accumulator >>= 32
            used -= 32
    if used > 0:
        tensor[word] = UInt32(accumulator)
    var device = context.enqueue_create_buffer[DType.uint32](word_count)
    context.enqueue_copy(device, host)
    context.synchronize()
    return device^


@always_inline
def _packed_coefficient_point[
    PointLayout: TensorLayout
](
    coefficient_points: TileTensor[DType.uint32, PointLayout, MutAnyOrigin],
    coefficient: Int,
    bits: Int,
) -> Int:
    comptime assert coefficient_points.flat_rank == 1
    var bit = coefficient * bits
    var word = bit // 32
    var shift = bit % 32
    var packed = UInt64(
        rebind[Scalar[DType.uint32]](coefficient_points[word])
    ) >> UInt64(shift)
    if shift + bits > 32:
        packed |= UInt64(
            rebind[Scalar[DType.uint32]](coefficient_points[word + 1])
        ) << UInt64(32 - shift)
    var mask = (UInt64(1) << UInt64(bits)) - UInt64(1)
    return Int(packed & mask)


def sparse_slice_dot_kernel[
    FieldLayout: TensorLayout,
    SliceLayout: TensorLayout,
    CoefficientPointLayout: TensorLayout,
    CoefficientLayout: TensorLayout,
    PointLayout: TensorLayout,
](
    field_point_offsets: TileTensor[DType.uint64, FieldLayout, MutAnyOrigin],
    slice_metadata: TileTensor[DType.uint64, SliceLayout, MutAnyOrigin],
    slice_offsets: TileTensor[DType.uint64, SliceLayout, MutAnyOrigin],
    coefficient_points: TileTensor[
        DType.uint32, CoefficientPointLayout, MutAnyOrigin
    ],
    coefficients: TileTensor[FDCB_ACCELERATOR_DTYPE, CoefficientLayout, MutAnyOrigin],
    particles: TileTensor[FDCB_ACCELERATOR_DTYPE, PointLayout, MutAnyOrigin],
    output: TileTensor[FDCB_ACCELERATOR_DTYPE, SliceLayout, MutAnyOrigin],
    slice_count: Int,
    coefficient_index_bits: Int,
):
    comptime assert field_point_offsets.flat_rank == 1
    comptime assert slice_metadata.flat_rank == 1
    comptime assert slice_offsets.flat_rank == 1
    comptime assert coefficient_points.flat_rank == 1
    comptime assert coefficients.flat_rank == 1
    comptime assert particles.flat_rank == 1
    comptime assert output.flat_rank == 1
    var lane = lane_id()
    var slice = global_idx.x // WARP_SIZE
    var total: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    if slice < slice_count:
        var metadata = rebind[Scalar[DType.uint64]](slice_metadata[slice])
        var field = Int(metadata >> 32)
        var point_base = Int(
            rebind[Scalar[DType.uint64]](field_point_offsets[field])
        )
        var offset = Int(rebind[Scalar[DType.uint64]](slice_offsets[slice]))
        var count = Int(metadata & UInt64(0xFFFFFFFF))
        for entry in range(Int(lane), count, WARP_SIZE):
            var coefficient = offset + entry
            var point = point_base + _packed_coefficient_point(
                coefficient_points, coefficient, coefficient_index_bits
            )
            total += Scalar[FDCB_ACCELERATOR_DTYPE](
                rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](coefficients[coefficient])
            ) * rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](particles[point])
    var sums = stack_allocation[
        FDCB_ACCELERATOR_BLOCK_SIZE,
        Scalar[FDCB_ACCELERATOR_DTYPE],
        address_space=AddressSpace.SHARED,
    ]()
    var thread = thread_idx.x
    sums[thread] = total
    barrier()
    var stride = WARP_SIZE // 2
    while stride > 0:
        if lane < stride:
            sums[thread] += sums[thread + stride]
        barrier()
        stride //= 2
    if lane == 0 and slice < slice_count:
        total = sums[thread]
        output[slice] = rebind[output.ElementType](total)


def biological_moment_kernel[
    ScenarioLayout: TensorLayout,
    SliceLayout: TensorLayout,
    StateLayout: TensorLayout,
    FactorLayout: TensorLayout,
    MomentLayout: TensorLayout,
](
    scenario_slice_offsets: TileTensor[
        DType.uint64, ScenarioLayout, MutAnyOrigin
    ],
    scenario_slice_counts: TileTensor[
        DType.uint32, ScenarioLayout, MutAnyOrigin
    ],
    scenario_state: TileTensor[
        FDCB_ACCELERATOR_DTYPE, StateLayout, MutAnyOrigin
    ],
    slice_factors: TileTensor[FDCB_ACCELERATOR_DTYPE, FactorLayout, MutAnyOrigin],
    slice_dots: TileTensor[FDCB_ACCELERATOR_DTYPE, SliceLayout, MutAnyOrigin],
    moments: TileTensor[FDCB_ACCELERATOR_DTYPE, MomentLayout, MutAnyOrigin],
    scenario_count: Int,
):
    comptime assert scenario_slice_offsets.flat_rank == 1
    comptime assert scenario_slice_counts.flat_rank == 1
    comptime assert scenario_state.flat_rank == 1
    comptime assert slice_factors.flat_rank == 1
    comptime assert slice_dots.flat_rank == 1
    comptime assert moments.flat_rank == 1
    var scenario = global_idx.x
    if scenario >= scenario_count:
        return
    var state = scenario * 4
    var dose = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](scenario_state[state])
    var alpha = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        scenario_state[state + 1]
    )
    var sqrt_beta = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        scenario_state[state + 2]
    )
    var let_mix = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        scenario_state[state + 3]
    )
    var let_bar: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    var slice_offset = Int(
        rebind[Scalar[DType.uint64]](scenario_slice_offsets[scenario])
    )
    var slice_count = Int(
        rebind[Scalar[DType.uint32]](scenario_slice_counts[scenario])
    )
    for local_slice in range(slice_count):
        var slice = slice_offset + local_slice
        var dot = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_dots[slice])
        var factor = slice * 5
        dose += dot * Scalar[FDCB_ACCELERATOR_DTYPE](
            rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[factor])
        )
        alpha += dot * Scalar[FDCB_ACCELERATOR_DTYPE](
            rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[factor + 1])
        )
        sqrt_beta += dot * Scalar[FDCB_ACCELERATOR_DTYPE](
            rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[factor + 2])
        )
        let_mix += dot * Scalar[FDCB_ACCELERATOR_DTYPE](
            rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[factor + 3])
        )
        let_bar += dot * Scalar[FDCB_ACCELERATOR_DTYPE](
            rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[factor + 4])
        )
    var output = scenario * 5
    moments[output] = rebind[moments.ElementType](dose)
    moments[output + 1] = rebind[moments.ElementType](alpha)
    moments[output + 2] = rebind[moments.ElementType](sqrt_beta)
    moments[output + 3] = rebind[moments.ElementType](let_mix)
    moments[output + 4] = rebind[moments.ElementType](let_bar)


def fdcb_objective_kernel[
    VoxelDataLayout: TensorLayout,
    MomentLayout: TensorLayout,
    BioLayout: TensorLayout,
    ResultLayout: TensorLayout,
    IndexLayout: TensorLayout,
](
    voxel_data: TileTensor[
        FDCB_ACCELERATOR_DTYPE, VoxelDataLayout, MutAnyOrigin
    ],
    moments: TileTensor[FDCB_ACCELERATOR_DTYPE, MomentLayout, MutAnyOrigin],
    scenario_bio: TileTensor[FDCB_ACCELERATOR_DTYPE, BioLayout, MutAnyOrigin],
    voxel_results: TileTensor[
        FDCB_ACCELERATOR_DTYPE, ResultLayout, MutAnyOrigin
    ],
    scenario_indices: TileTensor[DType.int32, IndexLayout, MutAnyOrigin],
    voxel_count: Int,
    scenario_count: Int,
    flags: UInt32,
    biological: UInt32,
):
    comptime assert voxel_data.flat_rank == 1
    comptime assert moments.flat_rank == 1
    comptime assert scenario_bio.flat_rank == 1
    comptime assert voxel_results.flat_rank == 1
    comptime assert scenario_indices.flat_rank == 1
    var voxel = global_idx.x
    if voxel >= voxel_count:
        return
    var voxel_offset = voxel * 13
    var prescribed = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset]
    )
    var prescribed_abs = prescribed
    if prescribed_abs < 0.0:
        prescribed_abs = -prescribed_abs
    var index = voxel * 2
    var initial_min_scenario = Int(
        rebind[Scalar[DType.int32]](scenario_indices[index])
    )
    var initial_max_scenario = Int(
        rebind[Scalar[DType.int32]](scenario_indices[index + 1])
    )
    var dose_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 1]
    )
    var dose_divisor = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 2]
    )
    var maximum_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 3]
    )
    var initial_dose = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 4]
    )
    var prescribed_let = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 5]
    )
    var let_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 6]
    )
    var rbe_cut = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 8]
    )
    var rbe_alpha = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 9]
    )
    var rbe_beta = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 10]
    )
    var rbe_slope = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 11]
    )
    var rbe_damage_cut = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 12]
    )
    var scenario_base = voxel * scenario_count
    var dose_min: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    var dose_max: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    var min_scenario = 0
    var max_scenario = 0
    for scenario in range(scenario_count):
        var scenario_index = scenario_base + scenario
        var moment = scenario_index * 5
        var dose_phys = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](moments[moment])
        var alpha = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](moments[moment + 1])
        var sqrt_beta = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            moments[moment + 2]
        )
        var let_mix = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            moments[moment + 3]
        )
        var let_bar_sum = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            moments[moment + 4]
        )
        var dose = initial_dose
        var let_bar: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
        var dose_phs = let_mix * FDCB_ACCELERATOR_MEV_TO_GY
        var denominator: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
        if biological == UInt32(0):
            dose += dose_phys * FDCB_ACCELERATOR_MEV_TO_GY
        elif dose_phys > 0.0 and let_mix > 0.0:
            let_bar = let_bar_sum / let_mix
            if dose_phs <= rbe_cut:
                var damage = FDCB_ACCELERATOR_MEV_TO_GY * (
                    sqrt_beta * sqrt_beta * FDCB_ACCELERATOR_MEV_TO_GY + alpha
                )
                if rbe_beta != 0.0:
                    denominator = fdcb_accelerator_sqrt(
                        damage * rbe_beta * 4.0 + rbe_alpha * rbe_alpha
                    )
                    dose += (
                        dose_phys
                        * FDCB_ACCELERATOR_MEV_TO_GY
                        * ((denominator - rbe_alpha) / (rbe_beta * 2.0))
                        / dose_phs
                    )
                elif rbe_alpha != 0.0:
                    denominator = rbe_alpha
                    dose += (
                        dose_phys
                        * FDCB_ACCELERATOR_MEV_TO_GY
                        * (damage / rbe_alpha)
                        / dose_phs
                    )
            elif rbe_slope != 0.0:
                var cut_scale = rbe_cut / let_mix
                var damage = (
                    sqrt_beta * sqrt_beta * cut_scale + alpha
                ) * cut_scale + (dose_phs - rbe_cut) * rbe_slope
                var bio_dose = (damage - rbe_damage_cut) / rbe_slope + rbe_cut
                denominator = rbe_slope
                dose += (
                    dose_phys * FDCB_ACCELERATOR_MEV_TO_GY * bio_dose / dose_phs
                )
        var bio = scenario_index * 4
        scenario_bio[bio] = rebind[scenario_bio.ElementType](dose)
        scenario_bio[bio + 1] = rebind[scenario_bio.ElementType](let_bar)
        scenario_bio[bio + 2] = rebind[scenario_bio.ElementType](dose_phs)
        scenario_bio[bio + 3] = rebind[scenario_bio.ElementType](denominator)
        if scenario == 0:
            dose_min = dose
            dose_max = dose
        else:
            if dose < dose_min:
                dose_min = dose
                min_scenario = scenario
            if dose > dose_max:
                dose_max = dose
                max_scenario = scenario
    if prescribed_abs == 0.0:
        dose_min = 0.0
        dose_max = 0.0
        min_scenario = initial_min_scenario
        max_scenario = initial_max_scenario
    var selected = min_scenario
    var selected_dose = dose_min
    if prescribed < 0.0:
        selected = max_scenario
        selected_dose = dose_max
    var chi2: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    var weighted: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    if prescribed_abs != 0.0:
        var residual = prescribed_abs - selected_dose
        var weight = dose_weight / dose_divisor
        var selected_let = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            scenario_bio[(scenario_base + selected) * 4 + 1]
        )
        var let_residual: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
        if (
            biological != UInt32(0)
            and let_weight > FDCB_ACCELERATOR_EPSILON
            and prescribed_let > 0.0
        ):
            let_residual = prescribed_let - selected_let
            if prescribed > 0.0:
                if let_residual < 0.0:
                    let_residual = 0.0
            elif let_residual > 0.0:
                let_residual = 0.0
            let_residual *= let_weight * prescribed / prescribed_let
        if residual < 0.0 or prescribed > 0.0:
            chi2 += (residual * weight) * (residual * weight)
            chi2 += let_residual * let_residual
        weighted += (weight * prescribed) * (weight * prescribed)
        if (flags & FDCB_FLAG_ROBUST_INCLUDE_DMAX) != UInt32(
            0
        ) and prescribed > 0.0:
            var max_weight = maximum_weight / dose_divisor
            var max_residual = prescribed_abs - dose_max
            chi2 += (max_residual * max_weight) * (max_residual * max_weight)
            weighted += (max_weight * prescribed) * (max_weight * prescribed)
    var result = voxel * 4
    voxel_results[result] = rebind[voxel_results.ElementType](dose_min)
    voxel_results[result + 1] = rebind[voxel_results.ElementType](dose_max)
    voxel_results[result + 2] = rebind[voxel_results.ElementType](chi2)
    voxel_results[result + 3] = rebind[voxel_results.ElementType](weighted)
    scenario_indices[index] = rebind[scenario_indices.ElementType](
        Int32(min_scenario)
    )
    scenario_indices[index + 1] = rebind[scenario_indices.ElementType](
        Int32(max_scenario)
    )


def fdcb_slice_gradient_kernel[
    VoxelDataLayout: TensorLayout,
    ScenarioLayout: TensorLayout,
    MomentLayout: TensorLayout,
    BioLayout: TensorLayout,
    ResultLayout: TensorLayout,
    IndexLayout: TensorLayout,
    SliceLayout: TensorLayout,
    FactorLayout: TensorLayout,
](
    voxel_data: TileTensor[
        FDCB_ACCELERATOR_DTYPE, VoxelDataLayout, MutAnyOrigin
    ],
    scenario_slice_offsets: TileTensor[
        DType.uint64, ScenarioLayout, MutAnyOrigin
    ],
    scenario_slice_counts: TileTensor[
        DType.uint32, ScenarioLayout, MutAnyOrigin
    ],
    moments: TileTensor[FDCB_ACCELERATOR_DTYPE, MomentLayout, MutAnyOrigin],
    scenario_bio: TileTensor[FDCB_ACCELERATOR_DTYPE, BioLayout, MutAnyOrigin],
    voxel_results: TileTensor[
        FDCB_ACCELERATOR_DTYPE, ResultLayout, MutAnyOrigin
    ],
    scenario_indices: TileTensor[DType.int32, IndexLayout, MutAnyOrigin],
    slice_factors: TileTensor[FDCB_ACCELERATOR_DTYPE, FactorLayout, MutAnyOrigin],
    slice_gradient: TileTensor[
        FDCB_ACCELERATOR_DTYPE, SliceLayout, MutAnyOrigin
    ],
    voxel_count: Int,
    scenarios_per_voxel: Int,
    flags: UInt32,
    overdose_weight: Scalar[FDCB_ACCELERATOR_DTYPE],
    biological: UInt32,
):
    comptime assert voxel_data.flat_rank == 1
    comptime assert scenario_slice_offsets.flat_rank == 1
    comptime assert scenario_slice_counts.flat_rank == 1
    comptime assert moments.flat_rank == 1
    comptime assert scenario_bio.flat_rank == 1
    comptime assert voxel_results.flat_rank == 1
    comptime assert scenario_indices.flat_rank == 1
    comptime assert slice_factors.flat_rank == 1
    comptime assert slice_gradient.flat_rank == 1
    var voxel = global_idx.x
    if voxel >= voxel_count:
        return
    var voxel_offset = voxel * 13
    var prescribed = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset]
    )
    var prescribed_abs = prescribed
    if prescribed_abs < 0.0:
        prescribed_abs = -prescribed_abs
    var dose_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 1]
    )
    var dose_divisor = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 2]
    )
    var maximum_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 3]
    )
    var prescribed_let = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 5]
    )
    var let_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 6]
    )
    var overdose_tolerance = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 7]
    )
    var rbe_cut = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 8]
    )
    var rbe_slope = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[voxel_offset + 11]
    )
    var scenario_base = voxel * scenarios_per_voxel
    var index = voxel * 2
    var min_scenario = Int(rebind[Scalar[DType.int32]](scenario_indices[index]))
    var max_scenario = Int(
        rebind[Scalar[DType.int32]](scenario_indices[index + 1])
    )
    var pass_count = 1
    if (flags & FDCB_FLAG_ROBUST_INCLUDE_DMAX) != UInt32(
        0
    ) and prescribed > 0.0:
        pass_count = 2
    for gradient_pass in range(pass_count):
        var selected = max_scenario
        var selected_dose = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            voxel_results[voxel * 4 + 1]
        )
        var pass_weight = dose_weight
        if gradient_pass == 0:
            if prescribed > 0.0:
                selected = min_scenario
                selected_dose = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
                    voxel_results[voxel * 4]
                )
        else:
            pass_weight = maximum_weight
        var residual = prescribed_abs - selected_dose
        var divisor = dose_divisor
        if prescribed < 0.0 or gradient_pass == 1:
            if residual > 0.0:
                residual = 0.0
        elif residual < 0.0:
            if residual > prescribed * -overdose_tolerance:
                residual = 0.0
            else:
                residual += prescribed * overdose_tolerance
                divisor /= overdose_weight
        var scenario = scenario_base + selected
        var slice_offset = Int(
            rebind[Scalar[DType.uint64]](scenario_slice_offsets[scenario])
        )
        var slice_count = Int(
            rebind[Scalar[DType.uint32]](scenario_slice_counts[scenario])
        )
        var scaled_weight = pass_weight / divisor
        var factor = residual * scaled_weight * (2.0 * scaled_weight)
        if biological == UInt32(0):
            if residual == 0.0:
                continue
            for local_slice in range(slice_count):
                var slice = slice_offset + local_slice
                var dose_factor = Scalar[FDCB_ACCELERATOR_DTYPE](
                    rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[slice * 5])
                )
                var prior = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
                    slice_gradient[slice]
                )
                slice_gradient[slice] = rebind[slice_gradient.ElementType](
                    prior + factor * dose_factor * FDCB_ACCELERATOR_MEV_TO_GY
                )
            continue
        var moment = scenario * 5
        var let_mix = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            moments[moment + 3]
        )
        if residual == 0.0 or let_mix <= 0.0:
            continue
        var alpha = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](moments[moment + 1])
        var sqrt_beta = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            moments[moment + 2]
        )
        var bio = scenario * 4
        var let_bar = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            scenario_bio[bio + 1]
        )
        var dose_phs = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            scenario_bio[bio + 2]
        )
        var denominator = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            scenario_bio[bio + 3]
        )
        var let_residual: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
        if let_weight > FDCB_ACCELERATOR_EPSILON and prescribed_let > 0.0:
            let_residual = prescribed_let - let_bar
            if prescribed > 0.0:
                if let_residual < 0.0:
                    let_residual = 0.0
            elif let_residual > 0.0:
                let_residual = 0.0
            let_residual *= let_weight * prescribed / prescribed_let
        for local_slice in range(slice_count):
            var slice = slice_offset + local_slice
            var packed = slice * 5
            var slice_alpha = Scalar[FDCB_ACCELERATOR_DTYPE](
                rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[packed + 1])
            )
            var slice_sqrt_beta = Scalar[FDCB_ACCELERATOR_DTYPE](
                rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[packed + 2])
            )
            var slice_let_mix = Scalar[FDCB_ACCELERATOR_DTYPE](
                rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[packed + 3])
            )
            var slice_let_bar = Scalar[FDCB_ACCELERATOR_DTYPE](
                rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[packed + 4])
            )
            var let_prim = (slice_let_bar - let_bar * slice_let_mix) / let_mix
            var grad_bio: Scalar[FDCB_ACCELERATOR_DTYPE]
            if dose_phs <= rbe_cut:
                grad_bio = (
                    slice_alpha
                    + FDCB_ACCELERATOR_MEV_TO_GY
                    * sqrt_beta
                    * 2.0
                    * slice_sqrt_beta
                ) * FDCB_ACCELERATOR_MEV_TO_GY
            else:
                var cut_scale = rbe_cut / let_mix
                grad_bio = (
                    (
                        slice_alpha
                        - slice_let_mix * alpha / let_mix
                        + (
                            slice_sqrt_beta
                            - slice_let_mix * sqrt_beta / let_mix
                        )
                        * sqrt_beta
                        * 2.0
                        * cut_scale
                    )
                    * cut_scale
                    + slice_let_mix * FDCB_ACCELERATOR_MEV_TO_GY * rbe_slope
                )
            var scale = (
                factor * grad_bio / denominator + let_residual * let_prim
            )
            var prior = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
                slice_gradient[slice]
            )
            slice_gradient[slice] = rebind[slice_gradient.ElementType](
                prior + scale
            )


def sparse_gradient_backprojection_kernel[
    FieldLayout: TensorLayout,
    SliceLayout: TensorLayout,
    CoefficientPointLayout: TensorLayout,
    CoefficientLayout: TensorLayout,
    PointLayout: TensorLayout,
    ActiveLayout: TensorLayout,
](
    field_point_offsets: TileTensor[DType.uint64, FieldLayout, MutAnyOrigin],
    slice_metadata: TileTensor[DType.uint64, SliceLayout, MutAnyOrigin],
    slice_offsets: TileTensor[DType.uint64, SliceLayout, MutAnyOrigin],
    coefficient_points: TileTensor[
        DType.uint32, CoefficientPointLayout, MutAnyOrigin
    ],
    coefficients: TileTensor[FDCB_ACCELERATOR_DTYPE, CoefficientLayout, MutAnyOrigin],
    point_active: TileTensor[DType.uint8, PointLayout, MutAnyOrigin],
    particles: TileTensor[FDCB_ACCELERATOR_DTYPE, PointLayout, MutAnyOrigin],
    slice_gradient: TileTensor[
        FDCB_ACCELERATOR_DTYPE, SliceLayout, MutAnyOrigin
    ],
    active_slices: TileTensor[DType.uint32, ActiveLayout, MutAnyOrigin],
    gradient: TileTensor[FDCB_ACCELERATOR_DTYPE, PointLayout, MutAnyOrigin],
    active_count: Int,
    coefficient_index_bits: Int,
):
    comptime assert field_point_offsets.flat_rank == 1
    comptime assert slice_metadata.flat_rank == 1
    comptime assert slice_offsets.flat_rank == 1
    comptime assert coefficient_points.flat_rank == 1
    comptime assert coefficients.flat_rank == 1
    comptime assert point_active.flat_rank == 1
    comptime assert particles.flat_rank == 1
    comptime assert slice_gradient.flat_rank == 1
    comptime assert active_slices.flat_rank == 1
    comptime assert gradient.flat_rank == 1
    var active = global_idx.x // WARP_SIZE
    if active >= active_count:
        return
    var slice = Int(rebind[Scalar[DType.uint32]](active_slices[active]))
    var scale = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_gradient[slice])
    if scale == 0.0:
        return
    var metadata = rebind[Scalar[DType.uint64]](slice_metadata[slice])
    var field = Int(metadata >> 32)
    var point_base = Int(
        rebind[Scalar[DType.uint64]](field_point_offsets[field])
    )
    var offset = Int(rebind[Scalar[DType.uint64]](slice_offsets[slice]))
    var count = Int(metadata & UInt64(0xFFFFFFFF))
    for entry in range(Int(lane_id()), count, WARP_SIZE):
        var coefficient = offset + entry
        var point = point_base + _packed_coefficient_point(
            coefficient_points, coefficient, coefficient_index_bits
        )
        if (
            rebind[Scalar[DType.uint8]](point_active[point]) != UInt8(0)
            and rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](particles[point]) != 0.0
        ):
            var contribution = scale * Scalar[FDCB_ACCELERATOR_DTYPE](
                rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](coefficients[coefficient])
            )
            _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
                gradient.ptr + point, contribution
            )


def active_slice_list_kernel[
    SliceLayout: TensorLayout,
    CountLayout: TensorLayout,
    ActiveLayout: TensorLayout,
](
    slice_gradient: TileTensor[
        FDCB_ACCELERATOR_DTYPE, SliceLayout, MutAnyOrigin
    ],
    active_count: TileTensor[DType.uint32, CountLayout, MutAnyOrigin],
    active_slices: TileTensor[DType.uint32, ActiveLayout, MutAnyOrigin],
    slice_count: Int,
):
    comptime assert slice_gradient.flat_rank == 1
    comptime assert active_count.flat_rank == 1
    comptime assert active_slices.flat_rank == 1
    var slice = global_idx.x
    if slice >= slice_count:
        return
    if rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_gradient[slice]) != 0.0:
        var write = Atomic.fetch_add[ordering=Ordering.RELAXED](
            active_count.ptr, UInt32(1)
        )
        active_slices[Int(write)] = rebind[active_slices.ElementType](
            UInt32(slice)
        )


def metric_partial_reduction_kernel[
    ResultLayout: TensorLayout,
    PointLayout: TensorLayout,
    PartialLayout: TensorLayout,
](
    voxel_results: TileTensor[
        FDCB_ACCELERATOR_DTYPE, ResultLayout, MutAnyOrigin
    ],
    gradient: TileTensor[FDCB_ACCELERATOR_DTYPE, PointLayout, MutAnyOrigin],
    partials: TileTensor[FDCB_ACCELERATOR_DTYPE, PartialLayout, MutAnyOrigin],
    voxel_count: Int,
    point_count: Int,
):
    comptime assert voxel_results.flat_rank == 1
    comptime assert gradient.flat_rank == 1
    comptime assert partials.flat_rank == 1
    var thread = thread_idx.x
    var block = global_idx.x // FDCB_REDUCTION_BLOCK_SIZE
    var index = block * FDCB_REDUCTION_BLOCK_SIZE * 2 + thread
    var chi2: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    var weighted: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    var norm2: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    if index < voxel_count:
        chi2 = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            voxel_results[index * 4 + 2]
        )
        weighted = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            voxel_results[index * 4 + 3]
        )
    if index < point_count:
        var value = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](gradient[index])
        norm2 = value * value
    var next = index + FDCB_REDUCTION_BLOCK_SIZE
    if next < voxel_count:
        chi2 += rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            voxel_results[next * 4 + 2]
        )
        weighted += rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            voxel_results[next * 4 + 3]
        )
    if next < point_count:
        var value = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](gradient[next])
        norm2 += value * value
    var chi_shared = stack_allocation[
        FDCB_REDUCTION_BLOCK_SIZE,
        Scalar[FDCB_ACCELERATOR_DTYPE],
        address_space=AddressSpace.SHARED,
    ]()
    var weighted_shared = stack_allocation[
        FDCB_REDUCTION_BLOCK_SIZE,
        Scalar[FDCB_ACCELERATOR_DTYPE],
        address_space=AddressSpace.SHARED,
    ]()
    var norm_shared = stack_allocation[
        FDCB_REDUCTION_BLOCK_SIZE,
        Scalar[FDCB_ACCELERATOR_DTYPE],
        address_space=AddressSpace.SHARED,
    ]()
    chi_shared[thread] = chi2
    weighted_shared[thread] = weighted
    norm_shared[thread] = norm2
    barrier()
    var stride = FDCB_REDUCTION_BLOCK_SIZE // 2
    while stride > 0:
        if thread < stride:
            chi_shared[thread] += chi_shared[thread + stride]
            weighted_shared[thread] += weighted_shared[thread + stride]
            norm_shared[thread] += norm_shared[thread + stride]
        barrier()
        stride //= 2
    if thread == 0:
        var output = block * 3
        partials[output] = rebind[partials.ElementType](chi_shared[0])
        partials[output + 1] = rebind[partials.ElementType](weighted_shared[0])
        partials[output + 2] = rebind[partials.ElementType](norm_shared[0])


def exact_step_terms_kernel[
    VoxelDataLayout: TensorLayout,
    ScenarioLayout: TensorLayout,
    SliceLayout: TensorLayout,
    FactorLayout: TensorLayout,
    ResultLayout: TensorLayout,
    IndexLayout: TensorLayout,
    TermLayout: TensorLayout,
](
    voxel_data: TileTensor[
        FDCB_ACCELERATOR_DTYPE, VoxelDataLayout, MutAnyOrigin
    ],
    scenario_slice_offsets: TileTensor[
        DType.uint64, ScenarioLayout, MutAnyOrigin
    ],
    scenario_slice_counts: TileTensor[
        DType.uint32, ScenarioLayout, MutAnyOrigin
    ],
    slice_factors: TileTensor[FDCB_ACCELERATOR_DTYPE, FactorLayout, MutAnyOrigin],
    slice_dots: TileTensor[FDCB_ACCELERATOR_DTYPE, SliceLayout, MutAnyOrigin],
    voxel_results: TileTensor[
        FDCB_ACCELERATOR_DTYPE, ResultLayout, MutAnyOrigin
    ],
    scenario_indices: TileTensor[DType.int32, IndexLayout, MutAnyOrigin],
    terms: TileTensor[FDCB_ACCELERATOR_DTYPE, TermLayout, MutAnyOrigin],
    voxel_count: Int,
    scenarios_per_voxel: Int,
    flags: UInt32,
    fractions: Scalar[FDCB_ACCELERATOR_DTYPE],
):
    comptime assert voxel_data.flat_rank == 1
    comptime assert scenario_slice_offsets.flat_rank == 1
    comptime assert scenario_slice_counts.flat_rank == 1
    comptime assert slice_factors.flat_rank == 1
    comptime assert slice_dots.flat_rank == 1
    comptime assert voxel_results.flat_rank == 1
    comptime assert scenario_indices.flat_rank == 1
    comptime assert terms.flat_rank == 1
    var voxel = global_idx.x
    if voxel >= voxel_count:
        return
    var data = voxel * 13
    var prescribed = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](voxel_data[data])
    var term = voxel * 2
    if prescribed == 0.0:
        terms[term] = 0.0
        terms[term + 1] = 0.0
        return
    var prescribed_abs = prescribed
    if prescribed_abs < 0.0:
        prescribed_abs = -prescribed_abs
    var dose_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[data + 1]
    )
    var divisor = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](voxel_data[data + 2])
    var maximum_weight = rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_data[data + 3]
    )
    var index = voxel * 2
    var min_scenario = Int(rebind[Scalar[DType.int32]](scenario_indices[index]))
    var max_scenario = Int(
        rebind[Scalar[DType.int32]](scenario_indices[index + 1])
    )
    var rmin: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    var rmax: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
    for selection in range(2):
        var selected = min_scenario
        if selection == 1:
            selected = max_scenario
        var scenario = voxel * scenarios_per_voxel + selected
        var offset = Int(
            rebind[Scalar[DType.uint64]](scenario_slice_offsets[scenario])
        )
        var count = Int(
            rebind[Scalar[DType.uint32]](scenario_slice_counts[scenario])
        )
        var response: Scalar[FDCB_ACCELERATOR_DTYPE] = 0.0
        for local_slice in range(count):
            var slice = offset + local_slice
            response += rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
                slice_dots[slice]
            ) * Scalar[FDCB_ACCELERATOR_DTYPE](
                rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](slice_factors[slice * 5])
            )
        response *= FDCB_ACCELERATOR_MEV_TO_GY * fractions
        if selection == 0:
            rmin = response
        else:
            rmax = response
    var dose_result = voxel * 4
    var residual = prescribed_abs - rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
        voxel_results[dose_result]
    )
    var response = rmin
    if prescribed < 0.0:
        residual = prescribed_abs - rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            voxel_results[dose_result + 1]
        )
        response = rmax
        if residual > 0.0:
            residual = 0.0
    var weighted_response = response * dose_weight / divisor
    var numerator = residual * dose_weight / divisor * weighted_response
    var denominator = weighted_response * weighted_response
    if (flags & FDCB_FLAG_ROBUST_INCLUDE_DMAX) != UInt32(
        0
    ) and prescribed > 0.0:
        residual = prescribed_abs - rebind[Scalar[FDCB_ACCELERATOR_DTYPE]](
            voxel_results[dose_result + 1]
        )
        weighted_response = rmax * maximum_weight / divisor
        numerator += residual * maximum_weight / divisor * weighted_response
        denominator += weighted_response * weighted_response
    terms[term] = rebind[terms.ElementType](numerator)
    terms[term + 1] = rebind[terms.ElementType](denominator)


struct FDCBAccelerator(Movable):
    var context: DeviceContext
    var field_count: Int
    var slice_count: Int
    var coefficient_count: Int
    var coefficient_index_bits: Int
    var coefficient_index_word_count: Int
    var point_count: Int
    var voxel_count: Int
    var scenarios_per_voxel: Int
    var scenario_count: Int
    var metric_partial_count: Int
    var active_slice_capacity: Int
    var flags: UInt32
    var overdose_weight: Float64
    var fractions: Float64
    var field_offsets: DeviceBuffer[DType.uint64]
    var slice_metadata: DeviceBuffer[DType.uint64]
    var slice_offsets: DeviceBuffer[DType.uint64]
    var coefficient_points: DeviceBuffer[DType.uint32]
    var coefficients: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var particles: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var point_active: DeviceBuffer[DType.uint8]
    var slice_dot_output: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var scenario_slice_offsets: DeviceBuffer[DType.uint64]
    var scenario_slice_counts: DeviceBuffer[DType.uint32]
    var scenario_state: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var slice_factors: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var scenario_moments: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var voxel_data: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var scenario_bio: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var voxel_results: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var scenario_indices: DeviceBuffer[DType.int32]
    var active_slice_count: DeviceBuffer[DType.uint32]
    var active_slices: DeviceBuffer[DType.uint32]
    var gradient: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var exact_step_terms: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]
    var metric_partials: DeviceBuffer[FDCB_ACCELERATOR_DTYPE]

    def __init__(out self, problem: FDCBProblemV1) raises:
        comptime if FDCB_ACCELERATOR_MIXED32:
            problem.validate(FDCB_PRECISION_MIXED32)
        else:
            problem.validate()
        if len(problem.slices) == 0:
            raise Error("packed FDCB accelerator requires sparse slices")
        self.context = DeviceContext()
        self.field_count = len(problem.field_slices)
        self.slice_count = len(problem.slices)
        self.coefficient_count = len(problem.coefficients)
        self.coefficient_index_bits = _coefficient_index_bits(problem)
        self.coefficient_index_word_count = (
            self.coefficient_count * self.coefficient_index_bits + 31
        ) // 32
        self.point_count = len(problem.particles)
        self.voxel_count = len(problem.voxels)
        self.scenarios_per_voxel = Int(problem.scenario_count)
        self.scenario_count = len(problem.voxel_scenarios)
        var coefficient_point_device = _pack_coefficient_indices_to_device(
            self.context,
            problem.coefficient_point_indices,
            self.coefficient_index_bits,
        )
        var coefficient_device = _copy_coefficients_to_device(
            self.context, problem.coefficients
        )
        var metric_count = self.voxel_count
        if self.point_count > metric_count:
            metric_count = self.point_count
        self.metric_partial_count = (
            metric_count + FDCB_REDUCTION_BLOCK_SIZE * 2 - 1
        ) // (FDCB_REDUCTION_BLOCK_SIZE * 2)
        var active_bound = 0
        for voxel in range(self.voxel_count):
            var maximum_slices = 0
            for scenario in range(self.scenarios_per_voxel):
                var count = Int(
                    problem.voxel_scenarios[
                        voxel * self.scenarios_per_voxel + scenario
                    ].slice_count
                )
                if count > maximum_slices:
                    maximum_slices = count
            var passes = 1
            if (
                problem.settings.flags & FDCB_FLAG_ROBUST_INCLUDE_DMAX
            ) != UInt32(0) and problem.voxels[voxel].prescribed_dose > 0.0:
                passes = 2
            active_bound += passes * maximum_slices
        self.active_slice_capacity = active_bound
        if self.active_slice_capacity > self.slice_count:
            self.active_slice_capacity = self.slice_count
        if self.active_slice_capacity == 0:
            self.active_slice_capacity = 1
        self.flags = problem.settings.flags
        self.overdose_weight = problem.settings.overdose_weight
        self.fractions = problem.settings.fractions
        var field_layout = row_major(Idx(self.field_count))
        var slice_layout = row_major(Idx(self.slice_count))
        var scenario_layout = row_major(Idx(self.scenario_count))
        var state_layout = row_major(Idx(self.scenario_count * 4))
        var factor_layout = row_major(Idx(self.slice_count * 5))
        var voxel_layout = row_major(Idx(self.voxel_count * 13))
        var voxel_index_layout = row_major(Idx(self.voxel_count * 2))
        var field_host = self.context.enqueue_create_host_buffer[DType.uint64](
            self.field_count
        )
        var slice_metadata_host = self.context.enqueue_create_host_buffer[
            DType.uint64
        ](self.slice_count)
        var slice_offset_host = self.context.enqueue_create_host_buffer[
            DType.uint64
        ](self.slice_count)
        var scenario_offset_host = self.context.enqueue_create_host_buffer[
            DType.uint64
        ](self.scenario_count)
        var scenario_count_host = self.context.enqueue_create_host_buffer[
            DType.uint32
        ](self.scenario_count)
        var scenario_state_host = self.context.enqueue_create_host_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.scenario_count * 4)
        var slice_factor_host = self.context.enqueue_create_host_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.slice_count * 5)
        var voxel_host = self.context.enqueue_create_host_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.voxel_count * 13)
        var active_host = self.context.enqueue_create_host_buffer[DType.uint8](
            self.point_count
        )
        var voxel_index_host = self.context.enqueue_create_host_buffer[
            DType.int32
        ](self.voxel_count * 2)
        var field_tensor = TileTensor(field_host, field_layout)
        var slice_metadata_tensor = TileTensor(
            slice_metadata_host, slice_layout
        )
        var slice_offset_tensor = TileTensor(slice_offset_host, slice_layout)
        var scenario_offset_tensor = TileTensor(
            scenario_offset_host, scenario_layout
        )
        var scenario_count_tensor = TileTensor(
            scenario_count_host, scenario_layout
        )
        var scenario_state_tensor = TileTensor(
            scenario_state_host, state_layout
        )
        var slice_factor_tensor = TileTensor(slice_factor_host, factor_layout)
        var voxel_tensor = TileTensor(voxel_host, voxel_layout)
        var active_tensor = TileTensor(
            active_host, row_major(Idx(self.point_count))
        )
        var voxel_index_tensor = TileTensor(
            voxel_index_host, voxel_index_layout
        )
        comptime assert field_tensor.flat_rank == 1
        comptime assert slice_metadata_tensor.flat_rank == 1
        comptime assert slice_offset_tensor.flat_rank == 1
        comptime assert scenario_offset_tensor.flat_rank == 1
        comptime assert scenario_count_tensor.flat_rank == 1
        comptime assert scenario_state_tensor.flat_rank == 1
        comptime assert slice_factor_tensor.flat_rank == 1
        comptime assert voxel_tensor.flat_rank == 1
        comptime assert active_tensor.flat_rank == 1
        comptime assert voxel_index_tensor.flat_rank == 1
        for i in range(self.field_count):
            field_tensor[i] = problem.field_slices[i].point_offset
        for i in range(self.slice_count):
            slice_metadata_tensor[i] = (
                UInt64(problem.slices[i].field_slice_index) << 32
            ) | UInt64(problem.slices[i].coefficient_count)
            slice_offset_tensor[i] = problem.slices[i].coefficient_offset
            var factor = i * 5
            slice_factor_tensor[factor] = Scalar[FDCB_ACCELERATOR_DTYPE](
                problem.slices[i].dose_coefficient
            )
            slice_factor_tensor[factor + 1] = Scalar[
                FDCB_ACCELERATOR_DTYPE
            ](problem.slices[i].alpha_coefficient)
            slice_factor_tensor[factor + 2] = Scalar[
                FDCB_ACCELERATOR_DTYPE
            ](problem.slices[i].sqrt_beta_coefficient)
            slice_factor_tensor[factor + 3] = Scalar[
                FDCB_ACCELERATOR_DTYPE
            ](problem.slices[i].let_mix_coefficient)
            slice_factor_tensor[factor + 4] = Scalar[
                FDCB_ACCELERATOR_DTYPE
            ](problem.slices[i].let_bar_coefficient)
        for i in range(self.point_count):
            active_tensor[i] = problem.point_active[i]
        for i in range(self.scenario_count):
            scenario_offset_tensor[i] = problem.voxel_scenarios[i].slice_offset
            scenario_count_tensor[i] = problem.voxel_scenarios[i].slice_count
            var state = i * 4
            scenario_state_tensor[state] = Scalar[FDCB_ACCELERATOR_DTYPE](
                problem.scenario_states[i].dose_minor
            )
            scenario_state_tensor[state + 1] = Scalar[FDCB_ACCELERATOR_DTYPE](
                problem.scenario_states[i].alpha_minor
            )
            scenario_state_tensor[state + 2] = Scalar[FDCB_ACCELERATOR_DTYPE](
                problem.scenario_states[i].sqrt_beta_minor
            )
            scenario_state_tensor[state + 3] = Scalar[FDCB_ACCELERATOR_DTYPE](
                problem.scenario_states[i].let_mix_minor
            )
        for i in range(self.voxel_count):
            var voxel = problem.voxels[i].copy()
            var offset = i * 13
            voxel_tensor[offset] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.prescribed_dose
            )
            voxel_tensor[offset + 1] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.dose_weight
            )
            voxel_tensor[offset + 2] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.dose_divisor
            )
            voxel_tensor[offset + 3] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.maximum_dose_weight
            )
            voxel_tensor[offset + 4] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.initial_dose
            )
            voxel_tensor[offset + 5] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.prescribed_let
            )
            voxel_tensor[offset + 6] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.let_weight
            )
            voxel_tensor[offset + 7] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.overdose_tolerance
            )
            voxel_tensor[offset + 8] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.rbe_cut
            )
            voxel_tensor[offset + 9] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.rbe_alpha
            )
            voxel_tensor[offset + 10] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.rbe_beta
            )
            voxel_tensor[offset + 11] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.rbe_slope_max
            )
            voxel_tensor[offset + 12] = Scalar[FDCB_ACCELERATOR_DTYPE](
                voxel.rbe_damage_cut
            )
            voxel_index_tensor[i * 2] = voxel.initial_min_scenario
            voxel_index_tensor[i * 2 + 1] = voxel.initial_max_scenario

        self.field_offsets = self.context.enqueue_create_buffer[DType.uint64](
            self.field_count
        )
        self.slice_metadata = self.context.enqueue_create_buffer[DType.uint64](
            self.slice_count
        )
        self.slice_offsets = self.context.enqueue_create_buffer[DType.uint64](
            self.slice_count
        )
        self.coefficient_points = coefficient_point_device^
        self.coefficients = coefficient_device^
        self.particles = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.point_count)
        self.point_active = self.context.enqueue_create_buffer[DType.uint8](
            self.point_count
        )
        self.slice_dot_output = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.slice_count)
        self.scenario_slice_offsets = self.context.enqueue_create_buffer[
            DType.uint64
        ](self.scenario_count)
        self.scenario_slice_counts = self.context.enqueue_create_buffer[
            DType.uint32
        ](self.scenario_count)
        self.scenario_state = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.scenario_count * 4)
        self.slice_factors = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.slice_count * 5)
        self.scenario_moments = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.scenario_count * 5)
        self.voxel_data = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.voxel_count * 13)
        self.scenario_bio = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.scenario_count * 4)
        self.voxel_results = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.voxel_count * 4)
        self.scenario_indices = self.context.enqueue_create_buffer[DType.int32](
            self.voxel_count * 2
        )
        self.active_slice_count = self.context.enqueue_create_buffer[
            DType.uint32
        ](1)
        self.active_slices = self.context.enqueue_create_buffer[DType.uint32](
            self.active_slice_capacity
        )
        self.gradient = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.point_count)
        self.exact_step_terms = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.voxel_count * 2)
        self.metric_partials = self.context.enqueue_create_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.metric_partial_count * 3)
        self.context.enqueue_copy(self.field_offsets, field_host)
        self.context.enqueue_copy(self.slice_metadata, slice_metadata_host)
        self.context.enqueue_copy(self.slice_offsets, slice_offset_host)
        self.context.enqueue_copy(self.point_active, active_host)
        self.context.enqueue_copy(
            self.scenario_slice_offsets, scenario_offset_host
        )
        self.context.enqueue_copy(
            self.scenario_slice_counts, scenario_count_host
        )
        self.context.enqueue_copy(self.scenario_state, scenario_state_host)
        self.context.enqueue_copy(self.slice_factors, slice_factor_host)
        self.context.enqueue_copy(self.voxel_data, voxel_host)
        self.context.enqueue_copy(self.scenario_indices, voxel_index_host)

    def _enqueue_slice_dots(mut self, particles: List[Float64]) raises:
        if len(particles) != self.point_count:
            raise Error("FDCB accelerator particle vector length mismatch")
        var field_layout = row_major(Idx(self.field_count))
        var slice_layout = row_major(Idx(self.slice_count))
        var coefficient_point_layout = row_major(
            Idx(self.coefficient_index_word_count)
        )
        var coefficient_layout = row_major(Idx(self.coefficient_count))
        var point_layout = row_major(Idx(self.point_count))
        var particle_host = self.context.enqueue_create_host_buffer[
            FDCB_ACCELERATOR_DTYPE
        ](self.point_count)
        var particle_tensor = TileTensor(particle_host, point_layout)
        comptime assert particle_tensor.flat_rank == 1
        for i in range(self.point_count):
            particle_tensor[i] = Scalar[FDCB_ACCELERATOR_DTYPE](particles[i])
        self.context.enqueue_copy(self.particles, particle_host)
        comptime kernel = sparse_slice_dot_kernel[
            type_of(field_layout),
            type_of(slice_layout),
            type_of(coefficient_point_layout),
            type_of(coefficient_layout),
            type_of(point_layout),
        ]
        self.context.enqueue_function[kernel](
            TileTensor(self.field_offsets, field_layout),
            TileTensor(self.slice_metadata, slice_layout),
            TileTensor(self.slice_offsets, slice_layout),
            TileTensor(self.coefficient_points, coefficient_point_layout),
            TileTensor(self.coefficients, coefficient_layout),
            TileTensor(self.particles, point_layout),
            TileTensor(self.slice_dot_output, slice_layout),
            self.slice_count,
            self.coefficient_index_bits,
            grid_dim=(
                self.slice_count * WARP_SIZE + FDCB_ACCELERATOR_BLOCK_SIZE - 1
            )
            // FDCB_ACCELERATOR_BLOCK_SIZE,
            block_dim=FDCB_ACCELERATOR_BLOCK_SIZE,
        )

    def slice_dots(mut self, particles: List[Float64]) raises -> List[Float64]:
        self._enqueue_slice_dots(particles)
        var slice_layout = row_major(Idx(self.slice_count))
        var output = List[Float64]()
        output.reserve(self.slice_count)
        with self.slice_dot_output.map_to_host() as mapped:
            var tensor = TileTensor(mapped, slice_layout)
            comptime assert tensor.flat_rank == 1
            for i in range(self.slice_count):
                output.append(Float64(tensor[i]))
        return output^

    def moments(mut self, particles: List[Float64]) raises -> List[Float64]:
        self._enqueue_scenario_moments(particles)
        var moment_layout = row_major(Idx(self.scenario_count * 5))
        var output = List[Float64]()
        output.reserve(self.scenario_count * 5)
        with self.scenario_moments.map_to_host() as mapped:
            var tensor = TileTensor(mapped, moment_layout)
            comptime assert tensor.flat_rank == 1
            for i in range(self.scenario_count * 5):
                output.append(Float64(tensor[i]))
        return output^

    def _enqueue_scenario_moments(mut self, particles: List[Float64]) raises:
        self._enqueue_slice_dots(particles)
        var scenario_layout = row_major(Idx(self.scenario_count))
        var slice_layout = row_major(Idx(self.slice_count))
        var state_layout = row_major(Idx(self.scenario_count * 4))
        var factor_layout = row_major(Idx(self.slice_count * 5))
        var moment_layout = row_major(Idx(self.scenario_count * 5))
        comptime kernel = biological_moment_kernel[
            type_of(scenario_layout),
            type_of(slice_layout),
            type_of(state_layout),
            type_of(factor_layout),
            type_of(moment_layout),
        ]
        self.context.enqueue_function[kernel](
            TileTensor(self.scenario_slice_offsets, scenario_layout),
            TileTensor(self.scenario_slice_counts, scenario_layout),
            TileTensor(self.scenario_state, state_layout),
            TileTensor(self.slice_factors, factor_layout),
            TileTensor(self.slice_dot_output, slice_layout),
            TileTensor(self.scenario_moments, moment_layout),
            self.scenario_count,
            grid_dim=(self.scenario_count + FDCB_ACCELERATOR_BLOCK_SIZE - 1)
            // FDCB_ACCELERATOR_BLOCK_SIZE,
            block_dim=FDCB_ACCELERATOR_BLOCK_SIZE,
        )

    def exact_step(mut self, direction: List[Float64]) raises -> Float64:
        self._enqueue_slice_dots(direction)
        var voxel_layout = row_major(Idx(self.voxel_count * 13))
        var scenario_layout = row_major(Idx(self.scenario_count))
        var slice_layout = row_major(Idx(self.slice_count))
        var factor_layout = row_major(Idx(self.slice_count * 5))
        var result_layout = row_major(Idx(self.voxel_count * 4))
        var index_layout = row_major(Idx(self.voxel_count * 2))
        var term_layout = row_major(Idx(self.voxel_count * 2))
        comptime kernel = exact_step_terms_kernel[
            type_of(voxel_layout),
            type_of(scenario_layout),
            type_of(slice_layout),
            type_of(factor_layout),
            type_of(result_layout),
            type_of(index_layout),
            type_of(term_layout),
        ]
        self.context.enqueue_function[kernel](
            TileTensor(self.voxel_data, voxel_layout),
            TileTensor(self.scenario_slice_offsets, scenario_layout),
            TileTensor(self.scenario_slice_counts, scenario_layout),
            TileTensor(self.slice_factors, factor_layout),
            TileTensor(self.slice_dot_output, slice_layout),
            TileTensor(self.voxel_results, result_layout),
            TileTensor(self.scenario_indices, index_layout),
            TileTensor(self.exact_step_terms, term_layout),
            self.voxel_count,
            self.scenarios_per_voxel,
            self.flags,
            Scalar[FDCB_ACCELERATOR_DTYPE](self.fractions),
            grid_dim=(self.voxel_count + FDCB_ACCELERATOR_BLOCK_SIZE - 1)
            // FDCB_ACCELERATOR_BLOCK_SIZE,
            block_dim=FDCB_ACCELERATOR_BLOCK_SIZE,
        )
        var numerator = 0.0
        var denominator = 0.0
        with self.exact_step_terms.map_to_host() as mapped:
            var tensor = TileTensor(mapped, term_layout)
            comptime assert tensor.flat_rank == 1
            for voxel in range(self.voxel_count):
                numerator += Float64(tensor[voxel * 2])
                denominator += Float64(tensor[voxel * 2 + 1])
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def _reduced_metrics(mut self) raises -> FDCBAcceleratorMetrics:
        var result_layout = row_major(Idx(self.voxel_count * 4))
        var point_layout = row_major(Idx(self.point_count))
        var partial_layout = row_major(Idx(self.metric_partial_count * 3))
        comptime kernel = metric_partial_reduction_kernel[
            type_of(result_layout),
            type_of(point_layout),
            type_of(partial_layout),
        ]
        self.context.enqueue_function[kernel](
            TileTensor(self.voxel_results, result_layout),
            TileTensor(self.gradient, point_layout),
            TileTensor(self.metric_partials, partial_layout),
            self.voxel_count,
            self.point_count,
            grid_dim=self.metric_partial_count,
            block_dim=FDCB_REDUCTION_BLOCK_SIZE,
        )
        var chi2 = 0.0
        var weighted = 0.0
        var norm2 = 0.0
        with self.metric_partials.map_to_host() as mapped:
            var tensor = TileTensor(mapped, partial_layout)
            comptime assert tensor.flat_rank == 1
            for block in range(self.metric_partial_count):
                chi2 += Float64(tensor[block * 3])
                weighted += Float64(tensor[block * 3 + 1])
                norm2 += Float64(tensor[block * 3 + 2])
        return FDCBAcceleratorMetrics(chi2, weighted, fdcb_sqrt64(norm2))

    def evaluation(
        mut self,
        particles: List[Float64],
        include_diagnostics: Bool = True,
    ) raises -> FDCBAcceleratorEvaluation:
        self._enqueue_scenario_moments(particles)
        var biological = self.flags & FDCB_FLAG_BIOLOGICAL
        var voxel_layout = row_major(Idx(self.voxel_count * 13))
        var moment_layout = row_major(Idx(self.scenario_count * 5))
        var bio_layout = row_major(Idx(self.scenario_count * 4))
        var result_layout = row_major(Idx(self.voxel_count * 4))
        var index_layout = row_major(Idx(self.voxel_count * 2))
        comptime kernel = fdcb_objective_kernel[
            type_of(voxel_layout),
            type_of(moment_layout),
            type_of(bio_layout),
            type_of(result_layout),
            type_of(index_layout),
        ]
        self.context.enqueue_function[kernel](
            TileTensor(self.voxel_data, voxel_layout),
            TileTensor(self.scenario_moments, moment_layout),
            TileTensor(self.scenario_bio, bio_layout),
            TileTensor(self.voxel_results, result_layout),
            TileTensor(self.scenario_indices, index_layout),
            self.voxel_count,
            self.scenarios_per_voxel,
            self.flags,
            biological,
            grid_dim=(self.voxel_count + FDCB_ACCELERATOR_BLOCK_SIZE - 1)
            // FDCB_ACCELERATOR_BLOCK_SIZE,
            block_dim=FDCB_ACCELERATOR_BLOCK_SIZE,
        )
        # Forward dots are no longer needed once scenario moments are complete.
        # Reuse their resident slice workspace for gradient scales.
        self.slice_dot_output.enqueue_fill(0.0)
        self.gradient.enqueue_fill(0.0)
        var scenario_layout = row_major(Idx(self.scenario_count))
        var slice_layout = row_major(Idx(self.slice_count))
        var factor_layout = row_major(Idx(self.slice_count * 5))
        comptime scale_kernel = fdcb_slice_gradient_kernel[
            type_of(voxel_layout),
            type_of(scenario_layout),
            type_of(moment_layout),
            type_of(bio_layout),
            type_of(result_layout),
            type_of(index_layout),
            type_of(slice_layout),
            type_of(factor_layout),
        ]
        self.context.enqueue_function[scale_kernel](
            TileTensor(self.voxel_data, voxel_layout),
            TileTensor(self.scenario_slice_offsets, scenario_layout),
            TileTensor(self.scenario_slice_counts, scenario_layout),
            TileTensor(self.scenario_moments, moment_layout),
            TileTensor(self.scenario_bio, bio_layout),
            TileTensor(self.voxel_results, result_layout),
            TileTensor(self.scenario_indices, index_layout),
            TileTensor(self.slice_factors, factor_layout),
            TileTensor(self.slice_dot_output, slice_layout),
            self.voxel_count,
            self.scenarios_per_voxel,
            self.flags,
            Scalar[FDCB_ACCELERATOR_DTYPE](self.overdose_weight),
            biological,
            grid_dim=(self.voxel_count + FDCB_ACCELERATOR_BLOCK_SIZE - 1)
            // FDCB_ACCELERATOR_BLOCK_SIZE,
            block_dim=FDCB_ACCELERATOR_BLOCK_SIZE,
        )
        self.active_slice_count.enqueue_fill(UInt32(0))
        var count_layout = row_major(Idx(1))
        var active_layout = row_major(Idx(self.active_slice_capacity))
        comptime list_kernel = active_slice_list_kernel[
            type_of(slice_layout),
            type_of(count_layout),
            type_of(active_layout),
        ]
        self.context.enqueue_function[list_kernel](
            TileTensor(self.slice_dot_output, slice_layout),
            TileTensor(self.active_slice_count, count_layout),
            TileTensor(self.active_slices, active_layout),
            self.slice_count,
            grid_dim=(self.slice_count + FDCB_ACCELERATOR_BLOCK_SIZE - 1)
            // FDCB_ACCELERATOR_BLOCK_SIZE,
            block_dim=FDCB_ACCELERATOR_BLOCK_SIZE,
        )
        var active_count: Int
        with self.active_slice_count.map_to_host() as mapped:
            var tensor = TileTensor(mapped, count_layout)
            comptime assert tensor.flat_rank == 1
            active_count = Int(tensor[0])
        var field_layout = row_major(Idx(self.field_count))
        var coefficient_point_layout = row_major(
            Idx(self.coefficient_index_word_count)
        )
        var coefficient_layout = row_major(Idx(self.coefficient_count))
        var point_layout = row_major(Idx(self.point_count))
        comptime backprojection_kernel = sparse_gradient_backprojection_kernel[
            type_of(field_layout),
            type_of(slice_layout),
            type_of(coefficient_point_layout),
            type_of(coefficient_layout),
            type_of(point_layout),
            type_of(active_layout),
        ]
        if active_count > 0:
            self.context.enqueue_function[backprojection_kernel](
                TileTensor(self.field_offsets, field_layout),
                TileTensor(self.slice_metadata, slice_layout),
                TileTensor(self.slice_offsets, slice_layout),
                TileTensor(self.coefficient_points, coefficient_point_layout),
                TileTensor(self.coefficients, coefficient_layout),
                TileTensor(self.point_active, point_layout),
                TileTensor(self.particles, point_layout),
                TileTensor(self.slice_dot_output, slice_layout),
                TileTensor(self.active_slices, active_layout),
                TileTensor(self.gradient, point_layout),
                active_count,
                self.coefficient_index_bits,
                grid_dim=(
                    active_count * WARP_SIZE + FDCB_ACCELERATOR_BLOCK_SIZE - 1
                )
                // FDCB_ACCELERATOR_BLOCK_SIZE,
                block_dim=FDCB_ACCELERATOR_BLOCK_SIZE,
            )
        var metrics = self._reduced_metrics()
        var scenario_bio = List[Float64]()
        if include_diagnostics:
            scenario_bio.reserve(self.scenario_count * 4)
            with self.scenario_bio.map_to_host() as mapped:
                var tensor = TileTensor(mapped, bio_layout)
                comptime assert tensor.flat_rank == 1
                for i in range(self.scenario_count * 4):
                    scenario_bio.append(Float64(tensor[i]))
        var dose_min = List[Float64]()
        var dose_max = List[Float64]()
        var chi2 = List[Float64]()
        var weighted = List[Float64]()
        if include_diagnostics:
            dose_min.reserve(self.voxel_count)
            dose_max.reserve(self.voxel_count)
            chi2.reserve(self.voxel_count)
            weighted.reserve(self.voxel_count)
            with self.voxel_results.map_to_host() as mapped:
                var tensor = TileTensor(mapped, result_layout)
                comptime assert tensor.flat_rank == 1
                for i in range(self.voxel_count):
                    var offset = i * 4
                    dose_min.append(Float64(tensor[offset]))
                    dose_max.append(Float64(tensor[offset + 1]))
                    chi2.append(Float64(tensor[offset + 2]))
                    weighted.append(Float64(tensor[offset + 3]))
        var min_scenario = List[Int32]()
        var max_scenario = List[Int32]()
        if include_diagnostics:
            min_scenario.reserve(self.voxel_count)
            max_scenario.reserve(self.voxel_count)
            with self.scenario_indices.map_to_host() as mapped:
                var tensor = TileTensor(mapped, index_layout)
                comptime assert tensor.flat_rank == 1
                for i in range(self.voxel_count):
                    min_scenario.append(Int32(tensor[i * 2]))
                    max_scenario.append(Int32(tensor[i * 2 + 1]))
        var gradient = List[Float64]()
        gradient.reserve(self.point_count)
        with self.gradient.map_to_host() as mapped:
            var tensor = TileTensor(mapped, point_layout)
            comptime assert tensor.flat_rank == 1
            for i in range(self.point_count):
                gradient.append(Float64(tensor[i]))
        return FDCBAcceleratorEvaluation(
            scenario_bio^,
            dose_min^,
            dose_max^,
            min_scenario^,
            max_scenario^,
            chi2^,
            weighted^,
            gradient^,
            metrics.chi2,
            metrics.weighted_dose2,
            metrics.gradient_norm,
        )


def accelerator_slice_dots(
    problem: FDCBProblemV1, particles: List[Float64]
) raises -> List[Float64]:
    var accelerator = FDCBAccelerator(problem)
    return accelerator.slice_dots(particles)
