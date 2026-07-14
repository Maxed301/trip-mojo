"""Accelerator construction of TRiP's compact lateral FDCB matrix."""

from std.gpu import WARP_SIZE, barrier, global_idx, thread_idx
from std.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import prefix_sum, shuffle_idx
from std.math import ceildiv, exp
from std.memory import OpaquePointer, UnsafePointer, alloc, stack_allocation
from std.sys.info import size_of
from std.time import perf_counter_ns


comptime FDCB_MATRIX_ABI_VERSION_V1 = UInt32(1)
comptime FDCB_MATRIX_FLAG_DEVICE_ONLY = UInt32(1)
comptime FDCB_MATRIX_BLOCK_SIZE = 128
comptime FDCB_MATRIX_FILL_BLOCK_SIZE = 128
comptime FDCB_MATRIX_FILL_WARPS = FDCB_MATRIX_FILL_BLOCK_SIZE // WARP_SIZE
comptime FDCB_MATRIX_REDUCTION_GROUP_SIZE = 32
comptime FDCB_MATRIX_REDUCTION_GROUPS = (
    FDCB_MATRIX_BLOCK_SIZE // FDCB_MATRIX_REDUCTION_GROUP_SIZE
)
comptime FDCB_MATRIX_PI = 3.141592653589793238462643383279502884
# TRiP's historical TRP_8LN2 is defined as 0.6932 * 8, then halved.
comptime FDCB_MATRIX_4LN2 = 2.7728


@fieldwise_init
struct FDCBMatrixEnergySliceV1(Copyable, Movable):
    var point_offset: UInt64
    var point_count: UInt32
    var reserved: UInt32


@fieldwise_init
struct FDCBMatrixPointV1(Copyable, Movable):
    var x: Float64
    var y: Float64
    var f2_max: Float64


@fieldwise_init
struct FDCBMatrixGroupV1(Copyable, Movable):
    var slice_offset: UInt64
    var slice_count: UInt32
    var reserved: UInt32
    var bev_x: Float64
    var bev_y: Float64
    var relative_cutoff: Float64
    var point_shift_x: Float64
    var point_shift_y: Float64


@fieldwise_init
struct FDCBMatrixRawEnergyV1(Copyable, Movable):
    var energy_slice: UInt32
    var ddd_table: UInt32
    var depth_shift_mm: Float64
    var focus_squared: Float64
    var lateral_limit_scale: Float64
    var fallback_scale: Float64


@fieldwise_init
struct FDCBMatrixRawGroupV1(Copyable, Movable):
    var energy_offset: UInt32
    var energy_count: UInt32
    var depth_mm: Float64
    var divergence_x: Float64
    var divergence_y: Float64


@fieldwise_init
struct FDCBMatrixDDDTableV1(Copyable, Movable):
    var entry_offset: UInt32
    var entry_count: UInt32


@fieldwise_init
struct FDCBMatrixDDDEntryV1(Copyable, Movable):
    var depth_cm: Float64
    var dose: Float64
    var fwhm1: Float64
    var mixture: Float64
    var fwhm2: Float64


@fieldwise_init
struct FDCBMatrixProblemViewV1(Copyable, Movable):
    var version: UInt32
    var group_count: UInt32
    var energy_slice_count: UInt32
    var maximum_group_slices: UInt32
    var slice_count: UInt64
    var point_count: UInt64
    var ddd_table_count: UInt32
    var flags: UInt32
    var ddd_entry_count: UInt64
    var energy_slices: UnsafePointer[FDCBMatrixEnergySliceV1, MutExternalOrigin]
    var points: UnsafePointer[FDCBMatrixPointV1, MutExternalOrigin]
    var groups: UnsafePointer[FDCBMatrixGroupV1, MutExternalOrigin]
    var raw_energies: UnsafePointer[FDCBMatrixRawEnergyV1, MutExternalOrigin]
    var raw_groups: UnsafePointer[FDCBMatrixRawGroupV1, MutExternalOrigin]
    var ddd_tables: UnsafePointer[FDCBMatrixDDDTableV1, MutExternalOrigin]
    var ddd_entries: UnsafePointer[FDCBMatrixDDDEntryV1, MutExternalOrigin]


@fieldwise_init
struct FDCBMatrixResultV1(Copyable, Movable):
    var entry_count: UInt64
    var slice_count: UInt64
    var group_count: UInt32
    var reserved: UInt32
    var group_maximum: UnsafePointer[Float64, MutAnyOrigin]
    var group_entry_counts: UnsafePointer[UInt32, MutAnyOrigin]
    var slice_entry_counts: UnsafePointer[UInt32, MutAnyOrigin]
    var slice_dose: UnsafePointer[Float64, MutAnyOrigin]
    var point_indices: UnsafePointer[UInt16, MutAnyOrigin]
    var coefficients: UnsafePointer[Float64, MutAnyOrigin]


@fieldwise_init
struct FDCBMatrixStorageV1(Movable):
    var context: DeviceContext
    var device_only: Bool
    var entry_count: Int
    var point_indices_device: DeviceBuffer[DType.uint16]
    var coefficients_device: DeviceBuffer[DType.float64]
    var group_maximum: HostBuffer[DType.float64]
    var group_entry_counts: HostBuffer[DType.uint32]
    var slice_entry_counts: HostBuffer[DType.uint32]
    var slice_dose: HostBuffer[DType.float64]
    var point_indices: HostBuffer[DType.uint16]
    var coefficients: HostBuffer[DType.float64]


@fieldwise_init
struct _DDDValue(Copyable, Movable):
    var valid: Bool
    var dose: Float64
    var fwhm1: Float64
    var mixture: Float64
    var fwhm2: Float64


def _copy_bytes_to_device[
    origin: Origin, //
](
    context: DeviceContext,
    source: UnsafePointer[UInt8, origin],
    byte_count: Int,
) raises -> DeviceBuffer[DType.uint8]:
    var allocation_count = byte_count
    if allocation_count == 0:
        allocation_count = 1
    var host = context.enqueue_create_host_buffer[DType.uint8](allocation_count)
    for index in range(byte_count):
        host[index] = source[index]
    var device = context.enqueue_create_buffer[DType.uint8](allocation_count)
    context.enqueue_copy(device, host)
    return device^


@always_inline("nodebug")
def _interpolate_ddd(
    table: FDCBMatrixDDDTableV1,
    entries: UnsafePointer[FDCBMatrixDDDEntryV1, MutAnyOrigin],
    depth_cm: Float64,
) -> _DDDValue:
    var first = entries[Int(table.entry_offset)].copy()
    var last = entries[
        Int(table.entry_offset + table.entry_count - UInt32(1))
    ].copy()
    if depth_cm < first.depth_cm or depth_cm > last.depth_cm:
        return _DDDValue(False, 0.0, 0.0, 0.0, 0.0)
    var lo = UInt32(0)
    var hi = table.entry_count - UInt32(1)
    var middle: UInt32
    while True:
        middle = (lo + hi) >> UInt32(1)
        if middle == lo:
            break
        var value = entries[Int(table.entry_offset + middle)].depth_cm
        if depth_cm == value:
            break
        if depth_cm < value:
            hi = middle
        else:
            lo = middle
    lo = middle
    hi = middle + UInt32(1)
    if hi >= table.entry_count:
        hi = table.entry_count - UInt32(1)
    var low = entries[Int(table.entry_offset + lo)].copy()
    var high = entries[Int(table.entry_offset + hi)].copy()
    var fraction = high.depth_cm - low.depth_cm
    if fraction != 0.0:
        fraction = (depth_cm - low.depth_cm) / fraction
    return _DDDValue(
        True,
        (high.dose - low.dose) * fraction + low.dose,
        (high.fwhm1 - low.fwhm1) * fraction + low.fwhm1,
        (high.mixture - low.mixture) * fraction + low.mixture,
        (high.fwhm2 - low.fwhm2) * fraction + low.fwhm2,
    )


def _materialize_raw_slices_kernel(
    groups: UnsafePointer[FDCBMatrixGroupV1, MutAnyOrigin],
    raw_energies: UnsafePointer[FDCBMatrixRawEnergyV1, MutAnyOrigin],
    raw_groups: UnsafePointer[FDCBMatrixRawGroupV1, MutAnyOrigin],
    ddd_tables: UnsafePointer[FDCBMatrixDDDTableV1, MutAnyOrigin],
    ddd_entries: UnsafePointer[FDCBMatrixDDDEntryV1, MutAnyOrigin],
    group_count: Int,
    maximum_group_slices: Int,
    slice_energy: UnsafePointer[UInt32, MutAnyOrigin],
    slice_values: UnsafePointer[Float64, MutAnyOrigin],
    slice_dose: UnsafePointer[Float64, MutAnyOrigin],
):
    var linear = global_idx.x
    var group_index = linear // maximum_group_slices
    var local_slice = linear - group_index * maximum_group_slices
    if group_index >= group_count:
        return
    var group = groups[group_index].copy()
    if local_slice >= Int(group.slice_count):
        return
    var raw_group = raw_groups[group_index].copy()
    var beam = Int(raw_group.energy_count) - 1 - local_slice
    var raw_energy = raw_energies[Int(raw_group.energy_offset) + beam].copy()
    var slice_index = Int(group.slice_offset) + local_slice
    var depth = _interpolate_ddd(
        ddd_tables[Int(raw_energy.ddd_table)].copy(),
        ddd_entries,
        (raw_group.depth_mm + raw_energy.depth_shift_mm) * 0.1,
    )
    var offset = slice_index * 10
    slice_energy[slice_index] = raw_energy.energy_slice
    for value_index in range(10):
        slice_values[offset + value_index] = 0.0
    slice_values[offset + 7] = raw_energy.focus_squared
    slice_values[offset + 8] = raw_group.divergence_x
    slice_values[offset + 9] = raw_group.divergence_y
    if depth.valid:
        var f2_0 = raw_energy.focus_squared + depth.fwhm1 * depth.fwhm1
        var isig2_0 = FDCB_MATRIX_4LN2 / f2_0
        slice_values[offset] = depth.dose
        slice_values[offset + 1] = isig2_0
        slice_values[offset + 3] = (
            isig2_0 / FDCB_MATRIX_PI * (1.0 - depth.mixture)
        )
        slice_values[offset + 5] = raw_energy.lateral_limit_scale * f2_0
        if depth.mixture > 0.0:
            var f2_1 = raw_energy.focus_squared + depth.fwhm2 * depth.fwhm2
            var isig2_1 = FDCB_MATRIX_4LN2 / f2_1
            slice_values[offset + 2] = isig2_1
            slice_values[offset + 4] = isig2_1 / FDCB_MATRIX_PI * depth.mixture
            slice_values[offset + 6] = raw_energy.lateral_limit_scale * f2_1
        var maximum_fwhm = depth.fwhm1
        if depth.fwhm2 > maximum_fwhm:
            maximum_fwhm = depth.fwhm2
        slice_values[offset + 7] = (
            raw_energy.focus_squared + maximum_fwhm * maximum_fwhm
        ) * raw_energy.fallback_scale
    slice_dose[slice_index] = slice_values[offset]


@always_inline("nodebug")
def _radius_squared(
    energy_slices: UnsafePointer[FDCBMatrixEnergySliceV1, MutAnyOrigin],
    points: UnsafePointer[FDCBMatrixPointV1, MutAnyOrigin],
    group: FDCBMatrixGroupV1,
    slice_energy: UnsafePointer[UInt32, MutAnyOrigin],
    slice_values: UnsafePointer[Float64, MutAnyOrigin],
    slice_index: Int,
    local_point: Int,
) -> Float64:
    var energy = energy_slices[Int(slice_energy[slice_index])].copy()
    var point = points[Int(energy.point_offset) + local_point].copy()
    var px = point.x + group.point_shift_x
    var py = point.y + group.point_shift_y
    var values = slice_index * 10
    var dx = group.bev_x - px
    if slice_values[values + 8] != 0.0:
        dx += slice_values[values + 8] * px
    var dx2 = dx * dx
    var f2_max = point.f2_max
    if f2_max == 0.0:
        f2_max = slice_values[values + 7]
    if dx2 > f2_max:
        return -1.0
    var dy = group.bev_y - py
    if slice_values[values + 9] != 0.0:
        dy += slice_values[values + 9] * py
    var dy2 = dy * dy
    if dy2 > f2_max:
        return -1.0
    return dx2 + dy2


@always_inline("nodebug")
def _coefficient_from_radius(
    slice_values: UnsafePointer[Float64, MutAnyOrigin],
    slice_index: Int,
    radius2: Float64,
) -> Float64:
    var values = slice_index * 10
    var coefficient = 0.0
    if radius2 < slice_values[values + 5]:
        coefficient = (
            exp(-radius2 * slice_values[values + 1]) * slice_values[values + 3]
        )
    if slice_values[values + 4] != 0.0 and radius2 < slice_values[values + 6]:
        coefficient += (
            exp(-radius2 * slice_values[values + 2]) * slice_values[values + 4]
        )
    return coefficient


@always_inline("nodebug")
def _contribution(
    energy_slices: UnsafePointer[FDCBMatrixEnergySliceV1, MutAnyOrigin],
    points: UnsafePointer[FDCBMatrixPointV1, MutAnyOrigin],
    group: FDCBMatrixGroupV1,
    slice_energy: UnsafePointer[UInt32, MutAnyOrigin],
    slice_values: UnsafePointer[Float64, MutAnyOrigin],
    slice_index: Int,
    local_point: Int,
) -> Float64:
    var radius2 = _radius_squared(
        energy_slices,
        points,
        group,
        slice_energy,
        slice_values,
        slice_index,
        local_point,
    )
    if radius2 < 0.0:
        return 0.0
    return _coefficient_from_radius(slice_values, slice_index, radius2)


def _count_kernel(
    energy_slices: UnsafePointer[FDCBMatrixEnergySliceV1, MutAnyOrigin],
    points: UnsafePointer[FDCBMatrixPointV1, MutAnyOrigin],
    groups: UnsafePointer[FDCBMatrixGroupV1, MutAnyOrigin],
    slice_energy: UnsafePointer[UInt32, MutAnyOrigin],
    slice_values: UnsafePointer[Float64, MutAnyOrigin],
    group_count: Int,
    group_maximum: UnsafePointer[Float64, MutAnyOrigin],
    group_entry_counts: UnsafePointer[UInt32, MutAnyOrigin],
    slice_entry_counts: UnsafePointer[UInt32, MutAnyOrigin],
):
    var thread = thread_idx.x
    var group_index = global_idx.x // FDCB_MATRIX_BLOCK_SIZE
    if group_index >= group_count:
        return
    var group = groups[group_index].copy()
    var minimum_shared = stack_allocation[
        FDCB_MATRIX_BLOCK_SIZE,
        Float64,
        address_space=AddressSpace.SHARED,
    ]()
    var subgroup_minima = stack_allocation[
        FDCB_MATRIX_REDUCTION_GROUPS,
        Float64,
        address_space=AddressSpace.SHARED,
    ]()
    var maximum_shared = stack_allocation[
        1,
        Float64,
        address_space=AddressSpace.SHARED,
    ]()
    if thread == 0:
        maximum_shared[0] = 0.0
    var subgroup = thread // FDCB_MATRIX_REDUCTION_GROUP_SIZE
    var subgroup_lane = thread % FDCB_MATRIX_REDUCTION_GROUP_SIZE
    barrier()
    for local_slice in range(Int(group.slice_count)):
        var slice_index = Int(group.slice_offset) + local_slice
        var dose = slice_values[slice_index * 10]
        var minimum = 1.7976931348623157e308
        if dose != 0.0:
            var energy = energy_slices[Int(slice_energy[slice_index])].copy()
            for point in range(
                thread, Int(energy.point_count), FDCB_MATRIX_BLOCK_SIZE
            ):
                var radius2 = _radius_squared(
                    energy_slices,
                    points,
                    group,
                    slice_energy,
                    slice_values,
                    slice_index,
                    point,
                )
                if radius2 >= 0.0 and radius2 < minimum:
                    minimum = radius2
        minimum_shared[thread] = minimum
        barrier()
        if subgroup_lane == 0:
            var subgroup_minimum = minimum_shared[thread]
            for offset in range(1, FDCB_MATRIX_REDUCTION_GROUP_SIZE):
                if minimum_shared[thread + offset] < subgroup_minimum:
                    subgroup_minimum = minimum_shared[thread + offset]
            subgroup_minima[subgroup] = subgroup_minimum
        barrier()
        if thread == 0:
            var block_minimum = subgroup_minima[0]
            for subgroup_index in range(1, FDCB_MATRIX_REDUCTION_GROUPS):
                if subgroup_minima[subgroup_index] < block_minimum:
                    block_minimum = subgroup_minima[subgroup_index]
            minimum_shared[0] = block_minimum
        barrier()
        if thread == 0 and minimum_shared[0] < 1.7976931348623157e308:
            var physical = (
                _coefficient_from_radius(
                    slice_values, slice_index, minimum_shared[0]
                )
                * dose
            )
            if physical > maximum_shared[0]:
                maximum_shared[0] = physical
        barrier()
    var maximum = maximum_shared[0]
    var threshold = maximum * group.relative_cutoff
    var count_shared = stack_allocation[
        FDCB_MATRIX_BLOCK_SIZE,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var subgroup_counts = stack_allocation[
        FDCB_MATRIX_REDUCTION_GROUPS,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var group_entries = UInt32(0)
    for local_slice in range(Int(group.slice_count)):
        var slice_index = Int(group.slice_offset) + local_slice
        var dose = slice_values[slice_index * 10]
        var energy = energy_slices[Int(slice_energy[slice_index])].copy()
        var count = UInt32(0)
        for point in range(
            thread, Int(energy.point_count), FDCB_MATRIX_BLOCK_SIZE
        ):
            var coefficient = 0.0
            if dose != 0.0:
                coefficient = _contribution(
                    energy_slices,
                    points,
                    group,
                    slice_energy,
                    slice_values,
                    slice_index,
                    point,
                )
                if coefficient * dose > threshold:
                    count += UInt32(1)
        count_shared[thread] = count
        barrier()
        if subgroup_lane == 0:
            var subgroup_count = UInt32(0)
            for offset in range(FDCB_MATRIX_REDUCTION_GROUP_SIZE):
                subgroup_count += count_shared[thread + offset]
            subgroup_counts[subgroup] = subgroup_count
        barrier()
        if thread == 0:
            var block_count = UInt32(0)
            for subgroup_index in range(FDCB_MATRIX_REDUCTION_GROUPS):
                block_count += subgroup_counts[subgroup_index]
            slice_entry_counts[slice_index] = block_count
            group_entries += block_count
        barrier()
    if thread == 0:
        group_maximum[group_index] = maximum
        group_entry_counts[group_index] = group_entries


def _fill_kernel(
    energy_slices: UnsafePointer[FDCBMatrixEnergySliceV1, MutAnyOrigin],
    points: UnsafePointer[FDCBMatrixPointV1, MutAnyOrigin],
    groups: UnsafePointer[FDCBMatrixGroupV1, MutAnyOrigin],
    slice_energy: UnsafePointer[UInt32, MutAnyOrigin],
    slice_values: UnsafePointer[Float64, MutAnyOrigin],
    slice_offsets: UnsafePointer[UInt64, MutAnyOrigin],
    group_maximum: UnsafePointer[Float64, MutAnyOrigin],
    group_count: Int,
    point_indices: UnsafePointer[UInt16, MutAnyOrigin],
    coefficients: UnsafePointer[Float64, MutAnyOrigin],
):
    var thread = thread_idx.x
    var group_index = global_idx.x // FDCB_MATRIX_FILL_BLOCK_SIZE
    if group_index >= group_count:
        return
    var group = groups[group_index].copy()
    var threshold = group_maximum[group_index] * group.relative_cutoff
    var warp = thread // WARP_SIZE
    var lane = thread - warp * WARP_SIZE
    for local_slice in range(
        warp, Int(group.slice_count), FDCB_MATRIX_FILL_WARPS
    ):
        var slice_index = Int(group.slice_offset) + local_slice
        var dose = slice_values[slice_index * 10]
        if dose == 0.0:
            continue
        var energy = energy_slices[Int(slice_energy[slice_index])].copy()
        var output = Int(slice_offsets[slice_index])
        var batches = ceildiv(Int(energy.point_count), WARP_SIZE)
        for batch in range(batches):
            var point = batch * WARP_SIZE + lane
            var coefficient = 0.0
            var keep = UInt32(0)
            if point < Int(energy.point_count):
                coefficient = _contribution(
                    energy_slices,
                    points,
                    group,
                    slice_energy,
                    slice_values,
                    slice_index,
                    point,
                )
                if coefficient * dose > threshold:
                    keep = UInt32(1)
            var position = prefix_sum[DType.uint32, exclusive=True](keep)
            var batch_count = shuffle_idx(
                position + keep, UInt32(WARP_SIZE - 1)
            )
            if keep != UInt32(0):
                var index = output + Int(position)
                point_indices[index] = UInt16(point)
                coefficients[index] = coefficient
            output += Int(batch_count)


def _validate_matrix_view(view: FDCBMatrixProblemViewV1) raises:
    if view.version != FDCB_MATRIX_ABI_VERSION_V1:
        raise Error("unsupported FDCB matrix ABI version")
    if view.flags & ~FDCB_MATRIX_FLAG_DEVICE_ONLY != UInt32(0):
        raise Error("FDCB matrix flags are invalid")
    if view.group_count == UInt32(0):
        raise Error("FDCB matrix requires at least one group")
    if view.maximum_group_slices == UInt32(0):
        raise Error("FDCB matrix maximum group slices must be nonzero")
    for index in range(Int(view.energy_slice_count)):
        var energy = view.energy_slices[index].copy()
        if energy.reserved != UInt32(0):
            raise Error("FDCB matrix energy reserved field must be zero")
        if energy.point_offset + UInt64(energy.point_count) > view.point_count:
            raise Error("FDCB matrix energy point range is invalid")
    for index in range(Int(view.group_count)):
        var group = view.groups[index].copy()
        var raw_group = view.raw_groups[index].copy()
        if group.reserved != UInt32(0):
            raise Error("FDCB matrix group reserved field must be zero")
        if group.slice_offset + UInt64(group.slice_count) > view.slice_count:
            raise Error("FDCB matrix group slice range is invalid")
        if group.slice_count > view.maximum_group_slices:
            raise Error("FDCB matrix group exceeds maximum slice count")
        if raw_group.energy_count != group.slice_count:
            raise Error("FDCB matrix raw group energy count is invalid")
        if UInt64(raw_group.energy_offset) + UInt64(
            raw_group.energy_count
        ) > UInt64(view.energy_slice_count):
            raise Error("FDCB matrix raw group energy range is invalid")
    for index in range(Int(view.energy_slice_count)):
        var raw_energy = view.raw_energies[index].copy()
        if raw_energy.energy_slice >= view.energy_slice_count:
            raise Error("FDCB matrix raw energy index is invalid")
        if raw_energy.ddd_table >= view.ddd_table_count:
            raise Error("FDCB matrix DDD table index is invalid")
    for index in range(Int(view.ddd_table_count)):
        var table = view.ddd_tables[index].copy()
        if table.entry_count == UInt32(0):
            raise Error("FDCB matrix DDD table must not be empty")
        if (
            UInt64(table.entry_offset) + UInt64(table.entry_count)
            > view.ddd_entry_count
        ):
            raise Error("FDCB matrix DDD table range is invalid")


def build_fdcb_matrix_accelerator_v1(
    view: FDCBMatrixProblemViewV1,
    storage_out: UnsafePointer[
        OpaquePointer[MutExternalOrigin], MutExternalOrigin
    ],
    result_out: UnsafePointer[FDCBMatrixResultV1, MutExternalOrigin],
) raises -> Int32:
    var start_ns = perf_counter_ns()
    _validate_matrix_view(view)
    var group_count = Int(view.group_count)
    var energy_count = Int(view.energy_slice_count)
    var point_count = Int(view.point_count)
    var slice_count = Int(view.slice_count)
    var table_count = Int(view.ddd_table_count)
    var entry_count = Int(view.ddd_entry_count)
    var context = DeviceContext()
    var energy_device_bytes = _copy_bytes_to_device(
        context,
        view.energy_slices.bitcast[UInt8](),
        energy_count * size_of[FDCBMatrixEnergySliceV1](),
    )
    var point_device_bytes = _copy_bytes_to_device(
        context,
        view.points.bitcast[UInt8](),
        point_count * size_of[FDCBMatrixPointV1](),
    )
    var group_device_bytes = _copy_bytes_to_device(
        context,
        view.groups.bitcast[UInt8](),
        group_count * size_of[FDCBMatrixGroupV1](),
    )
    var raw_energy_device_bytes = _copy_bytes_to_device(
        context,
        view.raw_energies.bitcast[UInt8](),
        energy_count * size_of[FDCBMatrixRawEnergyV1](),
    )
    var raw_group_device_bytes = _copy_bytes_to_device(
        context,
        view.raw_groups.bitcast[UInt8](),
        group_count * size_of[FDCBMatrixRawGroupV1](),
    )
    var table_device_bytes = _copy_bytes_to_device(
        context,
        view.ddd_tables.bitcast[UInt8](),
        table_count * size_of[FDCBMatrixDDDTableV1](),
    )
    var entry_device_bytes = _copy_bytes_to_device(
        context,
        view.ddd_entries.bitcast[UInt8](),
        entry_count * size_of[FDCBMatrixDDDEntryV1](),
    )
    var energy_device = energy_device_bytes.unsafe_ptr().bitcast[
        FDCBMatrixEnergySliceV1
    ]()
    var point_device = point_device_bytes.unsafe_ptr().bitcast[
        FDCBMatrixPointV1
    ]()
    var group_device = group_device_bytes.unsafe_ptr().bitcast[
        FDCBMatrixGroupV1
    ]()
    var raw_energy_device = raw_energy_device_bytes.unsafe_ptr().bitcast[
        FDCBMatrixRawEnergyV1
    ]()
    var raw_group_device = raw_group_device_bytes.unsafe_ptr().bitcast[
        FDCBMatrixRawGroupV1
    ]()
    var table_device = table_device_bytes.unsafe_ptr().bitcast[
        FDCBMatrixDDDTableV1
    ]()
    var entry_device = entry_device_bytes.unsafe_ptr().bitcast[
        FDCBMatrixDDDEntryV1
    ]()
    var slice_energy_device = context.enqueue_create_buffer[DType.uint32](
        slice_count
    )
    var slice_values_device = context.enqueue_create_buffer[DType.float64](
        slice_count * 10
    )
    var slice_dose_device = context.enqueue_create_buffer[DType.float64](
        slice_count
    )
    var group_maximum_device = context.enqueue_create_buffer[DType.float64](
        group_count
    )
    var group_counts_device = context.enqueue_create_buffer[DType.uint32](
        group_count
    )
    var slice_counts_device = context.enqueue_create_buffer[DType.uint32](
        slice_count
    )
    var raw_work = group_count * Int(view.maximum_group_slices)
    context.enqueue_function[_materialize_raw_slices_kernel](
        group_device,
        raw_energy_device,
        raw_group_device,
        table_device,
        entry_device,
        group_count,
        Int(view.maximum_group_slices),
        slice_energy_device.unsafe_ptr(),
        slice_values_device.unsafe_ptr(),
        slice_dose_device.unsafe_ptr(),
        grid_dim=ceildiv(raw_work, FDCB_MATRIX_BLOCK_SIZE),
        block_dim=FDCB_MATRIX_BLOCK_SIZE,
    )
    context.enqueue_function[_count_kernel](
        energy_device,
        point_device,
        group_device,
        slice_energy_device.unsafe_ptr(),
        slice_values_device.unsafe_ptr(),
        group_count,
        group_maximum_device.unsafe_ptr(),
        group_counts_device.unsafe_ptr(),
        slice_counts_device.unsafe_ptr(),
        grid_dim=group_count,
        block_dim=FDCB_MATRIX_BLOCK_SIZE,
    )
    var group_maximum_host = context.enqueue_create_host_buffer[DType.float64](
        group_count
    )
    var group_counts_host = context.enqueue_create_host_buffer[DType.uint32](
        group_count
    )
    var slice_counts_host = context.enqueue_create_host_buffer[DType.uint32](
        slice_count
    )
    var slice_dose_host = context.enqueue_create_host_buffer[DType.float64](
        slice_count
    )
    context.enqueue_copy(group_maximum_host, group_maximum_device)
    context.enqueue_copy(group_counts_host, group_counts_device)
    context.enqueue_copy(slice_counts_host, slice_counts_device)
    context.enqueue_copy(slice_dose_host, slice_dose_device)
    context.synchronize()
    var count_done_ns = perf_counter_ns()
    var slice_offsets_host = context.enqueue_create_host_buffer[DType.uint64](
        slice_count + 1
    )
    var total_entries = UInt64(0)
    slice_offsets_host[0] = UInt64(0)
    for index in range(slice_count):
        total_entries += UInt64(slice_counts_host[index])
        slice_offsets_host[index + 1] = total_entries
    var slice_offsets_device = context.enqueue_create_buffer[DType.uint64](
        slice_count + 1
    )
    context.enqueue_copy(slice_offsets_device, slice_offsets_host)
    var output_count = Int(total_entries)
    var allocation_count = output_count
    if allocation_count == 0:
        allocation_count = 1
    var point_indices_device = context.enqueue_create_buffer[DType.uint16](
        allocation_count
    )
    var coefficients_device = context.enqueue_create_buffer[DType.float64](
        allocation_count
    )
    context.enqueue_function[_fill_kernel](
        energy_device,
        point_device,
        group_device,
        slice_energy_device.unsafe_ptr(),
        slice_values_device.unsafe_ptr(),
        slice_offsets_device.unsafe_ptr(),
        group_maximum_device.unsafe_ptr(),
        group_count,
        point_indices_device.unsafe_ptr(),
        coefficients_device.unsafe_ptr(),
        grid_dim=group_count,
        block_dim=FDCB_MATRIX_FILL_BLOCK_SIZE,
    )
    var device_only = (view.flags & FDCB_MATRIX_FLAG_DEVICE_ONLY) != UInt32(0)
    var host_output_count = allocation_count
    if device_only:
        host_output_count = 1
    var point_indices_host = context.enqueue_create_host_buffer[DType.uint16](
        host_output_count
    )
    var coefficients_host = context.enqueue_create_host_buffer[DType.float64](
        host_output_count
    )
    if not device_only:
        context.enqueue_copy(point_indices_host, point_indices_device)
        context.enqueue_copy(coefficients_host, coefficients_device)
    context.synchronize()
    var fill_done_ns = perf_counter_ns()
    print(
        "<TIME> FDCB matrix accelerator: count ",
        Float64(count_done_ns - start_ns) * 1.0e-9,
        " sec fill ",
        Float64(fill_done_ns - count_done_ns) * 1.0e-9,
        " sec",
        sep="",
    )
    var storage = alloc[FDCBMatrixStorageV1](1)
    storage.init_pointee_move(
        FDCBMatrixStorageV1(
            context^,
            device_only,
            output_count,
            point_indices_device^,
            coefficients_device^,
            group_maximum_host^,
            group_counts_host^,
            slice_counts_host^,
            slice_dose_host^,
            point_indices_host^,
            coefficients_host^,
        )
    )
    var result_point_indices = storage[].point_indices.unsafe_ptr()
    var result_coefficients = storage[].coefficients.unsafe_ptr()
    if device_only:
        result_point_indices = storage[].point_indices_device.unsafe_ptr()
        result_coefficients = storage[].coefficients_device.unsafe_ptr()
    result_out[] = FDCBMatrixResultV1(
        total_entries,
        view.slice_count,
        view.group_count,
        UInt32(0),
        storage[].group_maximum.unsafe_ptr(),
        storage[].group_entry_counts.unsafe_ptr(),
        storage[].slice_entry_counts.unsafe_ptr(),
        storage[].slice_dose.unsafe_ptr(),
        result_point_indices,
        result_coefficients,
    )
    storage_out[] = storage.bitcast[NoneType]()
    return Int32(0)


def fdcb_matrix_build_accelerator_abi_v1(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    storage_out: UnsafePointer[
        OpaquePointer[MutExternalOrigin], MutExternalOrigin
    ],
    result_out: UnsafePointer[FDCBMatrixResultV1, MutExternalOrigin],
) -> Int32:
    try:
        var view = problem_pointer.bitcast[FDCBMatrixProblemViewV1]()[].copy()
        return build_fdcb_matrix_accelerator_v1(view, storage_out, result_out)
    except error:
        print("FDCB matrix accelerator ABI error:", error)
        return Int32(-1)


def fdcb_matrix_storage_destroy_abi_v1(
    storage_pointer: OpaquePointer[MutExternalOrigin],
) -> Int32:
    var storage = storage_pointer.bitcast[FDCBMatrixStorageV1]()
    storage.destroy_pointee()
    storage.free()
    return Int32(0)
