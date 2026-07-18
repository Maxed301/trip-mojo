"""Shared-ABI accelerator backend for packed clinical direct dose."""

from std.gpu import global_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import ceildiv, exp
from std.memory import OpaquePointer, UnsafePointer
from std.sys.info import size_of

from clinical_dose import (
    BioInterpolation,
    ClinicalDoseBioEntry,
    ClinicalDoseBioTable,
    ClinicalDoseCTState,
    ClinicalDoseDDDEntry,
    ClinicalDoseDDDTable,
    ClinicalDoseEnergy,
    ClinicalDoseField,
    ClinicalDoseGridField,
    ClinicalDoseGrid,
    ClinicalDoseOutput,
    ClinicalDosePointRun,
    ClinicalDosePoint,
    ClinicalDosePosition,
    ClinicalDoseProblem,
    ClinicalDoseState,
    DDDInterpolation,
    CLINICAL_DOSE_8LN2,
    CLINICAL_DOSE_ALGORITHM_MSDB,
    CLINICAL_DOSE_BIOLOGY_LOW_DOSE,
    CLINICAL_DOSE_PI,
    build_dense_hlut,
    interpolate_bio,
    interpolate_ddd,
    siddon_h2o,
    validate_clinical_dose,
)


comptime CLINICAL_DOSE_ACCELERATOR_BLOCK_SIZE = 128


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


def _copy_float64_list_to_device(
    context: DeviceContext, values: List[Float64]
) raises -> DeviceBuffer[DType.float64]:
    var host = context.enqueue_create_host_buffer[DType.float64](len(values))
    for index in range(len(values)):
        host[index] = values[index]
    var device = context.enqueue_create_buffer[DType.float64](len(values))
    context.enqueue_copy(device, host)
    return device^


def _copy_int64_list_to_device(
    context: DeviceContext, values: List[Int64]
) raises -> DeviceBuffer[DType.int64]:
    var host = context.enqueue_create_host_buffer[DType.int64](len(values))
    for index in range(len(values)):
        host[index] = values[index]
    var device = context.enqueue_create_buffer[DType.int64](len(values))
    context.enqueue_copy(device, host)
    return device^


@always_inline("nodebug")
def _grid_from_metadata(
    integer_metadata: UnsafePointer[Int64, MutAnyOrigin],
    floating_metadata: UnsafePointer[Float64, MutAnyOrigin],
    integer_offset: Int,
    floating_offset: Int,
) -> ClinicalDoseGrid:
    return ClinicalDoseGrid(
        Int32(integer_metadata[integer_offset]),
        Int32(integer_metadata[integer_offset + 1]),
        Int32(integer_metadata[integer_offset + 2]),
        floating_metadata[floating_offset],
        floating_metadata[floating_offset + 1],
        floating_metadata[floating_offset + 2],
        floating_metadata[floating_offset + 3],
        floating_metadata[floating_offset + 4],
        floating_metadata[floating_offset + 5],
        floating_metadata[floating_offset + 6],
        floating_metadata[floating_offset + 7],
        floating_metadata[floating_offset + 8],
    )


def clinical_dose_kernel(
    integer_metadata: UnsafePointer[Int64, MutAnyOrigin],
    floating_metadata: UnsafePointer[Float64, MutAnyOrigin],
    voxel_voi: UnsafePointer[Int32, MutAnyOrigin],
    ct_states: UnsafePointer[ClinicalDoseCTState, MutAnyOrigin],
    states: UnsafePointer[ClinicalDoseState, MutAnyOrigin],
    transformed_voxels: UnsafePointer[ClinicalDosePosition, MutAnyOrigin],
    ct_data: UnsafePointer[Int16, MutAnyOrigin],
    ct_boundaries: UnsafePointer[Float64, MutAnyOrigin],
    dose_axis_centers: UnsafePointer[Float64, MutAnyOrigin],
    dense_hlut: UnsafePointer[Float64, MutAnyOrigin],
    grid_fields: UnsafePointer[ClinicalDoseGridField, MutAnyOrigin],
    fields: UnsafePointer[ClinicalDoseField, MutAnyOrigin],
    energies: UnsafePointer[ClinicalDoseEnergy, MutAnyOrigin],
    points: UnsafePointer[ClinicalDosePoint, MutAnyOrigin],
    ddd_tables: UnsafePointer[ClinicalDoseDDDTable, MutAnyOrigin],
    ddd_entries: UnsafePointer[ClinicalDoseDDDEntry, MutAnyOrigin],
    bio_tables: UnsafePointer[ClinicalDoseBioTable, MutAnyOrigin],
    bio_entries: UnsafePointer[ClinicalDoseBioEntry, MutAnyOrigin],
    energy_run_offsets: UnsafePointer[UInt32, MutAnyOrigin],
    point_runs: UnsafePointer[ClinicalDosePointRun, MutAnyOrigin],
    output: UnsafePointer[ClinicalDoseOutput, MutAnyOrigin],
):
    var work_index = global_idx.x
    var voxel_count = Int(integer_metadata[0])
    var state_count = Int(integer_metadata[4])
    if work_index >= voxel_count * state_count:
        return
    var state_index = work_index // voxel_count
    var voxel = work_index - state_index * voxel_count

    var result = ClinicalDoseOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var field_count = Int(integer_metadata[2])
    var voi_count = Int(integer_metadata[3])
    var biological = (
        UInt32(integer_metadata[1]) == CLINICAL_DOSE_BIOLOGY_LOW_DOSE
    )
    var divergent = UInt32(integer_metadata[11]) == CLINICAL_DOSE_ALGORITHM_MSDB
    var voi = -1
    if biological:
        voi = Int(voxel_voi[voxel])
        if voi < 0 or voi >= voi_count:
            output[work_index] = result^
            return

    var dose_grid = _grid_from_metadata(
        integer_metadata, floating_metadata, 5, 0
    )
    var ct_state = ct_states[state_index].copy()
    var nx = Int(dose_grid.nx)
    var ny = Int(dose_grid.ny)
    var ix = voxel % nx
    var iy = (voxel // nx) % ny
    var iz = voxel // (nx * ny)
    var px = dose_axis_centers[Int(integer_metadata[8]) + ix]
    var py = dose_axis_centers[Int(integer_metadata[9]) + iy]
    var pz = dose_axis_centers[Int(integer_metadata[10]) + iz]
    var field_offset = 0
    var state_field_count = field_count
    if state_count > 1:
        var state = states[state_index].copy()
        field_offset = Int(state.field_offset)
        state_field_count = Int(state.field_count)
        var position = transformed_voxels[work_index].copy()
        px = position.x
        py = position.y
        pz = position.z

    for field_local in range(field_offset, field_offset + state_field_count):
        var grid_field = grid_fields[field_local].copy()
        if (
            grid_field.gated != Int32(0)
            or Int(grid_field.field_index) >= field_count
        ):
            continue
        var gx = (
            Float64(grid_field.m00) * px
            + Float64(grid_field.m01) * py
            + Float64(grid_field.m02) * pz
            + Float64(grid_field.m03)
        )
        var gy = (
            Float64(grid_field.m10) * px
            + Float64(grid_field.m11) * py
            + Float64(grid_field.m12) * pz
            + Float64(grid_field.m13)
        )
        var gz = (
            Float64(grid_field.m20) * px
            + Float64(grid_field.m21) * py
            + Float64(grid_field.m22) * pz
            + Float64(grid_field.m23)
        )
        if (
            gx < Float64(grid_field.window_x0)
            or gx > Float64(grid_field.window_x1)
            or gy < Float64(grid_field.window_y0)
            or gy > Float64(grid_field.window_y1)
        ):
            continue
        var h2o = siddon_h2o(
            ct_state,
            ct_data,
            ct_boundaries,
            dense_hlut,
            px,
            py,
            pz,
            Float64(grid_field.direction_x),
            Float64(grid_field.direction_y),
            Float64(grid_field.direction_z),
        )
        h2o += Float64(grid_field.off_h2o) + Float64(grid_field.bolus)
        var field = fields[Int(grid_field.field_index)].copy()
        var divergence_x = 0.0
        var divergence_y = 0.0
        if divergent:
            if field.scanner_x != 0.0:
                divergence_x = gz / Float64(field.scanner_x)
            if field.scanner_y != 0.0:
                divergence_y = gz / Float64(field.scanner_y)
        for energy_local in range(Int(field.energy_count)):
            var energy_index = Int(field.energy_offset) + energy_local
            var energy = energies[energy_index].copy()
            if (
                gx < Float64(energy.window_x0)
                or gx > Float64(energy.window_x1)
                or gy < Float64(energy.window_y0)
                or gy > Float64(energy.window_y1)
            ):
                continue
            var table = ddd_tables[Int(energy.ddd_table)].copy()
            var focus2 = Float64(energy.focus) * Float64(energy.focus)
            var bio_table = ClinicalDoseBioTable(UInt32(0), UInt32(0), 0.0)
            var bio = BioInterpolation(0.0, 0.0, 0.0, 0.0, 0.0)
            if biological:
                bio_table = bio_tables[
                    Int(energy.bio_table_offset) + voi
                ].copy()
            var last_z = Float64.MAX
            var depth = DDDInterpolation(False, 0.0, 0.0, 0.0, 0.0)
            for run_index in range(
                Int(energy_run_offsets[energy_index]),
                Int(energy_run_offsets[energy_index + 1]),
            ):
                var run = point_runs[run_index].copy()
                var dy = gy - Float64(run.y)
                if divergence_y != 0.0:
                    dy += divergence_y * Float64(run.y)
                var dy2 = dy * dy
                if dy2 > Float64(run.max_f2):
                    continue
                for point_index in range(
                    Int(run.point_offset),
                    Int(run.point_offset + run.point_count),
                ):
                    var point = points[point_index].copy()
                    if point.particles == 0.0:
                        continue
                    var dx = gx - Float64(point.x)
                    if divergence_x != 0.0:
                        dx += divergence_x * Float64(point.x)
                    var dx2 = dx * dx
                    if dx2 > Float64(point.f2_max) or dy2 > Float64(
                        point.f2_max
                    ):
                        continue
                    var r2 = dx2 + dy2
                    if r2 >= Float64(field.dose_extension2) * Float64(
                        point.f2_max
                    ):
                        continue
                    var z_cm = (
                        h2o
                        + Float64(energy.range_shifter)
                        + Float64(point.delta_z)
                    ) * 0.1
                    if z_cm != last_z:
                        depth = interpolate_ddd(table, ddd_entries, z_cm)
                        last_z = z_cm
                    if not depth.valid:
                        continue
                    var lateral = 0.0
                    var f2_0 = focus2 + depth.fwhm1 * depth.fwhm1
                    if f2_0 <= 0.0:
                        continue
                    var isig2_0 = CLINICAL_DOSE_8LN2 / f2_0 * 0.5
                    if r2 < Float64(field.dose_extension2) * f2_0:
                        lateral = (
                            exp(-r2 * isig2_0)
                            * (isig2_0 / CLINICAL_DOSE_PI)
                            * (1.0 - depth.mix)
                        )
                    if depth.mix > 0.0:
                        var f2_1 = focus2 + depth.fwhm2 * depth.fwhm2
                        if (
                            f2_1 > 0.0
                            and r2 < Float64(field.dose_extension2) * f2_1
                        ):
                            var isig2_1 = CLINICAL_DOSE_8LN2 / f2_1 * 0.5
                            lateral += (
                                exp(-r2 * isig2_1)
                                * (isig2_1 / CLINICAL_DOSE_PI)
                                * depth.mix
                            )
                    if biological and lateral > 0.0 and bio.alpha == 0.0:
                        var bio_z = (
                            h2o
                            + Float64(energy.range_shifter)
                            + Float64(point.delta_z)
                        ) * Float64(bio_table.z_scale)
                        bio = interpolate_bio(bio_table, bio_entries, bio_z)
                    var weighted_fluence = lateral * point.particles
                    result.absorbed_dose += weighted_fluence * depth.dose
                    if biological:
                        result.alpha += weighted_fluence * bio.alpha
                        result.sqrt_beta += weighted_fluence * bio.sqrt_beta
                        result.let_mix += weighted_fluence * bio.let_mix
                        result.let_bar += weighted_fluence * bio.let_bar
                        result.let_dm_sum += weighted_fluence * bio.let_dm_sum
    output[work_index] = result^


def reduce_clinical_dose_states_kernel(
    state_output: UnsafePointer[ClinicalDoseOutput, MutAnyOrigin],
    output: UnsafePointer[ClinicalDoseOutput, MutAnyOrigin],
    voxel_count: Int,
    state_count: Int,
):
    var voxel = global_idx.x
    if voxel >= voxel_count:
        return
    var result = ClinicalDoseOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for state_index in range(state_count):
        var value = state_output[state_index * voxel_count + voxel].copy()
        result.absorbed_dose += value.absorbed_dose
        result.alpha += value.alpha
        result.sqrt_beta += value.sqrt_beta
        result.let_mix += value.let_mix
        result.let_bar += value.let_bar
        result.let_dm_sum += value.let_dm_sum
    output[voxel] = result^


def compute_clinical_dose_accelerator(
    view: ClinicalDoseProblem,
    output: UnsafePointer[ClinicalDoseOutput, MutExternalOrigin],
) raises:
    validate_clinical_dose(view)
    var dense_hlut = build_dense_hlut(view, Int(view.max_threads))
    var energy_run_offsets = List[UInt32](capacity=Int(view.energy_count) + 1)
    var point_runs = List[ClinicalDosePointRun]()
    for energy_index in range(Int(view.energy_count)):
        energy_run_offsets.append(UInt32(len(point_runs)))
        var energy = view.energies[energy_index].copy()
        var point_index = Int(energy.point_offset)
        var point_end = point_index + Int(energy.point_count)
        while point_index < point_end:
            var run_start = point_index
            var y = view.points[point_index].y
            var max_f2 = view.points[point_index].f2_max
            point_index += 1
            while point_index < point_end and view.points[point_index].y == y:
                var f2 = view.points[point_index].f2_max
                if f2 > max_f2 or f2 != f2:
                    max_f2 = f2
                point_index += 1
            point_runs.append(
                ClinicalDosePointRun(
                    UInt32(run_start),
                    UInt32(point_index - run_start),
                    y,
                    max_f2,
                )
            )
    energy_run_offsets.append(UInt32(len(point_runs)))

    var integer_metadata: List[Int64] = [
        Int64(view.grid_voxel_count),
        Int64(view.biology_model),
        Int64(view.field_count),
        Int64(view.voi_count),
        Int64(view.state_count),
        Int64(view.dose_grid.nx),
        Int64(view.dose_grid.ny),
        Int64(view.dose_grid.nz),
        Int64(view.dose_x_offset),
        Int64(view.dose_y_offset),
        Int64(view.dose_z_offset),
        Int64(view.algorithm),
    ]
    var floating_metadata: List[Float64] = [
        view.dose_grid.x0,
        view.dose_grid.y0,
        view.dose_grid.z0,
        view.dose_grid.dx,
        view.dose_grid.dy,
        view.dose_grid.dz,
        view.dose_grid.boundary_x0,
        view.dose_grid.boundary_y0,
        view.dose_grid.boundary_z0,
    ]

    var context = DeviceContext()
    var integer_device = _copy_int64_list_to_device(context, integer_metadata)
    var floating_device = _copy_float64_list_to_device(
        context, floating_metadata
    )
    var voi_device = _copy_bytes_to_device(
        context,
        view.voxel_voi.bitcast[UInt8](),
        (
            Int(view.grid_voxel_count) * size_of[Int32]() if view.biology_model
            == CLINICAL_DOSE_BIOLOGY_LOW_DOSE else 0
        ),
    )
    var ct_state_device = _copy_bytes_to_device(
        context,
        view.ct_states.bitcast[UInt8](),
        Int(view.state_count) * size_of[ClinicalDoseCTState](),
    )
    var state_device = _copy_bytes_to_device(
        context,
        view.states.bitcast[UInt8](),
        (
            Int(view.state_count)
            * size_of[ClinicalDoseState]() if view.state_count
            > UInt32(1) else 0
        ),
    )
    var transformed_device = _copy_bytes_to_device(
        context,
        view.transformed_voxels.bitcast[UInt8](),
        Int(view.transformed_voxel_count) * size_of[ClinicalDosePosition](),
    )
    var ct_device = _copy_bytes_to_device(
        context,
        view.ct_data.bitcast[UInt8](),
        Int(view.ct_value_count) * size_of[Int16](),
    )
    var dose_axis_device = _copy_bytes_to_device(
        context,
        view.dose_axis_centers.bitcast[UInt8](),
        Int(view.dose_axis_count) * size_of[Float64](),
    )
    var ct_boundary_device = _copy_bytes_to_device(
        context,
        view.ct_boundaries.bitcast[UInt8](),
        Int(view.ct_boundary_count) * size_of[Float64](),
    )
    var dense_device = _copy_bytes_to_device(
        context,
        Span(dense_hlut).unsafe_ptr().bitcast[UInt8](),
        len(dense_hlut) * size_of[Float64](),
    )
    var grid_field_device = _copy_bytes_to_device(
        context,
        view.grid_fields.bitcast[UInt8](),
        Int(view.field_count) * size_of[ClinicalDoseGridField](),
    )
    var field_device = _copy_bytes_to_device(
        context,
        view.fields.bitcast[UInt8](),
        Int(view.field_count) * size_of[ClinicalDoseField](),
    )
    var energy_device = _copy_bytes_to_device(
        context,
        view.energies.bitcast[UInt8](),
        Int(view.energy_count) * size_of[ClinicalDoseEnergy](),
    )
    var point_device = _copy_bytes_to_device(
        context,
        view.points.bitcast[UInt8](),
        Int(view.point_count) * size_of[ClinicalDosePoint](),
    )
    var ddd_table_device = _copy_bytes_to_device(
        context,
        view.ddd_tables.bitcast[UInt8](),
        Int(view.ddd_table_count) * size_of[ClinicalDoseDDDTable](),
    )
    var ddd_entry_device = _copy_bytes_to_device(
        context,
        view.ddd_entries.bitcast[UInt8](),
        Int(view.ddd_entry_count) * size_of[ClinicalDoseDDDEntry](),
    )
    var bio_table_device = _copy_bytes_to_device(
        context,
        view.bio_tables.bitcast[UInt8](),
        Int(view.bio_table_count) * size_of[ClinicalDoseBioTable](),
    )
    var bio_entry_device = _copy_bytes_to_device(
        context,
        view.bio_entries.bitcast[UInt8](),
        Int(view.bio_entry_count) * size_of[ClinicalDoseBioEntry](),
    )
    var run_offset_device = _copy_bytes_to_device(
        context,
        Span(energy_run_offsets).unsafe_ptr().bitcast[UInt8](),
        len(energy_run_offsets) * size_of[UInt32](),
    )
    var run_device = _copy_bytes_to_device(
        context,
        Span(point_runs).unsafe_ptr().bitcast[UInt8](),
        len(point_runs) * size_of[ClinicalDosePointRun](),
    )
    var state_output_device = context.enqueue_create_buffer[DType.float64](
        Int(view.grid_voxel_count) * Int(view.state_count) * 6
    )
    var output_device = context.enqueue_create_buffer[DType.float64](
        Int(view.grid_voxel_count) * 6
    )
    context.enqueue_function[clinical_dose_kernel](
        integer_device.unsafe_ptr(),
        floating_device.unsafe_ptr(),
        voi_device.unsafe_ptr().bitcast[Int32](),
        ct_state_device.unsafe_ptr().bitcast[ClinicalDoseCTState](),
        state_device.unsafe_ptr().bitcast[ClinicalDoseState](),
        transformed_device.unsafe_ptr().bitcast[ClinicalDosePosition](),
        ct_device.unsafe_ptr().bitcast[Int16](),
        ct_boundary_device.unsafe_ptr().bitcast[Float64](),
        dose_axis_device.unsafe_ptr().bitcast[Float64](),
        dense_device.unsafe_ptr().bitcast[Float64](),
        grid_field_device.unsafe_ptr().bitcast[ClinicalDoseGridField](),
        field_device.unsafe_ptr().bitcast[ClinicalDoseField](),
        energy_device.unsafe_ptr().bitcast[ClinicalDoseEnergy](),
        point_device.unsafe_ptr().bitcast[ClinicalDosePoint](),
        ddd_table_device.unsafe_ptr().bitcast[ClinicalDoseDDDTable](),
        ddd_entry_device.unsafe_ptr().bitcast[ClinicalDoseDDDEntry](),
        bio_table_device.unsafe_ptr().bitcast[ClinicalDoseBioTable](),
        bio_entry_device.unsafe_ptr().bitcast[ClinicalDoseBioEntry](),
        run_offset_device.unsafe_ptr().bitcast[UInt32](),
        run_device.unsafe_ptr().bitcast[ClinicalDosePointRun](),
        state_output_device.unsafe_ptr().bitcast[ClinicalDoseOutput](),
        grid_dim=ceildiv(
            Int(view.grid_voxel_count) * Int(view.state_count),
            CLINICAL_DOSE_ACCELERATOR_BLOCK_SIZE,
        ),
        block_dim=CLINICAL_DOSE_ACCELERATOR_BLOCK_SIZE,
    )
    context.enqueue_function[reduce_clinical_dose_states_kernel](
        state_output_device.unsafe_ptr().bitcast[ClinicalDoseOutput](),
        output_device.unsafe_ptr().bitcast[ClinicalDoseOutput](),
        Int(view.grid_voxel_count),
        Int(view.state_count),
        grid_dim=ceildiv(
            Int(view.grid_voxel_count), CLINICAL_DOSE_ACCELERATOR_BLOCK_SIZE
        ),
        block_dim=CLINICAL_DOSE_ACCELERATOR_BLOCK_SIZE,
    )
    var output_host = context.enqueue_create_host_buffer[DType.float64](
        Int(view.grid_voxel_count) * 6
    )
    context.enqueue_copy(output_host, output_device)
    context.synchronize()
    var output_values = output_host.unsafe_ptr()
    for voxel in range(Int(view.grid_voxel_count)):
        var offset = voxel * 6
        output[voxel] = ClinicalDoseOutput(
            output_values[offset],
            output_values[offset + 1],
            output_values[offset + 2],
            output_values[offset + 3],
            output_values[offset + 4],
            output_values[offset + 5],
        )


def clinical_dose_compute_accelerator_abi(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    output: UnsafePointer[ClinicalDoseOutput, MutExternalOrigin],
    output_count: UInt64,
) -> Int32:
    try:
        var view = problem_pointer.bitcast[ClinicalDoseProblem]()[].copy()
        if output_count != UInt64(view.grid_voxel_count):
            return Int32(-2)
        compute_clinical_dose_accelerator(view, output)
        return Int32(0)
    except error:
        print("Clinical dose accelerator ABI error:", error)
        return Int32(-1)
