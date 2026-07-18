"""Backend-neutral packed clinical direct-dose CPU kernel."""

from std.algorithm import parallelize
from std.memory import OpaquePointer, UnsafePointer
from std.sys.info import size_of

from reference_math import reference_exp


comptime CLINICAL_DOSE_ALGORITHM_MS = UInt32(1)
comptime CLINICAL_DOSE_ALGORITHM_MSDB = UInt32(2)
comptime CLINICAL_DOSE_BIOLOGY_NONE = UInt32(0)
comptime CLINICAL_DOSE_BIOLOGY_LOW_DOSE = UInt32(1)
comptime CLINICAL_DOSE_THREADS = 12
comptime CLINICAL_DOSE_PI = 3.14159265358979323846
comptime CLINICAL_DOSE_8LN2 = 0.6932 * 8.0


@fieldwise_init
struct ClinicalDosePoint(Copyable, Movable):
    var x: Float64
    var y: Float64
    var delta_z: Float64
    var f2_max: Float64
    var particles: Float64


@fieldwise_init
struct ClinicalDoseDDDEntry(Copyable, Movable):
    var z_cm: Float64
    var dose: Float64
    var fwhm1: Float64
    var mix: Float64
    var fwhm2: Float64


@fieldwise_init
struct ClinicalDoseDDDTable(Copyable, Movable):
    var entry_offset: UInt32
    var entry_count: UInt32
    var column_count: UInt32


@fieldwise_init
struct ClinicalDoseBioEntry(Copyable, Movable):
    var z_cm: Float64
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64
    var let_dm_sum: Float64


@fieldwise_init
struct ClinicalDoseBioTable(Copyable, Movable):
    var entry_offset: UInt32
    var entry_count: UInt32
    var z_scale: Float64


@fieldwise_init
struct ClinicalDoseEnergy(Copyable, Movable):
    var point_offset: UInt32
    var point_count: UInt32
    var ddd_table: UInt32
    var bio_table_offset: UInt32
    var focus: Float64
    var range_shifter: Float64
    var window_x0: Float64
    var window_x1: Float64
    var window_y0: Float64
    var window_y1: Float64


@fieldwise_init
struct ClinicalDoseField(Copyable, Movable):
    var energy_offset: UInt32
    var energy_count: UInt32
    var dose_extension2: Float64
    var scanner_x: Float64
    var scanner_y: Float64


@fieldwise_init
struct ClinicalDosePointRun(Copyable, Movable):
    var point_offset: UInt32
    var point_count: UInt32
    var y: Float64
    var max_f2: Float64


@fieldwise_init
struct ClinicalDoseGrid(Copyable, Movable):
    var nx: Int32
    var ny: Int32
    var nz: Int32
    var x0: Float64
    var y0: Float64
    var z0: Float64
    var dx: Float64
    var dy: Float64
    var dz: Float64
    var boundary_x0: Float64
    var boundary_y0: Float64
    var boundary_z0: Float64


@fieldwise_init
struct ClinicalDoseCTState(Copyable, Movable):
    var grid: ClinicalDoseGrid
    var data_offset: UInt64
    var x_boundary_offset: UInt64
    var y_boundary_offset: UInt64
    var z_boundary_offset: UInt64
    var pmod: Float64
    var pore_size: Float64
    var byte_swap: Int32


@fieldwise_init
struct ClinicalDoseState(Copyable, Movable):
    var field_offset: UInt32
    var field_count: UInt32


@fieldwise_init
struct ClinicalDosePosition(Copyable, Movable):
    var x: Float64
    var y: Float64
    var z: Float64


@fieldwise_init
struct ClinicalDoseGridField(Copyable, Movable):
    var field_index: UInt32
    var m00: Float64
    var m01: Float64
    var m02: Float64
    var m03: Float64
    var m10: Float64
    var m11: Float64
    var m12: Float64
    var m13: Float64
    var m20: Float64
    var m21: Float64
    var m22: Float64
    var m23: Float64
    var direction_x: Float64
    var direction_y: Float64
    var direction_z: Float64
    var off_h2o: Float64
    var bolus: Float64
    var window_x0: Float64
    var window_x1: Float64
    var window_y0: Float64
    var window_y1: Float64
    var gated: Int32


@fieldwise_init
struct ClinicalDoseOutput(Copyable, Movable):
    var absorbed_dose: Float64
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64
    var let_dm_sum: Float64


@fieldwise_init
struct ClinicalDoseProblem(Copyable, Movable):
    var grid_voxel_count: UInt32
    var state_count: UInt32
    var field_count: UInt32
    var energy_count: UInt32
    var point_count: UInt32
    var ddd_table_count: UInt32
    var ddd_entry_count: UInt32
    var bio_table_count: UInt32
    var bio_entry_count: UInt32
    var voi_count: UInt32
    var hlut_count: UInt32
    var algorithm: UInt32
    var biology_model: UInt32
    var max_threads: UInt32
    var struct_size: UInt32
    var ct_value_count: UInt64
    var ct_boundary_count: UInt64
    var dose_axis_count: UInt64
    var dose_x_offset: UInt64
    var dose_y_offset: UInt64
    var dose_z_offset: UInt64
    var dose_grid: ClinicalDoseGrid
    var voxel_voi: UnsafePointer[Int32, MutExternalOrigin]
    var ct_states: UnsafePointer[ClinicalDoseCTState, MutExternalOrigin]
    var ct_data: UnsafePointer[Int16, MutExternalOrigin]
    var ct_boundaries: UnsafePointer[Float64, MutExternalOrigin]
    var dose_axis_centers: UnsafePointer[Float64, MutExternalOrigin]
    var hlut_x: UnsafePointer[Float64, MutExternalOrigin]
    var hlut_y: UnsafePointer[Float64, MutExternalOrigin]
    var grid_fields: UnsafePointer[ClinicalDoseGridField, MutExternalOrigin]
    var fields: UnsafePointer[ClinicalDoseField, MutExternalOrigin]
    var energies: UnsafePointer[ClinicalDoseEnergy, MutExternalOrigin]
    var points: UnsafePointer[ClinicalDosePoint, MutExternalOrigin]
    var ddd_tables: UnsafePointer[ClinicalDoseDDDTable, MutExternalOrigin]
    var ddd_entries: UnsafePointer[ClinicalDoseDDDEntry, MutExternalOrigin]
    var bio_tables: UnsafePointer[ClinicalDoseBioTable, MutExternalOrigin]
    var bio_entries: UnsafePointer[ClinicalDoseBioEntry, MutExternalOrigin]
    var transformed_voxel_count: UInt64
    var states: UnsafePointer[ClinicalDoseState, MutExternalOrigin]
    var transformed_voxels: UnsafePointer[
        ClinicalDosePosition, MutExternalOrigin
    ]


@fieldwise_init
struct DDDInterpolation(Copyable, Movable):
    var valid: Bool
    var dose: Float64
    var fwhm1: Float64
    var mix: Float64
    var fwhm2: Float64


@fieldwise_init
struct BioInterpolation(Copyable, Movable):
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64
    var let_dm_sum: Float64


@always_inline("nodebug")
def min3(a: Float64, b: Float64, c: Float64) -> Float64:
    var result = a
    if b < result:
        result = b
    if c < result:
        result = c
    return result


@always_inline("nodebug")
def max2(a: Float64, b: Float64) -> Float64:
    if a > b:
        return a
    return b


@always_inline("nodebug")
def abs64(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


@always_inline("nodebug")
def swap_i16(value: Int16) -> Int16:
    var bits = rebind[UInt16](value)
    return rebind[Int16]((bits << 8) | (bits >> 8))


def validate_clinical_dose(view: ClinicalDoseProblem) raises:
    if view.struct_size != UInt32(size_of[ClinicalDoseProblem]()):
        raise Error("clinical dose ABI struct size mismatch")
    if view.state_count == UInt32(0):
        raise Error("clinical dose requires at least one CT state")
    if view.grid_voxel_count == UInt32(0) or view.field_count == UInt32(0):
        raise Error("clinical dose grid and field counts must be nonzero")
    if (
        view.dose_grid.nx <= Int32(0)
        or view.dose_grid.ny <= Int32(0)
        or view.dose_grid.nz <= Int32(0)
    ):
        raise Error("clinical dose grid dimensions must be positive")
    if (
        view.algorithm != CLINICAL_DOSE_ALGORITHM_MS
        and view.algorithm != CLINICAL_DOSE_ALGORITHM_MSDB
    ):
        raise Error("clinical dose supports only ms and msdb")
    var biological = view.biology_model == CLINICAL_DOSE_BIOLOGY_LOW_DOSE
    if biological and view.biology_model != CLINICAL_DOSE_BIOLOGY_LOW_DOSE:
        raise Error("biological clinical dose  requires low-dose biology")
    if not biological and view.biology_model != CLINICAL_DOSE_BIOLOGY_NONE:
        raise Error("physical clinical dose must not request a biology model")
    if view.max_threads == UInt32(0):
        raise Error("clinical dose thread count is invalid")
    var expected = (
        UInt64(view.dose_grid.nx)
        * UInt64(view.dose_grid.ny)
        * UInt64(view.dose_grid.nz)
    )
    if expected != UInt64(view.grid_voxel_count):
        raise Error("clinical dose grid dimensions do not match voxel count")
    if (
        view.dose_x_offset + UInt64(view.dose_grid.nx) > view.dose_axis_count
        or view.dose_y_offset + UInt64(view.dose_grid.ny) > view.dose_axis_count
        or view.dose_z_offset + UInt64(view.dose_grid.nz) > view.dose_axis_count
    ):
        raise Error("clinical dose axes exceed packed center data")
    if view.hlut_count == UInt32(0) or view.ct_value_count == UInt64(0):
        raise Error("clinical dose requires CT and HLUT data")
    var four_d = view.state_count > UInt32(1)
    var expected_transformed = UInt64(view.state_count) * UInt64(
        view.grid_voxel_count
    )
    if four_d and view.transformed_voxel_count != expected_transformed:
        raise Error("4D clinical dose requires one transformed voxel per state")
    if not four_d and view.transformed_voxel_count != UInt64(0):
        raise Error("static clinical dose must not provide transformed voxels")
    var next_field = UInt64(0)
    for state_index in range(Int(view.state_count)):
        var ct_state = view.ct_states[state_index].copy()
        var ct_grid = ct_state.grid.copy()
        var ct_end = ct_state.data_offset + UInt64(ct_grid.nx) * UInt64(
            ct_grid.ny
        ) * UInt64(ct_grid.nz)
        if (
            ct_grid.nx <= Int32(0)
            or ct_grid.ny <= Int32(0)
            or ct_grid.nz <= Int32(0)
            or ct_end > view.ct_value_count
            or ct_state.x_boundary_offset + UInt64(ct_grid.nx) + 1
            > view.ct_boundary_count
            or ct_state.y_boundary_offset + UInt64(ct_grid.ny) + 1
            > view.ct_boundary_count
            or ct_state.z_boundary_offset + UInt64(ct_grid.nz) + 1
            > view.ct_boundary_count
        ):
            raise Error("clinical dose CT state exceeds packed CT data")
        if four_d:
            var state = view.states[state_index].copy()
            if UInt64(state.field_offset) != next_field:
                raise Error("clinical dose state fields must be contiguous")
            next_field += UInt64(state.field_count)
            if next_field > UInt64(view.field_count):
                raise Error("clinical dose state field range is out of bounds")
    if four_d and next_field != UInt64(view.field_count):
        raise Error("clinical dose states do not cover all fields")
    if biological:
        if view.voi_count == UInt32(0) or view.bio_table_count == UInt32(0):
            raise Error(
                "biological clinical dose requires VOI and biology tables"
            )
    for field_index in range(Int(view.field_count)):
        var grid_field = view.grid_fields[field_index].copy()
        if grid_field.field_index >= view.field_count:
            raise Error("clinical dose grid field index is out of range")
        var field = view.fields[Int(grid_field.field_index)].copy()
        if UInt64(field.energy_offset) + UInt64(field.energy_count) > UInt64(
            view.energy_count
        ):
            raise Error("clinical dose field energy range is out of bounds")
    for energy_index in range(Int(view.energy_count)):
        var energy = view.energies[energy_index].copy()
        if (
            UInt64(energy.point_offset) + UInt64(energy.point_count)
            > UInt64(view.point_count)
            or energy.ddd_table >= view.ddd_table_count
            or energy.point_count == UInt32(0)
        ):
            raise Error("clinical dose energy range is out of bounds")
        for point_local in range(Int(energy.point_count)):
            var point = view.points[
                Int(energy.point_offset) + point_local
            ].copy()
            if point.f2_max < 0.0 or point.particles < 0.0:
                raise Error(
                    "clinical dose point has invalid support or particles"
                )
        if biological and UInt64(energy.bio_table_offset) + UInt64(
            view.voi_count
        ) > UInt64(view.bio_table_count):
            raise Error("clinical dose biology table range is out of bounds")
    for table_index in range(Int(view.ddd_table_count)):
        var table = view.ddd_tables[table_index].copy()
        if UInt64(table.entry_offset) + UInt64(table.entry_count) > UInt64(
            view.ddd_entry_count
        ):
            raise Error("clinical dose DDD table range is out of bounds")
    if biological:
        for table_index in range(Int(view.bio_table_count)):
            var table = view.bio_tables[table_index].copy()
            if UInt64(table.entry_offset) + UInt64(table.entry_count) > UInt64(
                view.bio_entry_count
            ):
                raise Error(
                    "clinical dose biology table range is out of bounds"
                )


@always_inline("nodebug")
def hlut_lookup(view: ClinicalDoseProblem, hu: Float64) -> Float64:
    var count = Int(view.hlut_count)
    if hu <= Float64(view.hlut_x[0]):
        return max2(0.0, Float64(view.hlut_y[0]))
    var low = 0
    var high = count - 1
    while high - low > 1:
        var middle = (low + high) // 2
        if hu < Float64(view.hlut_x[middle]):
            high = middle
        else:
            low = middle
    var x0 = Float64(view.hlut_x[low])
    var x1 = Float64(view.hlut_x[high])
    var t = 0.0
    if x1 != x0:
        t = (hu - x0) / (x1 - x0)
    return max2(
        0.0,
        Float64(view.hlut_y[low])
        + (Float64(view.hlut_y[high]) - Float64(view.hlut_y[low])) * t,
    )


def build_dense_hlut(
    view: ClinicalDoseProblem, max_threads: Int
) -> List[Float64]:
    var dense = List[Float64]()
    dense.resize(65536, 0.0)

    @parameter
    def fill(index: Int):
        dense[index] = hlut_lookup(view, Float64(index - 32768))

    parallelize[fill](65536, max_threads)
    return dense^


@always_inline("nodebug")
def boundary_bin[
    boundary_origin: Origin, //
](
    boundaries: UnsafePointer[Float64, boundary_origin],
    offset: UInt64,
    count: Int,
    position: Float64,
) -> Int:
    if (
        position < boundaries[Int(offset)]
        or position > boundaries[Int(offset) + count]
    ):
        return -1
    var low = 0
    var high = count
    while high - low > 1:
        var middle = (low + high) // 2
        if position < boundaries[Int(offset) + middle]:
            high = middle
        else:
            low = middle
    return low


@always_inline("nodebug")
def siddon_h2o[
    ct_origin: Origin, boundary_origin: Origin, dense_origin: Origin, //
](
    state: ClinicalDoseCTState,
    ct_data: UnsafePointer[Int16, ct_origin],
    ct_boundaries: UnsafePointer[Float64, boundary_origin],
    dense_hlut: UnsafePointer[Float64, dense_origin],
    px: Float64,
    py: Float64,
    pz: Float64,
    direction_x: Float64,
    direction_y: Float64,
    direction_z: Float64,
) -> Float64:
    var grid = state.grid.copy()
    var dlx = -direction_x
    var dly = -direction_y
    var dlz = -direction_z
    var lmax_x = 1.0e8
    var lmax_y = 1.0e8
    var lmax_z = 1.0e8
    if abs64(dlx) > 1.0e-6:
        lmax_x = max2(
            (ct_boundaries[Int(state.x_boundary_offset)] - px) / dlx,
            (ct_boundaries[Int(state.x_boundary_offset) + Int(grid.nx)] - px)
            / dlx,
        )
    if abs64(dly) > 1.0e-6:
        lmax_y = max2(
            (ct_boundaries[Int(state.y_boundary_offset)] - py) / dly,
            (ct_boundaries[Int(state.y_boundary_offset) + Int(grid.ny)] - py)
            / dly,
        )
    if abs64(dlz) > 1.0e-6:
        lmax_z = max2(
            (ct_boundaries[Int(state.z_boundary_offset)] - pz) / dlz,
            (ct_boundaries[Int(state.z_boundary_offset) + Int(grid.nz)] - pz)
            / dlz,
        )
    var lmax = min3(lmax_x, lmax_y, lmax_z)

    var first_x = 0
    var first_y = 0
    var first_z = 0
    var step_x = 0
    var step_y = 0
    var step_z = 0
    if dlx > 0.0:
        first_x = 1 + boundary_bin(
            ct_boundaries, state.x_boundary_offset, Int(grid.nx), px
        )
        step_x = 1
    elif dlx < 0.0:
        first_x = boundary_bin(
            ct_boundaries, state.x_boundary_offset, Int(grid.nx), px
        )
        step_x = -1
    if dly > 0.0:
        first_y = 1 + boundary_bin(
            ct_boundaries, state.y_boundary_offset, Int(grid.ny), py
        )
        step_y = 1
    elif dly < 0.0:
        first_y = boundary_bin(
            ct_boundaries, state.y_boundary_offset, Int(grid.ny), py
        )
        step_y = -1
    if dlz > 0.0:
        first_z = 1 + boundary_bin(
            ct_boundaries, state.z_boundary_offset, Int(grid.nz), pz
        )
        step_z = 1
    elif dlz < 0.0:
        first_z = boundary_bin(
            ct_boundaries, state.z_boundary_offset, Int(grid.nz), pz
        )
        step_z = -1

    var next_x = 1.0e8
    var next_y = 1.0e8
    var next_z = 1.0e8
    var delta_x = 0.0
    var delta_y = 0.0
    var delta_z = 0.0
    if abs64(dlx) > 1.0e-6:
        while True:
            next_x = (
                ct_boundaries[Int(state.x_boundary_offset) + first_x] - px
            ) / dlx
            first_x += step_x
            if next_x >= 1.0e-6:
                break
        delta_x = abs64(grid.dx / dlx)
    if abs64(dly) > 1.0e-6:
        while True:
            next_y = (
                ct_boundaries[Int(state.y_boundary_offset) + first_y] - py
            ) / dly
            first_y += step_y
            if next_y >= 1.0e-6:
                break
        delta_y = abs64(grid.dy / dly)
    if abs64(dlz) > 1.0e-6:
        while True:
            next_z = (
                ct_boundaries[Int(state.z_boundary_offset) + first_z] - pz
            ) / dlz
            first_z += step_z
            if next_z >= 1.0e-6:
                break
        delta_z = abs64(grid.dz / dlz)

    var middle = min3(next_x, next_y, next_z) * 0.5
    var ix = boundary_bin(
        ct_boundaries, state.x_boundary_offset, Int(grid.nx), px + middle * dlx
    )
    var iy = boundary_bin(
        ct_boundaries, state.y_boundary_offset, Int(grid.ny), py + middle * dly
    )
    var iz = boundary_bin(
        ct_boundaries, state.z_boundary_offset, Int(grid.nz), pz + middle * dlz
    )
    if ix < 0 or iy < 0 or iz < 0:
        return 0.0
    var nx = Int(grid.nx)
    var ny = Int(grid.ny)
    var current = 0.0
    var h2o = 0.0
    var limit = lmax * (1.0 - 1.0e-7)
    while current < limit:
        if (
            ix < 0
            or ix >= nx
            or iy < 0
            or iy >= ny
            or iz < 0
            or iz >= Int(grid.nz)
        ):
            break
        var raw = ct_data[Int(state.data_offset) + (iz * ny + iy) * nx + ix]
        if state.byte_swap != Int32(0):
            raw = swap_i16(raw)
        var equivalent = dense_hlut[Int(raw) + 32768]
        var segment: Float64
        if next_x < next_y and next_x < next_z:
            segment = next_x - current
            current = next_x
            next_x += delta_x
            ix += step_x
        elif next_y < next_z:
            segment = next_y - current
            current = next_y
            next_y += delta_y
            iy += step_y
        else:
            segment = next_z - current
            current = next_z
            next_z += delta_z
            iz += step_z
        h2o += segment * equivalent
    return h2o


@always_inline("nodebug")
def interpolate_ddd[
    entry_origin: Origin, //
](
    table: ClinicalDoseDDDTable,
    entries: UnsafePointer[ClinicalDoseDDDEntry, entry_origin],
    z_cm: Float64,
) -> DDDInterpolation:
    if table.entry_count == UInt32(0):
        return DDDInterpolation(False, 0.0, 0.0, 0.0, 0.0)
    var offset = Int(table.entry_offset)
    var count = Int(table.entry_count)
    if z_cm < Float64(entries[offset].z_cm) or z_cm > Float64(
        entries[offset + count - 1].z_cm
    ):
        return DDDInterpolation(False, 0.0, 0.0, 0.0, 0.0)
    var low = 0
    var high = count - 1
    while True:
        var middle = (low + high) // 2
        if middle == low:
            low = middle
            break
        if z_cm == Float64(entries[offset + middle].z_cm):
            low = middle
            break
        if z_cm < Float64(entries[offset + middle].z_cm):
            high = middle
        else:
            low = middle
    high = low + 1
    if high >= count:
        high = count - 1
    var a = entries[offset + low].copy()
    var b = entries[offset + high].copy()
    var t = 0.0
    var dz = Float64(b.z_cm) - Float64(a.z_cm)
    if dz != 0.0:
        t = (z_cm - Float64(a.z_cm)) / dz
    var dose = Float64(a.dose) + (Float64(b.dose) - Float64(a.dose)) * t
    if table.column_count <= UInt32(2):
        return DDDInterpolation(True, dose, 0.0, 0.0, 0.0)
    return DDDInterpolation(
        True,
        dose,
        Float64(a.fwhm1) + (Float64(b.fwhm1) - Float64(a.fwhm1)) * t,
        Float64(a.mix) + (Float64(b.mix) - Float64(a.mix)) * t,
        Float64(a.fwhm2) + (Float64(b.fwhm2) - Float64(a.fwhm2)) * t,
    )


@always_inline("nodebug")
def interpolate_bio[
    entry_origin: Origin, //
](
    table: ClinicalDoseBioTable,
    entries: UnsafePointer[ClinicalDoseBioEntry, entry_origin],
    z_cm: Float64,
) -> BioInterpolation:
    if table.entry_count == UInt32(0):
        return BioInterpolation(0.0, 0.0, 0.0, 0.0, 0.0)
    var offset = Int(table.entry_offset)
    var count = Int(table.entry_count)
    var low = 0
    var high = count - 1
    while high - low > 1:
        var middle = (low + high) // 2
        if z_cm < Float64(entries[offset + middle].z_cm):
            high = middle
        else:
            low = middle
    high = low + 1
    if high >= count:
        high = low
    var a = entries[offset + low].copy()
    var b = entries[offset + high].copy()
    var t = z_cm - Float64(a.z_cm)
    var dz = Float64(b.z_cm) - Float64(a.z_cm)
    if dz != 0.0:
        t /= dz
    if t < 0.0 or t > 1.0:
        return BioInterpolation(0.0, 0.0, 0.0, 0.0, 0.0)
    var w0 = 1.0 - t
    var result = BioInterpolation(
        w0 * Float64(a.alpha) + t * Float64(b.alpha),
        w0 * Float64(a.sqrt_beta) + t * Float64(b.sqrt_beta),
        w0 * Float64(a.let_mix) + t * Float64(b.let_mix),
        w0 * Float64(a.let_bar) + t * Float64(b.let_bar),
        w0 * Float64(a.let_dm_sum) + t * Float64(b.let_dm_sum),
    )
    if (
        result.alpha < 0.0
        or result.sqrt_beta < 0.0
        or result.let_mix < 0.0
        or result.let_bar < 0.0
        or result.let_dm_sum < 0.0
    ):
        return BioInterpolation(0.0, 0.0, 0.0, 0.0, 0.0)
    return result^


@always_inline("nodebug")
def compute_clinical_dose_voxel[
    dense_origin: Origin, offset_origin: Origin, run_origin: Origin, //
](
    view: ClinicalDoseProblem,
    dense_hlut: UnsafePointer[Float64, dense_origin],
    energy_run_offsets: UnsafePointer[UInt32, offset_origin],
    point_runs: UnsafePointer[ClinicalDosePointRun, run_origin],
    voxel: Int,
    state_index: Int,
    field_offset: Int,
    field_count: Int,
    px: Float64,
    py: Float64,
    pz: Float64,
) -> ClinicalDoseOutput:
    var output = ClinicalDoseOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var biological = view.biology_model == CLINICAL_DOSE_BIOLOGY_LOW_DOSE
    var voi = -1
    if biological:
        voi = Int(view.voxel_voi[voxel])
        if voi < 0 or voi >= Int(view.voi_count):
            return output^
    for field_local in range(field_offset, field_offset + field_count):
        var grid_field = view.grid_fields[field_local].copy()
        if (
            grid_field.gated != Int32(0)
            or grid_field.field_index >= view.field_count
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
            view.ct_states[state_index],
            view.ct_data,
            view.ct_boundaries,
            dense_hlut,
            px,
            py,
            pz,
            Float64(grid_field.direction_x),
            Float64(grid_field.direction_y),
            Float64(grid_field.direction_z),
        )
        h2o += Float64(grid_field.off_h2o) + Float64(grid_field.bolus)
        var field = view.fields[Int(grid_field.field_index)].copy()
        var divergence_x = 0.0
        var divergence_y = 0.0
        if view.algorithm == CLINICAL_DOSE_ALGORITHM_MSDB:
            if field.scanner_x != 0.0:
                divergence_x = gz / Float64(field.scanner_x)
            if field.scanner_y != 0.0:
                divergence_y = gz / Float64(field.scanner_y)
        for energy_local in range(Int(field.energy_count)):
            var energy = view.energies[
                Int(field.energy_offset) + energy_local
            ].copy()
            if (
                gx < Float64(energy.window_x0)
                or gx > Float64(energy.window_x1)
                or gy < Float64(energy.window_y0)
                or gy > Float64(energy.window_y1)
            ):
                continue
            var table = view.ddd_tables[Int(energy.ddd_table)].copy()
            var focus2 = Float64(energy.focus) * Float64(energy.focus)
            var bio_table = ClinicalDoseBioTable(UInt32(0), UInt32(0), 0.0)
            var bio = BioInterpolation(0.0, 0.0, 0.0, 0.0, 0.0)
            if biological:
                bio_table = view.bio_tables[
                    Int(energy.bio_table_offset) + voi
                ].copy()
            var last_z = Float64.MAX
            var depth = DDDInterpolation(False, 0.0, 0.0, 0.0, 0.0)
            var energy_index = Int(field.energy_offset) + energy_local
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
                    var point = view.points[point_index].copy()
                    if point.particles == 0.0:
                        continue
                    var dx = gx - Float64(point.x)
                    if divergence_x != 0.0:
                        dx += divergence_x * Float64(point.x)
                    var dx2 = dx * dx
                    if dx2 > Float64(point.f2_max):
                        continue
                    if dy2 > Float64(point.f2_max):
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
                        depth = interpolate_ddd(table, view.ddd_entries, z_cm)
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
                            reference_exp(-r2 * isig2_0)
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
                                reference_exp(-r2 * isig2_1)
                                * (isig2_1 / CLINICAL_DOSE_PI)
                                * depth.mix
                            )
                    if biological and lateral > 0.0 and bio.alpha == 0.0:
                        var bio_z = (
                            h2o
                            + Float64(energy.range_shifter)
                            + Float64(point.delta_z)
                        ) * Float64(bio_table.z_scale)
                        bio = interpolate_bio(
                            bio_table, view.bio_entries, bio_z
                        )
                    var weighted_fluence = lateral * point.particles
                    output.absorbed_dose += weighted_fluence * depth.dose
                    if biological:
                        output.alpha += weighted_fluence * bio.alpha
                        output.sqrt_beta += weighted_fluence * bio.sqrt_beta
                        output.let_mix += weighted_fluence * bio.let_mix
                        output.let_bar += weighted_fluence * bio.let_bar
                        output.let_dm_sum += weighted_fluence * bio.let_dm_sum
    return output^


def compute_clinical_dose(
    view: ClinicalDoseProblem,
    output: UnsafePointer[ClinicalDoseOutput, MutExternalOrigin],
) raises:
    validate_clinical_dose(view)
    comptime assert CLINICAL_DOSE_THREADS > 0
    var max_threads = Int(view.max_threads)
    var dense_hlut = build_dense_hlut(view, max_threads)
    var dense_hlut_pointer = Span(dense_hlut).unsafe_ptr()
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
    var energy_run_offsets_pointer = Span(energy_run_offsets).unsafe_ptr()
    var point_runs_pointer = Span(point_runs).unsafe_ptr()

    @parameter
    def compute_row(row: Int):
        var start = row * Int(view.dose_grid.nx)
        var end = start + Int(view.dose_grid.nx)
        for voxel in range(start, end):
            var total = ClinicalDoseOutput(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            var nx = Int(view.dose_grid.nx)
            var ny = Int(view.dose_grid.ny)
            var ix = voxel % nx
            var iy = (voxel // nx) % ny
            var iz = voxel // (nx * ny)
            var static_x = view.dose_axis_centers[Int(view.dose_x_offset) + ix]
            var static_y = view.dose_axis_centers[Int(view.dose_y_offset) + iy]
            var static_z = view.dose_axis_centers[Int(view.dose_z_offset) + iz]
            for state_index in range(Int(view.state_count)):
                var field_offset = 0
                var field_count = Int(view.field_count)
                var px = static_x
                var py = static_y
                var pz = static_z
                if view.state_count > UInt32(1):
                    var state = view.states[state_index].copy()
                    field_offset = Int(state.field_offset)
                    field_count = Int(state.field_count)
                    var position = view.transformed_voxels[
                        state_index * Int(view.grid_voxel_count) + voxel
                    ].copy()
                    px = position.x
                    py = position.y
                    pz = position.z
                var state_output = compute_clinical_dose_voxel(
                    view,
                    dense_hlut_pointer,
                    energy_run_offsets_pointer,
                    point_runs_pointer,
                    voxel,
                    state_index,
                    field_offset,
                    field_count,
                    px,
                    py,
                    pz,
                )
                total.absorbed_dose += state_output.absorbed_dose
                total.alpha += state_output.alpha
                total.sqrt_beta += state_output.sqrt_beta
                total.let_mix += state_output.let_mix
                total.let_bar += state_output.let_bar
                total.let_dm_sum += state_output.let_dm_sum
            output[voxel] = total^

    parallelize[compute_row](
        Int(view.dose_grid.nz) * Int(view.dose_grid.ny), max_threads
    )


def clinical_dose_compute_abi(
    problem_pointer: OpaquePointer[MutExternalOrigin],
    output: UnsafePointer[ClinicalDoseOutput, MutExternalOrigin],
    output_count: UInt64,
) -> Int32:
    try:
        var view = problem_pointer.bitcast[ClinicalDoseProblem]()[].copy()
        if output_count != UInt64(view.grid_voxel_count):
            return Int32(-2)
        compute_clinical_dose(view, output)
        return Int32(0)
    except error:
        print("Clinical dose CPU ABI error:", error)
        return Int32(-1)
