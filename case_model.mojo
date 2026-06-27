from ct_ray import CTVolume, read_ct_volume
from exec_parser import ExecPlan, FieldSpec, OptimizationSpec, parse_exec_file
from geometry import Vec3
from hlut import HLUT, read_hlut
from voi import VOI, VOISet, load_voi_set_from_binfo, round_to_int


@fieldwise_init
struct NativeField(Copyable, Movable):
    var number: Int
    var gantry_degrees: Float64
    var couch_degrees: Float64
    var raster_x_mm: Float64
    var raster_y_mm: Float64
    var contour_extension: Float64
    var z_step_mm: Float64
    var distal_extension_mm: Float64
    var robust_range_mm_h2o: Float64
    var robust_position_mm: Float64
    var target_voi: String
    var optimized_rst_output_path: String


@fieldwise_init
struct NativeCase(Copyable, Movable):
    var source_exec_path: String
    var ct_path: String
    var voi_binfo_path: String
    var hlut_path: String
    var ddd_path_pattern: String
    var spc_path_pattern: String
    var dedx_path: String
    var target_tissue: String
    var residual_tissue: String
    var prescription_dose_gy: Float64
    var scancap_bolus_mm_h2o: Float64
    var off_h2o_mm: Float64
    var scancap_x_limit_mm: Float64
    var scancap_y_limit_mm: Float64
    var scancap_scanner_x_mm: Float64
    var scancap_scanner_y_mm: Float64
    var scancap_min_particles: Float64
    var sis_path: String
    var target_voi_name: String
    var target_isocenter: Vec3
    var fields: List[NativeField]
    var optimization: OptimizationSpec


@fieldwise_init
struct LoadedNativeCase(Copyable, Movable):
    var model: NativeCase
    var ct: CTVolume
    var hlut: HLUT
    var vois: VOISet
    var target: VOI


def native_case_from_exec_path(path: String) raises -> NativeCase:
    return native_case_from_exec(parse_exec_file(path))


def native_case_from_exec(plan: ExecPlan) raises -> NativeCase:
    var fields = List[NativeField]()
    var target_name = ""
    for i in range(len(plan.fields)):
        fields.append(native_field_from_exec(plan.fields[i]))
        if target_name.byte_length() == 0 and plan.fields[i].target_voi.byte_length() > 0:
            target_name = plan.fields[i].target_voi
    if target_name.byte_length() == 0:
        target_name = first_target_voi_name(plan)
    if target_name.byte_length() == 0:
        raise Error(".exec does not define a target VOI")

    var vois = load_voi_set_from_binfo(voi_binfo_path_from_exec_base(plan.voi_base_path))
    var target = vois.find(target_name)
    if target.empty():
        raise Error("target VOI from .exec was not found or is empty: " + target_name)

    return NativeCase(
        plan.source_path,
        nrrd_path_from_exec_base(plan.ct_base_path),
        voi_binfo_path_from_exec_base(plan.voi_base_path),
        plan.hlut_path,
        plan.ddd_path_pattern,
        plan.spc_path_pattern,
        plan.dedx_path,
        plan.target_tissue,
        plan.residual_tissue,
        plan.prescription_dose_gy,
        plan.scancap_bolus_mm_h2o,
        plan.scancap_off_h2o_mm,
        plan.scancap_x_limit_mm,
        plan.scancap_y_limit_mm,
        plan.scancap_scanner_x_mm,
        plan.scancap_scanner_y_mm,
        plan.scancap_min_particles,
        plan.sis_path,
        target_name,
        Vec3(
            (target.source_min_world.x + target.source_max_world.x) * 0.5,
            (target.source_min_world.y + target.source_max_world.y) * 0.5,
            (target.source_min_world.z + target.source_max_world.z) * 0.5,
        ),
        fields^,
        plan.optimization.copy(),
    )


def load_native_case(model: NativeCase) raises -> LoadedNativeCase:
    var ct = read_ct_volume(model.ct_path)
    var hlut = read_hlut(model.hlut_path)
    var vois = load_voi_set_from_binfo(model.voi_binfo_path)
    var target = vois.find(model.target_voi_name)
    if target.empty():
        raise Error("target VOI is empty when loading native case")
    return LoadedNativeCase(model.copy(), ct^, hlut^, vois^, target^)


def native_field_from_exec(field: FieldSpec) -> NativeField:
    return NativeField(
        field.number,
        field.gantry_degrees,
        field.couch_degrees,
        field.raster_x_mm,
        field.raster_y_mm,
        field.contour_extension,
        field.z_step_mm,
        field.distal_extension_mm,
        field.robust_range_mm_h2o,
        field.robust_position_mm,
        field.target_voi,
        field.write_rst_path,
    )


def first_target_voi_name(plan: ExecPlan) -> String:
    for i in range(len(plan.voi_commands)):
        if plan.voi_commands[i].action == "targetset":
            return plan.voi_commands[i].name
    return ""


def trip_target_window_center(target: VOI) -> Vec3:
    var nx = round_to_int((target.source_max_world.x - target.source_min_world.x) / target.grid.voxel_size.x)
    var ny = round_to_int((target.source_max_world.y - target.source_min_world.y) / target.grid.voxel_size.y)
    var nz = round_to_int((target.source_max_world.z - target.source_min_world.z) / target.grid.voxel_size.z)
    return Vec3(
        target.source_min_world.x + 0.5 * Float64(nx) * Float64(Float32(target.grid.voxel_size.x)),
        target.source_min_world.y + 0.5 * Float64(ny) * Float64(Float32(target.grid.voxel_size.y)),
        target.source_min_world.z + 0.5 * Float64(nz) * Float64(Float32(target.grid.voxel_size.z)),
    )


def nrrd_path_from_exec_base(path: String) -> String:
    if has_suffix(path, ".nrrd"):
        return path
    return path + ".nrrd"


def voi_binfo_path_from_exec_base(path: String) -> String:
    if has_suffix(path, ".binfo"):
        return path
    return path + ".binfo"


def has_suffix(value: String, suffix: String) -> Bool:
    if suffix.byte_length() > value.byte_length():
        return False
    var offset = value.byte_length() - suffix.byte_length()
    for i in range(suffix.byte_length()):
        if value[byte=offset + i] != suffix[byte=i]:
            return False
    return True
