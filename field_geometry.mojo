from std.algorithm import parallelize

from case_model import NativeField
from ct_ray import CTVolume, ct_grid_intersection_table, ray_water_depth_mm, vintpol1d
from geometry import Vec3, gantry_to_patient_point, patient_to_gantry_matrix, transform_point
from hlut import HLUT
from raster_grid import RasterGrid2D, raster_grid_from_window
from voi import VOI


@fieldwise_init
struct Bounds3f(Copyable, Movable):
    var min_x: Float64
    var max_x: Float64
    var min_y: Float64
    var max_y: Float64
    var min_z: Float64
    var max_z: Float64


def empty_bounds3f() -> Bounds3f:
    return Bounds3f(1.0e300, -1.0e300, 1.0e300, -1.0e300, 1.0e300, -1.0e300)


def include_point(mut bounds: Bounds3f, point: Vec3):
    if point.x < bounds.min_x:
        bounds.min_x = point.x
    if point.x > bounds.max_x:
        bounds.max_x = point.x
    if point.y < bounds.min_y:
        bounds.min_y = point.y
    if point.y > bounds.max_y:
        bounds.max_y = point.y
    if point.z < bounds.min_z:
        bounds.min_z = point.z
    if point.z > bounds.max_z:
        bounds.max_z = point.z


def target_voi_bev_center_bounds(field: NativeField, target: VOI, isocenter: Vec3) raises -> Bounds3f:
    if target.empty():
        raise Error("cannot compute BEV bounds for an empty target VOI")
    var patient_to_gantry = patient_to_gantry_matrix(isocenter, field.gantry_degrees, field.couch_degrees)
    var bounds = empty_bounds3f()
    for n in range(len(target.active_indices)):
        var index = target.grid.index_triplet(target.active_indices[n])
        var world = target.grid.world_at_index(Int(index.x), Int(index.y), Int(index.z))
        include_point(bounds, transform_point(patient_to_gantry, world))
    return bounds^


def target_voi_bev_binary_wdw_bounds(field: NativeField, target: VOI, isocenter: Vec3) raises -> Bounds3f:
    if target.empty():
        raise Error("cannot compute BEV bounds for an empty target VOI")
    var patient_to_gantry = patient_to_gantry_matrix(isocenter, field.gantry_degrees, field.couch_degrees)
    var bounds = empty_bounds3f()
    for n in range(len(target.active_indices)):
        var index = target.grid.index_triplet(target.active_indices[n])
        var i = Int(index.x)
        var j = Int(index.y)
        var k = Int(index.z)
        include_point(bounds, transform_point(patient_to_gantry, target.grid.world_at_index(i, j, k)))
        include_point(bounds, transform_point(patient_to_gantry, target.grid.world_at_index(i + 1, j + 1, k + 1)))
    return bounds^


def target_voi_bev_corner_bounds(field: NativeField, target: VOI, isocenter: Vec3) raises -> Bounds3f:
    if target.empty():
        raise Error("cannot compute BEV bounds for an empty target VOI")
    var patient_to_gantry = patient_to_gantry_matrix(isocenter, field.gantry_degrees, field.couch_degrees)
    var bounds = empty_bounds3f()
    var half_i = target.grid.voxel_size.x * 0.5
    var half_j = target.grid.voxel_size.y * 0.5
    var half_k = target.grid.voxel_size.z * 0.5
    for n in range(len(target.active_indices)):
        var index = target.grid.index_triplet(target.active_indices[n])
        var center = target.grid.world_at_index(Int(index.x), Int(index.y), Int(index.z))
        for si in range(2):
            var dx = half_i
            if si == 0:
                dx = -half_i
            for sj in range(2):
                var dy = half_j
                if sj == 0:
                    dy = -half_j
                for sk in range(2):
                    var dz = half_k
                    if sk == 0:
                        dz = -half_k
                    include_point(bounds, transform_point(patient_to_gantry, Vec3(center.x + dx, center.y + dy, center.z + dz)))
    return bounds^


def target_voi_bev_box_bounds(field: NativeField, target: VOI, isocenter: Vec3) raises -> Bounds3f:
    if target.empty():
        raise Error("cannot compute BEV bounds for an empty target VOI")
    var patient_to_gantry = patient_to_gantry_matrix(isocenter, field.gantry_degrees, field.couch_degrees)
    var bounds = empty_bounds3f()
    for si in range(2):
        var x = target.source_min_world.x
        if si == 1:
            x = target.source_max_world.x
        for sj in range(2):
            var y = target.source_min_world.y
            if sj == 1:
                y = target.source_max_world.y
            for sk in range(2):
                var z = target.source_min_world.z
                if sk == 1:
                    z = target.source_max_world.z
                include_point(bounds, transform_point(patient_to_gantry, Vec3(x, y, z)))
    return bounds^


def contour_extension_mm(field: NativeField) -> Float64:
    # TRPFldTargetVoiH2OWdw: dExtension = 3.0 * dvSteps[0] * dContourExtension
    return 3.0 * field.raster_x_mm * field.contour_extension


def with_lateral_contour_extension(bounds: Bounds3f, field: NativeField) -> Bounds3f:
    var ext = contour_extension_mm(field)
    return Bounds3f(
        bounds.min_x - ext,
        bounds.max_x + ext,
        bounds.min_y - ext,
        bounds.max_y + ext,
        bounds.min_z,
        bounds.max_z,
    )


def with_distal_extension(bounds: Bounds3f, field: NativeField) -> Bounds3f:
    return Bounds3f(
        bounds.min_x,
        bounds.max_x,
        bounds.min_y,
        bounds.max_y,
        bounds.min_z - field.distal_extension_mm,
        bounds.max_z + field.distal_extension_mm,
    )


def target_voi_bev_raster_window(field: NativeField, target: VOI, isocenter: Vec3) raises -> Bounds3f:
    var bounds = target_voi_bev_corner_bounds(field, target, isocenter)
    bounds = with_lateral_contour_extension(bounds, field)
    bounds = with_distal_extension(bounds, field)
    return bounds^


def target_voi_bev_h2o_window(
    field: NativeField,
    target: VOI,
    isocenter: Vec3,
    ct: CTVolume,
    hlut: HLUT,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
) raises -> Bounds3f:
    var bounds = target_voi_bev_binary_wdw_bounds(field, target, isocenter)
    var range_min = 1.0e300
    var range_max = -1.0e300
    var setup_bounds = with_distal_extension(bounds, field)
    var setup_grid = raster_grid_from_window(setup_bounds, field.raster_x_mm, field.raster_y_mm)
    include_h2o_range_for_setup(field, target, isocenter, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, 0.0, range_min, range_max)
    if field.robust_position_mm > 0.0:
        var shifted = Vec3(isocenter.x + field.robust_position_mm, isocenter.y, isocenter.z)
        include_h2o_range_for_setup(field, target, shifted, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, 0.0, range_min, range_max)
        shifted = Vec3(isocenter.x - field.robust_position_mm, isocenter.y, isocenter.z)
        include_h2o_range_for_setup(field, target, shifted, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, 0.0, range_min, range_max)
        shifted = Vec3(isocenter.x, isocenter.y + field.robust_position_mm, isocenter.z)
        include_h2o_range_for_setup(field, target, shifted, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, 0.0, range_min, range_max)
        shifted = Vec3(isocenter.x, isocenter.y - field.robust_position_mm, isocenter.z)
        include_h2o_range_for_setup(field, target, shifted, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, 0.0, range_min, range_max)
        shifted = Vec3(isocenter.x, isocenter.y, isocenter.z + field.robust_position_mm)
        include_h2o_range_for_setup(field, target, shifted, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, 0.0, range_min, range_max)
        shifted = Vec3(isocenter.x, isocenter.y, isocenter.z - field.robust_position_mm)
        include_h2o_range_for_setup(field, target, shifted, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, 0.0, range_min, range_max)
    if field.robust_range_mm_h2o > 0.0:
        include_h2o_range_for_setup(field, target, isocenter, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, field.robust_range_mm_h2o, range_min, range_max)
        include_h2o_range_for_setup(field, target, isocenter, ct, hlut, bolus_mm_h2o, off_h2o_mm, setup_grid, setup_bounds, -field.robust_range_mm_h2o, range_min, range_max)
    bounds = with_robust_position_expansion(bounds, field, isocenter)
    bounds.min_z = range_min - field.distal_extension_mm
    bounds.max_z = range_max + field.distal_extension_mm
    return bounds^


def gantry_to_patient_from_matrix(field: NativeField, isocenter: Vec3, point: Vec3) -> Vec3:
    return gantry_to_patient_point(isocenter, field.gantry_degrees, field.couch_degrees, point)


def point_inside_source_window(target: VOI, point: Vec3) -> Bool:
    return point.x >= target.source_min_world.x and point.x <= target.source_max_world.x and point.y >= target.source_min_world.y and point.y <= target.source_max_world.y and point.z >= target.source_min_world.z and point.z <= target.source_max_world.z


def point_inside_target_window(target: VOI, point: Vec3) -> Bool:
    var min_world = target.grid.world_at_index(target.bounds.min_i + 1, target.bounds.min_j + 1, target.bounds.min_k + 1)
    var max_world = target.grid.world_at_index(target.bounds.max_i, target.bounds.max_j, target.bounds.max_k)
    return point.x >= min_world.x and point.x <= max_world.x and point.y >= min_world.y and point.y <= max_world.y and point.z >= min_world.z and point.z <= max_world.z


def point_inside_target_mask(target: VOI, point: Vec3) -> Bool:
    return target.contains_point_vbind(point)


def include_h2o_range_for_setup(
    field: NativeField,
    target: VOI,
    isocenter: Vec3,
    ct: CTVolume,
    hlut: HLUT,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    grid: RasterGrid2D,
    bounds: Bounds3f,
    hlut_percent_shift: Float64,
    mut range_min: Float64,
    mut range_max: Float64,
):
    var scale = 1.0 + hlut_percent_shift * 0.01
    var point_min = List[Float64]()
    point_min.resize(len(grid.points), 1.0e300)
    var point_max = List[Float64]()
    point_max.resize(len(grid.points), -1.0e300)
    var num_workers = 12

    @parameter
    def scan_grid_point(p: Int):
        var local_min = 1.0e300
        var local_max = -1.0e300
        var ray_origin = gantry_to_patient_from_matrix(field, isocenter, Vec3(grid.points[p].x, grid.points[p].y, 0.0))
        var ray_next = gantry_to_patient_from_matrix(field, isocenter, Vec3(grid.points[p].x, grid.points[p].y, 1.0))
        var ray_dir = Vec3(ray_origin.x - ray_next.x, ray_origin.y - ray_next.y, ray_origin.z - ray_next.z)
        var table = ct_grid_intersection_table(ct, hlut, ray_origin, ray_dir, bolus_mm_h2o + off_h2o_mm, scale)
        if len(table.z) < 2:
            return
        var z_min = table.z[0]
        var z_max = table.z[len(table.z) - 1]
        if z_max < z_min:
            var tmp = z_min
            z_min = z_max
            z_max = tmp
        var z = z_min
        while z <= z_max:
            var patient = gantry_to_patient_from_matrix(field, isocenter, Vec3(grid.points[p].x, grid.points[p].y, z))
            if not point_inside_target_mask(target, patient):
                z += 0.2
                continue
            var h2o = vintpol1d(z, table.z, table.h2o)
            if h2o < local_min:
                local_min = h2o
            if h2o > local_max:
                local_max = h2o
            z += 0.2
        point_min[p] = local_min
        point_max[p] = local_max

    parallelize[scan_grid_point](len(grid.points), num_workers)

    for p in range(len(grid.points)):
        if point_min[p] < range_min:
            range_min = point_min[p]
        if point_max[p] > range_max:
            range_max = point_max[p]


def gantry_z_ct_interval(
    field: NativeField,
    isocenter: Vec3,
    ct: CTVolume,
    x: Float64,
    y: Float64,
    mut z_min: Float64,
    mut z_max: Float64,
) -> Bool:
    var p0 = gantry_to_patient_from_matrix(field, isocenter, Vec3(x, y, 0.0))
    var p1 = gantry_to_patient_from_matrix(field, isocenter, Vec3(x, y, 1.0))
    var d = Vec3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)
    z_min = -1.0e100
    z_max = 1.0e100
    if not clip_param_axis(p0.x, d.x, ct.header.origin.x, ct.header.origin.x + Float64(ct.header.size_i) * ct.header.directions[0, 0], z_min, z_max):
        return False
    if not clip_param_axis(p0.y, d.y, ct.header.origin.y, ct.header.origin.y + Float64(ct.header.size_j) * ct.header.directions[1, 1], z_min, z_max):
        return False
    if not clip_param_axis(p0.z, d.z, ct.header.origin.z, ct.header.origin.z + Float64(ct.header.size_k) * ct.header.directions[2, 2], z_min, z_max):
        return False
    return z_min <= z_max


def clip_param_axis(p: Float64, d: Float64, a: Float64, b: Float64, mut t0: Float64, mut t1: Float64) -> Bool:
    var lo = a
    var hi = b
    if hi < lo:
        lo = b
        hi = a
    if d == 0.0:
        return p >= lo and p <= hi
    var ta = (lo - p) / d
    var tb = (hi - p) / d
    if tb < ta:
        var tmp = ta
        ta = tb
        tb = tmp
    if ta > t0:
        t0 = ta
    if tb < t1:
        t1 = tb
    return t0 <= t1


def include_h2o_range_for_target_outline(
    field: NativeField,
    target: VOI,
    isocenter: Vec3,
    ct: CTVolume,
    hlut: HLUT,
    mut range_min: Float64,
    mut range_max: Float64,
):
    var origin0 = gantry_to_patient_from_matrix(field, isocenter, Vec3(0.0, 0.0, 0.0))
    var origin1 = gantry_to_patient_from_matrix(field, isocenter, Vec3(0.0, 0.0, 1.0))
    var direction = Vec3(origin1.x - origin0.x, origin1.y - origin0.y, origin1.z - origin0.z)
    for n in range(len(target.active_indices)):
        var idx = target.grid.index_triplet(target.active_indices[n])
        var i = Int(idx.x)
        var j = Int(idx.y)
        var k = Int(idx.z)
        if not is_outline_voxel(target, i, j, k):
            continue
        include_outline_corner_h2o(target, ct, hlut, direction, i, j, k, range_min, range_max)
        include_outline_corner_h2o(target, ct, hlut, direction, i + 1, j + 1, k + 1, range_min, range_max)


def include_outline_corner_h2o(
    target: VOI,
    ct: CTVolume,
    hlut: HLUT,
    direction: Vec3,
    i: Int,
    j: Int,
    k: Int,
    mut range_min: Float64,
    mut range_max: Float64,
):
    var point = target.grid.world_at_index(i, j, k)
    var water = ray_water_depth_mm(ct, hlut, point, direction)
    if water < range_min:
        range_min = water
    if water > range_max:
        range_max = water


def is_outline_voxel(target: VOI, i: Int, j: Int, k: Int) -> Bool:
    if i == target.bounds.min_i or i == target.bounds.max_i:
        return True
    if j == target.bounds.min_j or j == target.bounds.max_j:
        return True
    if k == target.bounds.min_k or k == target.bounds.max_k:
        return True
    if not target.contains_index(i - 1, j, k):
        return True
    if not target.contains_index(i + 1, j, k):
        return True
    if not target.contains_index(i, j - 1, k):
        return True
    if not target.contains_index(i, j + 1, k):
        return True
    if not target.contains_index(i, j, k - 1):
        return True
    if not target.contains_index(i, j, k + 1):
        return True
    return False


def shifted_point(point: Vec3, axis: Int, shift: Float64) -> Vec3:
    if axis == 0:
        return Vec3(point.x + shift, point.y, point.z)
    if axis == 1:
        return Vec3(point.x, point.y + shift, point.z)
    return Vec3(point.x, point.y, point.z + shift)


def with_robust_position_expansion(bounds: Bounds3f, field: NativeField, isocenter: Vec3) -> Bounds3f:
    if field.robust_position_mm <= 0.0:
        return bounds.copy()
    var nominal_matrix = patient_to_gantry_matrix(isocenter, field.gantry_degrees, field.couch_degrees)
    var centre = transform_point(nominal_matrix, isocenter)
    var min_dx = 0.0
    var max_dx = 0.0
    var min_dy = 0.0
    var max_dy = 0.0
    for axis in range(3):
        for sign_index in range(2):
            var shift = field.robust_position_mm
            if sign_index == 0:
                shift = -shift
            var shifted_isocenter = shifted_point(isocenter, axis, shift)
            var shifted_matrix = patient_to_gantry_matrix(shifted_isocenter, field.gantry_degrees, field.couch_degrees)
            var shifted = transform_point(shifted_matrix, isocenter)
            var dx = shifted.x - centre.x
            var dy = shifted.y - centre.y
            if dx < min_dx:
                min_dx = dx
            if dx > max_dx:
                max_dx = dx
            if dy < min_dy:
                min_dy = dy
            if dy > max_dy:
                max_dy = dy
    return Bounds3f(bounds.min_x + min_dx, bounds.max_x + max_dx, bounds.min_y + min_dy, bounds.max_y + max_dy, bounds.min_z, bounds.max_z)
