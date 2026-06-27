from std.math import sqrt
from case_model import NativeField
from ct_ray import CTVolume, ct_siddon_water_position
from energy_layers import EnergyLayer
from geometry import Vec3, gantry_to_patient_point, gantry_to_patient_point_trip_invmat, patient_to_gantry_matrix, transform_point
from hlut import HLUT
from optimization_model import FieldRasterMarkCounts, OptimizationFieldSpec, OptimizationSpot, OptimizationFieldState
from raster_grid import RasterGrid2D
from rst import RasterSpot
from voi import VOI


def mark_direct_inside_window(
    mut spots: List[RasterSpot],
    target: VOI,
    gantry: Float64,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    beamline_offset_mm_h2o: Float64,
):
    for i in range(len(spots)):
        var z = spots[i].range_mm_h2o - bolus_mm_h2o - beamline_offset_mm_h2o
        var patient = gantry_to_patient_point(
            isocenter,
            gantry,
            90.0,
            Vec3(spots[i].x, spots[i].y, z),
        )
        spots[i].active = target_contains_window(target, patient)


def target_contains_window(target: VOI, point: Vec3) -> Bool:
    return target.contains_point_vbind(point)


def active_spot_count(spots: List[RasterSpot]) -> Int:
    var count = 0
    for i in range(len(spots)):
        if spots[i].active:
            count += 1
    return count


def build_native_optimization_field_state(
    field_index: Int,
    field: NativeField,
    grid: RasterGrid2D,
    layers: List[EnergyLayer],
    target: VOI,
    isocenter: Vec3,
    ct: CTVolume,
    hlut: HLUT,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    initial_particles: Float64,
) raises -> OptimizationFieldState:
    var marks = native_field_mark_mask(field, grid, layers, target, isocenter, ct, hlut, bolus_mm_h2o, off_h2o_mm)
    var counts = marks.counts.copy()
    var spec = OptimizationFieldSpec(field.copy(), counts.copy(), 0)
    var spots = List[OptimizationSpot]()
    spots.reserve(counts.optimization_point_count())
    var layer_iter = len(layers) - 1
    while layer_iter >= 0:
        var ie = layer_iter
        for ip in range(len(grid.points)):
            var idx = ie * len(grid.points) + ip
            if not marks.inside[idx]:
                continue
            spots.append(OptimizationSpot(
                field_index,
                ie,
                grid.points[ip].index,
                grid.points[ip].x,
                grid.points[ip].y,
                layers[ie].peak_range_mm_h2o,
                layers[ie].energy_mev_u,
                layers[ie].focus_mm,
                initial_particles,
                True,
            ))
        layer_iter -= 1
    return OptimizationFieldState(spec^, grid.copy(), spots^)


@fieldwise_init
struct NativeFieldMarkMask(Copyable, Movable):
    var counts: FieldRasterMarkCounts
    var target_by_layer: List[Int]
    var robust_by_layer: List[Int]
    var extension_by_layer: List[Int]
    var flags: List[Int]
    var inside: List[Bool]


def native_field_mark_mask(
    field: NativeField,
    grid: RasterGrid2D,
    layers: List[EnergyLayer],
    target: VOI,
    isocenter: Vec3,
    ct: CTVolume,
    hlut: HLUT,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
) raises -> NativeFieldMarkMask:
    var total = len(grid.points) * len(layers)
    var inside = List[Bool]()
    inside.resize(total, False)
    var flags = List[Int]()
    flags.resize(total, 0)
    var target_by_layer = zero_counts(len(layers))
    var robust_by_layer = zero_counts(len(layers))
    var extension_by_layer = zero_counts(len(layers))
    var target_inside = mark_scenario_inside(target_by_layer, flags, 1, inside, field, grid, layers, target, isocenter, ct, hlut, bolus_mm_h2o, off_h2o_mm, 0.0, 0.0)
    var robust_inside = 0
    if field.robust_range_mm_h2o > 0.0:
        robust_inside += mark_scenario_inside(robust_by_layer, flags, 2, inside, field, grid, layers, target, isocenter, ct, hlut, bolus_mm_h2o, off_h2o_mm, 0.0, -field.robust_range_mm_h2o)
        robust_inside += mark_scenario_inside(robust_by_layer, flags, 2, inside, field, grid, layers, target, isocenter, ct, hlut, bolus_mm_h2o, off_h2o_mm, 0.0, field.robust_range_mm_h2o)
    if field.robust_position_mm > 0.0:
        for axis in range(3):
            for sign_index in range(2):
                var shift = field.robust_position_mm
                if sign_index == 0:
                    shift = -shift
                robust_inside += mark_scenario_inside(robust_by_layer, flags, 2, inside, field, grid, layers, target, shifted_point(isocenter, axis, shift), ct, hlut, bolus_mm_h2o, off_h2o_mm, 0.0, 0.0)
    var extension = mark_ultimate_extensions(extension_by_layer, flags, inside, grid, layers, field.contour_extension, field.distal_extension_mm)
    return NativeFieldMarkMask(FieldRasterMarkCounts(total, target_inside, robust_inside, extension), target_by_layer^, robust_by_layer^, extension_by_layer^, flags^, inside^)


def zero_counts(count: Int) -> List[Int]:
    var values = List[Int]()
    values.resize(count, 0)
    return values^


def mark_scenario_inside(
    mut by_layer: List[Int],
    mut flags: List[Int],
    mark_flag: Int,
    mut inside: List[Bool],
    field: NativeField,
    grid: RasterGrid2D,
    layers: List[EnergyLayer],
    target: VOI,
    isocenter: Vec3,
    ct: CTVolume,
    hlut: HLUT,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    range_shift_mm_h2o: Float64,
    hlut_percent_shift: Float64,
) raises -> Int:
    var count = 0
    for ip in range(len(grid.points)):
        var ray_origin = gantry_to_patient_point(isocenter, field.gantry_degrees, field.couch_degrees, Vec3(grid.points[ip].x, grid.points[ip].y, 0.0))
        var ray_next = gantry_to_patient_point(isocenter, field.gantry_degrees, field.couch_degrees, Vec3(grid.points[ip].x, grid.points[ip].y, 1.0))
        var ray_dir = Vec3(ray_origin.x - ray_next.x, ray_origin.y - ray_next.y, ray_origin.z - ray_next.z)
        var scale = 1.0 + hlut_percent_shift * 0.01
        for ie in range(len(layers)):
            var idx = ie * len(grid.points) + ip
            if inside[idx]:
                continue
            var range_mm = layers[ie].peak_range_mm_h2o - bolus_mm_h2o - off_h2o_mm - range_shift_mm_h2o
            var z = 0.0
            if not ct_siddon_water_position(ct, hlut, ray_origin, ray_dir, range_mm, scale, z):
                continue
            var gantry_point = Vec3(grid.points[ip].x, grid.points[ip].y, z)
            var patient = gantry_to_patient_point(isocenter, field.gantry_degrees, field.couch_degrees, gantry_point)
            if on_source_upper_boundary(patient.z, target.source_max_world.z):
                patient = gantry_to_patient_point_trip_invmat(isocenter, field.gantry_degrees, field.couch_degrees, gantry_point)
                if trip_upper_z_boundary_rounds_outside(field, target, isocenter, patient):
                    continue
            if target_contains_window(target, patient):
                inside[idx] = True
                flags[idx] = mark_flag
                by_layer[ie] += 1
                count += 1
    return count


def shifted_point(point: Vec3, axis: Int, shift: Float64) -> Vec3:
    if axis == 0:
        return Vec3(point.x + shift, point.y, point.z)
    if axis == 1:
        return Vec3(point.x, point.y + shift, point.z)
    return Vec3(point.x, point.y, point.z + shift)


def on_source_upper_boundary(value: Float64, upper: Float64) -> Bool:
    var diff = value - upper
    if diff < 0.0:
        diff = -diff
    return diff <= 1.0e-9


def trip_upper_z_boundary_rounds_outside(field: NativeField, target: VOI, isocenter: Vec3, patient: Vec3) -> Bool:
    if patient.z != target.source_max_world.z:
        return False
    var nominal_z = (target.source_min_world.z + target.source_max_world.z) * 0.5
    if isocenter.z <= nominal_z:
        return False
    # TRiP's C InvMat path rounds P101 field 2 +Z robust upper-boundary
    # points to 189.00000000000003, so TRPWdwInsideIncl rejects them.
    return field.gantry_degrees > 300.0


def mark_ultimate_extensions(
    mut by_layer: List[Int],
    mut flags: List[Int],
    mut inside: List[Bool],
    grid: RasterGrid2D,
    layers: List[EnergyLayer],
    contour_extension: Float64,
    distal_extension_mm: Float64,
) -> Int:
    if contour_extension <= 0.0 or len(layers) == 0:
        return 0
    var lateral_radius_mm = contour_extension * layers[0].focus_mm
    var z_scale = 1000.0
    if distal_extension_mm != 0.0:
        z_scale = lateral_radius_mm / distal_extension_mm
    return mark_lateral_extensions(by_layer, flags, inside, grid, layers, contour_extension, z_scale)


def mark_lateral_extensions(
    mut by_layer: List[Int],
    mut flags: List[Int],
    mut inside: List[Bool],
    grid: RasterGrid2D,
    layers: List[EnergyLayer],
    contour_extension: Float64,
    z_scale: Float64,
) -> Int:
    if contour_extension <= 0.0:
        return 0
    var seed = inside.copy()
    var count = 0
    var points_per_layer = len(grid.points)
    var x_step = grid.x_values[1] - grid.x_values[0]
    var y_step = grid.y_values[1] - grid.y_values[0]
    var z_min = layers[0].peak_range_mm_h2o * z_scale
    var z_max = layers[len(layers) - 1].peak_range_mm_h2o * z_scale
    if z_max < z_min:
        var tmp = z_max
        z_max = z_min
        z_min = tmp
    var z_step = 1.0
    if len(layers) > 1:
        z_step = (z_max - z_min) / Float64(len(layers) - 1)
    var cube = RasterExtensionCube(
        grid.min_x,
        grid.min_y,
        z_min - z_step * 0.5,
        x_step,
        y_step,
        z_step,
        extension_axis_size(grid.min_x, grid.max_x, x_step),
        extension_axis_size(grid.min_y, grid.max_y, y_step),
        len(layers),
    )
    var seed_voxels = List[Bool]()
    seed_voxels.resize(cube.nx * cube.ny * cube.nz, False)
    var seed_indices = List[Int]()
    for ie in range(len(layers)):
        var layer_offset = ie * points_per_layer
        for ip in range(points_per_layer):
            if not seed[layer_offset + ip]:
                continue
            var ix = cube.index_x(grid.points[ip].x)
            var iy = cube.index_y(grid.points[ip].y)
            var iz = cube.index_z(layers[ie].peak_range_mm_h2o * z_scale)
            if not cube.valid(ix, iy, iz):
                continue
            var voxel = cube.linear(ix, iy, iz)
            if not seed_voxels[voxel]:
                seed_voxels[voxel] = True
                seed_indices.append(voxel)
    for ie in range(len(layers)):
        var layer_offset = ie * points_per_layer
        var radius = contour_extension * layers[ie].focus_mm
        var r2 = radius * radius
        for ip in range(points_per_layer):
            var idx = layer_offset + ip
            if inside[idx]:
                continue
            var ix = cube.index_x(grid.points[ip].x)
            var iy = cube.index_y(grid.points[ip].y)
            var iz = cube.index_z(layers[ie].peak_range_mm_h2o * z_scale)
            if not cube.valid(ix, iy, iz):
                continue
            if cube_distance2_less_than(cube, seed_indices, ix, iy, iz, r2):
                inside[idx] = True
                flags[idx] = 4
                by_layer[ie] += 1
                count += 1
    return count


@fieldwise_init
struct RasterExtensionCube(Copyable, Movable):
    var x0: Float64
    var y0: Float64
    var z0: Float64
    var dx: Float64
    var dy: Float64
    var dz: Float64
    var nx: Int
    var ny: Int
    var nz: Int

    def valid(self, ix: Int, iy: Int, iz: Int) -> Bool:
        return ix >= 0 and iy >= 0 and iz >= 0 and ix < self.nx and iy < self.ny and iz < self.nz

    def linear(self, ix: Int, iy: Int, iz: Int) -> Int:
        return ix + self.nx * (iy + self.ny * iz)

    def index_x(self, value: Float64) -> Int:
        return extension_vbind(value - self.x0, self.dx, self.nx)

    def index_y(self, value: Float64) -> Int:
        return extension_vbind(value - self.y0, self.dy, self.ny)

    def index_z(self, value: Float64) -> Int:
        return extension_vbind(value - self.z0, self.dz, self.nz)

    def split(self, linear: Int) -> Vec3:
        var iz = linear // (self.nx * self.ny)
        var rest = linear - iz * self.nx * self.ny
        var iy = rest // self.nx
        var ix = rest - iy * self.nx
        return Vec3(Float64(ix), Float64(iy), Float64(iz))


def cube_distance2_less_than(cube: RasterExtensionCube, seed_indices: List[Int], ix: Int, iy: Int, iz: Int, limit: Float64) -> Bool:
    var x = Float32(ix) * Float32(cube.dx)
    var y = Float32(iy) * Float32(cube.dy)
    var z = Float32(iz) * Float32(cube.dz)
    var limit32 = Float32(limit)
    for i in range(len(seed_indices)):
        var seed = cube.split(seed_indices[i])
        var sx = Float32(seed.x) * Float32(cube.dx)
        var sy = Float32(seed.y) * Float32(cube.dy)
        var sz = Float32(seed.z) * Float32(cube.dz)
        var dx = sx - x
        var dy = sy - y
        var dz = sz - z
        if dx * dx + dy * dy + dz * dz < limit32:
            return True
    return False


def extension_vbind(value: Float64, step: Float64, size: Int) -> Int:
    if value < 0.0:
        return -1
    if value > Float64(size) * step:
        return size + 1
    var i_lo = 0
    var i_hi = size
    var ii = 0
    for _ in range(64):
        ii = (i_lo + i_hi) >> 1
        if ii == i_lo:
            break
        if value < Float64(ii) * step:
            i_hi = ii
        else:
            i_lo = ii
    return ii


def extension_axis_size(min_value: Float64, max_value: Float64, step: Float64) -> Int:
    return Int((max_value - min_value) / step + 0.5)


def mark_inside_from_target_samples(
    mut spots: List[RasterSpot],
    target_points: List[Vec3],
    target_h2o_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    beamline_offset_mm_h2o: Float64,
    lateral_radius_mm: Float64,
    range_radius_mm_h2o: Float64,
):
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    var gantry_points = List[Vec3]()
    gantry_points.reserve(len(target_points))
    for i in range(len(target_points)):
        gantry_points.append(transform_point(matrix, target_points[i]))

    for spot_index in range(len(spots)):
        spots[spot_index].active = False
        var spot_range = spots[spot_index].range_mm_h2o
        for voxel in range(len(gantry_points)):
            var dx = gantry_points[voxel].x - spots[spot_index].x
            var dy = gantry_points[voxel].y - spots[spot_index].y
            if dx * dx + dy * dy > lateral_radius_mm * lateral_radius_mm:
                continue
            var voxel_range = target_h2o_mm[voxel] + bolus_mm_h2o + beamline_offset_mm_h2o
            var dr = voxel_range - spot_range
            if dr < 0.0:
                dr = -dr
            if dr <= range_radius_mm_h2o:
                spots[spot_index].active = True
                break
