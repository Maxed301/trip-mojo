from std.math import sqrt

from geometry import Vec3
from hlut import HLUT, hlut_path_factor
from nrrd import NrrdHeader, payload_i16, read_nrrd_header, read_nrrd_payload


@fieldwise_init
struct CTVolume(Copyable, Movable):
    var header: NrrdHeader
    var payload: List[UInt8]


@fieldwise_init
struct CTIntersectionTable(Copyable, Movable):
    var z: List[Float64]
    var h2o: List[Float64]


def read_ct_volume(path: String) raises -> CTVolume:
    var header = read_nrrd_header(path)
    if header.dtype != "short":
        raise Error("CT volume must be short")
    var payload = read_nrrd_payload(header)
    if len(payload) != header.size_i * header.size_j * header.size_k * 2:
        raise Error("CT payload size mismatch")
    return CTVolume(header^, payload^)


def ct_value_at(volume: CTVolume, i: Int, j: Int, k: Int) -> Int:
    return payload_i16(volume.payload, i + volume.header.size_i * (j + volume.header.size_j * k))


def ray_water_depth_mm(volume: CTVolume, hlut: HLUT, origin: Vec3, direction: Vec3) -> Float64:
    return ray_water_depth_scaled_mm(volume, hlut, origin, direction, 1.0)


def ray_water_depth_scaled_mm(volume: CTVolume, hlut: HLUT, origin: Vec3, direction: Vec3, hlut_scale: Float64) -> Float64:
    var dir = normalize(direction)
    var t0 = -1.0e100
    var t1 = 1.0e100
    if not clip_axis(origin.x, dir.x, volume.header.origin.x, volume.header.origin.x + Float64(volume.header.size_i) * volume.header.directions[0, 0], t0, t1):
        return 0.0
    if not clip_axis(origin.y, dir.y, volume.header.origin.y, volume.header.origin.y + Float64(volume.header.size_j) * volume.header.directions[1, 1], t0, t1):
        return 0.0
    if not clip_axis(origin.z, dir.z, volume.header.origin.z, volume.header.origin.z + Float64(volume.header.size_k) * volume.header.directions[2, 2], t0, t1):
        return 0.0
    if t1 <= t0:
        return 0.0
    if t0 < 0.0:
        t0 = 0.0

    var t = t0
    var water = 0.0
    while t < t1:
        var p = add_scaled(origin, dir, t + 1.0e-7)
        var i = clamp_int(Int((p.x - volume.header.origin.x) / volume.header.directions[0, 0]), 0, volume.header.size_i - 1)
        var j = clamp_int(Int((p.y - volume.header.origin.y) / volume.header.directions[1, 1]), 0, volume.header.size_j - 1)
        var k = clamp_int(Int((p.z - volume.header.origin.z) / volume.header.directions[2, 2]), 0, volume.header.size_k - 1)
        var next_t = t1
        if dir.x > 0.0:
            next_t = min_float(next_t, (volume.header.origin.x + Float64(i + 1) * volume.header.directions[0, 0] - origin.x) / dir.x)
        elif dir.x < 0.0:
            next_t = min_float(next_t, (volume.header.origin.x + Float64(i) * volume.header.directions[0, 0] - origin.x) / dir.x)
        if dir.y > 0.0:
            next_t = min_float(next_t, (volume.header.origin.y + Float64(j + 1) * volume.header.directions[1, 1] - origin.y) / dir.y)
        elif dir.y < 0.0:
            next_t = min_float(next_t, (volume.header.origin.y + Float64(j) * volume.header.directions[1, 1] - origin.y) / dir.y)
        if dir.z > 0.0:
            next_t = min_float(next_t, (volume.header.origin.z + Float64(k + 1) * volume.header.directions[2, 2] - origin.z) / dir.z)
        elif dir.z < 0.0:
            next_t = min_float(next_t, (volume.header.origin.z + Float64(k) * volume.header.directions[2, 2] - origin.z) / dir.z)
        if next_t <= t:
            next_t = t + 1.0e-6
        if next_t > t1:
            next_t = t1
        water += (next_t - t) * hlut_path_factor(hlut, Float64(ct_value_at(volume, i, j, k))) * hlut_scale
        t = next_t
    return water


def grid_line_water_depth_mm(volume: CTVolume, hlut: HLUT, origin: Vec3, direction: Vec3) -> Float64:
    var dir = normalize(direction)
    var lambdas = List[Float64]()
    collect_axis_intersections(volume, origin, dir, 0, lambdas)
    collect_axis_intersections(volume, origin, dir, 1, lambdas)
    collect_axis_intersections(volume, origin, dir, 2, lambdas)
    sort_f64(lambdas)

    var water = 0.0
    if len(lambdas) < 2:
        return water

    for n in range(len(lambdas) - 1):
        var a = lambdas[n]
        var b = lambdas[n + 1]
        var dd = b - a
        if dd <= 0.0:
            continue
        var mid = add_scaled(origin, dir, (a + b) * 0.5)
        if not inside_volume(volume, mid):
            continue
        var i = grid_index(mid.x, volume.header.origin.x, volume.header.directions[0, 0], volume.header.size_i)
        var j = grid_index(mid.y, volume.header.origin.y, volume.header.directions[1, 1], volume.header.size_j)
        var k = grid_index(mid.z, volume.header.origin.z, volume.header.directions[2, 2], volume.header.size_k)
        water += dd * hlut_path_factor(hlut, Float64(ct_value_at(volume, i, j, k)))
    return water


def ct_grid_intersection_table(
    volume: CTVolume,
    hlut: HLUT,
    origin: Vec3,
    direction: Vec3,
    off_h2o_mm: Float64,
    hlut_scale: Float64,
) -> CTIntersectionTable:
    var lambdas = List[Float64]()
    collect_axis_intersections(volume, origin, direction, 0, lambdas)
    collect_axis_intersections(volume, origin, direction, 1, lambdas)
    collect_axis_intersections(volume, origin, direction, 2, lambdas)
    sort_f64(lambdas)

    var zs = List[Float64]()
    var h2os = List[Float64]()
    if len(lambdas) == 0:
        return CTIntersectionTable(zs^, h2os^)

    var lambda_last = lambdas[0]
    zs.append(lambda_last)
    h2os.append(0.0)
    for n in range(len(lambdas) - 1):
        var a = lambdas[n]
        var b = lambdas[n + 1]
        var dd = b - a
        lambda_last += dd
        if dd <= 0.0:
            continue
        var mid = add_scaled(origin, direction, (a + b) * 0.5)
        if not inside_volume(volume, mid):
            continue
        var i = grid_index(mid.x, volume.header.origin.x, volume.header.directions[0, 0], volume.header.size_i)
        var j = grid_index(mid.y, volume.header.origin.y, volume.header.directions[1, 1], volume.header.size_j)
        var k = grid_index(mid.z, volume.header.origin.z, volume.header.directions[2, 2], volume.header.size_k)
        var eqv_path = hlut_path_factor(hlut, Float64(ct_value_at(volume, i, j, k))) * hlut_scale
        var h2o_step = dd * eqv_path
        if h2o_step <= 0.0:
            continue
        zs.append(lambda_last)
        h2os.append(h2os[len(h2os) - 1] + h2o_step)

    for i in range(len(zs)):
        zs[i] = -zs[i]
        h2os[i] += off_h2o_mm

    return CTIntersectionTable(zs^, h2os^)


def vintpol1d(x: Float64, xs: List[Float64], ys: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    if len(xs) == 1:
        return ys[0]
    var last = len(xs) - 1
    var ascending = xs[0] <= xs[last]
    if ascending:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[last]:
            return ys[last]
        for i in range(last):
            if x >= xs[i] and x <= xs[i + 1]:
                return linear_interp(x, xs[i], xs[i + 1], ys[i], ys[i + 1])
    else:
        if x >= xs[0]:
            return ys[0]
        if x <= xs[last]:
            return ys[last]
        for i in range(last):
            if x <= xs[i] and x >= xs[i + 1]:
                return linear_interp(x, xs[i], xs[i + 1], ys[i], ys[i + 1])
    return ys[last]


def ct_siddon_water_position(
    volume: CTVolume,
    hlut: HLUT,
    start: Vec3,
    direction: Vec3,
    requested_h2o_mm: Float64,
    hlut_scale: Float64,
    mut position: Float64,
) -> Bool:
    var lambda_max_x = 1.0e8
    var lambda_max_y = 1.0e8
    var lambda_max_z = 1.0e8
    if abs_float(direction.x) > 1.0e-6:
        var l1 = (volume.header.origin.x - start.x) / -direction.x
        var l2 = (volume.header.origin.x + Float64(volume.header.size_i) * volume.header.directions[0, 0] - start.x) / -direction.x
        lambda_max_x = max_float(l1, l2)
    if abs_float(direction.y) > 1.0e-6:
        var l1 = (volume.header.origin.y - start.y) / -direction.y
        var l2 = (volume.header.origin.y + Float64(volume.header.size_j) * volume.header.directions[1, 1] - start.y) / -direction.y
        lambda_max_y = max_float(l1, l2)
    if abs_float(direction.z) > 1.0e-6:
        var l1 = (volume.header.origin.z - start.z) / -direction.z
        var l2 = (volume.header.origin.z + Float64(volume.header.size_k) * volume.header.directions[2, 2] - start.z) / -direction.z
        lambda_max_z = max_float(l1, l2)
    var lambda_iso = min_float(lambda_max_x, min_float(lambda_max_y, lambda_max_z))
    var p = Vec3(
        start.x - lambda_iso * direction.x + direction.x * 0.00001,
        start.y - lambda_iso * direction.y + direction.y * 0.00001,
        start.z - lambda_iso * direction.z + direction.z * 0.00001,
    )
    if not inside_volume_open(volume, p):
        return False
    var ix = grid_index(p.x, volume.header.origin.x, volume.header.directions[0, 0], volume.header.size_i)
    var iy = grid_index(p.y, volume.header.origin.y, volume.header.directions[1, 1], volume.header.size_j)
    var iz = grid_index(p.z, volume.header.origin.z, volume.header.directions[2, 2], volume.header.size_k)

    if abs_float(direction.x) > 1.0e-6:
        lambda_max_x = abs_float((Float64(volume.header.size_i) * volume.header.directions[0, 0]) / direction.x)
    else:
        lambda_max_x = 1.0e8
    if abs_float(direction.y) > 1.0e-6:
        lambda_max_y = abs_float((Float64(volume.header.size_j) * volume.header.directions[1, 1]) / direction.y)
    else:
        lambda_max_y = 1.0e8
    if abs_float(direction.z) > 1.0e-6:
        lambda_max_z = abs_float((Float64(volume.header.size_k) * volume.header.directions[2, 2]) / direction.z)
    else:
        lambda_max_z = 1.0e8
    var lambda_max = min_float(lambda_max_x, min_float(lambda_max_y, lambda_max_z))

    var step_ix = 1
    var step_iy = 1
    var step_iz = 1
    if direction.x < 0.0:
        step_ix = -1
    if direction.y < 0.0:
        step_iy = -1
    if direction.z < 0.0:
        step_iz = -1

    var next_x = 1.0e8
    var next_y = 1.0e8
    var next_z = 1.0e8
    var delta_x = 0.0
    var delta_y = 0.0
    var delta_z = 0.0
    if abs_float(direction.x) > 1.0e-6:
        var l1 = (volume.header.origin.x + Float64(ix + 1) * volume.header.directions[0, 0] - p.x) / direction.x
        var l2 = (volume.header.origin.x + Float64(ix) * volume.header.directions[0, 0] - p.x) / direction.x
        next_x = max_float(l1, l2)
        delta_x = abs_float(volume.header.directions[0, 0] / direction.x)
    if abs_float(direction.y) > 1.0e-6:
        var l1 = (volume.header.origin.y + Float64(iy + 1) * volume.header.directions[1, 1] - p.y) / direction.y
        var l2 = (volume.header.origin.y + Float64(iy) * volume.header.directions[1, 1] - p.y) / direction.y
        next_y = max_float(l1, l2)
        delta_y = abs_float(volume.header.directions[1, 1] / direction.y)
    if abs_float(direction.z) > 1.0e-6:
        var l1 = (volume.header.origin.z + Float64(iz + 1) * volume.header.directions[2, 2] - p.z) / direction.z
        var l2 = (volume.header.origin.z + Float64(iz) * volume.header.directions[2, 2] - p.z) / direction.z
        next_z = max_float(l1, l2)
        delta_z = abs_float(volume.header.directions[2, 2] / direction.z)

    var current = 0.0
    var h2o = 0.0
    var boundary = lambda_max * (1.0 - 1.0e-7)
    while current < boundary:
        if ix < 0 or ix >= volume.header.size_i or iy < 0 or iy >= volume.header.size_j or iz < 0 or iz >= volume.header.size_k:
            return False
        var eqv = hlut_path_factor(hlut, Float64(ct_value_at(volume, ix, iy, iz))) * hlut_scale
        var delta: Float64
        if next_x < next_y and next_x < next_z:
            delta = next_x - current
            current = next_x
            next_x += delta_x
            ix += step_ix
        elif next_y < next_z:
            delta = next_y - current
            current = next_y
            next_y += delta_y
            iy += step_iy
        else:
            delta = next_z - current
            current = next_z
            next_z += delta_z
            iz += step_iz
        h2o += delta * eqv
        if h2o > requested_h2o_mm:
            var geo = current - (h2o - requested_h2o_mm) / eqv
            position = lambda_iso - geo
            return True
    return False


def linear_interp(x: Float64, x0: Float64, x1: Float64, y0: Float64, y1: Float64) -> Float64:
    var dx = x1 - x0
    if dx == 0.0:
        return y0
    return y0 + (x - x0) / dx * (y1 - y0)


def collect_axis_intersections(
    volume: CTVolume,
    origin: Vec3,
    dir: Vec3,
    axis: Int,
    mut lambdas: List[Float64],
):
    var d = component(dir, axis)
    if d == 0.0:
        return
    var n = axis_size(volume, axis)
    var base = axis_origin(volume, axis)
    var step = axis_step(volume, axis)
    for idx in range(n + 1):
        var lam = (base + Float64(idx) * step - component(origin, axis)) / d
        var p = add_scaled(origin, dir, lam)
        if inside_other_axes(volume, p, axis):
            lambdas.append(lam)


def inside_volume(volume: CTVolume, p: Vec3) -> Bool:
    return inside_axis(p.x, volume.header.origin.x, volume.header.directions[0, 0], volume.header.size_i) and inside_axis(p.y, volume.header.origin.y, volume.header.directions[1, 1], volume.header.size_j) and inside_axis(p.z, volume.header.origin.z, volume.header.directions[2, 2], volume.header.size_k)


def inside_volume_open(volume: CTVolume, p: Vec3) -> Bool:
    return p.x >= volume.header.origin.x and p.x < volume.header.origin.x + Float64(volume.header.size_i) * volume.header.directions[0, 0] and p.y >= volume.header.origin.y and p.y < volume.header.origin.y + Float64(volume.header.size_j) * volume.header.directions[1, 1] and p.z >= volume.header.origin.z and p.z < volume.header.origin.z + Float64(volume.header.size_k) * volume.header.directions[2, 2]


def inside_other_axes(volume: CTVolume, p: Vec3, axis: Int) -> Bool:
    if axis != 0 and not inside_axis(p.x, volume.header.origin.x, volume.header.directions[0, 0], volume.header.size_i):
        return False
    if axis != 1 and not inside_axis(p.y, volume.header.origin.y, volume.header.directions[1, 1], volume.header.size_j):
        return False
    if axis != 2 and not inside_axis(p.z, volume.header.origin.z, volume.header.directions[2, 2], volume.header.size_k):
        return False
    return True


def inside_axis(value: Float64, origin: Float64, step: Float64, size: Int) -> Bool:
    var lo = origin
    var hi = origin + Float64(size) * step
    if hi < lo:
        var tmp = lo
        lo = hi
        hi = tmp
    return value >= lo and value <= hi


def grid_index(value: Float64, origin: Float64, step: Float64, size: Int) -> Int:
    if step < 0.0:
        return clamp_int(Int((origin - value) / (-step)), 0, size - 1)
    return clamp_int(Int((value - origin) / step), 0, size - 1)


def grid_index_unclamped(value: Float64, origin: Float64, step: Float64) -> Int:
    if step < 0.0:
        return Int((origin - value) / (-step))
    return Int((value - origin) / step)


def component(v: Vec3, axis: Int) -> Float64:
    if axis == 0:
        return v.x
    if axis == 1:
        return v.y
    return v.z


def axis_origin(volume: CTVolume, axis: Int) -> Float64:
    if axis == 0:
        return volume.header.origin.x
    if axis == 1:
        return volume.header.origin.y
    return volume.header.origin.z


def axis_step(volume: CTVolume, axis: Int) -> Float64:
    return volume.header.directions[axis, axis]


def axis_size(volume: CTVolume, axis: Int) -> Int:
    if axis == 0:
        return volume.header.size_i
    if axis == 1:
        return volume.header.size_j
    return volume.header.size_k


def sort_f64(mut values: List[Float64]):
    for i in range(1, len(values)):
        var current = values[i]
        var j = i - 1
        while j >= 0 and values[j] > current:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = current


def clip_axis(p: Float64, d: Float64, a: Float64, b: Float64, mut t0: Float64, mut t1: Float64) -> Bool:
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


def normalize(v: Vec3) -> Vec3:
    var n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return Vec3(v.x / n, v.y / n, v.z / n)


def add_scaled(p: Vec3, d: Vec3, t: Float64) -> Vec3:
    return Vec3(p.x + d.x * t, p.y + d.y * t, p.z + d.z * t)


def min_float(a: Float64, b: Float64) -> Float64:
    if a < b:
        return a
    return b


def max_float(a: Float64, b: Float64) -> Float64:
    if a > b:
        return a
    return b


def abs_float(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


def min_positive(best: Float64, candidate: Float64, current: Float64) -> Float64:
    if candidate > current + 1.0e-9 and candidate < best:
        return candidate
    return best


def clamp_int(value: Int, lo: Int, hi: Int) -> Int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
