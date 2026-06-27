from geometry import Vec3
from voi import VOI


@fieldwise_init
struct OptVoxel(Copyable, Movable):
    var i: Int
    var j: Int
    var k: Int
    var role: Int
    var high_density: Bool


@fieldwise_init
struct OptVoxelSet(Copyable, Movable):
    var voxels: List[OptVoxel]
    var target_count: Int
    var avoidance_count: Int
    var high_count: Int
    var low_count: Int


def build_p101_target_avoidance_opt_voxels(
    target: VOI,
    inner_margin_mm: Float64,
    outer_margin_mm: Float64,
    boundary_width_mm: Float64,
) -> OptVoxelSet:
    var pad_i = ceil_div_float(outer_margin_mm, target.grid.voxel_size.x) + 2
    var pad_j = ceil_div_float(outer_margin_mm, target.grid.voxel_size.y) + 2
    var pad_k = ceil_div_float(outer_margin_mm, target.grid.voxel_size.z) + 2
    var min_i = max_int(0, target.bounds.min_i - pad_i)
    var max_i = min_int(target.grid.shape.i - 1, target.bounds.max_i + pad_i)
    var min_j = max_int(0, target.bounds.min_j - pad_j)
    var max_j = min_int(target.grid.shape.j - 1, target.bounds.max_j + pad_j)
    var min_k = max_int(0, target.bounds.min_k - pad_k)
    var max_k = min_int(target.grid.shape.k - 1, target.bounds.max_k + pad_k)
    var nx = max_i - min_i + 1
    var ny = max_j - min_j + 1
    var nz = max_k - min_k + 1
    var total = nx * ny * nz
    var mask = List[Int]()
    mask.resize(total, 0)
    for n in range(len(target.active_indices)):
        var triplet = target.grid.index_triplet(target.active_indices[n])
        var i = Int(triplet.x) - min_i
        var j = Int(triplet.y) - min_j
        var k = Int(triplet.z) - min_k
        if i >= 0 and j >= 0 and k >= 0 and i < nx and j < ny and k < nz:
            mask[local_linear(i, j, k, nx, ny)] = 1

    var margin_source = List[Int]()
    margin_source.resize(total, 0)
    for n in range(len(target.active_indices)):
        var triplet = target.grid.index_triplet(target.active_indices[n])
        var global_i = Int(triplet.x)
        var global_j = Int(triplet.y)
        if global_i == target.bounds.max_i or global_j == target.bounds.max_j:
            continue
        var i = global_i - min_i
        var j = global_j - min_j
        var k = Int(triplet.z) - min_k
        if i >= 0 and j >= 0 and k >= 0 and i < nx and j < ny and k < nz:
            margin_source[local_linear(i, j, k, nx, ny)] = 1

    var outside_dt = squared_distance_transform(margin_source, nx, ny, nz, target.grid.voxel_size, False)
    var target_nx = target.bounds.max_i - target.bounds.min_i + 1
    var target_ny = target.bounds.max_j - target.bounds.min_j + 1
    var target_nz = target.bounds.max_k - target.bounds.min_k + 1
    var target_crop = List[Int]()
    target_crop.resize(target_nx * target_ny * target_nz, 0)
    for n in range(len(target.active_indices)):
        var triplet = target.grid.index_triplet(target.active_indices[n])
        var ti = Int(triplet.x) - target.bounds.min_i
        var tj = Int(triplet.y) - target.bounds.min_j
        var tk = Int(triplet.z) - target.bounds.min_k
        target_crop[local_linear(ti, tj, tk, target_nx, target_ny)] = 1
    var inside_dt = squared_distance_transform(target_crop, target_nx, target_ny, target_nz, target.grid.voxel_size, True)
    var shell_width_mm = outer_margin_mm - inner_margin_mm
    var shell2 = (shell_width_mm + 0.001) * (shell_width_mm + 0.001)
    var outer2 = (outer_margin_mm + 0.001) * (outer_margin_mm + 0.001)
    var boundary2 = boundary_width_mm * boundary_width_mm
    var expanded = List[Int]()
    expanded.resize(total, 0)
    var exp_min_i = nx
    var exp_max_i = -1
    var exp_min_j = ny
    var exp_max_j = -1
    var exp_min_k = nz
    var exp_max_k = -1
    for idx in range(total):
        if margin_source[idx] != 0 or outside_dt[idx] < Float32(outer2):
            expanded[idx] = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                var idx = local_linear(i, j, k, nx, ny)
                if expanded[idx] != 0:
                    if i < exp_min_i:
                        exp_min_i = i
                    if i > exp_max_i:
                        exp_max_i = i
                    if j < exp_min_j:
                        exp_min_j = j
                    if j > exp_max_j:
                        exp_max_j = j
                    if k < exp_min_k:
                        exp_min_k = k
                    if k > exp_max_k:
                        exp_max_k = k
    var cnx = exp_max_i - exp_min_i + 1
    var cny = exp_max_j - exp_min_j + 1
    var cnz = exp_max_k - exp_min_k + 1
    var ctotal = cnx * cny * cnz
    var cropped_expanded = List[Int]()
    cropped_expanded.resize(ctotal, 0)
    for k in range(cnz):
        for j in range(cny):
            for i in range(cnx):
                var src_idx = local_linear(i + exp_min_i, j + exp_min_j, k + exp_min_k, nx, ny)
                cropped_expanded[local_linear(i, j, k, cnx, cny)] = expanded[src_idx]
    var expanded_inside_dt = squared_distance_transform(cropped_expanded, cnx, cny, cnz, target.grid.voxel_size, True)
    var avoidance = List[Int]()
    avoidance.resize(total, 0)
    for k in range(cnz):
        for j in range(cny):
            for i in range(cnx):
                var cidx = local_linear(i, j, k, cnx, cny)
                if cropped_expanded[cidx] != 0 and expanded_inside_dt[cidx] <= Float32(shell2):
                    avoidance[local_linear(i + exp_min_i, j + exp_min_j, k + exp_min_k, nx, ny)] = 1
    var av_min_i = nx
    var av_max_i = -1
    var av_min_j = ny
    var av_max_j = -1
    var av_min_k = nz
    var av_max_k = -1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                var idx = local_linear(i, j, k, nx, ny)
                if avoidance[idx] != 0:
                    if i < av_min_i:
                        av_min_i = i
                    if i > av_max_i:
                        av_max_i = i
                    if j < av_min_j:
                        av_min_j = j
                    if j > av_max_j:
                        av_max_j = j
                    if k < av_min_k:
                        av_min_k = k
                    if k > av_max_k:
                        av_max_k = k
    var anx = av_max_i - av_min_i + 1
    var any = av_max_j - av_min_j + 1
    var anz = av_max_k - av_min_k + 1
    var avoidance_crop = List[Int]()
    avoidance_crop.resize(anx * any * anz, 0)
    for k in range(anz):
        for j in range(any):
            for i in range(anx):
                var src_idx = local_linear(i + av_min_i, j + av_min_j, k + av_min_k, nx, ny)
                avoidance_crop[local_linear(i, j, k, anx, any)] = avoidance[src_idx]
    var avoidance_inside_dt = squared_distance_transform(avoidance_crop, anx, any, anz, target.grid.voxel_size, True)
    var voxels = List[OptVoxel]()
    var target_count = 0
    var avoidance_count = 0
    var high_count = 0
    var low_count = 0
    var target_high = 0
    var avoidance_high = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                var idx = local_linear(i, j, k, nx, ny)
                var is_target = mask[idx] != 0
                var is_avoidance = False
                if not is_target:
                    is_avoidance = avoidance[idx] != 0
                if not is_target and not is_avoidance:
                    continue
                var high: Bool
                if is_target:
                    var ti = i + min_i - target.bounds.min_i
                    var tj = j + min_j - target.bounds.min_j
                    var tk = k + min_k - target.bounds.min_k
                    high = inside_dt[local_linear(ti, tj, tk, target_nx, target_ny)] < Float32(boundary2)
                    target_count += 1
                    if high:
                        target_high += 1
                else:
                    high = avoidance_inside_dt[local_linear(i - av_min_i, j - av_min_j, k - av_min_k, anx, any)] < Float32(boundary2)
                    avoidance_count += 1
                    if high:
                        avoidance_high += 1
                if high:
                    high_count += 1
                else:
                    low_count += 1
                var role = 1
                if is_avoidance:
                    role = 2
                voxels.append(OptVoxel(i + min_i, j + min_j, k + min_k, role, high))
    return OptVoxelSet(voxels^, target_count, avoidance_count, high_count, low_count)


def refine_opt_voxels_by_trip_max_fdc(
    opt_voxels: OptVoxelSet,
    summed_max_fdc: List[Float64],
    d_min_dos: Float64,
    avoidance_max_dose_fraction: Float64,
) raises -> OptVoxelSet:
    if len(opt_voxels.voxels) != len(summed_max_fdc):
        raise Error("opt voxel/max FDC length mismatch")
    var d_phys_max = 0.0
    for i in range(len(summed_max_fdc)):
        if summed_max_fdc[i] > d_phys_max:
            d_phys_max = summed_max_fdc[i]
    var threshold = d_min_dos * d_phys_max
    var voxels = List[OptVoxel]()
    var target_count = 0
    var avoidance_count = 0
    var high_count = 0
    var low_count = 0
    for i in range(len(opt_voxels.voxels)):
        var voxel = opt_voxels.voxels[i].copy()
        var keep = voxel.role == 1
        if not keep:
            keep = summed_max_fdc[i] > avoidance_max_dose_fraction * threshold
        if not keep:
            continue
        if voxel.role == 1:
            target_count += 1
        else:
            avoidance_count += 1
        if voxel.high_density:
            high_count += 1
        else:
            low_count += 1
        voxels.append(voxel^)
    return OptVoxelSet(voxels^, target_count, avoidance_count, high_count, low_count)


def squared_distance_transform(mask: List[Int], nx: Int, ny: Int, nz: Int, spacing: Vec3, inside: Bool) -> List[Float32]:
    var dist = List[Float32]()
    dist.resize(nx * ny * nz, Float32(1.0e8))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                var idx = local_linear(i, j, k, nx, ny)
                if (inside and mask[idx] == 0) or ((not inside) and mask[idx] != 0):
                    dist[idx] = Float32(0.0)
    distance_pass_x(dist, nx, ny, nz, Float32(spacing.x))
    distance_pass_y(dist, nx, ny, nz, Float32(spacing.y))
    distance_pass_z(dist, nx, ny, nz, Float32(spacing.z))
    return dist^


def distance_pass_x(mut dist: List[Float32], nx: Int, ny: Int, nz: Int, step: Float32):
    var g = List[Float32]()
    var h = List[Float32]()
    g.resize(nx, Float32(0.0))
    h.resize(nx, Float32(0.0))
    for k in range(nz):
        for j in range(ny):
            voronoi_line(dist, g, h, nx, step, 0, j, k, nx, ny)


def distance_pass_y(mut dist: List[Float32], nx: Int, ny: Int, nz: Int, step: Float32):
    var g = List[Float32]()
    var h = List[Float32]()
    g.resize(ny, Float32(0.0))
    h.resize(ny, Float32(0.0))
    for k in range(nz):
        for i in range(nx):
            voronoi_line(dist, g, h, ny, step, 1, i, k, nx, ny)


def distance_pass_z(mut dist: List[Float32], nx: Int, ny: Int, nz: Int, step: Float32):
    var g = List[Float32]()
    var h = List[Float32]()
    g.resize(nz, Float32(0.0))
    h.resize(nz, Float32(0.0))
    for j in range(ny):
        for i in range(nx):
            voronoi_line(dist, g, h, nz, step, 2, i, j, nx, ny)


def voronoi_line(
    mut dist: List[Float32],
    mut g: List[Float32],
    mut h: List[Float32],
    n: Int,
    step: Float32,
    dim: Int,
    a: Int,
    b: Int,
    nx: Int,
    ny: Int,
):
    var counter = -1
    for pos in range(n):
        var idx = line_index(pos, dim, a, b, nx, ny)
        var ri = dist[idx]
        if ri != Float32(1.0e8):
            var weight = Float32(pos) * step
            if counter < 1:
                counter += 1
                g[counter] = ri
                h[counter] = weight
            else:
                while counter >= 1 and dt_remove(g[counter - 1], g[counter], ri, h[counter - 1], h[counter], weight):
                    counter -= 1
                counter += 1
                g[counter] = ri
                h[counter] = weight
    if counter == -1:
        return
    var ns = counter
    counter = 0
    for pos in range(n):
        var weight = Float32(pos) * step
        var d1 = g[counter] + (h[counter] - weight) * (h[counter] - weight)
        while counter < ns:
            var d2 = g[counter + 1] + (h[counter + 1] - weight) * (h[counter + 1] - weight)
            if d1 <= d2:
                break
            counter += 1
            d1 = d2
        var idx = line_index(pos, dim, a, b, nx, ny)
        dist[idx] = d1


def dt_remove(d1: Float32, d2: Float32, df: Float32, x1: Float32, x2: Float32, xf: Float32) -> Bool:
    var a = x2 - x1
    var b = xf - x2
    var c = xf - x1
    return (c * d2 - b * d1 - a * df - a * b * c) > Float32(0.0)


def line_index(pos: Int, dim: Int, a: Int, b: Int, nx: Int, ny: Int) -> Int:
    if dim == 0:
        return local_linear(pos, a, b, nx, ny)
    if dim == 1:
        return local_linear(a, pos, b, nx, ny)
    return local_linear(a, b, pos, nx, ny)


def local_linear(i: Int, j: Int, k: Int, nx: Int, ny: Int) -> Int:
    return i + nx * (j + ny * k)


def ceil_div_float(value: Float64, step: Float64) -> Int:
    var q = value / step
    var i = Int(q)
    if Float64(i) < q:
        return i + 1
    return i


def min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


def max_int(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b
