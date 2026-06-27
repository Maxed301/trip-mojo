from std.math import sqrt


@fieldwise_init
struct SparseDoseEntry(Copyable, Movable):
    var voxel: Int
    var spot: Int
    var value: Float64


@fieldwise_init
struct SparseDoseMatrix(Copyable, Movable):
    var voxel_count: Int
    var spot_count: Int
    var entries: List[SparseDoseEntry]


@fieldwise_init
struct SparseOptimizationResult(Copyable, Movable):
    var particles: List[Float64]
    var dose: List[Float64]
    var chi2: Float64
    var relative_rmse: Float64
    var iterations: Int


def compute_sparse_dose(matrix: SparseDoseMatrix, particles: List[Float64]) raises -> List[Float64]:
    if len(particles) != matrix.spot_count:
        raise Error("particle vector length does not match sparse dose matrix")
    var dose = List[Float64]()
    dose.resize(matrix.voxel_count, 0.0)
    for i in range(len(matrix.entries)):
        dose[matrix.entries[i].voxel] += matrix.entries[i].value * particles[matrix.entries[i].spot]
    return dose^


def sparse_chi2(dose: List[Float64], target: List[Float64]) raises -> Float64:
    if len(dose) != len(target):
        raise Error("dose/target length mismatch")
    var value = 0.0
    for i in range(len(dose)):
        var residual = target[i] - dose[i]
        value += residual * residual
    return value


def sparse_relative_rmse(dose: List[Float64], target: List[Float64]) raises -> Float64:
    var chi2 = sparse_chi2(dose, target)
    var norm = 0.0
    for i in range(len(target)):
        norm += target[i] * target[i]
    if norm == 0.0:
        return 0.0
    return sqrt(chi2 / norm)


def optimize_sparse_dose(
    matrix: SparseDoseMatrix,
    target: List[Float64],
    initial_particles: List[Float64],
    iterations: Int,
    learning_rate: Float64,
    min_particles: Float64 = 0.0,
) raises -> SparseOptimizationResult:
    if len(target) != matrix.voxel_count:
        raise Error("target length does not match sparse dose matrix")
    if len(initial_particles) != matrix.spot_count:
        raise Error("initial particle length does not match sparse dose matrix")
    var particles = initial_particles.copy()
    var dose = compute_sparse_dose(matrix, particles)
    var gradient = List[Float64]()
    gradient.resize(matrix.spot_count, 0.0)
    var completed = 0
    for it in range(iterations):
        for s in range(matrix.spot_count):
            gradient[s] = 0.0
        for i in range(len(matrix.entries)):
            var residual = target[matrix.entries[i].voxel] - dose[matrix.entries[i].voxel]
            gradient[matrix.entries[i].spot] += -2.0 * matrix.entries[i].value * residual
        for s in range(matrix.spot_count):
            particles[s] -= learning_rate * gradient[s]
            if particles[s] < min_particles:
                particles[s] = 0.0
        dose = compute_sparse_dose(matrix, particles)
        completed = it + 1
    var chi2 = sparse_chi2(dose, target)
    var rel = sparse_relative_rmse(dose, target)
    return SparseOptimizationResult(particles^, dose^, chi2, rel, completed)


def sparse_normal_matvec(matrix: SparseDoseMatrix, vector: List[Float64]) raises -> List[Float64]:
    if len(vector) != matrix.spot_count:
        raise Error("vector length does not match sparse dose matrix")
    var av = List[Float64]()
    av.resize(matrix.voxel_count, 0.0)
    for i in range(len(matrix.entries)):
        av[matrix.entries[i].voxel] += matrix.entries[i].value * vector[matrix.entries[i].spot]
    var atav = List[Float64]()
    atav.resize(matrix.spot_count, 0.0)
    for i in range(len(matrix.entries)):
        atav[matrix.entries[i].spot] += matrix.entries[i].value * av[matrix.entries[i].voxel]
    return atav^


def sparse_transpose_matvec(matrix: SparseDoseMatrix, vector: List[Float64]) raises -> List[Float64]:
    if len(vector) != matrix.voxel_count:
        raise Error("vector length does not match sparse dose matrix voxel count")
    var out = List[Float64]()
    out.resize(matrix.spot_count, 0.0)
    for i in range(len(matrix.entries)):
        out[matrix.entries[i].spot] += matrix.entries[i].value * vector[matrix.entries[i].voxel]
    return out^


def dot(a: List[Float64], b: List[Float64]) raises -> Float64:
    if len(a) != len(b):
        raise Error("dot product length mismatch")
    var total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total


def optimize_sparse_dose_cg(
    matrix: SparseDoseMatrix,
    target: List[Float64],
    initial_particles: List[Float64],
    iterations: Int,
    tolerance: Float64 = 1.0e-12,
    min_particles: Float64 = 0.0,
) raises -> SparseOptimizationResult:
    if len(target) != matrix.voxel_count:
        raise Error("target length does not match sparse dose matrix")
    if len(initial_particles) != matrix.spot_count:
        raise Error("initial particle length does not match sparse dose matrix")

    var x = initial_particles.copy()
    var dose = compute_sparse_dose(matrix, x)
    var residual = List[Float64]()
    residual.reserve(matrix.voxel_count)
    for v in range(matrix.voxel_count):
        residual.append(target[v] - dose[v])

    var r = sparse_transpose_matvec(matrix, residual)
    var p = r.copy()
    var rr = dot(r, r)
    var rr_stop = tolerance * tolerance
    if rr > rr_stop:
        rr_stop = rr * tolerance * tolerance
    var completed = 0
    for it in range(iterations):
        if rr <= rr_stop:
            break
        var ap = sparse_normal_matvec(matrix, p)
        var denom = dot(p, ap)
        if denom == 0.0:
            break
        var alpha = rr / denom
        for s in range(matrix.spot_count):
            x[s] += alpha * p[s]
            if x[s] < min_particles:
                x[s] = 0.0
        var r_next = List[Float64]()
        r_next.resize(matrix.spot_count, 0.0)
        for s in range(matrix.spot_count):
            r_next[s] = r[s] - alpha * ap[s]
        var rr_next = dot(r_next, r_next)
        completed = it + 1
        if rr_next <= rr_stop:
            break
        var beta = rr_next / rr
        for s in range(matrix.spot_count):
            p[s] = r_next[s] + beta * p[s]
        r = r_next^
        rr = rr_next

    dose = compute_sparse_dose(matrix, x)
    var chi2 = sparse_chi2(dose, target)
    var rel = sparse_relative_rmse(dose, target)
    return SparseOptimizationResult(x^, dose^, chi2, rel, completed)
