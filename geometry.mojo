from std.math import cos, sin, pi
from std.collections import InlineArray
from std.memory import UnsafePointer


@fieldwise_init
struct Vec3(Copyable, Movable):
    var x: Float64
    var y: Float64
    var z: Float64


@fieldwise_init
struct Mat4(Copyable, Movable):
    var rows: InlineArray[SIMD[DType.float64, 4], 4]

    def __init__(out self, *values: Float64):
        self.rows = InlineArray[SIMD[DType.float64, 4], 4](fill=SIMD[DType.float64, 4](0))

        if len(values) == 16:
            for i in range(4):
                self.rows[i] = SIMD[DType.float64, 4](
                    values[i*4], values[i*4+1], values[i*4+2], values[i*4+3]
                )

    def __getitem__(self, row: Int, col: Int) -> Float64:
        return self.rows[row][col]

    def __setitem__(mut self, row: Int, col: Int, val: Float64):
        var vec = self.rows[row]
        vec[col] = val
        self.rows[row] = vec

    def __mul__(self, other: Mat4) -> Mat4:
        var result = Mat4()
        for i in range(4):
            for j in range(4):
                result.rows[i] += self.rows[i][j] * other.rows[j]
        # ^ to transfer value, otherwise ImplicitlyCopyable needed
        return result^


def transform_point(matrix: Mat4, point: Vec3) -> Vec3:
    return Vec3(
        x=matrix[0, 0] * point.x + matrix[0, 1] * point.y + matrix[0, 2] * point.z + matrix[0, 3],
        y=matrix[1, 0] * point.x + matrix[1, 1] * point.y + matrix[1, 2] * point.z + matrix[1, 3],
        z=matrix[2, 0] * point.x + matrix[2, 1] * point.y + matrix[2, 2] * point.z + matrix[2, 3],
    )


def gantry_to_patient_point(isocenter: Vec3, gantry_degrees: Float64, couch_degrees: Float64, point: Vec3) -> Vec3:
    var gantry = gantry_degrees / 180.0 * pi
    var x_room = point.x
    var y_room = cos(gantry) * point.y - sin(gantry) * point.z
    var z_room = sin(gantry) * point.y + cos(gantry) * point.z

    var couch = couch_degrees / 180.0 * pi
    var x_couch = cos(couch) * x_room + sin(couch) * z_room
    var y_couch = y_room
    var z_couch = -sin(couch) * x_room + cos(couch) * z_room

    return Vec3(
        isocenter.x - x_couch,
        isocenter.y - y_couch,
        z_couch + isocenter.z,
    )


def gantry_to_patient_point_trip_invmat(isocenter: Vec3, gantry_degrees: Float64, couch_degrees: Float64, point: Vec3) -> Vec3:
    return transform_point(trip_inv_mat4(patient_to_gantry_matrix(isocenter, gantry_degrees, couch_degrees)), point)


def patient_to_gantry_matrix(isocenter: Vec3, gantry_degrees: Float64, couch_degrees: Float64) -> Mat4:
    # Match TRiP98 TRPFldTrafoSet. P101 .exec uses this legacy field path,
    # not TRPFldTrafoSetIEC.
    var patient_to_couch = Mat4(
            -1,  0, 0, isocenter.x,
             0, -1, 0, isocenter.y,
             0,  0, 1, -isocenter.z,
            0,  0, 0, 1.0
            )

    var couch = couch_degrees / 90.0 * (pi * 0.5)
    var couch_cos = trip_trig_zero(cos(couch))
    var couch_sin = trip_trig_zero(sin(couch))
    var couch_to_room = Mat4(
            couch_cos, 0, -couch_sin, 0,
            0, 1, 0, 0,
            couch_sin, 0, couch_cos, 0,
            0, 0, 0, 1)

    var gantry = gantry_degrees / 90.0 * (pi * 0.5)
    var gantry_cos = trip_trig_zero(cos(gantry))
    var gantry_sin = trip_trig_zero(sin(gantry))
    var room_to_gantry = Mat4(
            1, 0, 0, 0,
            0, gantry_cos, gantry_sin, 0,
            0, -gantry_sin, gantry_cos, 0,
            0, 0, 0, 1)

    return room_to_gantry* couch_to_room* patient_to_couch


def trip_trig_zero(value: Float64) -> Float64:
    if value < 1.0e-6 and value > -1.0e-6:
        return 0.0
    return value


def trip_inv_mat4(matrix: Mat4) -> Mat4:
    var values = List[Float64]()
    values.resize(16, 0.0)
    for i in range(4):
        for j in range(4):
            values[i * 4 + j] = matrix[i, j]

    var rem_x = List[Int]()
    var rem_y = List[Int]()
    rem_x.resize(4, -1)
    rem_y.resize(4, -1)

    for _ in range(4):
        var pivot = 0.0
        var pivot_x = 0
        var pivot_y = 0
        for ix in range(4):
            if rem_x[ix] < 0:
                for iy in range(4):
                    if rem_y[iy] < 0:
                        var candidate = values[trip_inv_index(ix, iy)]
                        if abs_float(candidate) > abs_float(pivot):
                            pivot = candidate
                            pivot_x = ix
                            pivot_y = iy
        var aux = 0.0
        if pivot != 0.0:
            aux = 1.0 / pivot
        values[trip_inv_index(pivot_x, pivot_y)] = aux
        rem_x[pivot_x] = pivot_y
        rem_y[pivot_y] = pivot_x

        for jj in range(4):
            if jj != pivot_x:
                for kk in range(4):
                    if kk != pivot_y:
                        values[trip_inv_index(jj, kk)] -= values[trip_inv_index(jj, pivot_y)] * values[trip_inv_index(pivot_x, kk)] * aux

        for ix in range(4):
            if ix != pivot_x:
                values[trip_inv_index(ix, pivot_y)] *= aux
        for iy in range(4):
            if iy != pivot_y:
                values[trip_inv_index(pivot_x, iy)] *= -aux

    for ii in range(1, 4):
        var ix = ii - 1
        if rem_x[ix] != ix:
            var iy = ii
            for jj in range(ii, 4):
                iy = jj
                if rem_x[iy] == ix:
                    break
            for kk in range(4):
                var tmp = values[trip_inv_index(ix, kk)]
                values[trip_inv_index(ix, kk)] = values[trip_inv_index(iy, kk)]
                values[trip_inv_index(iy, kk)] = tmp
            rem_x[iy] = rem_x[ix]
        rem_x[ix] = ix

    for ii in range(1, 4):
        var ix = ii - 1
        if rem_y[ix] != ix:
            var iy = ii
            for jj in range(ii, 4):
                iy = jj
                if rem_y[iy] == ix:
                    break
            for kk in range(4):
                var tmp = values[trip_inv_index(kk, ix)]
                values[trip_inv_index(kk, ix)] = values[trip_inv_index(kk, iy)]
                values[trip_inv_index(kk, iy)] = tmp
            rem_y[iy] = rem_y[ix]
        rem_y[ix] = ix

    return Mat4(
        values[0], values[1], values[2], values[3],
        values[4], values[5], values[6], values[7],
        values[8], values[9], values[10], values[11],
        values[12], values[13], values[14], values[15],
    )


def abs_float(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


def trip_inv_index(row: Int, col: Int) -> Int:
    return col * 4 + row
