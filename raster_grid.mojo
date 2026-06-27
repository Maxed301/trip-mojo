from field_geometry import Bounds3f


@fieldwise_init
struct RasterPoint2D(Copyable, Movable):
    var x: Float64
    var y: Float64
    var index: Int


@fieldwise_init
struct RasterGrid2D(Copyable, Movable):
    var x_values: List[Float64]
    var y_values: List[Float64]
    var points: List[RasterPoint2D]
    var min_x: Float64
    var max_x: Float64
    var min_y: Float64
    var max_y: Float64


def raster_axis_values(min_value: Float64, max_value: Float64, step: Float64) raises -> List[Float64]:
    if step <= 0.0:
        raise Error("raster step must be positive")
    var values = List[Float64]()
    var dd = 0.0
    while dd <= max_value:
        if dd >= min_value:
            values.append(dd)
        dd += step
    dd = -step
    while dd >= min_value:
        if dd <= max_value:
            values.append(dd)
        dd -= step
    sort_values(values)
    return values^


def raster_grid_from_window(window: Bounds3f, step_x: Float64, step_y: Float64) raises -> RasterGrid2D:
    var xs = raster_axis_values(window.min_x, window.max_x, step_x)
    var ys = raster_axis_values(window.min_y, window.max_y, step_y)
    var points = List[RasterPoint2D]()
    points.reserve(len(xs) * len(ys))

    var index = 0
    var forward = True
    for iy in range(len(ys)):
        if forward:
            for ix in range(len(xs)):
                points.append(RasterPoint2D(xs[ix], ys[iy], index))
                index += 1
        else:
            var ix = len(xs) - 1
            while ix >= 0:
                points.append(RasterPoint2D(xs[ix], ys[iy], index))
                index += 1
                ix -= 1
        forward = not forward
    return RasterGrid2D(xs^, ys^, points^, window.min_x, window.max_x, window.min_y, window.max_y)


def sort_values(mut values: List[Float64]):
    for i in range(1, len(values)):
        var v = values[i]
        var j = i
        while j > 0 and values[j - 1] > v:
            values[j] = values[j - 1]
            j -= 1
        values[j] = v
