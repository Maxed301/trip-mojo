from std.testing import assert_true

from dedx import dedx_stopping_power, find_projectile, read_dedx
from trip_ffi import TripFFI


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var path = "/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/20040607.dedx"
    var ffi = TripFFI()
    var table = read_dedx(path)
    var carbon = find_projectile(table, 6, 12)
    var trip = ffi.dedx_eval(6, 12, 200.0)
    var mojo = dedx_stopping_power(carbon, 200.0)
    assert_close(mojo, trip, 1.0e-7)
