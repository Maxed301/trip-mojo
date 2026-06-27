from std.testing import assert_true

from ddd import interp
from trip_ffi import TripFFI


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def mojo_interp(v0: Float64, v1: Float64, depth_cm: Float64) -> Float64:
    var z = List[Float64]()
    z.append(1.0)
    z.append(3.0)
    var values = List[Float64]()
    values.append(v0)
    values.append(v1)
    return interp(z, values, depth_cm)


def main() raises:
    var trip = TripFFI()
    var depth = 1.75

    var trip_dose = trip.ddd2_interp_col(
        1.0, 10.0, 2.0, 0.2, 4.0,
        3.0, 18.0, 6.0, 0.6, 8.0,
        depth, 1,
    )
    assert_close(trip_dose, mojo_interp(10.0, 18.0, depth), 1.0e-12)

    var trip_fwhm1 = trip.ddd2_interp_col(
        1.0, 10.0, 2.0, 0.2, 4.0,
        3.0, 18.0, 6.0, 0.6, 8.0,
        depth, 2,
    )
    assert_close(trip_fwhm1, mojo_interp(2.0, 6.0, depth), 1.0e-12)

    var trip_mix = trip.ddd2_interp_col(
        1.0, 10.0, 2.0, 0.2, 4.0,
        3.0, 18.0, 6.0, 0.6, 8.0,
        depth, 3,
    )
    assert_close(trip_mix, mojo_interp(0.2, 0.6, depth), 1.0e-12)

    var trip_fwhm2 = trip.ddd2_interp_col(
        1.0, 10.0, 2.0, 0.2, 4.0,
        3.0, 18.0, 6.0, 0.6, 8.0,
        depth, 4,
    )
    assert_close(trip_fwhm2, mojo_interp(4.0, 8.0, depth), 1.0e-12)
