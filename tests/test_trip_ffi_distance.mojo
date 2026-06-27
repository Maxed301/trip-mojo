from std.testing import assert_true

from trip_ffi import TripFFI


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var trip = TripFFI()

    var dr2 = trip.dose_distance_dr2(0.0, 0.0, 64.0, 3.0, 4.0, 0.0, 0.0)
    assert_close(dr2, 25.0, 0.0)

    dr2 = trip.dose_distance_dr2(0.0, 0.0, 64.0, 9.0, 0.0, 0.0, 0.0)
    assert_close(dr2, -1.0, 0.0)

    dr2 = trip.dose_distance_dr2(10.0, -5.0, 100.0, 11.0, -4.0, 0.2, -0.1)
    assert_close(dr2, 11.25, 1.0e-12)
