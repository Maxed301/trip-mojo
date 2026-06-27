from std.testing import assert_true

from f2max import spot_f2max_from_distal_h2o
from rst import RasterSpot


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    # TRiP source behavior: dF2Max includes dose-extension-scaled focus.
    # If distal WET is outside the DDD table, no scatter term is added.
    var low_energy = RasterSpot(16.0, -6.0, 201.62, 1.0, 6.7, 0.0, 0.0, True, 348)
    assert_close(spot_f2max_from_distal_h2o(low_energy, 194.35641238224309, 1.2), 64.64159999999998, 1.0e-12)

    var in_range = RasterSpot(26.0, -12.0, 263.83, 1.0, 10.0, 0.0, 0.0, True, 276)
    assert_true(spot_f2max_from_distal_h2o(in_range, 189.69010405102355, 1.2) > 144.0)
