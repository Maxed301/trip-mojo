from dedx import dedx_stopping_power, find_projectile, parse_projectile, read_dedx
from std.testing import assert_equal


def assert_close(got: Float64, expected: Float64, tol: Float64) raises:
    if abs(got - expected) > tol:
        raise Error("value mismatch")


def main() raises:
    var h = parse_projectile("1H")
    assert_equal(h.atomic_number, 1)
    assert_equal(h.mass_number, 1)
    var c = parse_projectile("12C")
    assert_equal(c.atomic_number, 6)
    assert_equal(c.mass_number, 12)

    var table = read_dedx("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/20040607.dedx")
    assert_equal(table.material, "H2O")
    assert_close(table.density, 1.0, 1e-12)

    var proton = find_projectile(table, 1, 1)
    assert_equal(proton.projectile, "1H")
    assert_equal(len(proton.energy), 66)
    assert_close(proton.energy[0], 0.01, 1e-12)
    assert_close(proton.stopping_power[0], 745.06, 1e-12)
    assert_close(proton.range[0], 1.8968e-05, 1e-16)
    assert_close(proton.energy[len(proton.energy) - 1], 8000.0, 1e-12)
    assert_close(proton.stopping_power[len(proton.stopping_power) - 1], 2.1162, 1e-12)

    var carbon = find_projectile(table, 6, 12)
    assert_equal(carbon.projectile, "12C")
    assert_equal(len(carbon.energy), 66)
    assert_close(carbon.energy[0], 0.01, 1e-12)
    assert_close(carbon.stopping_power[0], 4122.8, 1e-12)
    assert_close(carbon.range[0], 4.6837e-05, 1e-16)
    assert_close(carbon.energy[51], 500.0, 1e-12)
    assert_close(carbon.stopping_power[51], 98.14, 1e-12)
    assert_close(dedx_stopping_power(proton, 300.0), 3.4993, 1e-8)
    assert_close(dedx_stopping_power(carbon, 300.0), 125.79, 1e-8)
