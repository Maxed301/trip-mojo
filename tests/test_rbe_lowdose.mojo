from dedx import read_dedx
from rbe_lowdose import find_rbe_projectile, low_dose_values_for_projectile, read_rbe_lowdose
from std.testing import assert_equal


def assert_close(got: Float64, expected: Float64, tol: Float64) raises:
    if abs(got - expected) > tol:
        raise Error("value mismatch")


def main() raises:
    var rbe = read_rbe_lowdose("/home/max/Projects/TRIP_DATA/Basedata/RBE/chordom02.rbe")
    assert_close(rbe.alpha, 0.1, 1e-12)
    assert_close(rbe.beta, 0.05, 1e-12)
    assert_close(rbe.cut_gy, 30.0, 1e-12)
    assert_close(rbe.nucleus_radius_um, 5.0, 1e-12)
    assert_close(rbe.nucleus_area_um2, 78.53981633974483, 1e-12)
    assert_close(rbe.slope_max, 3.1, 1e-12)
    assert_equal(len(rbe.projectiles), 23)

    var carbon = find_rbe_projectile(rbe, 6, 12)
    assert_equal(carbon.projectile, "12C")
    assert_equal(len(carbon.energy), 40)
    assert_close(carbon.energy[0], 0.126, 1e-12)
    assert_close(carbon.rbe[0], 30.858, 1e-12)
    assert_close(carbon.energy[20], 12.6, 1e-12)
    assert_close(carbon.rbe[20], 15.562, 1e-12)

    var dedx = read_dedx("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/20040607.dedx")
    var low0 = low_dose_values_for_projectile(rbe, dedx, 6, 12, 0)
    assert_close(low0.alpha_low_dose, 0.615075705465564, 1e-12)
    assert_close(low0.beta_low_dose, 9.402832319220746e-06, 1e-18)
    assert_close(low0.sqrt_beta_low_dose, 0.003066403808897443, 1e-15)
    var low20 = low_dose_values_for_projectile(rbe, dedx, 6, 12, 20)
    assert_close(low20.alpha_low_dose, 1.2664953573025763, 1e-12)
    assert_close(low20.beta_low_dose, 0.01704182763557329, 1e-15)
    assert_close(low20.sqrt_beta_low_dose, 0.13054435122046948, 1e-15)
