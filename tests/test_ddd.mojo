from std.math import exp
from std.testing import assert_true

from ddd import interp, load_ddd
from phys_dose import trip_8ln2


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var exact = load_ddd(280.0)
    assert_close(exact.z[0], 0.0, 0.0)
    assert_close(exact.dose[0], 130.4555, 1.0e-12)
    assert_close(exact.fwhm1[0], 0.0009982957, 1.0e-15)
    assert_close(exact.mix[0], 0.005519712, 1.0e-15)
    assert_close(exact.fwhm2[0], 0.001750047, 1.0e-15)
    assert_close(interp(exact.z, exact.dose, 1.720897), 131.0505, 1.0e-12)

    var curve = load_ddd(280.48)
    var depth_cm = 42.594128392349546 * 0.1
    var dose = interp(curve.z, curve.dose, depth_cm)
    var fwhm1 = interp(curve.z, curve.fwhm1, depth_cm)
    var mix = interp(curve.z, curve.mix, depth_cm)
    var fwhm2 = interp(curve.z, curve.fwhm2, depth_cm)

    assert_close(dose, 133.37042039724756, 1.0e-10)
    assert_close(fwhm1, 0.27192792099772534, 1.0e-12)
    assert_close(mix, 0.17821469253398028, 1.0e-12)
    assert_close(fwhm2, 0.49240374711694235, 1.0e-12)

    var simple_curve = load_ddd(252.4)
    var simple_depth_cm = 243.21072461033373 * 0.1
    assert_close(interp(simple_curve.z, simple_curve.dose, simple_depth_cm), 10.466597605763472, 1.0e-10)
    assert_close(interp(simple_curve.z, simple_curve.fwhm1, simple_depth_cm), 69.68779754525299, 1.0e-10)
    assert_close(interp(simple_curve.z, simple_curve.mix, simple_depth_cm), 0.33272728772621407, 1.0e-12)
    assert_close(interp(simple_curve.z, simple_curve.fwhm2, simple_depth_cm), 24.392257697485988, 1.0e-10)

    var focus = 7.5
    var dr2 = 49.322134592249
    var inv1 = trip_8ln2() * 0.5 / (focus * focus + fwhm1 * fwhm1)
    var g = exp(-dr2 * inv1) * inv1 / 3.141592653589793 * (1.0 - mix)
    var inv2 = trip_8ln2() * 0.5 / (focus * focus + fwhm2 * fwhm2)
    g += exp(-dr2 * inv2) * inv2 / 3.141592653589793 * mix
    assert_close(g, 0.0013832263059482928, 1.0e-14)
