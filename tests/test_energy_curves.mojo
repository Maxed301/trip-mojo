from std.testing import assert_equal, assert_true

from ddd import interp, load_ddd
from phys_dose import curve_index_for_energy, load_curves
from rst import RasterSpot


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var spots = List[RasterSpot]()
    spots.append(RasterSpot(0.0, 0.0, 280.48, 100.0, 7.5))
    spots.append(RasterSpot(2.0, 0.0, 280.48, 200.0, 7.5))
    spots.append(RasterSpot(4.0, 2.0, 303.14, 300.0, 7.6))

    var h2o = List[Float64]()
    h2o.append(42.594128392349546)
    h2o.append(264.88578510717286)

    var curves = load_curves(spots, h2o, False)
    assert_equal(len(curves), 2)
    assert_equal(curve_index_for_energy(curves, 280.48), 0)
    assert_equal(curve_index_for_energy(curves, 303.14), 1)

    var ddd280 = load_ddd(280.48)
    assert_close(curves[0].voxel_dose[0], interp(ddd280.z, ddd280.dose, h2o[0] * 0.1), 1.0e-12)
    assert_close(curves[0].voxel_fwhm1[0], interp(ddd280.z, ddd280.fwhm1, h2o[0] * 0.1), 1.0e-12)
    assert_close(curves[0].voxel_mix[0], interp(ddd280.z, ddd280.mix, h2o[0] * 0.1), 1.0e-12)
    assert_close(curves[0].voxel_fwhm2[0], interp(ddd280.z, ddd280.fwhm2, h2o[0] * 0.1), 1.0e-12)

    var ddd303 = load_ddd(303.14)
    assert_close(curves[1].voxel_dose[1], interp(ddd303.z, ddd303.dose, h2o[1] * 0.1), 1.0e-12)
    assert_close(curves[1].voxel_fwhm1[1], interp(ddd303.z, ddd303.fwhm1, h2o[1] * 0.1), 1.0e-12)
    assert_close(curves[1].voxel_mix[1], interp(ddd303.z, ddd303.mix, h2o[1] * 0.1), 1.0e-12)
    assert_close(curves[1].voxel_fwhm2[1], interp(ddd303.z, ddd303.fwhm2, h2o[1] * 0.1), 1.0e-12)

    assert_true(curves[0].max_fwhm2 >= curves[0].voxel_fwhm1[0] * curves[0].voxel_fwhm1[0])
    assert_true(curves[0].max_fwhm2 >= curves[0].voxel_fwhm2[0] * curves[0].voxel_fwhm2[0])
    assert_true(curves[1].max_fwhm2 >= curves[1].voxel_fwhm1[1] * curves[1].voxel_fwhm1[1])
    assert_true(curves[1].max_fwhm2 >= curves[1].voxel_fwhm2[1] * curves[1].voxel_fwhm2[1])
