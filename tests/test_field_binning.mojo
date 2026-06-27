from std.testing import assert_true

from geometry import Vec3, patient_to_gantry_matrix, transform_point
from phys_dose import (
    curve_index_for_energy,
    gaussian_support_fwhm_factor,
    load_curves,
    physical_field_replay,
    spot_voxel_contribution,
    trip_8ln2,
)
from rst import RasterSpot


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var points = List[Vec3]()
    points.append(Vec3(192.87849828600883, 193.8550982773304, 157.5))
    points.append(Vec3(193.8550982773304, 193.8550982773304, 166.5))
    points.append(Vec3(194.83169826865196, 193.8550982773304, 166.5))
    points.append(Vec3(195.80829825997353, 193.8550982773304, 166.5))

    var h2o = List[Float64]()
    h2o.append(42.594128392349546)
    h2o.append(42.650671234616887)
    h2o.append(42.485506525985826)
    h2o.append(42.453117475890792)

    var spots = List[RasterSpot]()
    spots.append(RasterSpot(0.0, 0.0, 280.48, 100000000.0, 7.5))
    spots.append(RasterSpot(-10.0, 0.0, 280.48, 100000000.0, 7.5))
    spots.append(RasterSpot(14.0, -8.0, 280.48, 25000000.0, 7.5))

    var binned = physical_field_replay(points, spots, h2o, 280.0)
    var brute = brute_force(points, spots, h2o)
    for i in range(len(points)):
        assert_close(binned[i], brute[i], 1.0e-12)


def brute_force(points: List[Vec3], spots: List[RasterSpot], h2o: List[Float64]) raises -> List[Float64]:
    var isocenter = Vec3(220.223, 317.883, 162.0)
    var matrix = patient_to_gantry_matrix(isocenter, 280.0, 90.0)
    var curves = load_curves(spots, h2o, False)
    var out = List[Float64]()
    for voxel in range(len(points)):
        var p = transform_point(matrix, points[voxel])
        var value = 0.0
        for spot in range(len(spots)):
            value += spot_voxel_contribution(
                voxel,
                p,
                spots[spot],
                curves[curve_index_for_energy(curves, spots[spot].energy)],
                trip_8ln2(),
                0.15915494309189535,
                gaussian_support_fwhm_factor() * gaussian_support_fwhm_factor(),
                1.0,
                1.0,
                False,
            )
        out.append(value)
    return out^
