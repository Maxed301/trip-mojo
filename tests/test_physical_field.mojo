from std.testing import assert_true

from geometry import Vec3
from phys_dose import physical_field_replay_with_dose_extension
from rst import RasterSpot


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    # One voxel and one centered spot from TRIP_DEBUG_DOSE_DIRECT_CSV.
    var points = List[Vec3]()
    points.append(Vec3(192.87849828600883, 193.8550982773304, 157.5))
    var h2o = List[Float64]()
    h2o.append(42.594128392349546)
    var spots = List[RasterSpot]()
    spots.append(RasterSpot(0.0, 0.0, 280.48, 100000000.0, 7.5))
    var dose = physical_field_replay_with_dose_extension(points, spots, h2o, 280.0, 1.0)
    assert_close(dose[0], 0.29557418823465553, 1.0e-12)

    # Same path for an off-axis spot, also from TRIP_DEBUG_DOSE_DIRECT_CSV.
    points = List[Vec3]()
    points.append(Vec3(193.8550982773304, 193.8550982773304, 166.5))
    h2o = List[Float64]()
    h2o.append(42.650671234616887)
    spots = List[RasterSpot]()
    spots.append(RasterSpot(-10.0, 0.0, 280.48, 100000000.0, 7.5))
    dose = physical_field_replay_with_dose_extension(points, spots, h2o, 280.0, 1.0)
    assert_close(dose[0], 0.31056675714741161, 1.0e-12)

    # Accumulation is linear for independent spots at a fixed voxel.
    points = List[Vec3]()
    points.append(Vec3(192.87849828600883, 193.8550982773304, 157.5))
    h2o = List[Float64]()
    h2o.append(42.594128392349546)
    spots = List[RasterSpot]()
    spots.append(RasterSpot(0.0, 0.0, 280.48, 100000000.0, 7.5))
    spots.append(RasterSpot(0.0, 0.0, 280.48, 50000000.0, 7.5))
    dose = physical_field_replay_with_dose_extension(points, spots, h2o, 280.0, 1.0)
    assert_close(dose[0], 0.44336128235088024, 1.0e-12)

    # TRiP dose loops ignore raster points without TRPRSTPNT_INSIDE.
    points = List[Vec3]()
    points.append(Vec3(192.87849828600883, 193.8550982773304, 157.5))
    h2o = List[Float64]()
    h2o.append(42.594128392349546)
    spots = List[RasterSpot]()
    spots.append(RasterSpot(0.0, 0.0, 280.48, 100000000.0, 7.5, 0.0, 0.0, False, 0))
    dose = physical_field_replay_with_dose_extension(points, spots, h2o, 280.0, 1.0)
    assert_close(dose[0], 0.0, 1.0e-15)

    # Multiple voxels from the same off-axis TRiP debug run.
    points = List[Vec3]()
    h2o = List[Float64]()
    points.append(Vec3(193.8550982773304, 193.8550982773304, 166.5))
    h2o.append(42.650671234616887)
    points.append(Vec3(194.83169826865196, 193.8550982773304, 166.5))
    h2o.append(42.485506525985826)
    points.append(Vec3(191.90189829468727, 193.8550982773304, 169.5))
    h2o.append(42.770455573596706)
    points.append(Vec3(195.80829825997353, 193.8550982773304, 166.5))
    h2o.append(42.453117475890792)
    points.append(Vec3(192.87849828600883, 193.8550982773304, 169.5))
    h2o.append(42.645581374924312)
    spots = List[RasterSpot]()
    spots.append(RasterSpot(-10.0, 0.0, 280.48, 100000000.0, 7.5))
    dose = physical_field_replay_with_dose_extension(points, spots, h2o, 280.0, 1.0)
    assert_close(dose[0], 0.31056675714741161, 1.0e-12)
    assert_close(dose[1], 0.4511748030937616, 1.0e-12)
    assert_close(dose[2], 0.34951642808476691, 1.0e-12)
    assert_close(dose[3], 0.5985210509161764, 1.0e-12)
    assert_close(dose[4], 0.6092073879715779, 1.0e-12)

    # Gantry 325 row from TRIP_DEBUG_DOSE_DIRECT_CSV.
    points = List[Vec3]()
    points.append(Vec3(382.3388966023922, 402.84749642014503, 103.5))
    h2o = List[Float64]()
    h2o.append(261.18830502400516)
    spots = List[RasterSpot]()
    spots.append(RasterSpot(0.0, 0.0, 280.48, 100000000.0, 7.5))
    dose = physical_field_replay_with_dose_extension(points, spots, h2o, 325.0, 1.0)
    assert_close(dose[0], 0.00021430347458583493, 1.0e-15)

    # Actual field-2 first-spot energy/focus from the P101 RST.
    points = List[Vec3]()
    points.append(Vec3(383.31549659371376, 397.96449646353722, 112.5))
    h2o = List[Float64]()
    h2o.append(264.88578510717286)
    spots = List[RasterSpot]()
    spots.append(RasterSpot(4.0, 2.0, 303.14, 100000000.0, 7.6))
    dose = physical_field_replay_with_dose_extension(points, spots, h2o, 325.0, 1.0)
    assert_close(dose[0], 0.0003955877625178419, 1.0e-15)

    # Real contribution from a simple one-field P101 spot.
    points = List[Vec3]()
    points.append(Vec3(265.14689764380455, 388.19849655032158, 55.5))
    h2o = List[Float64]()
    h2o.append(243.21072461033373)
    spots = List[RasterSpot]()
    spots.append(RasterSpot(24.0, 16.0, 252.4, 650202.0, 7.2))
    dose = physical_field_replay_with_dose_extension(points, spots, h2o, 280.0, 1.2)
    assert_close(dose[0], 2.4744541419759494e-07, 1.0e-18)
