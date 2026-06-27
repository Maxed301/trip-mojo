from ct_ray import CTVolume, grid_line_water_depth_mm
from ddd import interp, load_ddd
from geometry import Vec3, gantry_to_patient_point
from hlut import HLUT
from rst import RasterSpot


def annotate_spot_f2max(
    mut spots: List[RasterSpot],
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    gantry: Float64,
    couch: Float64,
    dose_extension: Float64,
    off_h2o_mm: Float64,
) raises:
    var origin0 = gantry_to_patient_point(isocenter, gantry, couch, Vec3(0.0, 0.0, 0.0))
    var origin1 = gantry_to_patient_point(isocenter, gantry, couch, Vec3(0.0, 0.0, 1.0))
    var forward = Vec3(origin1.x - origin0.x, origin1.y - origin0.y, origin1.z - origin0.z)
    for i in range(len(spots)):
        var p = gantry_to_patient_point(isocenter, gantry, couch, Vec3(spots[i].x, spots[i].y, 0.0))
        var total_h2o_mm = grid_line_water_depth_mm(ct, hlut, p, forward) + off_h2o_mm
        spots[i].f2_max = spot_f2max_from_distal_h2o(spots[i], total_h2o_mm, dose_extension)


def spot_f2max_from_distal_h2o(spot: RasterSpot, distal_h2o_mm: Float64, dose_extension: Float64) raises -> Float64:
    var fmax = spot.focus * dose_extension
    var out = fmax * fmax
    var curve = load_ddd(spot.energy)
    var depth_cm = distal_h2o_mm * 0.1
    var fwhm1 = interp(curve.z, curve.fwhm1, depth_cm)
    var fwhm2 = interp(curve.z, curve.fwhm2, depth_cm)
    var scatter = fwhm1
    if fwhm2 > scatter:
        scatter = fwhm2
    if scatter > 0.0:
        var dd = scatter * dose_extension
        out += dd * dd
    return out
