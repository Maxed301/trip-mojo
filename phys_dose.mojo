from std.algorithm import parallelize
from std.math import exp, log, sqrt

from bio_coeff import BioDoseAggregates, empty_bio_dose_aggregates, finalize_low_dose_bio
from ddd import DDDCurve, interp, interp_clamped, load_ddd
from dedx import DEDXProjectile, DEDXTable, read_dedx
from geometry import Vec3, patient_to_gantry_matrix, transform_point
from rbe import chordoma_rbe_12c, rbe_at_energy, residual_energy_from_depth
from rbe_lowdose import RBEProjectile, RBELowDoseTable, read_rbe_lowdose
from rst import RasterSpot
from spc import SPCBioCoefficients, depth_bio_coefficients, interpolate_spc_bio_coefficients, read_spc_beam, read_spc_peak_cm


@fieldwise_init
struct EnergyCurve(Copyable, Movable):
    var energy: Float64
    var curve: DDDCurve
    var voxel_dose: List[Float64]
    var voxel_fwhm1: List[Float64]
    var voxel_mix: List[Float64]
    var voxel_fwhm2: List[Float64]
    var voxel_bio_factor: List[Float64]
    var voxel_alpha: List[Float64]
    var voxel_sqrt_beta: List[Float64]
    var voxel_let_mix: List[Float64]
    var voxel_let_bar: List[Float64]
    var max_fwhm2: Float64


@fieldwise_init
struct BioCoefficientTable(Copyable, Movable):
    var spectrum_code: Int
    var peak_cm: Float64
    var coeffs: List[SPCBioCoefficients]


def compute_physical_field_dose(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
) raises -> List[Float64]:
    return compute_physical_field_dose_with_support(points, spots, h2o, gantry, isocenter, gaussian_support_fwhm_factor())


def compute_physical_field_dose_with_support(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    dose_extension: Float64,
) raises -> List[Float64]:
    var dose = List[Float64]()
    for _ in range(len(points)):
        dose.append(0.0)
    accumulate_physical_field_with_dose_extension(dose, points, spots, h2o, gantry, isocenter, dose_extension)
    return dose^


def biological_field_aggregates_with_rbe_path(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    rbe_path: String,
) raises -> List[BioDoseAggregates]:
    var aggregates = List[BioDoseAggregates]()
    for _ in range(len(points)):
        aggregates.append(empty_bio_dose_aggregates())
    accumulate_biological_field(aggregates, points, spots, h2o, gantry, isocenter, gaussian_support_fwhm_factor(), rbe_path)
    return aggregates^


def lq_dose_from_effect(effect: Float64) -> Float64:
    var alpha = 0.1
    var beta = 0.05
    if effect <= 0.0:
        return 0.0
    var alpha_equivalent_dose = effect / alpha
    var mixed_effect = effect + 0.2 * beta * alpha_equivalent_dose * alpha_equivalent_dose
    return (-alpha + sqrt(alpha * alpha + 4.0 * beta * mixed_effect)) / (2.0 * beta)


def accumulate_physical_field(
    mut dose: List[Float64],
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
) raises:
    accumulate_physical_field_with_dose_extension(dose, points, spots, h2o_depth_mm, gantry, isocenter, gaussian_support_fwhm_factor())


def accumulate_physical_field_with_dose_extension(
    mut dose: List[Float64],
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    dose_extension: Float64,
) raises:
    accumulate_field(dose, points, spots, h2o_depth_mm, gantry, isocenter, False, dose_extension)


def accumulate_biological_field(
    mut aggregates: List[BioDoseAggregates],
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    dose_extension: Float64,
    rbe_path: String,
) raises:
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    var gantry_points = List[Vec3]()
    for i in range(len(points)):
        gantry_points.append(transform_point(matrix, points[i]))

    var curves = load_curves_with_rbe_path(spots, h2o_depth_mm, True, rbe_path)
    var eight_ln2 = trip_8ln2()
    var half_over_pi = 0.15915494309189535
    var dose_extension2 = dose_extension * dose_extension
    var spot_curve_indices = List[Int]()
    for spot_index in range(len(spots)):
        spot_curve_indices.append(curve_index_for_energy(curves, spots[spot_index].energy))
    var bin_size = 10.0
    var min_x = spots[0].x
    var max_x = spots[0].x
    var min_y = spots[0].y
    var max_y = spots[0].y
    var global_radius = 0.0
    for spot_index in range(len(spots)):
        if spots[spot_index].x < min_x:
            min_x = spots[spot_index].x
        if spots[spot_index].x > max_x:
            max_x = spots[spot_index].x
        if spots[spot_index].y < min_y:
            min_y = spots[spot_index].y
        if spots[spot_index].y > max_y:
            max_y = spots[spot_index].y
        var curve_index = spot_curve_indices[spot_index]
        var radius = sqrt(dose_extension2 * (spots[spot_index].focus * spots[spot_index].focus + curves[curve_index].max_fwhm2))
        if radius > global_radius:
            global_radius = radius
    global_radius += 10.0
    min_x -= global_radius
    min_y -= global_radius
    max_x += global_radius
    max_y += global_radius
    var bins_x = Int((max_x - min_x) / bin_size) + 1
    var bins_y = Int((max_y - min_y) / bin_size) + 1
    var spot_bins = List[List[Int]]()
    for _ in range(bins_x * bins_y):
        spot_bins.append(List[Int]())
    for spot_index in range(len(spots)):
        var bx = Int((spots[spot_index].x - min_x) / bin_size)
        var by = Int((spots[spot_index].y - min_y) / bin_size)
        spot_bins[by * bins_x + bx].append(spot_index)
    var bin_radius = Int(global_radius / bin_size) + 2

    var num_workers = 12

    @parameter
    def accumulate_bio_voxel(voxel: Int):
        var p = gantry_points[voxel].copy()
        var agg = empty_bio_dose_aggregates()
        var centre_bx = Int((p.x - min_x) / bin_size)
        var centre_by = Int((p.y - min_y) / bin_size)
        var bx0 = centre_bx - bin_radius
        var bx1 = centre_bx + bin_radius
        var by0 = centre_by - bin_radius
        var by1 = centre_by + bin_radius
        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= bins_x:
            bx1 = bins_x - 1
        if by1 >= bins_y:
            by1 = bins_y - 1
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                var bin_index = by * bins_x + bx
                for bin_item in range(len(spot_bins[bin_index])):
                    var spot_index = spot_bins[bin_index][bin_item]
                    var bio = spot_voxel_bio_contribution(
                        voxel,
                        p,
                        spots[spot_index],
                        curves[spot_curve_indices[spot_index]],
                        eight_ln2,
                        half_over_pi,
                        dose_extension2,
                    )
                    agg.absorbed_mev_per_g += bio.absorbed_mev_per_g
                    agg.alpha += bio.alpha
                    agg.sqrt_beta += bio.sqrt_beta
                    agg.let_mix += bio.let_mix
                    agg.let_bar += bio.let_bar
        aggregates[voxel] = agg^

    parallelize[accumulate_bio_voxel](len(points), num_workers)

def accumulate_field(
    mut dose: List[Float64],
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    biological: Bool,
    dose_extension: Float64,
) raises:
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    var gantry_points = List[Vec3]()
    for i in range(len(points)):
        gantry_points.append(transform_point(matrix, points[i]))

    var curves = load_curves(spots, h2o_depth_mm, biological)
    var eight_ln2 = trip_8ln2()
    var half_over_pi = 0.15915494309189535
    var dose_extension2 = dose_extension * dose_extension
    var focus_scale = 1.0
    var divergence_scale = 1.0
    var num_workers = 12
    var spot_curve_indices = List[Int]()
    for spot_index in range(len(spots)):
        spot_curve_indices.append(curve_index_for_energy(curves, spots[spot_index].energy))
    var bin_size = 10.0
    var min_x = spots[0].x
    var max_x = spots[0].x
    var min_y = spots[0].y
    var max_y = spots[0].y
    var global_radius = 0.0
    for spot_index in range(len(spots)):
        var spot = spots[spot_index].copy()
        if spot.x < min_x:
            min_x = spot.x
        if spot.x > max_x:
            max_x = spot.x
        if spot.y < min_y:
            min_y = spot.y
        if spot.y > max_y:
            max_y = spot.y
        var curve_index = spot_curve_indices[spot_index]
        var radius = sqrt(dose_extension2 * (spot.focus * spot.focus + curves[curve_index].max_fwhm2))
        if radius > global_radius:
            global_radius = radius
    global_radius += 10.0
    min_x -= global_radius
    min_y -= global_radius
    max_x += global_radius
    max_y += global_radius
    var bins_x = Int((max_x - min_x) / bin_size) + 1
    var bins_y = Int((max_y - min_y) / bin_size) + 1
    var spot_bins = List[List[Int]]()
    for _ in range(bins_x * bins_y):
        spot_bins.append(List[Int]())
    for spot_index in range(len(spots)):
        var bx = Int((spots[spot_index].x - min_x) / bin_size)
        var by = Int((spots[spot_index].y - min_y) / bin_size)
        spot_bins[by * bins_x + bx].append(spot_index)
    var bin_radius = Int(global_radius / bin_size) + 2

    @parameter
    def accumulate_voxel(voxel: Int):
        var value = 0.0
        var p = gantry_points[voxel].copy()
        var centre_bx = Int((p.x - min_x) / bin_size)
        var centre_by = Int((p.y - min_y) / bin_size)
        var bx0 = centre_bx - bin_radius
        var bx1 = centre_bx + bin_radius
        var by0 = centre_by - bin_radius
        var by1 = centre_by + bin_radius
        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= bins_x:
            bx1 = bins_x - 1
        if by1 >= bins_y:
            by1 = bins_y - 1
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                var bin_index = by * bins_x + bx
                for bin_item in range(len(spot_bins[bin_index])):
                    var spot_index = spot_bins[bin_index][bin_item]
                    value += spot_voxel_contribution(
                        voxel,
                        p,
                        spots[spot_index],
                        curves[spot_curve_indices[spot_index]],
                        eight_ln2,
                        half_over_pi,
                        dose_extension2,
                        focus_scale,
                        divergence_scale,
                        biological,
                    )
        dose[voxel] += value

    parallelize[accumulate_voxel](len(points), num_workers)


def spot_voxel_contribution(
    voxel: Int,
    p: Vec3,
    spot: RasterSpot,
    curve: EnergyCurve,
    eight_ln2: Float64,
    half_over_pi: Float64,
    dose_extension2: Float64,
    focus_scale: Float64,
    divergence_scale: Float64,
    biological: Bool,
) -> Float64:
    if not spot.active:
        return 0.0
    var focus = spot.focus * focus_scale
    var focus2 = focus * focus
    var div_x = 1.0 - divergence_scale * p.z / 8832.0
    var div_y = 1.0 - divergence_scale * p.z / 7806.0
    var dx = p.x - spot.x * div_x
    var dy = p.y - spot.y * div_y
    var max_f2 = focus2 + curve.max_fwhm2
    var axis_f2_max = dose_extension2 * max_f2
    var radial_f2_max = axis_f2_max
    if spot.f2_max > 0.0:
        axis_f2_max = spot.f2_max
        radial_f2_max = dose_extension2 * spot.f2_max
    if dx * dx > axis_f2_max:
        return 0.0
    if dy * dy > axis_f2_max:
        return 0.0
    var r2 = dx * dx + dy * dy
    if r2 >= radial_f2_max:
        return 0.0

    var ddd_d = curve.voxel_dose[voxel]
    if ddd_d == 0.0:
        return 0.0

    var fwhm1 = curve.voxel_fwhm1[voxel]
    var mix = curve.voxel_mix[voxel]
    var fwhm2 = curve.voxel_fwhm2[voxel]

    var f2_a = focus2 + fwhm1 * fwhm1
    var g_a = 0.0
    if r2 < dose_extension2 * f2_a:
        var isig2_a = eight_ln2 / f2_a
        g_a = exp(-0.5 * r2 * isig2_a) * isig2_a * half_over_pi * (1.0 - mix)

    var f2_b = focus2 + fwhm2 * fwhm2
    var g_b = 0.0
    if mix > 0.0 and r2 < dose_extension2 * f2_b:
        var isig2_b = eight_ln2 / f2_b
        g_b = exp(-0.5 * r2 * isig2_b) * isig2_b * half_over_pi * mix

    if g_a > 0.0 or g_b > 0.0:
        var contribution = ddd_d * (g_a + g_b) * spot.particles * mev_per_g_from_mm2_fluence_to_gy()
        if biological:
            return contribution * curve.voxel_bio_factor[voxel]
        return contribution
    return 0.0


def spot_voxel_optimizer_contribution(
    voxel: Int,
    p: Vec3,
    spot: RasterSpot,
    curve: EnergyCurve,
    eight_ln2: Float64,
    half_over_pi: Float64,
    dose_extension2: Float64,
    focus_scale: Float64,
    divergence_scale: Float64,
    biological: Bool,
) -> Float64:
    if not spot.active:
        return 0.0
    var focus = spot.focus * focus_scale
    var focus2 = focus * focus
    var div_x = 1.0 - divergence_scale * p.z / 8832.0
    var div_y = 1.0 - divergence_scale * p.z / 7806.0
    var dx = p.x - spot.x * div_x
    var dy = p.y - spot.y * div_y
    var max_f2 = focus2 + curve.max_fwhm2
    var axis_f2_max = dose_extension2 * max_f2
    if spot.f2_max > 0.0:
        axis_f2_max = spot.f2_max
    if dx * dx > axis_f2_max:
        return 0.0
    if dy * dy > axis_f2_max:
        return 0.0
    var r2 = dx * dx + dy * dy

    var ddd_d = curve.voxel_dose[voxel]
    if ddd_d == 0.0:
        return 0.0

    var fwhm1 = curve.voxel_fwhm1[voxel]
    var mix = curve.voxel_mix[voxel]
    var fwhm2 = curve.voxel_fwhm2[voxel]

    var f2_a = focus2 + fwhm1 * fwhm1
    var g_a = 0.0
    if r2 < dose_extension2 * f2_a:
        var isig2_a = eight_ln2 / f2_a
        g_a = exp(-0.5 * r2 * isig2_a) * isig2_a * half_over_pi * (1.0 - mix)

    var f2_b = focus2 + fwhm2 * fwhm2
    var g_b = 0.0
    if mix > 0.0 and r2 < dose_extension2 * f2_b:
        var isig2_b = eight_ln2 / f2_b
        g_b = exp(-0.5 * r2 * isig2_b) * isig2_b * half_over_pi * mix

    if g_a > 0.0 or g_b > 0.0:
        var contribution = ddd_d * (g_a + g_b) * spot.particles * mev_per_g_from_mm2_fluence_to_gy()
        if biological:
            return contribution * curve.voxel_bio_factor[voxel]
        return contribution
    return 0.0


def spot_voxel_bio_contribution(
    voxel: Int,
    p: Vec3,
    spot: RasterSpot,
    curve: EnergyCurve,
    eight_ln2: Float64,
    half_over_pi: Float64,
    dose_extension2: Float64,
) -> BioDoseAggregates:
    if not spot.active:
        return empty_bio_dose_aggregates()
    var focus = spot.focus
    var focus2 = focus * focus
    var div_x = 1.0 - p.z / 8832.0
    var div_y = 1.0 - p.z / 7806.0
    var dx = p.x - spot.x * div_x
    var dy = p.y - spot.y * div_y
    var max_f2 = focus2 + curve.max_fwhm2
    var axis_f2_max = dose_extension2 * max_f2
    var radial_f2_max = axis_f2_max
    if spot.f2_max > 0.0:
        axis_f2_max = spot.f2_max
        radial_f2_max = dose_extension2 * spot.f2_max
    if dx * dx > axis_f2_max:
        return empty_bio_dose_aggregates()
    if dy * dy > axis_f2_max:
        return empty_bio_dose_aggregates()
    var r2 = dx * dx + dy * dy
    if r2 >= radial_f2_max:
        return empty_bio_dose_aggregates()

    var ddd_d = curve.voxel_dose[voxel]
    if ddd_d == 0.0:
        return empty_bio_dose_aggregates()

    var fwhm1 = curve.voxel_fwhm1[voxel]
    var mix = curve.voxel_mix[voxel]
    var fwhm2 = curve.voxel_fwhm2[voxel]

    var f2_a = focus2 + fwhm1 * fwhm1
    var g_a = 0.0
    if r2 < dose_extension2 * f2_a:
        var isig2_a = eight_ln2 / f2_a
        g_a = exp(-0.5 * r2 * isig2_a) * isig2_a * half_over_pi * (1.0 - mix)

    var f2_b = focus2 + fwhm2 * fwhm2
    var g_b = 0.0
    if mix > 0.0 and r2 < dose_extension2 * f2_b:
        var isig2_b = eight_ln2 / f2_b
        g_b = exp(-0.5 * r2 * isig2_b) * isig2_b * half_over_pi * mix

    var g = g_a + g_b
    if g <= 0.0:
        return empty_bio_dose_aggregates()
    var fluence = g * spot.particles
    return BioDoseAggregates(
        ddd_d * fluence,
        fluence * curve.voxel_alpha[voxel],
        fluence * curve.voxel_sqrt_beta[voxel],
        fluence * curve.voxel_let_mix[voxel],
        fluence * curve.voxel_let_bar[voxel],
    )


def spot_voxel_optimizer_bio_contribution(
    voxel: Int,
    p: Vec3,
    spot: RasterSpot,
    curve: EnergyCurve,
    eight_ln2: Float64,
    half_over_pi: Float64,
    dose_extension2: Float64,
) -> BioDoseAggregates:
    if not spot.active:
        return empty_bio_dose_aggregates()
    var focus = spot.focus
    var focus2 = focus * focus
    var div_x = 1.0 - p.z / 8832.0
    var div_y = 1.0 - p.z / 7806.0
    var dx = p.x - spot.x * div_x
    var dy = p.y - spot.y * div_y
    var max_f2 = focus2 + curve.max_fwhm2
    var axis_f2_max = dose_extension2 * max_f2
    if spot.f2_max > 0.0:
        axis_f2_max = spot.f2_max
    if dx * dx > axis_f2_max:
        return empty_bio_dose_aggregates()
    if dy * dy > axis_f2_max:
        return empty_bio_dose_aggregates()
    var r2 = dx * dx + dy * dy

    var ddd_d = curve.voxel_dose[voxel]
    if ddd_d == 0.0:
        return empty_bio_dose_aggregates()

    var fwhm1 = curve.voxel_fwhm1[voxel]
    var mix = curve.voxel_mix[voxel]
    var fwhm2 = curve.voxel_fwhm2[voxel]

    var f2_a = focus2 + fwhm1 * fwhm1
    var g_a = 0.0
    if r2 < dose_extension2 * f2_a:
        var isig2_a = eight_ln2 / f2_a
        g_a = exp(-0.5 * r2 * isig2_a) * isig2_a * half_over_pi * (1.0 - mix)

    var f2_b = focus2 + fwhm2 * fwhm2
    var g_b = 0.0
    if mix > 0.0 and r2 < dose_extension2 * f2_b:
        var isig2_b = eight_ln2 / f2_b
        g_b = exp(-0.5 * r2 * isig2_b) * isig2_b * half_over_pi * mix

    var g = g_a + g_b
    if g <= 0.0:
        return empty_bio_dose_aggregates()
    var fluence = g * spot.particles
    return BioDoseAggregates(
        ddd_d * fluence,
        fluence * curve.voxel_alpha[voxel],
        fluence * curve.voxel_sqrt_beta[voxel],
        fluence * curve.voxel_let_mix[voxel],
        fluence * curve.voxel_let_bar[voxel],
    )



def mev_per_g_from_mm2_fluence_to_gy() -> Float64:
    return 1.602189e-8


def load_curves(spots: List[RasterSpot], h2o_depth_mm: List[Float64], include_bio: Bool) raises -> List[EnergyCurve]:
    return load_curves_with_rbe_path(spots, h2o_depth_mm, include_bio, "")


def load_curves_with_rbe_path(
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    include_bio: Bool,
    rbe_path: String,
) raises -> List[EnergyCurve]:
    var curves = List[EnergyCurve]()
    var rbe = chordoma_rbe_12c()
    var bio_tables = List[BioCoefficientTable]()
    var dedx = DEDXTable("", 0.0, List[DEDXProjectile]())
    var rbe_low = RBELowDoseTable(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, List[RBEProjectile]())
    if include_bio:
        if rbe_path.byte_length() == 0:
            raise Error("biological curve setup requires an explicit RBE low-dose table path")
        dedx = read_dedx("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/20040607.dedx")
        rbe_low = read_rbe_lowdose(rbe_path)
    for spot in range(len(spots)):
        if not has_curve(curves, spots[spot].energy):
            var curve = load_ddd(spots[spot].energy)
            var voxel_dose = List[Float64]()
            var voxel_fwhm1 = List[Float64]()
            var voxel_mix = List[Float64]()
            var voxel_fwhm2 = List[Float64]()
            var voxel_bio_factor = List[Float64]()
            var voxel_alpha = List[Float64]()
            var voxel_sqrt_beta = List[Float64]()
            var voxel_let_mix = List[Float64]()
            var voxel_let_bar = List[Float64]()
            var max_fwhm2 = 0.0
            var range_cm = curve.z[0]
            var max_depth_dose = curve.dose[0]
            for depth_index in range(1, len(curve.z)):
                if curve.dose[depth_index] > max_depth_dose:
                    max_depth_dose = curve.dose[depth_index]
                    range_cm = curve.z[depth_index]
            var z_scale = 1.0
            var table_index = -1
            if include_bio:
                table_index = bio_table_index_for_spectrum_code(bio_tables, spc_spectrum_code_for_slice_energy(spots[spot].energy))
                if table_index < 0:
                    bio_tables.append(build_bio_coefficient_table(spots[spot].energy, dedx, rbe_low))
                    table_index = len(bio_tables) - 1
                var slice_peak_cm = spc_interpolated_peak_cm_for_slice_energy(spots[spot].energy)
                if slice_peak_cm > 0.0:
                    z_scale = 0.1 * bio_tables[table_index].peak_cm / slice_peak_cm
            for voxel in range(len(h2o_depth_mm)):
                var depth_cm = h2o_depth_mm[voxel] * 0.1
                voxel_dose.append(interp(curve.z, curve.dose, depth_cm))
                var fwhm1 = interp_clamped(curve.z, curve.fwhm1, depth_cm)
                var fwhm2 = interp_clamped(curve.z, curve.fwhm2, depth_cm)
                voxel_fwhm1.append(fwhm1)
                voxel_mix.append(interp_clamped(curve.z, curve.mix, depth_cm))
                voxel_fwhm2.append(fwhm2)
                var residual_energy = residual_energy_from_depth(spots[spot].energy, depth_cm, range_cm)
                voxel_bio_factor.append(rbe_at_energy(rbe, residual_energy))
                if include_bio:
                    var coeffs = spc_bio_coefficients_at_depth(bio_tables[table_index].coeffs, h2o_depth_mm[voxel] * z_scale)
                    voxel_alpha.append(coeffs.alpha0x1)
                    voxel_sqrt_beta.append(coeffs.sqrt_beta0x1)
                    voxel_let_mix.append(coeffs.let_mix)
                    voxel_let_bar.append(coeffs.let_bar)
                else:
                    voxel_alpha.append(0.0)
                    voxel_sqrt_beta.append(0.0)
                    voxel_let_mix.append(0.0)
                    voxel_let_bar.append(0.0)
                if fwhm1 * fwhm1 > max_fwhm2:
                    max_fwhm2 = fwhm1 * fwhm1
                if fwhm2 * fwhm2 > max_fwhm2:
                    max_fwhm2 = fwhm2 * fwhm2
            curves.append(EnergyCurve(
                energy=spots[spot].energy,
                curve=curve^,
                voxel_dose=voxel_dose^,
                voxel_fwhm1=voxel_fwhm1^,
                voxel_mix=voxel_mix^,
                voxel_fwhm2=voxel_fwhm2^,
                voxel_bio_factor=voxel_bio_factor^,
                voxel_alpha=voxel_alpha^,
                voxel_sqrt_beta=voxel_sqrt_beta^,
                voxel_let_mix=voxel_let_mix^,
                voxel_let_bar=voxel_let_bar^,
                max_fwhm2=max_fwhm2,
            ))
    return curves^


def build_bio_coefficient_table(
    energy: Float64,
    dedx: DEDXTable,
    rbe_low: RBELowDoseTable,
) raises -> BioCoefficientTable:
    var spc_beam = read_spc_beam(spc_path_for_energy(energy))
    var spc_bio = List[SPCBioCoefficients]()
    for depth_index in range(len(spc_beam.depths)):
        spc_bio.append(depth_bio_coefficients(spc_beam.depths[depth_index], dedx, rbe_low))
    return BioCoefficientTable(
        spc_spectrum_code_for_slice_energy(energy),
        spc_beam.peak_cm,
        spc_bio^,
    )


def bio_table_index_for_spectrum_code(tables: List[BioCoefficientTable], spectrum_code: Int) -> Int:
    for i in range(len(tables)):
        if tables[i].spectrum_code == spectrum_code:
            return i
    return -1


def spc_spectrum_code_for_slice_energy(energy: Float64) -> Int:
    var slice_code = Int(energy * 100.0 + 0.5)
    var available = List[Int]()
    available.append(19000)
    available.append(19500)
    for code in range(20000, 34000, 1000):
        available.append(code)
    var best = available[0]
    var best_delta = abs(slice_code - best)
    for i in range(1, len(available)):
        var delta = abs(slice_code - available[i])
        if delta < best_delta:
            best = available[i]
            best_delta = delta
    return best


def lower_spc_spectrum_code(slice_code: Int) -> Int:
    if slice_code <= 19000:
        return 19000
    if slice_code <= 19500:
        return 19000
    if slice_code <= 20000:
        return 19500
    if slice_code >= 33000:
        return 33000
    return ((slice_code - 20000) // 1000) * 1000 + 20000


def upper_spc_spectrum_code(slice_code: Int) -> Int:
    if slice_code <= 19000:
        return 19000
    if slice_code <= 19500:
        return 19500
    if slice_code <= 20000:
        return 20000
    if slice_code >= 33000:
        return 33000
    return lower_spc_spectrum_code(slice_code) + 1000


def spc_path_for_energy(energy: Float64) -> String:
    return spc_path_for_code(spc_spectrum_code_for_slice_energy(energy))


def spc_path_for_code(code: Int) -> String:
    return "/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/SPC/12C.H2O.MeV" + String(code) + ".spc"


def spc_interpolated_peak_cm_for_slice_energy(energy: Float64) raises -> Float64:
    var slice_code = Int(energy * 100.0 + 0.5)
    var lo = lower_spc_spectrum_code(slice_code)
    var hi = upper_spc_spectrum_code(slice_code)
    var lo_energy = Float64(lo) * 0.01
    var hi_energy = Float64(hi) * 0.01
    var lo_peak = read_spc_peak_cm(spc_path_for_code(lo))
    var hi_peak = read_spc_peak_cm(spc_path_for_code(hi))
    if hi == lo or hi_energy == lo_energy:
        return lo_peak
    var fraction = (energy - lo_energy) / (hi_energy - lo_energy)
    return exp(log(lo_peak) + fraction * (log(hi_peak) - log(lo_peak)))


def spc_bio_coefficients_at_depth(
    coeffs: List[SPCBioCoefficients],
    depth_cm: Float64,
) -> SPCBioCoefficients:
    if len(coeffs) == 0:
        return SPCBioCoefficients(0.0, 0.0, 0.0, 0.0, 0.0)
    if depth_cm < coeffs[0].depth_cm or depth_cm > coeffs[len(coeffs) - 1].depth_cm:
        return SPCBioCoefficients(depth_cm, 0.0, 0.0, 0.0, 0.0)
    var lo = 0
    var hi = len(coeffs) - 1
    var iz: Int
    while True:
        iz = (lo + hi) >> 1
        if iz == lo:
            break
        if depth_cm - coeffs[iz].depth_cm < 0.0:
            hi = iz
        else:
            lo = iz
    lo = iz
    hi = lo + 1
    if hi >= len(coeffs):
        hi = lo
    return interpolate_spc_bio_coefficients(coeffs[lo], coeffs[hi], depth_cm)


def gaussian_support_fwhm_factor() -> Float64:
    return 1.2


def trip_8ln2() -> Float64:
    return 0.6932 * 8.0


def has_curve(curves: List[EnergyCurve], energy: Float64) -> Bool:
    for i in range(len(curves)):
        if curves[i].energy == energy:
            return True
    return False


def curve_index_for_energy(curves: List[EnergyCurve], energy: Float64) -> Int:
    for i in range(len(curves)):
        if curves[i].energy == energy:
            return i
    return 0
