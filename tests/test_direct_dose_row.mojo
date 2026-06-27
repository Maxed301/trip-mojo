from std.testing import assert_true

from ddd import interp, load_ddd
from geometry import Vec3
from phys_dose import EnergyCurve, mev_per_g_from_mm2_fluence_to_gy, spot_voxel_bio_contribution, spot_voxel_contribution, spot_voxel_optimizer_bio_contribution, spot_voxel_optimizer_contribution, trip_8ln2
from rst import RasterSpot


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def curve_for_depth(depth_mm: Float64) raises -> EnergyCurve:
    var pristine = load_ddd(280.48)
    var depth_cm = depth_mm * 0.1
    var voxel_dose = List[Float64]()
    var voxel_fwhm1 = List[Float64]()
    var voxel_mix = List[Float64]()
    var voxel_fwhm2 = List[Float64]()
    var voxel_bio_factor = List[Float64]()
    var voxel_alpha = List[Float64]()
    var voxel_sqrt_beta = List[Float64]()
    var voxel_let_mix = List[Float64]()
    var voxel_let_bar = List[Float64]()
    voxel_dose.append(interp(pristine.z, pristine.dose, depth_cm))
    voxel_fwhm1.append(interp(pristine.z, pristine.fwhm1, depth_cm))
    voxel_mix.append(interp(pristine.z, pristine.mix, depth_cm))
    voxel_fwhm2.append(interp(pristine.z, pristine.fwhm2, depth_cm))
    voxel_bio_factor.append(1.0)
    voxel_alpha.append(0.0)
    voxel_sqrt_beta.append(0.0)
    voxel_let_mix.append(0.0)
    voxel_let_bar.append(0.0)
    return EnergyCurve(
        energy=280.48,
        curve=pristine^,
        voxel_dose=voxel_dose^,
        voxel_fwhm1=voxel_fwhm1^,
        voxel_mix=voxel_mix^,
        voxel_fwhm2=voxel_fwhm2^,
        voxel_bio_factor=voxel_bio_factor^,
        voxel_alpha=voxel_alpha^,
        voxel_sqrt_beta=voxel_sqrt_beta^,
        voxel_let_mix=voxel_let_mix^,
        voxel_let_bar=voxel_let_bar^,
        max_fwhm2=1.0,
    )


def main() raises:
    # From TRIP_DEBUG_DOSE_DIRECT_CSV, centered spot row.
    var curve = curve_for_depth(42.594128392349546)
    var spot = RasterSpot(0.0, 0.0, 280.48, 100000000.0, 7.5)
    var p = Vec3(4.5, -5.3918581761994631, 126.89196209816603)
    var gy = spot_voxel_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 1.0, 1.0, 1.0, False
    )
    assert_close(gy, 0.29557418823465553, 1.0e-12)

    # From TRIP_DEBUG_DOSE_DIRECT_CSV, off-axis spot row with scanner divergence.
    curve = curve_for_depth(42.650671234616887)
    spot = RasterSpot(-10.0, 0.0, 280.48, 100000000.0, 7.5)
    p = Vec3(-4.5, -4.4300949331543507, 126.72237728936349)
    gy = spot_voxel_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 1.0, 1.0, 1.0, False
    )
    assert_close(gy, 0.31056675714741161, 1.0e-12)

    # Axis-wise support rejection from TRPRstPntDoseDistance.
    p = Vec3(8.0, 0.0, 0.0)
    gy = spot_voxel_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 1.0, 1.0, 0.0, False
    )
    assert_close(gy, 0.0, 0.0)

    # TRiP applies dose extension once in dF2Max and once in the radial gate.
    spot = RasterSpot(0.0, 0.0, 280.48, 100000000.0, 7.5)
    spot.f2_max = 100.0
    p = Vec3(8.0, 8.0, 0.0)
    gy = spot_voxel_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 4.0, 1.0, 0.0, False
    )
    assert_true(gy > 0.0)

    # TRPOptDoseMatrixPhys uses TRPRstPntDoseDistance for FDCS setup. There
    # dF2Max is an x/y gate only; the field-dose radial dF2Max gate is not used.
    spot = RasterSpot(0.0, 0.0, 280.48, 100000000.0, 20.0)
    spot.f2_max = 100.0
    p = Vec3(8.0, 8.0, 0.0)
    var field_path = spot_voxel_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 1.0, 1.0, 0.0, False
    )
    var optimizer_path = spot_voxel_optimizer_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 1.0, 1.0, 0.0, False
    )
    assert_close(field_path, 0.0, 0.0)
    assert_true(optimizer_path > 0.0)

    var field_bio = spot_voxel_bio_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 1.0
    )
    var optimizer_bio = spot_voxel_optimizer_bio_contribution(
        0, p, spot, curve, trip_8ln2(), 0.15915494309189535, 1.0
    )
    assert_close(field_bio.absorbed_mev_per_g, 0.0, 0.0)
    assert_true(optimizer_bio.absorbed_mev_per_g > 0.0)

    var unit = mev_per_g_from_mm2_fluence_to_gy()
    assert_close(unit, 1.602189e-8, 0.0)
