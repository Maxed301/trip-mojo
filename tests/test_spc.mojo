from dedx import find_projectile, read_dedx
from phys_dose import spc_bio_coefficients_at_depth
from rbe_lowdose import read_rbe_lowdose
from spc import SPCBioCoefficients, be_f64, be_u32, be_u64, depth_bio_coefficients, depth_let_moments, depth_rbe_coefficients, interpolate_spc_bio_coefficients, read_spc_beam, species_energy_moments, species_let_moments
from std.testing import assert_equal


def assert_close(got: Float64, expected: Float64, tol: Float64) raises:
    if abs(got - expected) > tol:
        raise Error("value mismatch")


def main() raises:
    var bytes = List[UInt8]()
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x01))
    bytes.append(UInt8(0x3f))
    bytes.append(UInt8(0xf0))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x00))
    bytes.append(UInt8(0x00))
    assert_equal(be_u32(bytes, 0), 1)
    assert_equal(be_u64(bytes, 4), UInt64(0x3ff0000000000000))
    assert_close(be_f64(bytes, 4), 1.0, 1e-12)

    var beam = read_spc_beam("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/SPC/12C.H2O.MeV28000.spc")
    assert_close(beam.energy_mev_u, 280.0, 1e-12)
    assert_close(beam.peak_cm, 15.077287622747752, 1e-12)
    assert_close(beam.normalization, 1.0, 1e-12)
    assert_equal(len(beam.depths), 161)
    assert_close(beam.depths[0].z_cm, 0.0, 1e-12)
    assert_close(beam.depths[0].normalization, 1.0, 1e-12)
    assert_equal(beam.depths[0].species_count, 18)
    assert_equal(len(beam.depths[0].species), 18)
    assert_close(beam.depths[0].species[0].charge, 6.0, 1e-12)
    assert_close(beam.depths[0].species[0].mass, 12.0, 1e-12)
    assert_equal(beam.depths[0].species[0].atomic_number, 6)
    assert_equal(beam.depths[0].species[0].mass_number, 12)
    assert_close(beam.depths[0].species[0].cumulative, 0.9889010697368196, 1e-12)
    assert_equal(len(beam.depths[0].species[0].energy_edges), 1000)
    assert_equal(len(beam.depths[0].species[0].histo), 999)
    assert_close(beam.depths[0].species[0].energy_edges[1], 0.01007130149600965, 1e-16)
    assert_close(beam.depths[0].species[0].histo[0], 0.0, 1e-16)
    assert_close(beam.depths[0].species[1].charge, 1.0, 1e-12)
    assert_equal(beam.depths[0].species[1].atomic_number, 1)
    assert_equal(beam.depths[0].species[1].mass_number, 1)
    assert_equal(len(beam.depths[0].species[1].energy_edges), 101)
    assert_equal(len(beam.depths[0].species[1].histo), 100)
    assert_close(beam.depths[0].species[1].energy_edges[1], 0.010751622488803432, 1e-16)
    assert_close(beam.depths[0].species[1].histo[0], 1.606737862581754e-07, 1e-20)

    var primary_moments = species_energy_moments(beam.depths[0].species[0])
    assert_close(primary_moments.sum, 0.9889010697368196, 1e-12)
    assert_close(primary_moments.mean, 277.40699476795385, 1e-12)
    var proton_moments = species_energy_moments(beam.depths[0].species[1])
    assert_close(proton_moments.sum, 0.005267718592073435, 1e-15)
    assert_close(proton_moments.mean, 256.67858016384486, 1e-12)

    var dedx = read_dedx("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/20040607.dedx")
    var carbon_dedx = find_projectile(dedx, 6, 12)
    var primary_let = species_let_moments(beam.depths[0].species[0], carbon_dedx)
    assert_close(primary_let.let_sum, 0.9889010697368196, 1e-12)
    assert_close(primary_let.let_mean, 131.46055924378157, 1e-10)
    assert_close(primary_let.dose_mean_let_sum, 130.00148766437616, 1e-10)
    assert_close(primary_let.dose_mean_let_mean, 131.46308506404853, 1e-10)

    var depth_let = depth_let_moments(beam.depths[0], dedx)
    assert_close(depth_let.let_sum, 1.0226313812784193, 1e-12)
    assert_close(depth_let.let_mean, 127.7596879568531, 1e-10)
    assert_close(depth_let.dose_mean_let_sum, 130.65106616701652, 1e-10)
    assert_close(depth_let.dose_mean_let_mean, 131.14874607219235, 1e-10)

    var rbe = read_rbe_lowdose("/home/max/Projects/TRIP_DATA/Basedata/RBE/chordom02.rbe")
    var rbe_coeffs = depth_rbe_coefficients(beam.depths[0], dedx, rbe)
    assert_close(rbe_coeffs.alpha0x1, 67.48126856794913, 1e-10)
    assert_close(rbe_coeffs.sqrt_beta0x1, 26.90136647215751, 1e-10)
    assert_close(rbe_coeffs.alpha_let_sum, 130.6510661670165, 1e-10)
    assert_close(rbe_coeffs.alpha_let_mean, 0.5164999456007892, 1e-12)
    assert_close(rbe_coeffs.beta_let_sum, 130.6510661670165, 1e-10)
    assert_close(rbe_coeffs.beta_let_mean, 0.04239579323792425, 1e-12)

    var beam200 = read_spc_beam("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/SPC/12C.H2O.MeV20000.spc")
    assert_close(beam200.energy_mev_u, 200.0, 1e-12)
    assert_close(beam200.peak_cm, 8.3887935109497995, 1e-12)
    assert_close(beam200.depths[161].z_cm, 14.163120738853223, 1e-12)
    assert_close(beam200.depths[162].z_cm, 14.936880029552771, 1e-12)
    var bio_low = depth_bio_coefficients(beam200.depths[161], dedx, rbe)
    var bio_high = depth_bio_coefficients(beam200.depths[162], dedx, rbe)
    assert_close(bio_low.alpha0x1, 7.1272440819246299, 1e-6)
    assert_close(bio_high.alpha0x1, 6.3290894938162063, 1e-6)
    assert_close(bio_low.sqrt_beta0x1, 2.3058580773899457, 1e-6)
    assert_close(bio_high.sqrt_beta0x1, 2.0760270097001059, 1e-6)
    assert_close(bio_low.let_mix, 11.476181961153467, 1e-6)
    assert_close(bio_high.let_mix, 10.308797605428257, 1e-6)
    assert_close(bio_low.let_bar, 113.69165717885134, 1e-6)
    assert_close(bio_high.let_bar, 94.651174762991388, 1e-6)
    var bio_interp = interpolate_spc_bio_coefficients(bio_low, bio_high, 14.279323180669309)
    assert_close(bio_interp.alpha0x1, 7.007378101348877, 1e-6)
    assert_close(bio_interp.sqrt_beta0x1, 2.2713422775268555, 1e-6)
    assert_close(bio_interp.let_mix, 11.300865173339844, 1e-6)
    assert_close(bio_interp.let_bar, 110.83217620849609, 1e-6)

    # Mirrors TRPFdcsBioContribution: if any interpolated bio coefficient is
    # negative, all four are cleared rather than only the negative channel.
    var negative = interpolate_spc_bio_coefficients(
        SPCBioCoefficients(0.0, 1.0, 1.0, 1.0, 1.0),
        SPCBioCoefficients(1.0, -3.0, 1.0, 1.0, 1.0),
        0.5,
    )
    assert_close(negative.alpha0x1, 0.0, 1e-12)
    assert_close(negative.sqrt_beta0x1, 0.0, 1e-12)
    assert_close(negative.let_mix, 0.0, 1e-12)
    assert_close(negative.let_bar, 0.0, 1e-12)

    var coeff_table = List[SPCBioCoefficients]()
    coeff_table.append(SPCBioCoefficients(1.0, 1.0, 2.0, 3.0, 4.0))
    coeff_table.append(SPCBioCoefficients(2.0, 5.0, 6.0, 7.0, 8.0))
    var before_table = spc_bio_coefficients_at_depth(coeff_table, 0.999999)
    var after_table = spc_bio_coefficients_at_depth(coeff_table, 2.000001)
    assert_close(before_table.alpha0x1, 0.0, 1e-12)
    assert_close(before_table.sqrt_beta0x1, 0.0, 1e-12)
    assert_close(before_table.let_mix, 0.0, 1e-12)
    assert_close(before_table.let_bar, 0.0, 1e-12)
    assert_close(after_table.alpha0x1, 0.0, 1e-12)
    assert_close(after_table.sqrt_beta0x1, 0.0, 1e-12)
    assert_close(after_table.let_mix, 0.0, 1e-12)
    assert_close(after_table.let_bar, 0.0, 1e-12)
