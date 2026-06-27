from bio_coeff import BioDepthCoefficients, accumulate_bio_contribution, empty_bio_dose_aggregates, finalize_low_dose_bio, interpolate_bio_depth_coefficients


def assert_close(got: Float64, expected: Float64, tol: Float64) raises:
    if abs(got - expected) > tol:
        raise Error("value mismatch")


def main() raises:
    var low = BioDepthCoefficients(10.0, 2.0, 3.0, 5.0, 7.0)
    var high = BioDepthCoefficients(14.0, 10.0, 11.0, 13.0, 15.0)
    var mid = interpolate_bio_depth_coefficients(low, high, 11.0)
    assert_close(mid.alpha, 4.0, 1e-12)
    assert_close(mid.sqrt_beta, 5.0, 1e-12)
    assert_close(mid.let_mix, 7.0, 1e-12)
    assert_close(mid.let_bar, 9.0, 1e-12)

    var negative = interpolate_bio_depth_coefficients(
        BioDepthCoefficients(0.0, 1.0, 1.0, 1.0, 1.0),
        BioDepthCoefficients(1.0, -3.0, 1.0, 1.0, 1.0),
        0.5,
    )
    assert_close(negative.alpha, 0.0, 1e-12)
    assert_close(negative.sqrt_beta, 0.0, 1e-12)
    assert_close(negative.let_mix, 0.0, 1e-12)
    assert_close(negative.let_bar, 0.0, 1e-12)

    var aggregates = empty_bio_dose_aggregates()
    accumulate_bio_contribution(
        aggregates,
        49048289.197767213,
        1.0,
        BioDepthCoefficients(
            0.0,
            45172107.196480192,
            8382307.2710413886,
            49191444.796643756,
            4021512080.6038694,
        ),
    )
    var bio = finalize_low_dose_bio(aggregates, 0.1, 0.05, 30.0)
    assert_close(bio.phys_gy, 0.78584629421481456, 1e-12)
    assert_close(bio.bio_gy, 2.9707242031727685, 1e-12)
    assert_close(bio.rbe, 3.7802865840844797, 1e-12)

    # First TRIP_DEBUG_DOSE_DIRECT_CSV row for P101 Tumor voxel (25,51,8).
    # This pins the per-spot bio accumulation contract below the final voxel
    # aggregate.
    var direct = empty_bio_dose_aggregates()
    accumulate_bio_contribution(
        direct,
        585.00916694163243,
        0.0012634501899528364 * 40795.800000000003,
        BioDepthCoefficients(
            144.85984080970056 * 0.1,
            7.007378101348877,
            2.2713422775268555,
            11.300865173339844,
            110.83217620849609,
        ),
    )
    assert_close(direct.alpha, 361.18452169598834, 1e-9)
    assert_close(direct.sqrt_beta, 117.07284268826557, 1e-9)
    assert_close(direct.let_mix, 582.48570625836533, 1e-9)
    assert_close(direct.let_bar, 5712.6739806840833, 1e-9)
    assert_close(direct.absorbed_mev_per_g, 585.00916694163243, 1e-12)
