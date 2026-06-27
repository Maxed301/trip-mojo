from bio_lq import trip_bio_from_aggregates


def assert_close(got: Float64, expected: Float64, tol: Float64) raises:
    if abs(got - expected) > tol:
        raise Error("value mismatch")


def main() raises:
    # TRiP_DEBUG_BIO_AGG_CSV for P101 Tumor local voxel (25,51,8),
    # bioalg(ld), chordom02, both replay fields.
    var result = trip_bio_from_aggregates(
        49048289.197767213,
        45172107.196480192,
        8382307.2710413886,
        49191444.796643756,
        0.1,
        0.05,
        30.0,
    )
    assert_close(result.phys_gy, 0.78584629421481456, 1e-12)
    assert_close(result.spc_phys_gy, 0.78813991747289869, 1e-12)
    assert_close(result.bio_gy, 2.9707242031727685, 1e-12)
    assert_close(result.rbe, 3.7802865840844797, 1e-12)
