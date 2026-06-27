from bio_lq import BioAggregate, trip_bio_from_aggregates


@fieldwise_init
struct BioDepthCoefficients(Copyable, Movable):
    var depth_cm: Float64
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64


@fieldwise_init
struct BioDoseAggregates(Copyable, Movable):
    var absorbed_mev_per_g: Float64
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64


def interpolate_bio_depth_coefficients(
    low: BioDepthCoefficients,
    high: BioDepthCoefficients,
    depth_cm: Float64,
) -> BioDepthCoefficients:
    var dz = high.depth_cm - low.depth_cm
    var t = 0.0
    if dz != 0.0:
        t = (depth_cm - low.depth_cm) / dz
    var one_minus_t = 1.0 - t
    var alpha = one_minus_t * low.alpha + t * high.alpha
    var sqrt_beta = one_minus_t * low.sqrt_beta + t * high.sqrt_beta
    var let_mix = one_minus_t * low.let_mix + t * high.let_mix
    var let_bar = one_minus_t * low.let_bar + t * high.let_bar
    if alpha < 0.0 or sqrt_beta < 0.0 or let_mix < 0.0 or let_bar < 0.0:
        alpha = 0.0
        sqrt_beta = 0.0
        let_mix = 0.0
        let_bar = 0.0
    return BioDepthCoefficients(depth_cm, alpha, sqrt_beta, let_mix, let_bar)


def empty_bio_dose_aggregates() -> BioDoseAggregates:
    return BioDoseAggregates(0.0, 0.0, 0.0, 0.0, 0.0)


def accumulate_bio_contribution(
    mut aggregates: BioDoseAggregates,
    absorbed_mev_per_g: Float64,
    fluence_per_mm2: Float64,
    coeffs: BioDepthCoefficients,
):
    if absorbed_mev_per_g <= 0.0 or fluence_per_mm2 <= 0.0:
        return
    aggregates.absorbed_mev_per_g += absorbed_mev_per_g
    aggregates.alpha += fluence_per_mm2 * coeffs.alpha
    aggregates.sqrt_beta += fluence_per_mm2 * coeffs.sqrt_beta
    aggregates.let_mix += fluence_per_mm2 * coeffs.let_mix
    aggregates.let_bar += fluence_per_mm2 * coeffs.let_bar


def finalize_low_dose_bio(
    aggregates: BioDoseAggregates,
    photon_alpha: Float64,
    photon_beta: Float64,
    cut_gy: Float64,
) -> BioAggregate:
    return trip_bio_from_aggregates(
        aggregates.absorbed_mev_per_g,
        aggregates.alpha,
        aggregates.sqrt_beta,
        aggregates.let_mix,
        photon_alpha,
        photon_beta,
        cut_gy,
    )
