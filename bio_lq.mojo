from std.math import sqrt


@fieldwise_init
struct BioAggregate(Copyable, Movable):
    var phys_gy: Float64
    var bio_gy: Float64
    var rbe: Float64
    var spc_phys_gy: Float64
    var spc_bio_gy: Float64
    var damage: Float64


def trip_bio_from_aggregates(
    absorbed_mev_per_g: Float64,
    alpha_acc: Float64,
    sqrt_beta_acc: Float64,
    letmix_acc: Float64,
    photon_alpha: Float64,
    photon_beta: Float64,
    cut_gy: Float64,
) -> BioAggregate:
    var mev_to_gy = 1.602189e-8
    var phys_gy = absorbed_mev_per_g * mev_to_gy
    if letmix_acc <= 0.0:
        return BioAggregate(phys_gy, 0.0, 0.0, 0.0, 0.0, 0.0)

    var spc_phys_gy = letmix_acc * mev_to_gy
    var damage_cut = (photon_beta * cut_gy + photon_alpha) * cut_gy
    var slope_max = photon_beta * cut_gy * 2.0 + photon_alpha
    var damage: Float64
    if spc_phys_gy <= cut_gy:
        damage = mev_to_gy * (sqrt_beta_acc * sqrt_beta_acc * mev_to_gy + alpha_acc)
    else:
        damage = (
            (sqrt_beta_acc * sqrt_beta_acc * (cut_gy / letmix_acc) + alpha_acc)
            * (cut_gy / letmix_acc)
        ) + (spc_phys_gy - cut_gy) * slope_max

    var spc_bio_gy: Float64
    if damage <= damage_cut:
        if photon_beta != 0.0:
            spc_bio_gy = (
                sqrt(damage * photon_beta * 4.0 + photon_alpha * photon_alpha)
                - photon_alpha
            ) / (photon_beta * 2.0)
        else:
            spc_bio_gy = damage / photon_alpha
    else:
        spc_bio_gy = (damage - damage_cut) / slope_max + cut_gy

    var rbe = spc_bio_gy / spc_phys_gy
    return BioAggregate(phys_gy, phys_gy * rbe, rbe, spc_phys_gy, spc_bio_gy, damage)
