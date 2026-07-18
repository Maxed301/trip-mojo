from std.math import sqrt

from tests.support.sparse_matrix import SparseDoseMatrix


@fieldwise_init
struct VoxelObjective(Copyable, Movable):
    var dose_p: Float64
    var dose_w: Float64
    var dose_d: Float64
    var dose_w_max: Float64
    var overdose_tolerance: Float64


@fieldwise_init
struct PhysicalScenarioSet(Copyable, Movable):
    var matrices: List[SparseDoseMatrix]


@fieldwise_init
struct BioLQParams(Copyable, Movable):
    var alpha: Float64
    var beta: Float64
    var cut_gy: Float64

    def slope_max(self) -> Float64:
        return self.beta * self.cut_gy * 2.0 + self.alpha

    def damage_cut(self) -> Float64:
        return (self.beta * self.cut_gy + self.alpha) * self.cut_gy


@fieldwise_init
struct BiologicalMatrixEntry(Copyable, Movable):
    var voxel: Int
    var spot: Int
    var g: Float64
    var ddd: Float64
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64


@fieldwise_init
struct BiologicalDoseMatrix(Copyable, Movable):
    var voxel_count: Int
    var spot_count: Int
    var entries: List[BiologicalMatrixEntry]


@fieldwise_init
struct BiologicalScenarioSet(Copyable, Movable):
    var matrices: List[BiologicalDoseMatrix]


@fieldwise_init
struct BioScenarioDose(Copyable, Movable):
    var dose_a: Float64
    var dose_phys: Float64
    var dose_phs: Float64
    var alpha: Float64
    var sqrt_beta: Float64
    var let_mix: Float64
    var let_bar: Float64
    var gradient_denominator: Float64


@fieldwise_init
struct BioRobustEvaluation(Copyable, Movable):
    var scenario_doses: List[List[BioScenarioDose]]
    var dose_a: List[Float64]
    var dose_min: List[Float64]
    var dose_max: List[Float64]
    var min_scenario: List[Int]
    var max_scenario: List[Int]
    var gradient: List[Float64]
    var chi2: Float64
    var dose_p_weighted_avg2: Float64
    var grad_norm: Float64


def evaluate_robust_biological(
    scenarios: BiologicalScenarioSet,
    objectives: List[VoxelObjective],
    lq_params: List[BioLQParams],
    particles: List[Float64],
    include_dmax: Bool = False,
) raises -> BioRobustEvaluation:
    validate_biological_inputs(scenarios, objectives, lq_params, particles)
    var scenario_doses = List[List[BioScenarioDose]]()
    scenario_doses.reserve(len(scenarios.matrices))
    for scenario in range(len(scenarios.matrices)):
        scenario_doses.append(
            compute_biological_scenario_dose(
                scenarios.matrices[scenario], lq_params, particles
            )
        )

    var dose_a = List[Float64]()
    var dose_min = List[Float64]()
    var dose_max = List[Float64]()
    var min_scenario = List[Int]()
    var max_scenario = List[Int]()
    dose_a.resize(len(objectives), 0.0)
    dose_min.resize(len(objectives), 0.0)
    dose_max.resize(len(objectives), 0.0)
    min_scenario.resize(len(objectives), 0)
    max_scenario.resize(len(objectives), 0)
    for voxel in range(len(objectives)):
        var dmin = scenario_doses[0][voxel].dose_a
        var dmax = scenario_doses[0][voxel].dose_a
        var imin = 0
        var imax = 0
        for scenario in range(1, len(scenario_doses)):
            var dose = scenario_doses[scenario][voxel].dose_a
            if dose < dmin:
                dmin = dose
                imin = scenario
            if dose > dmax:
                dmax = dose
                imax = scenario
        dose_min[voxel] = dmin
        dose_max[voxel] = dmax
        min_scenario[voxel] = imin
        max_scenario[voxel] = imax
        if objectives[voxel].dose_p > 0.0:
            dose_a[voxel] = dmin
        else:
            dose_a[voxel] = dmax

    var gradient = List[Float64]()
    gradient.resize(len(particles), 0.0)
    var chi2 = 0.0
    var dose_p_weighted_avg2 = 0.0
    for voxel in range(len(objectives)):
        var ddp = abs_float(objectives[voxel].dose_p)
        if ddp <= 0.0:
            continue
        var weight = objectives[voxel].dose_w / objectives[voxel].dose_d
        var ddose_chi = ddp - dose_a[voxel]
        if ddose_chi < 0.0 or objectives[voxel].dose_p > 0.0:
            chi2 += (ddose_chi * weight) * (ddose_chi * weight)
        dose_p_weighted_avg2 += (weight * objectives[voxel].dose_p) * (
            weight * objectives[voxel].dose_p
        )

        var scenario = min_scenario[voxel]
        if objectives[voxel].dose_p < 0.0:
            scenario = max_scenario[voxel]
        scatter_bio_gradient_for_selected_scenario(
            scenarios.matrices[scenario],
            scenario_doses[scenario],
            objectives[voxel],
            lq_params[voxel],
            voxel,
            scenario,
            gradient,
            False,
        )
        if include_dmax and objectives[voxel].dose_p > 0.0:
            var max_weight = (
                objectives[voxel].dose_w_max / objectives[voxel].dose_d
            )
            var ddose_max = ddp - dose_max[voxel]
            chi2 += (ddose_max * max_weight) * (ddose_max * max_weight)
            dose_p_weighted_avg2 += (max_weight * objectives[voxel].dose_p) * (
                max_weight * objectives[voxel].dose_p
            )
            scatter_bio_gradient_for_selected_scenario(
                scenarios.matrices[max_scenario[voxel]],
                scenario_doses[max_scenario[voxel]],
                objectives[voxel],
                lq_params[voxel],
                voxel,
                max_scenario[voxel],
                gradient,
                True,
            )

    var grad_norm2 = 0.0
    for i in range(len(gradient)):
        grad_norm2 += gradient[i] * gradient[i]
    return BioRobustEvaluation(
        scenario_doses^,
        dose_a^,
        dose_min^,
        dose_max^,
        min_scenario^,
        max_scenario^,
        gradient^,
        chi2,
        dose_p_weighted_avg2,
        sqrt(grad_norm2),
    )


def validate_physical_inputs(
    scenarios: PhysicalScenarioSet,
    objectives: List[VoxelObjective],
    particles: List[Float64],
) raises:
    if len(scenarios.matrices) == 0:
        raise Error("optimizer evaluation requires at least one scenario")
    var voxel_count = scenarios.matrices[0].voxel_count
    var spot_count = scenarios.matrices[0].spot_count
    if len(objectives) != voxel_count:
        raise Error("objective length does not match voxel count")
    if len(particles) != spot_count:
        raise Error("particle length does not match spot count")
    for i in range(1, len(scenarios.matrices)):
        if scenarios.matrices[i].voxel_count != voxel_count:
            raise Error("scenario voxel count mismatch")
        if scenarios.matrices[i].spot_count != spot_count:
            raise Error("scenario spot count mismatch")


def compute_biological_scenario_dose(
    matrix: BiologicalDoseMatrix,
    lq_params: List[BioLQParams],
    particles: List[Float64],
) raises -> List[BioScenarioDose]:
    var dose_phys = List[Float64]()
    var alpha = List[Float64]()
    var sqrt_beta = List[Float64]()
    var let_mix = List[Float64]()
    var let_bar = List[Float64]()
    dose_phys.resize(matrix.voxel_count, 0.0)
    alpha.resize(matrix.voxel_count, 0.0)
    sqrt_beta.resize(matrix.voxel_count, 0.0)
    let_mix.resize(matrix.voxel_count, 0.0)
    let_bar.resize(matrix.voxel_count, 0.0)
    for i in range(len(matrix.entries)):
        var entry = matrix.entries[i].copy()
        var fluence = particles[entry.spot] * entry.g
        dose_phys[entry.voxel] += fluence * entry.ddd
        alpha[entry.voxel] += fluence * entry.alpha
        sqrt_beta[entry.voxel] += fluence * entry.sqrt_beta
        let_mix[entry.voxel] += fluence * entry.let_mix
        let_bar[entry.voxel] += fluence * entry.let_bar

    var out = List[BioScenarioDose]()
    out.reserve(matrix.voxel_count)
    for voxel in range(matrix.voxel_count):
        var dose_a = 0.0
        var dose_phs = let_mix[voxel] * mev2gy()
        var gradient_denominator = lq_params[voxel].alpha
        var let_bar_value = 0.0
        if let_mix[voxel] > 0.0:
            let_bar_value = let_bar[voxel] / let_mix[voxel]
            var damage: Float64
            if dose_phs <= lq_params[voxel].cut_gy:
                damage = mev2gy() * (
                    sqrt_beta[voxel] * sqrt_beta[voxel] * mev2gy()
                    + alpha[voxel]
                )
                if lq_params[voxel].beta != 0.0:
                    gradient_denominator = sqrt(
                        damage * lq_params[voxel].beta * 4.0
                        + lq_params[voxel].alpha * lq_params[voxel].alpha
                    )
                    var bio_dose = (
                        gradient_denominator - lq_params[voxel].alpha
                    ) / (lq_params[voxel].beta * 2.0)
                    dose_a = dose_phys[voxel] * mev2gy() * bio_dose / dose_phs
                else:
                    gradient_denominator = lq_params[voxel].alpha
                    var bio_dose = damage / lq_params[voxel].alpha
                    dose_a = dose_phys[voxel] * mev2gy() * bio_dose / dose_phs
            else:
                var damage_cut = lq_params[voxel].damage_cut()
                var slope_max = lq_params[voxel].slope_max()
                damage = (
                    (
                        sqrt_beta[voxel]
                        * sqrt_beta[voxel]
                        * (lq_params[voxel].cut_gy / let_mix[voxel])
                        + alpha[voxel]
                    )
                    * (lq_params[voxel].cut_gy / let_mix[voxel])
                ) + (dose_phs - lq_params[voxel].cut_gy) * slope_max
                var bio_dose = (damage - damage_cut) / slope_max + lq_params[
                    voxel
                ].cut_gy
                gradient_denominator = slope_max
                dose_a = dose_phys[voxel] * mev2gy() * bio_dose / dose_phs
        out.append(
            BioScenarioDose(
                dose_a,
                dose_phys[voxel],
                dose_phs,
                alpha[voxel],
                sqrt_beta[voxel],
                let_mix[voxel],
                let_bar_value,
                gradient_denominator,
            )
        )
    return out^


def scatter_bio_gradient_for_selected_scenario(
    matrix: BiologicalDoseMatrix,
    scenario_doses: List[BioScenarioDose],
    objective: VoxelObjective,
    lq: BioLQParams,
    voxel: Int,
    scenario: Int,
    mut gradient: List[Float64],
    maxdose_pass: Bool,
) raises:
    _ = scenario
    var ddose = abs_float(objective.dose_p) - scenario_doses[voxel].dose_a
    var dose_d = objective.dose_d
    var dose_w = objective.dose_w
    if maxdose_pass:
        dose_w = objective.dose_w_max
    if objective.dose_p < 0.0 or maxdose_pass:
        if ddose > 0.0:
            ddose = 0.0
    elif ddose < 0.0:
        if ddose > objective.dose_p * -objective.overdose_tolerance:
            ddose = 0.0
        else:
            ddose += objective.dose_p * objective.overdose_tolerance
    if ddose == 0.0:
        return
    var factor = (ddose * dose_w / dose_d) * (2.0 * dose_w / dose_d)
    var aux = scenario_doses[voxel].copy()
    if aux.let_mix == 0.0:
        return
    for i in range(len(matrix.entries)):
        var entry = matrix.entries[i].copy()
        if entry.voxel != voxel:
            continue
        if aux.dose_phs <= lq.cut_gy:
            var grad_bio = (
                entry.alpha
                + mev2gy() * (aux.sqrt_beta + aux.sqrt_beta) * entry.sqrt_beta
            ) * mev2gy()
            gradient[entry.spot] += (
                factor * grad_bio / aux.gradient_denominator * entry.g
            )
        else:
            var dcut = lq.cut_gy / aux.let_mix
            var grad_bio = (
                entry.alpha
                - entry.let_mix * aux.alpha / aux.let_mix
                + (
                    entry.sqrt_beta
                    - entry.let_mix * aux.sqrt_beta / aux.let_mix
                )
                * (aux.sqrt_beta + aux.sqrt_beta)
                * dcut
            ) * dcut + (entry.let_mix * mev2gy()) * lq.slope_max()
            gradient[entry.spot] += (
                factor * grad_bio / aux.gradient_denominator * entry.g
            )


def abs_float(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value


def validate_biological_inputs(
    scenarios: BiologicalScenarioSet,
    objectives: List[VoxelObjective],
    lq_params: List[BioLQParams],
    particles: List[Float64],
) raises:
    if len(scenarios.matrices) == 0:
        raise Error("bio optimizer evaluation requires at least one scenario")
    var voxel_count = scenarios.matrices[0].voxel_count
    var spot_count = scenarios.matrices[0].spot_count
    if len(objectives) != voxel_count:
        raise Error("objective length does not match bio matrix voxel count")
    if len(lq_params) != voxel_count:
        raise Error("LQ parameter length does not match bio matrix voxel count")
    if len(particles) != spot_count:
        raise Error("particle length does not match bio matrix spot count")
    for i in range(1, len(scenarios.matrices)):
        if scenarios.matrices[i].voxel_count != voxel_count:
            raise Error("bio scenario voxel count mismatch")
        if scenarios.matrices[i].spot_count != spot_count:
            raise Error("bio scenario spot count mismatch")


def mev2gy() -> Float64:
    return 1.602189e-8
