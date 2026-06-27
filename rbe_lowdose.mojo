from std.math import exp

from dedx import DEDXProjectile, DEDXTable, dedx_stopping_power, find_projectile, parse_projectile


@fieldwise_init
struct RBEProjectile(Copyable, Movable):
    var projectile: String
    var atomic_number: Int
    var mass_number: Int
    var energy: List[Float64]
    var rbe: List[Float64]


@fieldwise_init
struct RBELowDoseTable(Copyable, Movable):
    var alpha: Float64
    var beta: Float64
    var cut_gy: Float64
    var nucleus_radius_um: Float64
    var nucleus_area_um2: Float64
    var slope_max: Float64
    var projectiles: List[RBEProjectile]


@fieldwise_init
struct LowDoseValues(Copyable, Movable):
    var alpha_low_dose: Float64
    var beta_low_dose: Float64
    var sqrt_beta_low_dose: Float64


def read_rbe_lowdose(path: String) raises -> RBELowDoseTable:
    var alpha = 0.0
    var beta = 0.0
    var cut_gy = 0.0
    var nucleus_radius = 0.0
    var projectiles = List[RBEProjectile]()
    var current_projectile = ""
    var current_z = 0
    var current_a = 0
    var energies = List[Float64]()
    var rbes = List[Float64]()
    var in_rbe = False

    with open(path, "r") as f:
        for raw_line in f.read().split("\n"):
            var line = String(raw_line.strip())
            if line.byte_length() == 0 or line[byte=0] == "#":
                continue
            var parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == "!alpha":
                alpha = Float64(parts[1])
                continue
            if parts[0] == "!beta":
                beta = Float64(parts[1])
                continue
            if parts[0] == "!cut":
                cut_gy = Float64(parts[1])
                continue
            if parts[0] == "!rnucleus":
                nucleus_radius = Float64(parts[1])
                continue
            if parts[0] == "!projectile":
                if current_projectile != "":
                    projectiles.append(RBEProjectile(
                        current_projectile,
                        current_z,
                        current_a,
                        energies.copy(),
                        rbes.copy(),
                    ))
                current_projectile = String(parts[1])
                var za = parse_projectile(current_projectile)
                current_z = za.atomic_number
                current_a = za.mass_number
                energies = List[Float64]()
                rbes = List[Float64]()
                in_rbe = False
                continue
            if parts[0] == "!rbe":
                in_rbe = True
                continue
            if line[byte=0] == "!":
                in_rbe = False
                continue
            if in_rbe:
                energies.append(Float64(parts[0]))
                rbes.append(Float64(parts[1]))

    if current_projectile != "":
        projectiles.append(RBEProjectile(
            current_projectile,
            current_z,
            current_a,
            energies.copy(),
            rbes.copy(),
        ))
    var area = nucleus_radius * nucleus_radius * 3.141592653589793
    return RBELowDoseTable(
        alpha,
        beta,
        cut_gy,
        nucleus_radius,
        area,
        beta * cut_gy * 2.0 + alpha,
        projectiles^,
    )


def find_rbe_projectile(table: RBELowDoseTable, atomic_number: Int, mass_number: Int) raises -> RBEProjectile:
    for i in range(len(table.projectiles)):
        if table.projectiles[i].atomic_number == atomic_number and table.projectiles[i].mass_number == mass_number:
            return table.projectiles[i].copy()
    raise Error("RBE projectile was not found")


def low_dose_values(
    rbe_table: RBELowDoseTable,
    rbe_projectile: RBEProjectile,
    dedx_projectile: DEDXProjectile,
    index: Int,
) raises -> LowDoseValues:
    var energy = rbe_projectile.energy[index]
    var rbe = rbe_projectile.rbe[index]
    var stopping = dedx_stopping_power(dedx_projectile, energy)
    var d1 = stopping * 1.602189e-8 / (rbe_table.nucleus_area_um2 * 1.0e-6)
    var alpha_low = (1.0 - exp(-rbe_table.alpha * rbe * d1)) / d1
    var factor = alpha_low / (rbe_table.alpha * rbe)
    var beta_low = factor * factor * (rbe_table.slope_max - rbe_table.alpha * rbe) / (rbe_table.cut_gy + rbe_table.cut_gy)
    if beta_low < 0.0:
        beta_low = 0.0
    return LowDoseValues(alpha_low, beta_low, sqrt_approx(beta_low))


def low_dose_values_for_projectile(
    rbe_table: RBELowDoseTable,
    dedx_table: DEDXTable,
    atomic_number: Int,
    mass_number: Int,
    index: Int,
) raises -> LowDoseValues:
    return low_dose_values(
        rbe_table,
        find_rbe_projectile(rbe_table, atomic_number, mass_number),
        find_projectile(dedx_table, atomic_number, mass_number),
        index,
    )


def low_dose_values_at_energy(
    rbe_table: RBELowDoseTable,
    rbe_projectile: RBEProjectile,
    dedx_projectile: DEDXProjectile,
    energy: Float64,
) raises -> LowDoseValues:
    var lo = 0
    var hi = len(rbe_projectile.energy) - 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if energy < rbe_projectile.energy[mid]:
            hi = mid
        else:
            lo = mid
    var log_e = log_value(energy)
    var t = 0.0
    var dx = log_value(rbe_projectile.energy[hi]) - log_value(rbe_projectile.energy[lo])
    if dx != 0.0:
        t = (log_e - log_value(rbe_projectile.energy[lo])) / dx
    var low_lo = low_dose_values(rbe_table, rbe_projectile, dedx_projectile, lo)
    var low_hi = low_dose_values(rbe_table, rbe_projectile, dedx_projectile, hi)
    var alpha_low = exp_value(log_value(low_lo.alpha_low_dose) * (1.0 - t) + log_value(low_hi.alpha_low_dose) * t)
    var beta_low = low_lo.beta_low_dose * (1.0 - t) + low_hi.beta_low_dose * t
    if beta_low < 0.0:
        beta_low = 0.0
    return LowDoseValues(alpha_low, beta_low, sqrt_approx(beta_low))


def log_value(value: Float64) -> Float64:
    from std.math import log

    return log(value)


def exp_value(value: Float64) -> Float64:
    from std.math import exp

    return exp(value)


def sqrt_approx(value: Float64) -> Float64:
    from std.math import sqrt

    return sqrt(value)
