from ddd import interp_clamped


@fieldwise_init
struct RBETable(Copyable, Movable):
    var energy: List[Float64]
    var rbe: List[Float64]


def load_rbe_12c(path: String) raises -> RBETable:
    var energy = List[Float64]()
    var rbe = List[Float64]()
    var active_projectile = False
    var active_rbe = False
    with open(path, "r") as f:
        var lines = f.read().split("\n")
        for raw_line in lines:
            var line = String(raw_line.strip())
            if line.byte_length() == 0 or line[byte=0] == "#":
                continue
            var parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == "!projectile":
                active_projectile = len(parts) > 1 and parts[1] == "12C"
                active_rbe = False
                continue
            if parts[0] == "!rbe":
                active_rbe = active_projectile
                continue
            if line[byte=0] == "!":
                if active_rbe:
                    break
                continue
            if active_rbe:
                energy.append(Float64(parts[0]))
                rbe.append(Float64(parts[1]))
    return RBETable(energy^, rbe^)


def chordoma_rbe_12c() raises -> RBETable:
    return load_rbe_12c("/home/max/Projects/TRIP_DATA/Basedata/RBE/chordom02.rbe")


def rbe_at_energy(table: RBETable, energy_mev_u: Float64) -> Float64:
    return interp_clamped(table.energy, table.rbe, energy_mev_u)


def residual_energy_from_depth(initial_energy: Float64, depth_cm: Float64, range_cm: Float64) -> Float64:
    if range_cm <= 0.0:
        return initial_energy
    var remaining = range_cm - depth_cm
    if remaining <= 0.0:
        return table_min_energy()
    var fraction = remaining / range_cm
    if fraction >= 1.0:
        return initial_energy
    # Carbon range is roughly proportional to E^1.75 in water.  This gives a
    # first-order residual energy for using TRiP's RBE(E) tables until native
    # spectra/LET accumulation is in place.
    return initial_energy * pow_approx(fraction, 0.5714285714285714)


def table_min_energy() -> Float64:
    return 0.126


def pow_approx(x: Float64, p: Float64) -> Float64:
    from std.math import exp, log

    if x <= 0.0:
        return 0.0
    return exp(log(x) * p)

