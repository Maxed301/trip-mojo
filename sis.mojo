@fieldwise_init
struct SISEnergy(Copyable, Movable):
    var energy_mev_u: Float64
    var focus_values_mm: List[Float64]


@fieldwise_init
struct SISTable(Copyable, Movable):
    var energies: List[SISEnergy]


def read_sis_table(path: String) raises -> SISTable:
    var energies = List[SISEnergy]()
    with open(path, "r") as f:
        for raw_line in f.read().split("\n"):
            var line = String(raw_line.strip())
            if line.byte_length() == 0:
                continue
            var parts = line.split()
            if len(parts) < 4 or String(parts[0]) != "energy":
                continue
            var values = List[Float64]()
            var i = 3
            while i < len(parts) and String(parts[i]) != "intensity":
                values.append(Float64(parts[i]))
                i += 1
            energies.append(SISEnergy(Float64(parts[1]), values^))
    return SISTable(energies^)


def nearest_sis_energy(table: SISTable, requested: Float64, floor_only: Bool, ceil_only: Bool) raises -> Float64:
    if len(table.energies) == 0:
        raise Error("empty SIS table")
    var best = table.energies[0].energy_mev_u
    var best_delta = 1.0e300
    for i in range(len(table.energies)):
        var e = table.energies[i].energy_mev_u
        if floor_only and e > requested:
            continue
        if ceil_only and e < requested:
            continue
        var delta = e - requested
        if delta < 0.0:
            delta = -delta
        if delta < best_delta:
            best_delta = delta
            best = e
    if best_delta == 1.0e300:
        raise Error("requested energy is outside SIS table")
    return best


def focus_ceil_for_energy(table: SISTable, energy: Float64, requested_focus: Float64) raises -> Float64:
    for i in range(len(table.energies)):
        if table.energies[i].energy_mev_u == energy:
            var fallback = table.energies[i].focus_values_mm[len(table.energies[i].focus_values_mm) - 1]
            for j in range(len(table.energies[i].focus_values_mm)):
                if table.energies[i].focus_values_mm[j] >= requested_focus:
                    return table.energies[i].focus_values_mm[j]
            return fallback
    raise Error("energy not found in SIS table")
