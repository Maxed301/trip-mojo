from ddd import DDDCurve, interp, load_ddd
from std.math import exp, log


@fieldwise_init
struct RiFiModel(Copyable, Movable):
    var id: String


@fieldwise_init
struct RiFiTable(Copyable, Movable):
    var offsets_cm: List[Float64]
    var weights: List[Float64]


def p101_rifi_model() raises -> RiFiModel:
    return RiFiModel(id="3")


def fold_curve_for_rifi(curve: DDDCurve, model: RiFiModel) raises -> DDDCurve:
    if model.id == "3":
        return fold_curve_for_rifi3(curve)
    raise Error("Unsupported RiFi model")


def fold_curve_for_rifi3(curve: DDDCurve) raises -> DDDCurve:
    return fold_curve_with_table(curve, load_rifi_table("/home/max/Projects/trip_temp/TEST/RIFI/3mm.rifi"))


def load_curve_for_rifi(energy: Float64, model: RiFiModel) raises -> DDDCurve:
    if model.id != "3":
        raise Error("Unsupported RiFi model")
    var lower_energy = Float64(Int(energy))
    if lower_energy == energy:
        return fold_curve_for_rifi3(load_ddd(energy))
    var upper_energy = lower_energy + 1.0
    var lower = fold_curve_for_rifi3(load_ddd(lower_energy))
    var upper = fold_curve_for_rifi3(load_ddd(upper_energy))
    return interpolate_folded_curves(energy, lower_energy, lower, upper_energy, upper)


def load_rifi_table(path: String) raises -> RiFiTable:
    var offsets = List[Float64]()
    var weights = List[Float64]()
    var active = False
    with open(path, "r") as f:
        for raw_line in f.read().split("\n"):
            var line = raw_line.strip()
            if line.byte_length() == 0:
                continue
            if line == "!rifi":
                active = True
                continue
            if not active:
                continue
            if line[byte=0] == "#" or line[byte=0] == "!":
                continue
            var parts = line.split()
            if len(parts) < 2:
                continue
            offsets.append(Float64(parts[0]) * 0.1)
            weights.append(Float64(parts[1]))
    return RiFiTable(offsets^, weights^)


def fold_curve_with_table(curve: DDDCurve, table: RiFiTable) raises -> DDDCurve:
    var dz = min_positive_spacing(curve.z)
    if dz <= 0.0:
        raise Error("DDD depth grid is not ascending")

    var source_max = curve.z[len(curve.z) - 1]
    var source_peak = peak_depth(curve.z, curve.dose)
    var master_z = make_ramp(source_max, dz)

    var folded_dose = fold_column(master_z, curve.z, curve.dose, table)
    var folded_fwhm1 = fold_column(master_z, curve.z, curve.fwhm1, table)
    var folded_mix = fold_column(master_z, curve.z, curve.mix, table)
    var folded_fwhm2 = fold_column(master_z, curve.z, curve.fwhm2, table)

    var folded_peak = peak_depth(master_z, folded_dose)
    var out_z = List[Float64]()
    for i in range(len(curve.z)):
        out_z.append((curve.z[i] / source_peak) * folded_peak)

    return DDDCurve(
        z=out_z^,
        dose=interpolate_column(master_z, folded_dose, out_z),
        fwhm1=interpolate_column(master_z, folded_fwhm1, out_z),
        mix=interpolate_column(master_z, folded_mix, out_z),
        fwhm2=interpolate_column(master_z, folded_fwhm2, out_z),
    )


def min_positive_spacing(z: List[Float64]) -> Float64:
    var best = 1.7976931348623157e308
    for i in range(len(z) - 1):
        var dz = z[i + 1] - z[i]
        if dz > 0.0 and dz < best:
            best = dz
    return best


def make_ramp(max_z: Float64, dz: Float64) -> List[Float64]:
    var n = Int(max_z / dz) + 1
    var out = List[Float64]()
    for i in range(n):
        out.append(Float64(i) * dz)
    return out^


def peak_depth(z: List[Float64], values: List[Float64]) -> Float64:
    var peak_value = values[0]
    var peak_index = 0
    for i in range(1, len(values)):
        if values[i] > peak_value:
            peak_value = values[i]
            peak_index = i
    return z[peak_index]


def fold_column(
    master_z: List[Float64],
    source_z: List[Float64],
    source_values: List[Float64],
    table: RiFiTable,
) -> List[Float64]:
    var out = List[Float64]()
    for _ in range(len(master_z)):
        out.append(0.0)

    var weight_sum = 0.0
    for r in range(len(table.offsets_cm)):
        var weight = table.weights[r]
        weight_sum += weight
        for i in range(len(master_z)):
            var shifted_z = master_z[i] + table.offsets_cm[r]
            var value = interp(source_z, source_values, shifted_z)
            if value < 0.0:
                value = 0.0
            out[i] += value * weight

    if weight_sum > 0.0:
        var inv_sum = 1.0 / weight_sum
        for i in range(len(out)):
            out[i] *= inv_sum
    return out^


def interpolate_column(
    source_z: List[Float64],
    source_values: List[Float64],
    target_z: List[Float64],
) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(target_z)):
        out.append(interp(source_z, source_values, target_z[i]))
    return out^


def interpolate_folded_curves(
    energy: Float64,
    lower_energy: Float64,
    lower: DDDCurve,
    upper_energy: Float64,
    upper: DDDCurve,
) raises -> DDDCurve:
    var lower_peak = peak_depth(lower.z, lower.dose)
    var upper_peak = peak_depth(upper.z, upper.dose)
    var target_peak = log_interpolate(lower_energy, upper_energy, lower_peak, upper_peak, energy)
    var target_norm_z = normalized_master_z(lower, lower_peak, upper, upper_peak)
    var target_z = List[Float64]()
    var target_dose = List[Float64]()
    var target_fwhm1 = List[Float64]()
    var target_mix = List[Float64]()
    var target_fwhm2 = List[Float64]()
    var t = (energy - lower_energy) / (upper_energy - lower_energy)
    for i in range(len(target_norm_z)):
        var nz = target_norm_z[i]
        target_z.append(nz * target_peak)
        var nd_lo = interp_normalized_dose(lower, lower_energy, lower_peak, nz)
        var nd_hi = interp_normalized_dose(upper, upper_energy, upper_peak, nz)
        var nd = nd_lo * (1.0 - t) + nd_hi * t
        target_dose.append(nd * ((energy * 12.0) / target_peak))
        target_fwhm1.append(
            interp_normalized_column(lower.z, lower.fwhm1, lower_peak, nz) * (1.0 - t)
            + interp_normalized_column(upper.z, upper.fwhm1, upper_peak, nz) * t
        )
        target_mix.append(
            interp_normalized_column(lower.z, lower.mix, lower_peak, nz) * (1.0 - t)
            + interp_normalized_column(upper.z, upper.mix, upper_peak, nz) * t
        )
        target_fwhm2.append(
            interp_normalized_column(lower.z, lower.fwhm2, lower_peak, nz) * (1.0 - t)
            + interp_normalized_column(upper.z, upper.fwhm2, upper_peak, nz) * t
        )
    return DDDCurve(target_z^, target_dose^, target_fwhm1^, target_mix^, target_fwhm2^)


def normalized_master_z(
    lower: DDDCurve,
    lower_peak: Float64,
    upper: DDDCurve,
    upper_peak: Float64,
) -> List[Float64]:
    var out = List[Float64]()
    if len(upper.z) > len(lower.z):
        for i in range(len(upper.z)):
            out.append(upper.z[i] / upper_peak)
    else:
        for i in range(len(lower.z)):
            out.append(lower.z[i] / lower_peak)
    return out^


def interp_normalized_dose(
    curve: DDDCurve,
    energy: Float64,
    peak: Float64,
    normalized_z: Float64,
) -> Float64:
    return interp_normalized_column(curve.z, curve.dose, peak, normalized_z) / ((energy * 12.0) / peak)


def interp_normalized_column(
    z: List[Float64],
    values: List[Float64],
    peak: Float64,
    normalized_z: Float64,
) -> Float64:
    var absolute_z = normalized_z * peak
    return interp(z, values, absolute_z)


def log_interpolate(x1: Float64, x2: Float64, y1: Float64, y2: Float64, x: Float64) -> Float64:
    var denom = log(x2) - log(x1)
    if denom == 0.0:
        return y1
    return exp((log(y2) - log(y1)) / denom * (log(x) - log(x1)) + log(y1))
