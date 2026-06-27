from std.math import exp, log


@fieldwise_init
struct DDDCurve(Copyable, Movable):
    var z: List[Float64]
    var dose: List[Float64]
    var fwhm1: List[Float64]
    var mix: List[Float64]
    var fwhm2: List[Float64]


def load_ddd(energy: Float64) raises -> DDDCurve:
    var key = round_energy_key(energy)
    var lower_key = ddd_lower_key(key)
    var upper_key = ddd_upper_key(key)
    if lower_key == upper_key:
        return read_ddd_file(ddd_path(lower_key))
    return interpolate_ddd(energy, lower_key, read_ddd_file(ddd_path(lower_key)), upper_key, read_ddd_file(ddd_path(upper_key)))


def read_ddd_file(path: String) raises -> DDDCurve:
    var z = List[Float64]()
    var dose = List[Float64]()
    var fwhm1 = List[Float64]()
    var mix = List[Float64]()
    var fwhm2 = List[Float64]()
    var active = False
    with open(path, "r") as f:
        for raw_line in f.read().split("\n"):
            var line = String(raw_line.strip())
            if line.byte_length() == 0:
                continue
            if line == "!ddd":
                active = True
                continue
            if not active:
                continue
            if line[byte=0] == "!" or line[byte=0] == "#":
                continue
            var parts = line.split()
            if len(parts) < 5:
                continue
            z.append(Float64(parts[0]))
            dose.append(Float64(parts[1]))
            fwhm1.append(Float64(parts[2]))
            mix.append(Float64(parts[3]))
            fwhm2.append(Float64(parts[4]))
    return DDDCurve(z^, dose^, fwhm1^, mix^, fwhm2^)


def interpolate_ddd(energy: Float64, lower_key: Int, lower: DDDCurve, upper_key: Int, upper: DDDCurve) -> DDDCurve:
    var elo = Float64(lower_key) * 0.01
    var ehi = Float64(upper_key) * 0.01
    var t = (energy - elo) / (ehi - elo)
    var lower_peak = peak_depth(lower.z, lower.dose)
    var upper_peak = peak_depth(upper.z, upper.dose)
    var log_t = (log(energy) - log(elo)) / (log(ehi) - log(elo))
    var peak = exp(log(lower_peak) * (1.0 - log_t) + log(upper_peak) * log_t)

    var master = lower.copy()
    var master_peak = lower_peak
    if len(upper.z) > len(lower.z):
        master = upper.copy()
        master_peak = upper_peak

    var x_norm = List[Float64]()
    x_norm.reserve(len(master.z))
    var out_z = List[Float64]()
    out_z.reserve(len(master.z))
    for i in range(len(master.z)):
        var x = master.z[i] / master_peak
        x_norm.append(x)
        out_z.append(x * peak)

    var lower_x = normalize_depth(lower.z, lower_peak)
    var upper_x = normalize_depth(upper.z, upper_peak)
    var lower_dose = normalize_dose(lower.dose, lower_peak, elo)
    var upper_dose = normalize_dose(upper.dose, upper_peak, ehi)

    var out_dose = interpolate_normalized_column(x_norm, lower_x, lower_dose, upper_x, upper_dose, t)
    var dose_scale = (energy * 12.0) / peak
    for i in range(len(out_dose)):
        out_dose[i] = out_dose[i] * dose_scale

    return DDDCurve(
        out_z^,
        out_dose^,
        interpolate_normalized_column(x_norm, lower_x, lower.fwhm1, upper_x, upper.fwhm1, t),
        interpolate_normalized_column(x_norm, lower_x, lower.mix, upper_x, upper.mix, t),
        interpolate_normalized_column(x_norm, lower_x, lower.fwhm2, upper_x, upper.fwhm2, t),
    )


def interp(xs: List[Float64], ys: List[Float64], x: Float64) -> Float64:
    if x < xs[0] or x > xs[len(xs) - 1]:
        return 0.0
    var lo = 0
    var hi = len(xs) - 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if x < xs[mid]:
            hi = mid
        else:
            lo = mid
    var dx = xs[hi] - xs[lo]
    if dx == 0.0:
        return ys[lo]
    var t = (x - xs[lo]) / dx
    return ys[lo] * (1.0 - t) + ys[hi] * t


def interp_clamped(xs: List[Float64], ys: List[Float64], x: Float64) -> Float64:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[len(xs) - 1]:
        return ys[len(ys) - 1]
    return interp(xs, ys, x)


def interpolate_normalized_column(
    x_norm: List[Float64],
    lower_x: List[Float64],
    lower_values: List[Float64],
    upper_x: List[Float64],
    upper_values: List[Float64],
    t: Float64,
) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(x_norm))
    for i in range(len(x_norm)):
        var lo = interp_linear_extrapolated(lower_x, lower_values, x_norm[i])
        var hi = interp_linear_extrapolated(upper_x, upper_values, x_norm[i])
        out.append(lo * (1.0 - t) + hi * t)
    return out^


def interp_linear_extrapolated(xs: List[Float64], ys: List[Float64], x: Float64) -> Float64:
    var lo = 0
    if x > xs[len(xs) - 1]:
        lo = len(xs) - 2
    elif x >= xs[0]:
        var hi = len(xs) - 1
        while hi - lo > 1:
            var mid = (lo + hi) // 2
            if x < xs[mid]:
                hi = mid
            else:
                lo = mid
        if lo >= len(xs) - 1:
            lo = len(xs) - 2
    var dx = xs[lo + 1] - xs[lo]
    if dx == 0.0:
        return ys[lo]
    return ys[lo] + (ys[lo + 1] - ys[lo]) * ((x - xs[lo]) / dx)


def normalize_depth(z: List[Float64], peak: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(z))
    for i in range(len(z)):
        out.append(z[i] / peak)
    return out^


def normalize_dose(dose: List[Float64], peak: Float64, energy: Float64) -> List[Float64]:
    var scale = (energy * 12.0) / peak
    var out = List[Float64]()
    out.reserve(len(dose))
    for i in range(len(dose)):
        out.append(dose[i] / scale)
    return out^


def peak_depth(z: List[Float64], values: List[Float64]) -> Float64:
    var best = values[0]
    var index = 0
    for i in range(1, len(values)):
        if values[i] > best:
            best = values[i]
            index = i
    return z[index]


def peak_depth_for_energy_cm(energy: Float64) raises -> Float64:
    var curve = load_ddd(energy)
    return peak_depth(curve.z, curve.dose)


def energy_for_peak_depth_cm(depth_cm: Float64) raises -> Float64:
    var sis_like = ddd_energy_keys()
    var prev_energy = Float64(sis_like[0]) * 0.01
    var prev_peak = peak_depth_for_energy_cm(prev_energy)
    for i in range(1, len(sis_like)):
        var energy = Float64(sis_like[i]) * 0.01
        var peak = peak_depth_for_energy_cm(energy)
        if depth_cm <= peak:
            return log_interpolate(prev_peak, peak, prev_energy, energy, depth_cm)
        prev_energy = energy
        prev_peak = peak
    return prev_energy


def log_interpolate(x1: Float64, x2: Float64, y1: Float64, y2: Float64, x: Float64) -> Float64:
    var denom = log(x2) - log(x1)
    if denom == 0.0:
        return y1
    return exp((log(y2) - log(y1)) / denom * (log(x) - log(x1)) + log(y1))


def round_energy_key(energy: Float64) -> Int:
    if energy >= 0.0:
        return Int(energy * 100.0 + 0.5)
    return Int(energy * 100.0 - 0.5)


def ddd_lower_key(key: Int) -> Int:
    var keys = ddd_energy_keys()
    var lower = keys[0]
    for i in range(len(keys)):
        if keys[i] <= key:
            lower = keys[i]
    return lower


def ddd_upper_key(key: Int) -> Int:
    var keys = ddd_energy_keys()
    for i in range(len(keys)):
        if keys[i] >= key:
            return keys[i]
    return keys[len(keys) - 1]


def ddd_energy_keys() -> List[Int]:
    var keys = List[Int]()
    keys.append(3000)
    keys.append(4000)
    keys.append(5000)
    keys.append(6000)
    keys.append(7000)
    keys.append(8000)
    keys.append(9000)
    keys.append(10000)
    keys.append(11000)
    keys.append(12000)
    keys.append(13000)
    keys.append(13500)
    keys.append(14000)
    keys.append(15000)
    keys.append(16000)
    keys.append(17000)
    keys.append(18000)
    keys.append(19000)
    keys.append(19500)
    keys.append(20000)
    keys.append(21000)
    keys.append(22000)
    keys.append(23000)
    keys.append(24000)
    keys.append(25000)
    keys.append(26000)
    keys.append(27000)
    keys.append(28000)
    keys.append(29000)
    keys.append(30000)
    keys.append(31000)
    keys.append(32000)
    keys.append(33000)
    keys.append(34000)
    keys.append(35000)
    keys.append(36000)
    keys.append(37000)
    keys.append(38000)
    keys.append(39000)
    keys.append(40000)
    keys.append(41000)
    keys.append(41260)
    keys.append(42000)
    keys.append(43000)
    keys.append(44000)
    keys.append(45000)
    keys.append(46000)
    keys.append(47000)
    keys.append(48000)
    keys.append(49000)
    keys.append(50000)
    return keys^


def ddd_path(key: Int) -> String:
    return "/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/DDD/12C.H2O.MeV" + five_digits(key) + ".ddd"


def five_digits(value: Int) -> String:
    if value < 10:
        return String.write("0000", value)
    if value < 100:
        return String.write("000", value)
    if value < 1000:
        return String.write("00", value)
    if value < 10000:
        return String.write("0", value)
    return String.write(value)


def gaussian_depth_fold(curve: DDDCurve, sigma_cm: Float64) -> DDDCurve:
    if sigma_cm <= 0.0:
        return curve.copy()

    return DDDCurve(
        z=curve.z.copy(),
        dose=_fold_values(curve.z, curve.dose, sigma_cm),
        fwhm1=_fold_values(curve.z, curve.fwhm1, sigma_cm),
        mix=_fold_values(curve.z, curve.mix, sigma_cm),
        fwhm2=_fold_values(curve.z, curve.fwhm2, sigma_cm),
    )


def _fold_values(z: List[Float64], values: List[Float64], sigma_cm: Float64) -> List[Float64]:
    var out = List[Float64]()
    var radius = 2.0 * sigma_cm
    var inv_two_sigma2 = 0.5 / (sigma_cm * sigma_cm)
    for i in range(len(z)):
        var sum = 0.0
        var weight_sum = 0.0
        for j in range(len(z)):
            var dz = z[j] - z[i]
            if dz >= -radius and dz <= radius:
                var weight = exp(-dz * dz * inv_two_sigma2)
                sum += values[j] * weight
                weight_sum += weight
        if weight_sum > 0.0:
            out.append(sum / weight_sum)
        else:
            out.append(values[i])
    return out^
