
from std.math import exp, log, sqrt


@fieldwise_init
struct DEDXProjectile(Copyable, Movable):
    var projectile: String
    var atomic_number: Int
    var mass_number: Int
    var energy: List[Float64]
    var stopping_power: List[Float64]
    var range: List[Float64]


@fieldwise_init
struct DEDXTable(Copyable, Movable):
    var material: String
    var density: Float64
    var projectiles: List[DEDXProjectile]


def read_dedx(path: String) raises -> DEDXTable:
    var material = ""
    var density = 0.0
    var projectiles = List[DEDXProjectile]()
    var current_projectile = ""
    var current_z = 0
    var current_a = 0
    var energy = List[Float64]()
    var stopping = List[Float64]()
    var ranges = List[Float64]()
    var in_dedx = False

    with open(path, "r") as f:
        for raw_line in f.read().split("\n"):
            var line = String(raw_line.strip())
            if line.byte_length() == 0 or line[byte=0] == "#":
                continue
            var parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == "!material":
                if len(parts) > 1:
                    material = String(parts[1])
                continue
            if parts[0] == "!density":
                if len(parts) > 1:
                    density = Float64(parts[1])
                continue
            if parts[0] == "!projectile":
                if current_projectile != "":
                    projectiles.append(DEDXProjectile(
                        current_projectile,
                        current_z,
                        current_a,
                        energy.copy(),
                        stopping.copy(),
                        ranges.copy(),
                    ))
                current_projectile = String(parts[1])
                var za = parse_projectile(current_projectile)
                current_z = za.atomic_number
                current_a = za.mass_number
                energy = List[Float64]()
                stopping = List[Float64]()
                ranges = List[Float64]()
                in_dedx = False
                continue
            if parts[0] == "!dedx":
                in_dedx = True
                continue
            if line[byte=0] == "!":
                in_dedx = False
                continue
            if in_dedx:
                if len(parts) < 2:
                    raise Error("Malformed dE/dx row")
                energy.append(Float64(parts[0]))
                stopping.append(Float64(parts[1]))
                if len(parts) > 2:
                    ranges.append(Float64(parts[2]))
                else:
                    ranges.append(0.0)

    if current_projectile != "":
        projectiles.append(DEDXProjectile(
            current_projectile,
            current_z,
            current_a,
            energy.copy(),
            stopping.copy(),
            ranges.copy(),
        ))
    return DEDXTable(material, density, projectiles^)


@fieldwise_init
struct ProjectileZA(Copyable, Movable):
    var atomic_number: Int
    var mass_number: Int


def parse_projectile(token: String) raises -> ProjectileZA:
    var split = 0
    while split < token.byte_length():
        var c = token[byte=split]
        if c < "0" or c > "9":
            break
        split += 1
    if split == 0:
        raise Error("Projectile mass number is missing")
    var mass_number = Int(substr(token, 0, split))
    var symbol = substr(token, split, token.byte_length())
    return ProjectileZA(element_atomic_number(symbol), mass_number)


def element_atomic_number(symbol: String) raises -> Int:
    if symbol == "H":
        return 1
    if symbol == "He":
        return 2
    if symbol == "Li":
        return 3
    if symbol == "Be":
        return 4
    if symbol == "B":
        return 5
    if symbol == "C":
        return 6
    if symbol == "N":
        return 7
    if symbol == "O":
        return 8
    if symbol == "F":
        return 9
    if symbol == "Ne":
        return 10
    if symbol == "Ar":
        return 18
    if symbol == "Ni":
        return 28
    if symbol == "Kr":
        return 36
    if symbol == "Xe":
        return 54
    if symbol == "Au":
        return 79
    if symbol == "U":
        return 92
    raise Error("Unsupported projectile element")


def find_projectile(table: DEDXTable, atomic_number: Int, mass_number: Int) raises -> DEDXProjectile:
    for i in range(len(table.projectiles)):
        if table.projectiles[i].atomic_number == atomic_number and table.projectiles[i].mass_number == mass_number:
            return table.projectiles[i].copy()
    raise Error("dE/dx projectile was not found")


def dedx_stopping_power(projectile: DEDXProjectile, energy_mev_u: Float64) raises -> Float64:
    if energy_mev_u <= 0.0:
        return 0.0
    var x = List[Float64]()
    var y = List[Float64]()
    for i in range(len(projectile.energy)):
        var beta2_value = beta2(projectile.energy[i])
        x.append(log(projectile.energy[i]))
        y.append(log(projectile.stopping_power[i] * beta2_value))
    var coeffs = spline_local_coefficients(x, y)
    var log_energy = log(energy_mev_u)
    if log_energy < coeffs[0] or log_energy > coeffs[len(projectile.energy) - 1]:
        return 0.0
    return exp(spline_eval(coeffs, len(projectile.energy), log_energy)) / beta2(energy_mev_u)


def beta2(energy_mev_u: Float64) -> Float64:
    var gamma_inv = 931.481 / (931.481 + energy_mev_u)
    return 1.0 - gamma_inv * gamma_inv


def spline_local_coefficients(x: List[Float64], y: List[Float64]) raises -> List[Float64]:
    var n = len(x)
    if n < 2:
        raise Error("Need at least two points for spline")
    var coeff = List[Float64]()
    for _ in range(n * 5):
        coeff.append(0.0)
    if n < 3:
        coeff[0] = x[0]
        coeff[n] = y[0]
        coeff[n * 2] = 0.0
        if x[1] - x[0] != 0.0:
            coeff[n * 2] = (y[1] - y[0]) / (x[1] - x[0])
        coeff[n * 3] = 0.0
        coeff[n * 4] = 0.0
        coeff[1] = x[1]
        coeff[n + 1] = y[1]
        coeff[n * 2 + 1] = coeff[n * 2]
        coeff[n * 3 + 1] = 0.0
        coeff[n * 4 + 1] = 0.0
        return coeff^

    for i in range(n - 1):
        var dh = x[i + 1] - x[i]
        if dh <= 0.0:
            raise Error("Spline x values must increase")
        coeff[n + i] = (y[i + 1] - y[i]) / dh
    for i in range(1, n - 1):
        coeff[n * 2 + i] = (coeff[n + i] + coeff[n + i - 1]) * 0.5
    coeff[n * 2] = coeff[n]
    coeff[n * 2 + n - 1] = coeff[n + n - 2]
    for i in range(n - 2):
        if abs(coeff[n + i]) <= eps64():
            coeff[n * 2 + i + 1] = 0.0
            coeff[n * 2 + i] = 0.0
        elif coeff[n + i] * coeff[n + i + 1] < 0.0:
            coeff[n * 2 + i + 1] = 0.0
    if abs(coeff[n + n - 2]) <= eps64():
        coeff[n * 2 + n - 1] = 0.0
        coeff[n * 2 + n - 2] = 0.0

    for i in range(n - 1):
        if coeff[n * 2 + i] * coeff[n + i] < 0.0:
            coeff[n * 2 + i] = 0.0
        if coeff[n * 2 + i + 1] * coeff[n + i] < 0.0:
            coeff[n * 2 + i + 1] = 0.0
    if coeff[n * 2 + n - 1] * coeff[n + n - 2] < 0.0:
        coeff[n * 2 + n - 1] = 0.0

    for i in range(n - 1):
        var dpk = 0.0
        var dqk = 0.0
        if abs(coeff[n * 2 + i]) + abs(coeff[n * 2 + i + 1]) > eps64():
            var dh = 1.0 / (coeff[n + i] * 3.0)
            var drk = (coeff[n * 2 + i] + coeff[n * 2 + i] + coeff[n * 2 + i + 1]) * dh
            var dsk = (coeff[n * 2 + i + 1] + coeff[n * 2 + i + 1] + coeff[n * 2 + i]) * dh
            if (1.0 - drk) * (dsk - 1.0) < 0.0:
                dpk = min_float(drk, dsk)
        if abs(coeff[n + i]) > eps64():
            var dh = 1.0 / (coeff[n + i] * 3.0)
            dqk = (coeff[n * 2 + i] + coeff[n * 2 + i + 1]) * dh - sqrt(coeff[n * 2 + i + 1] * coeff[n * 2 + i]) * abs(dh)
        coeff[n + i] = max_float(max_float(dqk, dpk), 1.0)
    coeff[n * 2] = coeff[n * 2] / coeff[n]
    coeff[n * 2 + n - 1] = coeff[n * 2 + n - 1] / coeff[n + n - 2]
    for i in range(n - 2):
        var dh = max_float(coeff[n + i], coeff[n + i + 1])
        coeff[n * 2 + i + 1] = coeff[n * 2 + i + 1] / dh

    for i in range(n - 1):
        var dh = 1.0 / (x[i + 1] - x[i])
        var dydx = (y[i + 1] - y[i]) * dh
        coeff[n + i] = y[i]
        coeff[n * 3 + i] = (dydx * 3.0 - coeff[n * 2 + i] * 2.0 - coeff[n * 2 + i + 1]) * dh
        coeff[n * 4 + i] = (dydx * -2.0 + coeff[n * 2 + i] + coeff[n * 2 + i + 1]) * dh * dh
    coeff[n + n - 1] = y[n - 1]
    coeff[n * 3 + n - 1] = 0.0
    coeff[n * 4 + n - 1] = 0.0
    for i in range(n):
        coeff[i] = x[i]
    return coeff^


def spline_eval(coeff: List[Float64], n: Int, x: Float64) -> Float64:
    var lo = 0
    var hi = n
    var ii: Int
    while True:
        ii = (lo + hi) // 2
        if ii == lo:
            break
        if x == coeff[ii]:
            break
        if x < coeff[ii]:
            hi = ii
        else:
            lo = ii
    if ii + 1 >= n:
        ii -= 1
    var dx = x - coeff[ii]
    if dx == 0.0:
        return coeff[n + ii]
    var value = 0.0
    for kk in range(4):
        value *= dx
        value += coeff[(4 - kk) * n + ii]
    return value


def eps64() -> Float64:
    return 2.220446049250313e-16


def min_float(a: Float64, b: Float64) -> Float64:
    if a < b:
        return a
    return b


def max_float(a: Float64, b: Float64) -> Float64:
    if a > b:
        return a
    return b


def substr(text: String, start: Int, end: Int) -> String:
    var out = String()
    for i in range(start, end):
        out += String(text[byte=i])
    return out^
