from std.memory import bitcast
from dedx import DEDXProjectile, DEDXTable, dedx_stopping_power, find_projectile
from rbe_lowdose import RBELowDoseTable, find_rbe_projectile, low_dose_values_at_energy


@fieldwise_init
struct SPCSpecies(Copyable, Movable):
    var charge: Float64
    var mass: Float64
    var atomic_number: Int
    var mass_number: Int
    var cumulative: Float64
    var coefficient_count: Int
    var energy_edges: List[Float64]
    var histo: List[Float64]


@fieldwise_init
struct SPCDepth(Copyable, Movable):
    var z_cm: Float64
    var normalization: Float64
    var species_count: Int
    var species: List[SPCSpecies]


@fieldwise_init
struct SPCBeam(Copyable, Movable):
    var energy_mev_u: Float64
    var peak_cm: Float64
    var normalization: Float64
    var depths: List[SPCDepth]


@fieldwise_init
struct SPCMoments(Copyable, Movable):
    var sum: Float64
    var mean: Float64
    var variance: Float64
    var skewness: Float64


@fieldwise_init
struct SPCLETMoments(Copyable, Movable):
    var let_sum: Float64
    var let_mean: Float64
    var dose_mean_let_sum: Float64
    var dose_mean_let_mean: Float64


@fieldwise_init
struct SPCRBECoefficients(Copyable, Movable):
    var alpha0x1: Float64
    var sqrt_beta0x1: Float64
    var alpha_let_sum: Float64
    var alpha_let_mean: Float64
    var beta_let_sum: Float64
    var beta_let_mean: Float64


@fieldwise_init
struct SPCBioCoefficients(Copyable, Movable):
    var depth_cm: Float64
    var alpha0x1: Float64
    var sqrt_beta0x1: Float64
    var let_mix: Float64
    var let_bar: Float64


def read_spc_beam(path: String) raises -> SPCBeam:
    var payload: List[UInt8]
    with open(path, "r") as f:
        payload = f.read_bytes()
    if payload.byte_length() < 16:
        raise Error("SPC file is too small")
    var little_endian = False
    if be_u32(payload, 0) != 1:
        if le_u32(payload, 0) == 1:
            little_endian = True
        else:
            raise Error("SPC file does not start with filetype tag")
    var filetype_size = tagged_u32(payload, 4, little_endian)
    if filetype_size < 4:
        raise Error("SPC filetype tag is truncated")
    if not (
        payload[8] == UInt8(0x53)
        and payload[9] == UInt8(0x50)
        and payload[10] == UInt8(0x43)
        and (payload[11] == UInt8(0x4D) or payload[11] == UInt8(0x49))
    ):
        raise Error("Unsupported SPC filetype")

    var energy = 0.0
    var peak = 0.0
    var normalization = 0.0
    var expected_depths = 0
    var depths = List[SPCDepth]()
    var current_z = 0.0
    var current_norm = 0.0
    var current_species_count = 0
    var current_species = List[SPCSpecies]()
    var species_charge = 0.0
    var species_mass = 0.0
    var species_atomic_number = 0
    var species_mass_number = 0
    var species_cumulative = 0.0
    var species_coefficient_count = 0
    var species_energy_count = 0
    var species_energy_edges = List[Float64]()
    var have_depth = False
    var have_species = False
    var offset = 0
    while offset + 8 <= payload.byte_length():
        var tag = Int(tagged_u32(payload, offset, little_endian))
        var size = Int(tagged_u32(payload, offset + 4, little_endian))
        var data = offset + 8
        if data + size > payload.byte_length():
            raise Error("SPC tag extends past end of file")
        if tag == 6:
            energy = tagged_f64(payload, data, little_endian)
        elif tag == 7:
            peak = tagged_f64(payload, data, little_endian)
        elif tag == 8:
            normalization = tagged_f64(payload, data, little_endian)
        elif tag == 9:
            expected_depths = Int(tagged_u64(payload, data, little_endian))
        elif tag == 10:
            current_z = tagged_f64(payload, data, little_endian)
            current_norm = 0.0
            current_species_count = 0
            current_species = List[SPCSpecies]()
            have_depth = True
        elif tag == 11 and have_depth:
            current_norm = tagged_f64(payload, data, little_endian)
        elif tag == 12 and have_depth:
            current_species_count = Int(tagged_u64(payload, data, little_endian))
        elif tag == 13 and have_depth:
            species_charge = tagged_f64(payload, data, little_endian)
            species_mass = tagged_f64(payload, data + 8, little_endian)
            species_atomic_number = tagged_i32(payload, data + 16, little_endian)
            species_mass_number = tagged_i32(payload, data + 20, little_endian)
            species_cumulative = 0.0
            species_coefficient_count = 0
            species_energy_count = 0
            species_energy_edges = List[Float64]()
            have_species = True
        elif tag == 14 and have_species:
            species_cumulative = tagged_f64(payload, data, little_endian)
        elif tag == 15 and have_species:
            species_coefficient_count = Int(tagged_u64(payload, data, little_endian))
        elif tag == 16 and have_species:
            species_energy_count = Int(tagged_u64(payload, data, little_endian))
        elif tag == 17 and have_species:
            species_energy_edges = read_tagged_f64_list(payload, data, species_energy_count + 1, little_endian)
        elif tag == 18 and have_species:
            var ref_index = Int(tagged_u64(payload, data, little_endian))
            if ref_index < 0 or ref_index >= len(current_species):
                raise Error("SPC energy reference is out of range")
            species_energy_edges = current_species[ref_index].energy_edges.copy()
        elif tag == 19 and have_species:
            current_species.append(SPCSpecies(
                species_charge,
                species_mass,
                species_atomic_number,
                species_mass_number,
                species_cumulative,
                species_coefficient_count,
                species_energy_edges.copy(),
                read_tagged_f64_list(payload, data, species_energy_count, little_endian),
            ))
            have_species = False
            if len(current_species) == current_species_count:
                depths.append(SPCDepth(current_z, current_norm, current_species_count, current_species.copy()))
                have_depth = False
        offset = data + size
    if expected_depths != 0 and len(depths) != expected_depths:
        raise Error("SPC depth count mismatch")
    return SPCBeam(energy, peak, normalization, depths^)


def read_spc_peak_cm(path: String) raises -> Float64:
    var payload: List[UInt8]
    with open(path, "r") as f:
        payload = f.read_bytes()
    if payload.byte_length() < 16:
        raise Error("SPC file is too small")
    var little_endian = False
    if be_u32(payload, 0) != 1:
        if le_u32(payload, 0) == 1:
            little_endian = True
        else:
            raise Error("SPC file does not start with filetype tag")
    var offset = 0
    while offset + 8 <= payload.byte_length():
        var tag = Int(tagged_u32(payload, offset, little_endian))
        var size = Int(tagged_u32(payload, offset + 4, little_endian))
        var data = offset + 8
        if data + size > payload.byte_length():
            raise Error("SPC tag extends past end of file")
        if tag == 7:
            return tagged_f64(payload, data, little_endian)
        offset = data + size
    raise Error("SPC peak tag is missing")


def read_spc_beam_be(path: String) raises -> SPCBeam:
    var payload: List[UInt8]
    with open(path, "r") as f:
        payload = f.read_bytes()
    if payload.byte_length() < 16:
        raise Error("SPC file is too small")
    if be_u32(payload, 0) != 1:
        raise Error("SPC file does not start with filetype tag")
    if be_u32(payload, 4) < 4:
        raise Error("SPC filetype tag is truncated")
    if not (
        payload[8] == UInt8(0x53)
        and payload[9] == UInt8(0x50)
        and payload[10] == UInt8(0x43)
        and (payload[11] == UInt8(0x4D) or payload[11] == UInt8(0x49))
    ):
        raise Error("Unsupported SPC filetype")

    var energy = 0.0
    var peak = 0.0
    var normalization = 0.0
    var expected_depths = 0
    var depths = List[SPCDepth]()
    var current_z = 0.0
    var current_norm = 0.0
    var current_species_count = 0
    var current_species = List[SPCSpecies]()
    var species_charge = 0.0
    var species_mass = 0.0
    var species_atomic_number = 0
    var species_mass_number = 0
    var species_cumulative = 0.0
    var species_coefficient_count = 0
    var species_energy_count = 0
    var species_energy_edges = List[Float64]()
    var have_depth = False
    var have_species = False
    var offset = 0
    while offset + 8 <= payload.byte_length():
        var tag = be_u32(payload, offset)
        var size = Int(be_u32(payload, offset + 4))
        var data = offset + 8
        if data + size > payload.byte_length():
            raise Error("SPC tag extends past end of file")
        if tag == 6:
            energy = be_f64(payload, data)
        elif tag == 7:
            peak = be_f64(payload, data)
        elif tag == 8:
            normalization = be_f64(payload, data)
        elif tag == 9:
            expected_depths = Int(be_u64(payload, data))
        elif tag == 10:
            current_z = be_f64(payload, data)
            current_norm = 0.0
            current_species_count = 0
            current_species = List[SPCSpecies]()
            have_depth = True
        elif tag == 11 and have_depth:
            current_norm = be_f64(payload, data)
        elif tag == 12 and have_depth:
            current_species_count = Int(be_u64(payload, data))
        elif tag == 13 and have_depth:
            species_charge = be_f64(payload, data)
            species_mass = be_f64(payload, data + 8)
            species_atomic_number = be_i32(payload, data + 16)
            species_mass_number = be_i32(payload, data + 20)
            species_cumulative = 0.0
            species_coefficient_count = 0
            species_energy_count = 0
            species_energy_edges = List[Float64]()
            have_species = True
        elif tag == 14 and have_species:
            species_cumulative = be_f64(payload, data)
        elif tag == 15 and have_species:
            species_coefficient_count = Int(be_u64(payload, data))
        elif tag == 16 and have_species:
            species_energy_count = Int(be_u64(payload, data))
        elif tag == 17 and have_species:
            species_energy_edges = read_f64_list(payload, data, species_energy_count + 1)
        elif tag == 18 and have_species:
            var ref_index = Int(be_u64(payload, data))
            if ref_index < 0 or ref_index >= len(current_species):
                raise Error("SPC energy reference is out of range")
            species_energy_edges = current_species[ref_index].energy_edges.copy()
        elif tag == 19 and have_species:
            current_species.append(SPCSpecies(
                species_charge,
                species_mass,
                species_atomic_number,
                species_mass_number,
                species_cumulative,
                species_coefficient_count,
                species_energy_edges.copy(),
                read_f64_list(payload, data, species_energy_count),
            ))
            have_species = False
            if len(current_species) == current_species_count:
                depths.append(SPCDepth(current_z, current_norm, current_species_count, current_species.copy()))
                have_depth = False
        offset = data + size
    if expected_depths != 0 and len(depths) != expected_depths:
        raise Error("SPC depth count mismatch")
    return SPCBeam(energy, peak, normalization, depths^)


def tagged_u32(payload: List[UInt8], offset: Int, little_endian: Bool) -> UInt32:
    if little_endian:
        return le_u32(payload, offset)
    return be_u32(payload, offset)


def tagged_u64(payload: List[UInt8], offset: Int, little_endian: Bool) -> UInt64:
    if little_endian:
        return (
            UInt64(le_u32(payload, offset))
            | (UInt64(le_u32(payload, offset + 4)) << 32)
        )
    return be_u64(payload, offset)


def tagged_i32(payload: List[UInt8], offset: Int, little_endian: Bool) -> Int:
    return Int(bitcast[DType.int32](tagged_u32(payload, offset, little_endian)))


def tagged_f64(payload: List[UInt8], offset: Int, little_endian: Bool) -> Float64:
    return bitcast[DType.float64](tagged_u64(payload, offset, little_endian))


def read_tagged_f64_list(payload: List[UInt8], offset: Int, count: Int, little_endian: Bool) -> List[Float64]:
    var values = List[Float64]()
    for i in range(count):
        values.append(tagged_f64(payload, offset + i * 8, little_endian))
    return values^


def be_u32(payload: List[UInt8], offset: Int) -> UInt32:
    return (
        (UInt32(payload[offset]) << 24)
        | (UInt32(payload[offset + 1]) << 16)
        | (UInt32(payload[offset + 2]) << 8)
        | UInt32(payload[offset + 3])
    )


def le_u32(payload: List[UInt8], offset: Int) -> UInt32:
    return (
        UInt32(payload[offset])
        | (UInt32(payload[offset + 1]) << 8)
        | (UInt32(payload[offset + 2]) << 16)
        | (UInt32(payload[offset + 3]) << 24)
    )


def be_u64(payload: List[UInt8], offset: Int) -> UInt64:
    return (
        (UInt64(be_u32(payload, offset)) << 32)
        | UInt64(be_u32(payload, offset + 4))
    )


def be_i32(payload: List[UInt8], offset: Int) -> Int:
    return Int(bitcast[DType.int32](be_u32(payload, offset)))


def be_f64(payload: List[UInt8], offset: Int) -> Float64:
    return bitcast[DType.float64](be_u64(payload, offset))


def read_f64_list(payload: List[UInt8], offset: Int, count: Int) -> List[Float64]:
    var values = List[Float64]()
    for i in range(count):
        values.append(be_f64(payload, offset + i * 8))
    return values^


def species_energy_moments(species: SPCSpecies) -> SPCMoments:
    var sum = 0.0
    var weighted = 0.0
    for i in range(len(species.histo)):
        var width = species.energy_edges[i + 1] - species.energy_edges[i]
        var count = species.histo[i] * width
        var center = (species.energy_edges[i + 1] + species.energy_edges[i]) * 0.5
        sum += count
        weighted += count * center
    var mean = 0.0
    if sum > 0.0:
        mean = weighted / sum

    var variance = 0.0
    var skewness = 0.0
    for i in range(len(species.histo)):
        var width = species.energy_edges[i + 1] - species.energy_edges[i]
        var count = species.histo[i] * width
        var center = (species.energy_edges[i + 1] + species.energy_edges[i]) * 0.5
        var delta = center - mean
        variance += delta * delta * count
        skewness += delta * delta * delta * count
    if sum > 0.0:
        variance /= sum
        skewness /= sum
    return SPCMoments(sum, mean, variance, skewness)


def species_let_moments(species: SPCSpecies, projectile: DEDXProjectile) raises -> SPCLETMoments:
    var let_sum = 0.0
    var let_weighted = 0.0
    var dose_mean_let_sum = 0.0
    var dose_mean_let_weighted = 0.0
    for i in range(len(species.histo)):
        var width = species.energy_edges[i + 1] - species.energy_edges[i]
        var count = species.histo[i] * width
        var center = (species.energy_edges[i + 1] + species.energy_edges[i]) * 0.5
        var stopping = dedx_stopping_power(projectile, center)
        let_sum += count
        let_weighted += count * stopping
        dose_mean_let_sum += count * stopping
        dose_mean_let_weighted += count * stopping * stopping
    var let_mean = 0.0
    if let_sum > 0.0:
        let_mean = let_weighted / let_sum
    var dose_mean_let_mean = 0.0
    if dose_mean_let_sum > 0.0:
        dose_mean_let_mean = dose_mean_let_weighted / dose_mean_let_sum
    return SPCLETMoments(let_sum, let_mean, dose_mean_let_sum, dose_mean_let_mean)


def depth_let_moments(depth: SPCDepth, dedx: DEDXTable) raises -> SPCLETMoments:
    var let_sum = 0.0
    var let_weighted = 0.0
    var dose_mean_let_sum = 0.0
    var dose_mean_let_weighted = 0.0
    for i in range(len(depth.species)):
        var projectile = find_projectile(
            dedx,
            depth.species[i].atomic_number,
            depth.species[i].mass_number,
        )
        var moments = species_let_moments(depth.species[i], projectile)
        let_sum += moments.let_sum
        let_weighted += moments.let_sum * moments.let_mean
        dose_mean_let_sum += moments.dose_mean_let_sum
        dose_mean_let_weighted += moments.dose_mean_let_sum * moments.dose_mean_let_mean
    var let_mean = 0.0
    if let_sum > 0.0:
        let_mean = let_weighted / let_sum
    var dose_mean_let_mean = 0.0
    if dose_mean_let_sum > 0.0:
        dose_mean_let_mean = dose_mean_let_weighted / dose_mean_let_sum
    return SPCLETMoments(let_sum, let_mean, dose_mean_let_sum, dose_mean_let_mean)


def depth_rbe_coefficients(
    depth: SPCDepth,
    dedx: DEDXTable,
    rbe: RBELowDoseTable,
) raises -> SPCRBECoefficients:
    var alpha_let_sum = 0.0
    var alpha_let_weighted = 0.0
    var beta_let_sum = 0.0
    var beta_let_weighted = 0.0
    for i in range(len(depth.species)):
        var dedx_projectile = find_projectile(
            dedx,
            depth.species[i].atomic_number,
            depth.species[i].mass_number,
        )
        var rbe_projectile = find_rbe_projectile(
            rbe,
            depth.species[i].atomic_number,
            depth.species[i].mass_number,
        )
        for e in range(len(depth.species[i].histo)):
            var width = depth.species[i].energy_edges[e + 1] - depth.species[i].energy_edges[e]
            var count = depth.species[i].histo[e] * width
            var center = (depth.species[i].energy_edges[e + 1] + depth.species[i].energy_edges[e]) * 0.5
            var stopping = dedx_stopping_power(dedx_projectile, center)
            var low = low_dose_values_at_energy(rbe, rbe_projectile, dedx_projectile, center)
            alpha_let_sum += count * stopping
            alpha_let_weighted += count * low.alpha_low_dose * stopping
            beta_let_sum += count * stopping
            beta_let_weighted += count * low.beta_low_dose * stopping
    var alpha_mean = 0.0
    if alpha_let_sum > 0.0:
        alpha_mean = alpha_let_weighted / alpha_let_sum
    var beta_mean = 0.0
    if beta_let_sum > 0.0:
        beta_mean = beta_let_weighted / beta_let_sum
    return SPCRBECoefficients(
        alpha_let_weighted,
        sqrt_approx_spc(beta_let_weighted * beta_let_sum),
        alpha_let_sum,
        alpha_mean,
        beta_let_sum,
        beta_mean,
    )


def depth_bio_coefficients(
    depth: SPCDepth,
    dedx: DEDXTable,
    rbe: RBELowDoseTable,
) raises -> SPCBioCoefficients:
    var rbe_coeffs = depth_rbe_coefficients(depth, dedx, rbe)
    var let = depth_let_moments(depth, dedx)
    return SPCBioCoefficients(
        depth.z_cm,
        rbe_coeffs.alpha0x1,
        rbe_coeffs.sqrt_beta0x1,
        let.let_sum * let.let_mean,
        0.1 * let.dose_mean_let_sum * let.dose_mean_let_mean,
    )


def interpolate_spc_bio_coefficients(
    low: SPCBioCoefficients,
    high: SPCBioCoefficients,
    depth_cm: Float64,
) -> SPCBioCoefficients:
    var fraction = depth_cm - low.depth_cm
    var width = high.depth_cm - low.depth_cm
    if width != 0.0:
        fraction /= width
    var low_weight = 1.0 - fraction
    var alpha0x1 = low_weight * low.alpha0x1 + fraction * high.alpha0x1
    var sqrt_beta0x1 = low_weight * low.sqrt_beta0x1 + fraction * high.sqrt_beta0x1
    var let_mix = low_weight * low.let_mix + fraction * high.let_mix
    var let_bar = low_weight * low.let_bar + fraction * high.let_bar
    if alpha0x1 < 0.0 or sqrt_beta0x1 < 0.0 or let_mix < 0.0 or let_bar < 0.0:
        alpha0x1 = 0.0
        sqrt_beta0x1 = 0.0
        let_mix = 0.0
        let_bar = 0.0
    return SPCBioCoefficients(
        depth_cm,
        alpha0x1,
        sqrt_beta0x1,
        let_mix,
        let_bar,
    )


def sqrt_approx_spc(value: Float64) -> Float64:
    from std.math import sqrt

    return sqrt(value)
