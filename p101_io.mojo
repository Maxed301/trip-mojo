from std.memory import bitcast
from std.math import sqrt
from geometry import Vec3
from nrrd import NrrdHeader, payload_f32, read_nrrd_header, read_nrrd_payload
from voi import VOI, load_voi_from_binfo


def load_tumor_world_points() raises -> List[Vec3]:
    var tumor = load_voi_from_binfo("/home/max/Projects/beamline_transport/CT/P101.binfo", "Tumor")
    var points = List[Vec3]()
    points.reserve(len(tumor))
    for i in range(len(tumor)):
        points.append(tumor.active_world_point(i))
    return points^


def p101_reference_path() -> String:
    return "reference/trip4d_p101_cpu_20260511_140457"


def write_tumor_dose(path: String, dose: List[Float64]) raises:
    var tumor = load_voi_from_binfo("/home/max/Projects/beamline_transport/CT/P101.binfo", "Tumor")
    if len(dose) != len(tumor):
        raise Error("Dose length does not match P101 tumor VOI length")

    var voxel_count = tumor.grid.shape.i * tumor.grid.shape.j * tumor.grid.shape.k
    var payload = List[UInt8]()
    payload.resize(voxel_count * 4, 0)
    for i in range(len(dose)):
        write_le_f32(payload, tumor.active_indices[i] * 4, Float32(dose[i]))

    var data_file = raw_data_file_for_nrrd(path)
    with open(path, "w") as f:
        f.write(nrrd_float_header(data_file, tumor))
    with open(data_path_for_nrrd(path, data_file), "w") as f:
        f.write_bytes(payload)


def write_le_f32(mut payload: List[UInt8], offset: Int, value: Float32):
    var bits = bitcast[DType.uint32](value)
    payload[offset] = UInt8(bits & UInt32(0xFF))
    payload[offset + 1] = UInt8((bits >> 8) & UInt32(0xFF))
    payload[offset + 2] = UInt8((bits >> 16) & UInt32(0xFF))
    payload[offset + 3] = UInt8((bits >> 24) & UInt32(0xFF))


def nrrd_float_header(data_file: String, tumor: VOI) -> String:
    var grid = tumor.grid.copy()
    var m = grid.index_to_world.copy()
    return String.write(
        "NRRD0005\n",
        "type: float\n",
        "dimension: 3\n",
        "space: left-posterior-superior\n",
        "sizes: ", grid.shape.i, " ", grid.shape.j, " ", grid.shape.k, "\n",
        "space directions: (", m[0, 0], ",", m[1, 0], ",", m[2, 0], ") (", m[0, 1], ",", m[1, 1], ",", m[2, 1], ") (", m[0, 2], ",", m[1, 2], ",", m[2, 2], ")\n",
        "kinds: domain domain domain\n",
        "endian: little\n",
        "encoding: raw\n",
        "space origin: (", m[0, 3], ",", m[1, 3], ",", m[2, 3], ")\n",
        "data file: ", data_file, "\n",
    )


def raw_data_file_for_nrrd(path: String) -> String:
    return without_nrrd_suffix(basename(path)) + ".raw"


def data_path_for_nrrd(path: String, data_file: String) -> String:
    var base = dirname(path)
    if base == ".":
        return data_file
    return base + "/" + data_file


def basename(path: String) -> String:
    var last = -1
    for i in range(path.byte_length()):
        if path[byte=i] == "/":
            last = i
    var out = String()
    for i in range(last + 1, path.byte_length()):
        out += String(path[byte=i])
    return out^


def dirname(path: String) -> String:
    var last = -1
    for i in range(path.byte_length()):
        if path[byte=i] == "/":
            last = i
    if last < 0:
        return "."
    var out = String()
    for i in range(last):
        out += String(path[byte=i])
    return out^


def without_nrrd_suffix(name: String) -> String:
    if name.byte_length() <= 5:
        return String(name)
    var suffix_start = name.byte_length() - 5
    if String(name[byte=suffix_start]) == "." and String(name[byte=suffix_start + 1]) == "n" and String(name[byte=suffix_start + 2]) == "r" and String(name[byte=suffix_start + 3]) == "r" and String(name[byte=suffix_start + 4]) == "d":
        var out = String()
        for i in range(suffix_start):
            out += String(name[byte=i])
        return out^
    return String(name)


def clamp_int(value: Int, lo: Int, hi: Int) -> Int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def compare_to_trip_bio(path: String) raises:
    var stats = compare_tumor_dose(path, read_latest_reference_path_native() + "/3Ddose_P101_2110_iGTV.bio.nrrd", True)
    write_bio_stats_json("/tmp/trip_mojo_modular_bio_stats.json", stats)
    print("Bio relative RMSE %:", stats.relative_rmse_percent)
    print("Bio mean error %:", stats.mean_error_percent)
    print("Bio error range:", stats.min_error, stats.max_error, "max abs:", stats.max_abs_error)
    enforce_relative_rmse_limit("Bio", stats)


def compare_to_trip_phys(path: String) raises:
    var stats = compare_tumor_dose(path, read_latest_reference_path_native() + "/3Ddose_P101_2110_iGTV.phys.nrrd", False)
    write_phys_stats_json("/tmp/trip_mojo_modular_phys_stats.json", stats)
    print("Phys relative RMSE %:", stats.relative_rmse_percent)
    print("Phys mean error %:", stats.mean_error_percent)
    print("Phys error range:", stats.min_error, stats.max_error, "max abs:", stats.max_abs_error)
    enforce_relative_rmse_limit("Phys", stats)


@fieldwise_init
struct TumorDoseStats(Copyable, Movable):
    var trip_mean: Float64
    var trip_std: Float64
    var trip_rmse_to_3: Float64
    var mojo_mean: Float64
    var mojo_std: Float64
    var mojo_rmse_to_3: Float64
    var mojo_vs_trip_rmse: Float64
    var relative_rmse_percent: Float64
    var mean_error_percent: Float64
    var max_abs_error: Float64
    var min_error: Float64
    var max_error: Float64


def compare_tumor_dose(mojo_path: String, trip_path: String, include_rmse_to_3: Bool) raises -> TumorDoseStats:
    var tumor = load_voi_from_binfo("/home/max/Projects/beamline_transport/CT/P101.binfo", "Tumor")
    var trip_header = read_nrrd_header(trip_path)
    var mojo_header = read_nrrd_header(mojo_path)
    validate_same_grid(tumor, trip_header)
    validate_same_grid(tumor, mojo_header)
    var trip_payload = read_nrrd_payload(trip_header)
    var mojo_payload = read_nrrd_payload(mojo_header)

    var count = Float64(len(tumor))
    var trip_sum = 0.0
    var mojo_sum = 0.0
    var trip_sum2 = 0.0
    var mojo_sum2 = 0.0
    var abs_trip_sum = 0.0
    var diff2 = 0.0
    var max_abs_error = 0.0
    var min_error = 0.0
    var max_error = 0.0
    var max_abs_error_voxel = -1
    var max_abs_error_grid_i = 0
    var max_abs_error_grid_j = 0
    var max_abs_error_grid_k = 0
    var max_abs_error_trip = 0.0
    var max_abs_error_mojo = 0.0
    var trip_to_3_diff2 = 0.0
    var mojo_to_3_diff2 = 0.0

    for i in range(len(tumor)):
        var index = tumor.active_indices[i]
        var trip = Float64(payload_f32(trip_payload, index))
        var mojo = Float64(payload_f32(mojo_payload, index))
        trip_sum += trip
        mojo_sum += mojo
        trip_sum2 += trip * trip
        mojo_sum2 += mojo * mojo
        if trip >= 0.0:
            abs_trip_sum += trip
        else:
            abs_trip_sum -= trip
        var diff = mojo - trip
        if i == 0 or diff < min_error:
            min_error = diff
        if i == 0 or diff > max_error:
            max_error = diff
        var abs_diff = diff
        if abs_diff < 0.0:
            abs_diff = -abs_diff
        if abs_diff > max_abs_error:
            max_abs_error = abs_diff
            max_abs_error_voxel = i
            max_abs_error_grid_i = index % tumor.grid.shape.i
            max_abs_error_grid_j = (index // tumor.grid.shape.i) % tumor.grid.shape.j
            max_abs_error_grid_k = index // (tumor.grid.shape.i * tumor.grid.shape.j)
            max_abs_error_trip = trip
            max_abs_error_mojo = mojo
        diff2 += diff * diff
        if include_rmse_to_3:
            var trip_to_3 = trip - 3.0
            var mojo_to_3 = mojo - 3.0
            trip_to_3_diff2 += trip_to_3 * trip_to_3
            mojo_to_3_diff2 += mojo_to_3 * mojo_to_3

    if max_abs_error_voxel >= 0:
        print(
            "Worst tumor voxel:",
            max_abs_error_voxel,
            "grid:",
            max_abs_error_grid_i,
            max_abs_error_grid_j,
            max_abs_error_grid_k,
            "trip:",
            max_abs_error_trip,
            "mojo:",
            max_abs_error_mojo,
        )

    var trip_mean = trip_sum / count
    var mojo_mean = mojo_sum / count
    var trip_var = trip_sum2 / count - trip_mean * trip_mean
    var mojo_var = mojo_sum2 / count - mojo_mean * mojo_mean
    var rmse = sqrt(diff2 / count)
    return TumorDoseStats(
        trip_mean,
        sqrt_nonnegative(trip_var),
        sqrt(trip_to_3_diff2 / count),
        mojo_mean,
        sqrt_nonnegative(mojo_var),
        sqrt(mojo_to_3_diff2 / count),
        rmse,
        100.0 * rmse / (abs_trip_sum / count),
        100.0 * (mojo_mean - trip_mean) / trip_mean,
        max_abs_error,
        min_error,
        max_error,
    )


def validate_same_grid(tumor: VOI, header: NrrdHeader) raises:
    if header.dtype != "float":
        raise Error("Dose NRRD must be float")
    if header.size_i != tumor.grid.shape.i or header.size_j != tumor.grid.shape.j or header.size_k != tumor.grid.shape.k:
        raise Error("Dose NRRD grid shape does not match tumor grid")


def sqrt_nonnegative(value: Float64) -> Float64:
    if value <= 0.0:
        return 0.0
    return sqrt(value)


def max_allowed_relative_rmse_percent(label: String) -> Float64:
    if label == "Phys":
        return 0.002
    if label == "Bio":
        return 0.017
    return 0.1


def enforce_relative_rmse_limit(label: String, stats: TumorDoseStats) raises:
    var limit = max_allowed_relative_rmse_percent(label)
    if stats.relative_rmse_percent > limit:
        raise Error(String.write(
            label,
            " relative RMSE exceeds tolerance: ",
            stats.relative_rmse_percent,
            "% > ",
            limit,
            "%",
        ))


def read_latest_reference_path_native() raises -> String:
    return p101_reference_path()


def write_phys_stats_json(path: String, stats: TumorDoseStats) raises:
    with open(path, "w") as f:
        f.write(String.write(
            "{\n",
            "  \"trip_mean\": ", stats.trip_mean, ",\n",
            "  \"trip_std\": ", stats.trip_std, ",\n",
            "  \"mojo_mean\": ", stats.mojo_mean, ",\n",
            "  \"mojo_std\": ", stats.mojo_std, ",\n",
            "  \"mojo_vs_trip_rmse\": ", stats.mojo_vs_trip_rmse, ",\n",
            "  \"relative_rmse_percent\": ", stats.relative_rmse_percent, ",\n",
            "  \"mean_error_percent\": ", stats.mean_error_percent, ",\n",
            "  \"max_abs_error\": ", stats.max_abs_error, ",\n",
            "  \"min_error\": ", stats.min_error, ",\n",
            "  \"max_error\": ", stats.max_error, "\n",
            "}\n",
        ))


def write_bio_stats_json(path: String, stats: TumorDoseStats) raises:
    with open(path, "w") as f:
        f.write(String.write(
            "{\n",
            "  \"trip_mean\": ", stats.trip_mean, ",\n",
            "  \"trip_rmse_to_3\": ", stats.trip_rmse_to_3, ",\n",
            "  \"mojo_mean\": ", stats.mojo_mean, ",\n",
            "  \"mojo_rmse_to_3\": ", stats.mojo_rmse_to_3, ",\n",
            "  \"mojo_vs_trip_rmse\": ", stats.mojo_vs_trip_rmse, ",\n",
            "  \"relative_rmse_percent\": ", stats.relative_rmse_percent, ",\n",
            "  \"mean_error_percent\": ", stats.mean_error_percent, ",\n",
            "  \"max_abs_error\": ", stats.max_abs_error, ",\n",
            "  \"min_error\": ", stats.min_error, ",\n",
            "  \"max_error\": ", stats.max_error, "\n",
            "}\n",
        ))
