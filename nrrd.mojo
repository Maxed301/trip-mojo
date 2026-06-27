from std.memory import bitcast

from geometry import Mat4, Vec3


@fieldwise_init
struct NrrdHeader(Copyable, Movable):
    var path: String
    var base_path: String
    var data_file: String
    var dtype: String
    var size_i: Int
    var size_j: Int
    var size_k: Int
    var origin: Vec3
    var directions: Mat4


def read_nrrd_header(path: String) raises -> NrrdHeader:
    var dtype = ""
    var endian = ""
    var encoding = ""
    var data_file = ""
    var size_i = 0
    var size_j = 0
    var size_k = 0
    var origin = Vec3(0.0, 0.0, 0.0)
    var directions = Mat4()

    with open(path, "r") as f:
        for raw_line in f.read().split("\n"):
            var line = String(raw_line.strip())
            if line.byte_length() == 0 or line[byte=0] == "#":
                continue
            var pos = find_byte(line, ":")
            if pos < 0:
                continue
            var key = trim(substr(line, 0, pos))
            var value = trim(substr(line, pos + 1, line.byte_length()))
            if key == "type":
                dtype = value
            elif key == "endian":
                endian = value
            elif key == "encoding":
                encoding = value
            elif key == "data file":
                data_file = value
            elif key == "sizes":
                var p = value.split()
                size_i = Int(p[0])
                size_j = Int(p[1])
                size_k = Int(p[2])
            elif key == "space origin":
                origin = parse_vec3(value)
            elif key == "space directions":
                directions = parse_directions(value)

    if encoding != "raw":
        raise Error("Only raw NRRD encoding is supported")
    if endian != "" and endian != "little":
        raise Error("Only little-endian NRRD payloads are supported")
    if data_file == "":
        raise Error("Detached NRRD data file is required")

    return NrrdHeader(
        path,
        dirname(path),
        data_file,
        dtype,
        size_i,
        size_j,
        size_k,
        origin.copy(),
        directions.copy(),
    )


def read_nrrd_payload(header: NrrdHeader) raises -> List[UInt8]:
    with open(header.base_path + "/" + header.data_file, "r") as f:
        return f.read_bytes()


def nrrd_type_bytes(dtype: String) raises -> Int:
    if dtype == "short":
        return 2
    if dtype == "int" or dtype == "float":
        return 4
    raise Error("Unsupported NRRD type")


# Little-endian convenience methods
def le_i16(b0: UInt8, b1: UInt8) -> Int:
    return Int(bitcast[DType.int16](UInt16(b0) | (UInt16(b1) << 8)))


def le_i32(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> Int:
    return Int(bitcast[DType.int32](le_u32(b0, b1, b2, b3)))


def le_u32(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> UInt32:
    return UInt32(b0) | (UInt32(b1) << 8) | (UInt32(b2) << 16) | (UInt32(b3) << 24)


def le_f32(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> Float32:
    return bitcast[DType.float32](le_u32(b0, b1, b2, b3))


def payload_i32(payload: List[UInt8], index: Int) -> Int:
    var o = index * 4
    return le_i32(payload[o], payload[o + 1], payload[o + 2], payload[o + 3])


def payload_i16(payload: List[UInt8], index: Int) -> Int:
    var o = index * 2
    return le_i16(payload[o], payload[o + 1])


def payload_f32(payload: List[UInt8], index: Int) -> Float32:
    var o = index * 4
    return le_f32(payload[o], payload[o + 1], payload[o + 2], payload[o + 3])


def parse_vec3(text: String) raises -> Vec3:
    var p = trim(strip_parens(text)).split(",")
    return Vec3(Float64(p[0]), Float64(p[1]), Float64(p[2]))


def parse_directions(text: String) raises -> Mat4:
    var v = List[Vec3]()
    var start = -1
    for i in range(text.byte_length()):
        if text[byte=i] == "(":
            start = i
        elif text[byte=i] == ")" and start >= 0:
            v.append(parse_vec3(substr(text, start, i + 1)))
            start = -1
    if len(v) != 3:
        raise Error("Expected three NRRD space direction vectors")
    return Mat4(
        v[0].x, v[0].y, v[0].z, 0.0,
        v[1].x, v[1].y, v[1].z, 0.0,
        v[2].x, v[2].y, v[2].z, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def strip_parens(text: String) -> String:
    if text.byte_length() >= 2 and text[byte=0] == "(" and text[byte=text.byte_length() - 1] == ")":
        return substr(text, 1, text.byte_length() - 1)
    return String(text)


def dirname(path: String) -> String:
    var last = -1
    for i in range(path.byte_length()):
        if path[byte=i] == "/":
            last = i
    if last < 0:
        return "."
    return substr(path, 0, last)


def find_byte(text: String, needle: String) -> Int:
    for i in range(text.byte_length()):
        if text[byte=i] == needle:
            return i
    return -1


def trim(text: String) -> String:
    return String(text.strip())


def substr(text: String, start: Int, end: Int) -> String:
    var out = String()
    for i in range(start, end):
        out += String(text[byte=i])
    return out^
