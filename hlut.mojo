@fieldwise_init
struct HLUT(Copyable, Movable):
    var ct: List[Float64]
    var path: List[Float64]


def read_hlut(path: String) raises -> HLUT:
    var ct = List[Float64]()
    var h2o = List[Float64]()
    var active = False
    with open(path, "r") as f:
        for raw_line in f.read().split("\n"):
            var line = String(raw_line.strip())
            if line.byte_length() == 0:
                continue
            if line == "!hlut":
                active = True
                continue
            if not active or line[byte=0] == "#":
                continue
            var parts = line.split()
            if len(parts) >= 4:
                ct.append(Float64(parts[0]))
                h2o.append(Float64(parts[3]))
    if len(ct) < 2:
        raise Error("HLUT has too few entries")
    return HLUT(ct^, h2o^)


def hlut_path_factor(table: HLUT, value: Float64) -> Float64:
    if value <= table.ct[0]:
        return table.path[0]
    var last = len(table.ct) - 1
    if value >= table.ct[last]:
        return table.path[last]
    var lo = 0
    var hi = last
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if value < table.ct[mid]:
            hi = mid
        else:
            lo = mid
    var dx = table.ct[hi] - table.ct[lo]
    if dx == 0.0:
        return table.path[lo]
    var t = (value - table.ct[lo]) / dx
    return table.path[lo] * (1.0 - t) + table.path[hi] * t
