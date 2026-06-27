from geometry import Mat4, Vec3, transform_point
from nrrd import nrrd_type_bytes, read_nrrd_header, read_nrrd_payload


@fieldwise_init
struct Bounds3i(Copyable, Movable):
    var min_i: Int
    var min_j: Int
    var min_k: Int
    var max_i: Int
    var max_j: Int
    var max_k: Int


@fieldwise_init
struct GridShape(Copyable, Movable):
    var i: Int
    var j: Int
    var k: Int


@fieldwise_init
struct VoxelGrid(Copyable, Movable):
    var shape: GridShape
    var voxel_size: Vec3
    var index_to_world: Mat4

    def linear_index(self, i: Int, j: Int, k: Int) -> Int:
        return i + self.shape.i * (j + self.shape.j * k)

    def index_triplet(self, linear: Int) -> Vec3:
        var i = linear % self.shape.i
        var plane = linear // self.shape.i
        var j = plane % self.shape.j
        var k = plane // self.shape.j
        return Vec3(Float64(i), Float64(j), Float64(k))

    def world_at_index(self, i: Int, j: Int, k: Int) -> Vec3:
        return transform_point(self.index_to_world, Vec3(Float64(i), Float64(j), Float64(k)))


@fieldwise_init
struct VOIRole(Copyable, Movable):
    var code: Int

    @staticmethod
    def residual() -> VOIRole:
        return VOIRole(0)

    @staticmethod
    def target() -> VOIRole:
        return VOIRole(1)

    @staticmethod
    def oar() -> VOIRole:
        return VOIRole(2)


@fieldwise_init
struct BinfoDescriptor(Copyable, Movable):
    var voi_name: String
    var bit: Int
    var usage: Int
    var states_mask: UInt32
    var name: String
    var description: String


@fieldwise_init
struct BinfoStructure(Copyable, Movable):
    var name: String
    var tissue: String
    var lvdx_type: Int
    var flags: UInt32
    var subcube_origin_mm: Vec3
    var update_wdw_from_bit: Int
    var descriptors: List[BinfoDescriptor]

    def active_mask(self) -> UInt32:
        for i in range(len(self.descriptors)):
            if self.descriptors[i].name == "INTRSC":
                return UInt32(1) << UInt32(self.descriptors[i].bit)
        return UInt32(1) << UInt32(self.update_wdw_from_bit)


struct VOI(Copyable, Movable, Sized):
    var id: Int
    var name: String
    var tissue: String
    var role: VOIRole
    var grid: VoxelGrid
    var active_indices: List[Int]
    var active_mask: UInt32
    var bounds: Bounds3i
    var source_min_world: Vec3
    var source_max_world: Vec3
    var center_index: Vec3
    var center_world: Vec3
    var volume_mm3: Float64

    def __init__(
        out self,
        id: Int,
        name: String,
        tissue: String,
        role: VOIRole,
        grid: VoxelGrid,
        var active_indices: List[Int],
        active_mask: UInt32 = 1,
        source_min_world: Vec3 = Vec3(0.0, 0.0, 0.0),
        source_max_world: Vec3 = Vec3(0.0, 0.0, 0.0),
    ):
        self.id = id
        self.name = name
        self.tissue = tissue
        self.role = role.copy()
        self.grid = grid.copy()
        self.active_indices = active_indices^
        self.active_mask = active_mask
        self.bounds = Bounds3i(0, 0, 0, -1, -1, -1)
        self.source_min_world = source_min_world.copy()
        self.source_max_world = source_max_world.copy()
        self.center_index = Vec3(0.0, 0.0, 0.0)
        self.center_world = transform_point(self.grid.index_to_world, self.center_index)
        self.volume_mm3 = 0.0
        self._refresh_cached_geometry()

    def _refresh_cached_geometry(mut self):
        if len(self.active_indices) == 0:
            return

        var min_i = self.grid.shape.i
        var min_j = self.grid.shape.j
        var min_k = self.grid.shape.k
        var max_i = -1
        var max_j = -1
        var max_k = -1
        var sum_i = 0.0
        var sum_j = 0.0
        var sum_k = 0.0

        for n in range(len(self.active_indices)):
            var idx = self.grid.index_triplet(self.active_indices[n])
            var i = Int(idx.x)
            var j = Int(idx.y)
            var k = Int(idx.z)

            if i < min_i:
                min_i = i
            if j < min_j:
                min_j = j
            if k < min_k:
                min_k = k
            if i > max_i:
                max_i = i
            if j > max_j:
                max_j = j
            if k > max_k:
                max_k = k

            sum_i += idx.x + 0.5
            sum_j += idx.y + 0.5
            sum_k += idx.z + 0.5

        var inv_count = 1.0 / Float64(len(self.active_indices))
        self.bounds = Bounds3i(min_i, min_j, min_k, max_i, max_j, max_k)
        self.center_index = Vec3(sum_i * inv_count, sum_j * inv_count, sum_k * inv_count)
        self.center_world = transform_point(self.grid.index_to_world, self.center_index)
        self.volume_mm3 = Float64(len(self.active_indices)) * self.grid.voxel_size.x * self.grid.voxel_size.y * self.grid.voxel_size.z

    def __len__(self) -> Int:
        return len(self.active_indices)

    def empty(self) -> Bool:
        return len(self.active_indices) == 0

    def active_world_point(self, n: Int) -> Vec3:
        var idx = self.grid.index_triplet(self.active_indices[n])
        return transform_point(self.grid.index_to_world, Vec3(idx.x + 0.5, idx.y + 0.5, idx.z + 0.5))

    def contains_linear(self, linear: Int) -> Bool:
        for n in range(len(self.active_indices)):
            if self.active_indices[n] == linear:
                return True
        return False

    def contains_index(self, i: Int, j: Int, k: Int) -> Bool:
        if i < 0 or j < 0 or k < 0:
            return False
        if i >= self.grid.shape.i or j >= self.grid.shape.j or k >= self.grid.shape.k:
            return False
        return self.contains_linear(self.grid.linear_index(i, j, k))

    def contains_point_vbind(self, point: Vec3) -> Bool:
        var i = self.source_axis_vbind(point.x, self.source_min_world.x, self.source_max_world.x, self.grid.voxel_size.x)
        var j = self.source_axis_vbind(point.y, self.source_min_world.y, self.source_max_world.y, self.grid.voxel_size.y)
        var k = self.source_axis_vbind(point.z, self.source_min_world.z, self.source_max_world.z, self.grid.voxel_size.z)
        return self.contains_index(i, j, k)

    def source_axis_vbind(self, value: Float64, source_min: Float64, source_max: Float64, step: Float64) -> Int:
        var source_size = round_to_int((source_max - source_min) / step)
        var local = vbind_uniform(value - source_min, step, source_size)
        var offset = round_to_int(source_min / step)
        return local + offset


def vbind_uniform(value: Float64, step: Float64, size: Int) -> Int:
    var i_lo = 0
    var i_hi = size
    var ii = 0
    for _ in range(64):
        ii = (i_lo + i_hi) >> 1
        if ii == i_lo:
            break
        if value < Float64(ii) * step:
            i_hi = ii
        else:
            i_lo = ii
    if value < 0.0:
        return -1
    if value > Float64(size) * step:
        return size + 1
    return ii


struct VOISet(Copyable, Movable):
    var patient_id: String
    var grid: VoxelGrid
    var vois: List[VOI]

    def __init__(out self, patient_id: String, grid: VoxelGrid):
        self.patient_id = patient_id
        self.grid = grid.copy()
        self.vois = List[VOI]()

    def append(mut self, var voi: VOI):
        self.vois.append(voi^)

    def find_index(self, name: String) -> Int:
        for i in range(len(self.vois)):
            if self.vois[i].name == name:
                return i
        return -1

    def find(self, name: String) -> VOI:
        var index = self.find_index(name)
        if index < 0:
            return VOI(0, "", "", VOIRole.residual(), self.grid, List[Int]())
        return self.vois[index].copy()


struct BinfoFile(Copyable, Movable):
    var path: String
    var base_path: String
    var patient_id: String
    var grid: VoxelGrid
    var structures: List[BinfoStructure]

    def __init__(out self, path: String) raises:
        self.path = path
        self.base_path = binfo_base_path(path)
        self.patient_id = ""
        self.grid = VoxelGrid(GridShape(0, 0, 0), Vec3(0.0, 0.0, 0.0), Mat4())
        self.structures = List[BinfoStructure]()

        with open(path, "r") as f:
            var lines = f.read().split("\n")
            for raw_line in lines:
                var line = raw_line.strip()
                if line.byte_length() == 0:
                    continue
                if line[byte=0] == "#":
                    continue
                var parts = line.split()
                if len(parts) == 0:
                    continue
                if parts[0] == "patient":
                    self.patient_id = String(parts[1])
                elif parts[0] == "geometry":
                    var voxel_size = Vec3(Float64(parts[1]), Float64(parts[2]), Float64(parts[3]))
                    var shape = GridShape(Int(parts[4]), Int(parts[5]), Int(parts[6]))
                    self.grid = VoxelGrid(shape.copy(), voxel_size.copy(), binfo_index_to_world(voxel_size))
                elif parts[0] == "binvoi":
                    var descriptors = List[BinfoDescriptor]()
                    self.structures.append(BinfoStructure(
                        String(parts[1]),
                        String(parts[2]),
                        Int(parts[3]),
                        UInt32(Int(parts[4])),
                        Vec3(Float64(parts[5]), Float64(parts[6]), Float64(parts[7])),
                        Int(parts[8]),
                        descriptors^,
                    ))
                elif parts[0] == "descriptor":
                    self._append_descriptor(String(line))

    def nrrd_path(self, structure_index: Int) -> String:
        return self.base_path + "/" + self.patient_id + self.structures[structure_index].name + ".nrrd"

    def _append_descriptor(mut self, line: String) raises:
        if len(self.structures) == 0:
            return
        var parts = line.split()
        if len(parts) < 7:
            return
        var descriptor = BinfoDescriptor(
            String(parts[1]),
            Int(parts[3]),
            Int(parts[4]),
            UInt32(Int(parts[5])),
            quoted_field(line, 0),
            quoted_field(line, 1),
        )
        for i in range(len(self.structures)):
            if self.structures[i].name == descriptor.voi_name:
                self.structures[i].descriptors.append(descriptor^)
                return

    def structure_index(self, name: String) -> Int:
        for i in range(len(self.structures)):
            if self.structures[i].name == name:
                return i
        return -1

    def load_voi(self, name: String) raises -> VOI:
        var index = self.structure_index(name)
        if index < 0:
            raise Error("VOI not found in binfo")
        return self.load_voi_at(index)

    def load_voi_at(self, index: Int) raises -> VOI:
        var header = read_nrrd_header(self.nrrd_path(index))
        var active_indices = load_binfo_structure_active_indices(
            self.nrrd_path(index),
            self.grid,
            self.structures[index].subcube_origin_mm,
            self.structures[index].active_mask(),
        )
        return VOI(
            index,
            self.structures[index].name,
            self.structures[index].tissue,
            role_from_trip_flags(self.structures[index].lvdx_type, self.structures[index].flags),
            self.grid,
            active_indices^,
            self.structures[index].active_mask(),
            header.origin,
            Vec3(
                header.origin.x + Float64(header.size_i) * header.directions[0, 0],
                header.origin.y + Float64(header.size_j) * header.directions[1, 1],
                header.origin.z + Float64(header.size_k) * header.directions[2, 2],
            ),
        )

    def load_voi_set(self) raises -> VOISet:
        var out = VOISet(self.patient_id, self.grid)
        for i in range(len(self.structures)):
            out.append(self.load_voi_at(i))
        return out^


def load_voi_set_from_binfo(path: String) raises -> VOISet:
    var binfo = BinfoFile(path)
    return binfo.load_voi_set()


def load_voi_from_binfo(path: String, name: String) raises -> VOI:
    var binfo = BinfoFile(path)
    return binfo.load_voi(name)


def role_from_trip_flags(lvdx_type: Int, flags: UInt32) -> VOIRole:
    if lvdx_type == 1 or (flags & UInt32(2)) != UInt32(0):
        return VOIRole.target()
    if lvdx_type == 2 or (flags & UInt32(1)) != UInt32(0):
        return VOIRole.oar()
    return VOIRole.residual()


def load_binfo_structure_active_indices(
    nrrd_path: String,
    grid: VoxelGrid,
    subcube_origin_mm: Vec3,
    active_mask: UInt32,
) raises -> List[Int]:
    var header = read_nrrd_header(nrrd_path)
    var payload = read_nrrd_payload(header)
    var value_bytes = nrrd_type_bytes(header.dtype)
    var expected_bytes = header.size_i * header.size_j * header.size_k * value_bytes
    if len(payload) != expected_bytes:
        raise Error("NRRD payload size does not match header sizes")

    var offset_i = round_to_int(header.origin.x / grid.voxel_size.x)
    var offset_j = round_to_int(header.origin.y / grid.voxel_size.y)
    var offset_k = round_to_int(header.origin.z / grid.voxel_size.z)
    if header.dtype == "int":
        return active_indices_from_i32_payload(payload, header.size_i, header.size_j, header.size_k, offset_i, offset_j, offset_k, grid, active_mask)
    if header.dtype == "short":
        return active_indices_from_i16_payload(payload, header.size_i, header.size_j, header.size_k, offset_i, offset_j, offset_k, grid, UInt16(active_mask))
    raise Error("Unsupported integer NRRD type")


def active_indices_from_i32_payload(
    payload: List[UInt8],
    size_i: Int,
    size_j: Int,
    size_k: Int,
    offset_i: Int,
    offset_j: Int,
    offset_k: Int,
    grid: VoxelGrid,
    mask: UInt32,
) -> List[Int]:
    var out = List[Int]()
    out.reserve(estimated_active_capacity(size_i * size_j * size_k))
    var slice_stride = size_i * size_j
    var k_start = max_int(0, -offset_k)
    var k_end = min_int(size_k, grid.shape.k - offset_k)
    var j_start = max_int(0, -offset_j)
    var j_end = min_int(size_j, grid.shape.j - offset_j)
    var i_start = max_int(0, -offset_i)
    var i_end = min_int(size_i, grid.shape.i - offset_i)
    for k in range(k_start, k_end):
        var global_k = k + offset_k
        var k_base = k * slice_stride
        var global_k_base = grid.shape.i * grid.shape.j * global_k
        for j in range(j_start, j_end):
            var global_j = j + offset_j
            var row_base = k_base + j * size_i
            var global_row_base = global_k_base + grid.shape.i * global_j
            for i in range(i_start, i_end):
                var global_i = i + offset_i
                var byte_offset = (row_base + i) * 4
                var value = UInt32(payload[byte_offset]) | (UInt32(payload[byte_offset + 1]) << 8) | (UInt32(payload[byte_offset + 2]) << 16) | (UInt32(payload[byte_offset + 3]) << 24)
                if (value & mask) != UInt32(0):
                    out.append(global_row_base + global_i)
    return out^


def active_indices_from_i16_payload(
    payload: List[UInt8],
    size_i: Int,
    size_j: Int,
    size_k: Int,
    offset_i: Int,
    offset_j: Int,
    offset_k: Int,
    grid: VoxelGrid,
    mask: UInt16,
) -> List[Int]:
    var out = List[Int]()
    out.reserve(estimated_active_capacity(size_i * size_j * size_k))
    var slice_stride = size_i * size_j
    var k_start = max_int(0, -offset_k)
    var k_end = min_int(size_k, grid.shape.k - offset_k)
    var j_start = max_int(0, -offset_j)
    var j_end = min_int(size_j, grid.shape.j - offset_j)
    var i_start = max_int(0, -offset_i)
    var i_end = min_int(size_i, grid.shape.i - offset_i)
    for k in range(k_start, k_end):
        var global_k = k + offset_k
        var k_base = k * slice_stride
        var global_k_base = grid.shape.i * grid.shape.j * global_k
        for j in range(j_start, j_end):
            var global_j = j + offset_j
            var row_base = k_base + j * size_i
            var global_row_base = global_k_base + grid.shape.i * global_j
            for i in range(i_start, i_end):
                var global_i = i + offset_i
                var byte_offset = (row_base + i) * 2
                var value = UInt16(payload[byte_offset]) | (UInt16(payload[byte_offset + 1]) << 8)
                if (value & mask) != UInt16(0):
                    out.append(global_row_base + global_i)
    return out^


def round_to_int(value: Float64) -> Int:
    if value >= 0.0:
        return Int(value + 0.5)
    return Int(value - 0.5)


def floor_to_int(value: Float64) -> Int:
    var truncated = Int(value)
    if value < 0.0 and Float64(truncated) != value:
        return truncated - 1
    return truncated


def estimated_active_capacity(voxels: Int) -> Int:
    var capacity = voxels // 4
    if capacity < 1024:
        return voxels
    return capacity


def min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b


def max_int(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b


def binfo_index_to_world(voxel_size: Vec3) -> Mat4:
    return Mat4(
        voxel_size.x, 0.0, 0.0, 0.0,
        0.0, voxel_size.y, 0.0, 0.0,
        0.0, 0.0, voxel_size.z, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def binfo_base_path(path: String) -> String:
    var last_slash = -1
    for i in range(path.byte_length()):
        if path[byte=i] == "/":
            last_slash = i
    if last_slash < 0:
        return "."
    var out = String()
    for i in range(last_slash):
        out += String(path[byte=i])
    return out^


def quoted_field(line: String, field_index: Int) -> String:
    var current = -1
    var in_quote = False
    var out = String()
    for i in range(line.byte_length()):
        var ch = line[byte=i]
        if ch == '"':
            if in_quote:
                if current == field_index:
                    return out^
                in_quote = False
            else:
                current += 1
                out = String()
                in_quote = True
        elif in_quote and current == field_index:
            out += String(ch)
    return out^
