from std.testing import assert_equal

from geometry import Mat4, Vec3
from nrrd import payload_f32
from voi import BinfoFile, GridShape, VOI, VOIRole, VoxelGrid, load_voi_from_binfo


def main() raises:
    var indices = List[Int]()
    indices.append(0)
    indices.append(1)
    indices.append(4)
    indices.append(5)

    var index_to_world = Mat4(
        2.0, 0.0, 0.0, 10.0,
        0.0, 3.0, 0.0, 20.0,
        0.0, 0.0, 4.0, 30.0,
        0.0, 0.0, 0.0, 1.0,
    )
    var grid = VoxelGrid(GridShape(4, 3, 2), Vec3(2.0, 3.0, 4.0), index_to_world.copy())
    var voi = VOI(7, "test", "soft", VOIRole.target(), grid, indices^)

    assert_equal(len(voi), 4)
    assert_equal(voi.bounds.min_i, 0)
    assert_equal(voi.bounds.max_i, 1)
    assert_equal(voi.bounds.min_j, 0)
    assert_equal(voi.bounds.max_j, 1)
    assert_equal(voi.bounds.min_k, 0)
    assert_equal(voi.bounds.max_k, 0)
    assert_equal(voi.contains_index(1, 1, 0), True)
    assert_equal(voi.contains_index(2, 1, 0), False)
    assert_equal(voi.volume_mm3, 96.0)
    assert_equal(voi.center_index.x, 1.0)
    assert_equal(voi.center_index.y, 1.0)
    assert_equal(voi.center_index.z, 0.5)
    assert_equal(voi.center_world.x, 12.0)
    assert_equal(voi.center_world.y, 23.0)
    assert_equal(voi.center_world.z, 32.0)

    var binfo = BinfoFile("/home/max/Projects/beamline_transport/CT/P101.binfo")
    assert_equal(binfo.patient_id, "P101")
    assert_equal(binfo.grid.shape.i, 512)
    assert_equal(binfo.grid.shape.j, 512)
    assert_equal(binfo.grid.shape.k, 149)
    assert_equal(len(binfo.structures), 5)
    assert_equal(binfo.structures[4].name, "Tumor")
    assert_equal(binfo.structures[4].update_wdw_from_bit, 0)
    assert_equal(binfo.structures[4].active_mask(), UInt32(2147483648))
    assert_equal(len(binfo.structures[4].descriptors), 12)

    var tumor = load_voi_from_binfo("/home/max/Projects/beamline_transport/CT/P101.binfo", "Tumor")
    assert_equal(tumor.name, "Tumor")
    assert_equal(len(tumor), 16131)
    assert_equal(tumor.bounds.min_i, 202)
    assert_equal(tumor.bounds.min_j, 298)
    assert_equal(tumor.bounds.min_k, 45)
    assert_equal(tumor.bounds.max_i, 248)
    assert_equal(tumor.bounds.max_j, 352)
    assert_equal(tumor.bounds.max_k, 62)

    var bytes = List[UInt8]()
    bytes.append(0)
    bytes.append(0)
    bytes.append(128)
    bytes.append(63)
    assert_equal(payload_f32(bytes, 0), Float32(1.0))
