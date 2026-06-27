from std.testing import assert_equal

from case_model import load_native_case, native_case_from_exec_path
from opt_voxels import OptVoxel, OptVoxelSet, build_p101_target_avoidance_opt_voxels, refine_opt_voxels_by_trip_max_fdc


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var loaded = load_native_case(native_case)
    var voxels = build_p101_target_avoidance_opt_voxels(loaded.target, 5.0, 10.0, 2.0)
    assert_equal(voxels.target_count, 16131)
    assert_equal(voxels.avoidance_count, 21606)
    assert_equal(len(voxels.voxels), 37737)
    assert_equal(voxels.high_count, 17363)
    assert_equal(voxels.low_count, 20374)

    var simple_voxels = List[OptVoxel]()
    simple_voxels.append(OptVoxel(0, 0, 0, 1, True))
    simple_voxels.append(OptVoxel(1, 0, 0, 2, False))
    simple_voxels.append(OptVoxel(2, 0, 0, 2, True))
    var simple = OptVoxelSet(simple_voxels^, 1, 2, 2, 1)
    var max_fdc = List[Float64]()
    max_fdc.append(1.0)
    max_fdc.append(9.1)
    max_fdc.append(8.0)
    var refined = refine_opt_voxels_by_trip_max_fdc(simple, max_fdc, 1.0, 0.9)
    assert_equal(len(refined.voxels), 2)
    assert_equal(refined.target_count, 1)
    assert_equal(refined.avoidance_count, 1)
    assert_equal(refined.high_count, 1)
    assert_equal(refined.low_count, 1)
