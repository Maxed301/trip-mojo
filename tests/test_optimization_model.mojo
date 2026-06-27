from std.testing import assert_equal

from case_model import native_case_from_exec_path
from optimization_model import FieldRasterMarkCounts, OptimizationFieldSpec, OptimizationProblemSpec
from rst import RSTPlan


def main() raises:
    var native_case = native_case_from_exec_path("reference/trip4d_p101_cpu_20260511_140457/P101_iGTV_3Dplan.exec")
    var ref_dir = "reference/trip4d_p101_cpu_20260511_140457"
    var field1_written = len(RSTPlan(ref_dir + "/StaticP101_2110_field1_iGTV_R.rst").field)
    var field2_written = len(RSTPlan(ref_dir + "/StaticP101_2110_field2_iGTV_R.rst").field)

    # From TRiP MarkInsideUltimate diagnostics for the same P101 reference run:
    # field 1: 209250 total, 3588 target-inside, 1997 robust-scenario, 6054 extension
    # field 2: 199640 total, 3592 target-inside, 1879 robust-scenario, 5781 extension
    # TRPOptH2OFldSetup then counts this in-memory INSIDE set as optimizer points.
    var field1_counts = FieldRasterMarkCounts(209250, 3588, 1997, 6054)
    var field2_counts = FieldRasterMarkCounts(199640, 3592, 1879, 5781)
    assert_equal(field1_counts.optimization_point_count(), 11639)
    assert_equal(field2_counts.optimization_point_count(), 11252)

    var fields = List[OptimizationFieldSpec]()
    fields.append(OptimizationFieldSpec(native_case.fields[0].copy(), field1_counts.copy(), field1_written))
    fields.append(OptimizationFieldSpec(native_case.fields[1].copy(), field2_counts.copy(), field2_written))
    var problem = OptimizationProblemSpec(native_case.copy(), fields^)

    assert_equal(problem.optimization_points(), 22891)
    assert_equal(problem.written_rst_spots(), 19137)
    assert_equal(problem.omitted_from_written_rst(), 3754)
    assert_equal(problem.fields[0].omitted_from_written_rst(), 1740)
    assert_equal(problem.fields[1].omitted_from_written_rst(), 2014)
