from std.testing import assert_equal, assert_true

from nrrd import payload_f32, read_nrrd_header, read_nrrd_payload
from p101_io import load_tumor_world_points, write_tumor_dose
from voi import load_voi_from_binfo


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def main() raises:
    var tumor = load_voi_from_binfo("/home/max/Projects/beamline_transport/CT/P101.binfo", "Tumor")
    var points = load_tumor_world_points()
    assert_equal(len(points), len(tumor))
    assert_equal(len(points), 16120)
    assert_close(points[0].x, 210.4573, 1.0e-12)
    assert_close(points[0].y, 333.5089, 1.0e-12)
    assert_close(points[0].z, 136.5, 1.0e-12)
    assert_close(points[len(points) - 1].x, 229.9893, 1.0e-12)
    assert_close(points[len(points) - 1].y, 337.4153, 1.0e-12)
    assert_close(points[len(points) - 1].z, 187.5, 1.0e-12)

    var dose = List[Float64]()
    dose.resize(len(tumor), 0.0)
    dose[0] = 1.25
    dose[len(dose) - 1] = 2.5
    var out_path = "/tmp/trip_mojo_p101_io_baseline.nrrd"
    write_tumor_dose(out_path, dose)

    var header = read_nrrd_header(out_path)
    assert_equal(header.dtype, "float")
    assert_equal(header.size_i, 512)
    assert_equal(header.size_j, 512)
    assert_equal(header.size_k, 149)

    var payload = read_nrrd_payload(header)
    assert_equal(payload_f32(payload, tumor.active_indices[0]), Float32(1.25))
    assert_equal(payload_f32(payload, tumor.active_indices[len(tumor) - 1]), Float32(2.5))
    assert_equal(payload_f32(payload, 0), Float32(0.0))
