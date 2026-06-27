from std.testing import assert_equal

from nrrd import (
    le_i16,
    le_i32,
    le_f32,
    nrrd_type_bytes,
    payload_f32,
    payload_i16,
    payload_i32,
    read_nrrd_header,
    read_nrrd_payload,
)


def main() raises:
    assert_equal(le_i16(UInt8(0xFF), UInt8(0x7F)), 32767)
    assert_equal(le_i16(UInt8(0x00), UInt8(0x80)), -32768)
    assert_equal(le_i32(UInt8(0x78), UInt8(0x56), UInt8(0x34), UInt8(0x12)), 305419896)
    assert_equal(le_i32(UInt8(0xFF), UInt8(0xFF), UInt8(0xFF), UInt8(0xFF)), -1)
    assert_equal(le_f32(UInt8(0), UInt8(0), UInt8(128), UInt8(63)), Float32(1.0))

    assert_equal(nrrd_type_bytes("short"), 2)
    assert_equal(nrrd_type_bytes("int"), 4)
    assert_equal(nrrd_type_bytes("float"), 4)

    var payload = List[UInt8]()
    payload.append(UInt8(0))
    payload.append(UInt8(128))
    payload.append(UInt8(255))
    payload.append(UInt8(127))
    assert_equal(payload_i16(payload, 0), -32768)
    assert_equal(payload_i16(payload, 1), 32767)

    var payload32 = List[UInt8]()
    payload32.append(UInt8(255))
    payload32.append(UInt8(255))
    payload32.append(UInt8(255))
    payload32.append(UInt8(255))
    payload32.append(UInt8(0))
    payload32.append(UInt8(0))
    payload32.append(UInt8(128))
    payload32.append(UInt8(63))
    assert_equal(payload_i32(payload32, 0), -1)
    assert_equal(payload_f32(payload32, 1), Float32(1.0))

    var path = "/tmp/trip_mojo_test_nrrd.nrrd"
    var raw = "/tmp/trip_mojo_test_nrrd.raw"
    with open(path, "w") as f:
        f.write(
            "NRRD0005\n"
            "type: float\n"
            "dimension: 3\n"
            "sizes: 2 1 1\n"
            "space directions: (1,0,0) (0,2,0) (0,0,3)\n"
            "space origin: (4,5,6)\n"
            "endian: little\n"
            "encoding: raw\n"
            "data file: trip_mojo_test_nrrd.raw\n"
        )
    var raw_bytes = List[UInt8]()
    raw_bytes.append(UInt8(0))
    raw_bytes.append(UInt8(0))
    raw_bytes.append(UInt8(128))
    raw_bytes.append(UInt8(63))
    raw_bytes.append(UInt8(0))
    raw_bytes.append(UInt8(0))
    raw_bytes.append(UInt8(0))
    raw_bytes.append(UInt8(64))
    with open(raw, "w") as f:
        f.write_bytes(raw_bytes)

    var header = read_nrrd_header(path)
    assert_equal(header.dtype, "float")
    assert_equal(header.size_i, 2)
    assert_equal(header.size_j, 1)
    assert_equal(header.size_k, 1)
    assert_equal(header.origin.x, 4.0)
    assert_equal(header.origin.y, 5.0)
    assert_equal(header.origin.z, 6.0)
    assert_equal(header.directions[0, 0], 1.0)
    assert_equal(header.directions[1, 1], 2.0)
    assert_equal(header.directions[2, 2], 3.0)
    var bytes = read_nrrd_payload(header)
    assert_equal(payload_f32(bytes, 0), Float32(1.0))
    assert_equal(payload_f32(bytes, 1), Float32(2.0))
