from std.testing import assert_equal, assert_true

from rst import RSTPlan


def latest_reference_path() -> String:
    return "reference/trip4d_p101_cpu_20260511_140457"


def contains(text: String, needle: String) -> Bool:
    if needle.byte_length() == 0:
        return True
    if text.byte_length() < needle.byte_length():
        return False
    for i in range(text.byte_length() - needle.byte_length() + 1):
        var matched = True
        for j in range(needle.byte_length()):
            if text[byte=i + j] != needle[byte=j]:
                matched = False
                break
        if matched:
            return True
    return False


def main() raises:
    var ref_dir = latest_reference_path()
    var field1 = RSTPlan(ref_dir + "/StaticP101_2110_field1_iGTV_R.rst")
    var field2 = RSTPlan(ref_dir + "/StaticP101_2110_field2_iGTV_R.rst")
    var written_spots = len(field1.field) + len(field2.field)
    assert_equal(written_spots, 19137)

    var stderr = String(open(ref_dir + "/stderr.log", "r").read())
    assert_true(contains(stderr, "FDCB H2O opt field setup"))
    assert_true(contains(stderr, "22891 raster points"))
    assert_true(contains(stderr, "Iter #  1/300"))
    assert_true(contains(stderr, "dChi2: 50057.3"))
    assert_true(contains(stderr, "myfac=0.5"))

    # This documents a real contract gap: the dose reference was generated
    # from in-memory optimized fields, not just from the nonzero written RSTs.
    assert_true(22891 > written_spots)
