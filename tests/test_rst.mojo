from std.testing import assert_equal

from rst import RSTPlan


def reference_path() -> String:
    return "reference/trip4d_p101_cpu_20260511_140457"


def main() raises:
    var base = reference_path()
    var field1 = RSTPlan(base + "/StaticP101_2110_field1_iGTV_R.rst")
    var field2 = RSTPlan(base + "/StaticP101_2110_field2_iGTV_R.rst")

    assert_equal(field1.patient_id, "P101")
    assert_equal(field1.projectile.name, "C")
    assert_equal(field1.projectile.charge, 6)
    assert_equal(field1.projectile.mass, 12)
    assert_equal(field1.gantry, 280)
    assert_equal(field1.couch, 90)
    assert_equal(field1.bolus, 40)
    assert_equal(field1.ripplefilter, 3)
    assert_equal(len(field1.field), 9899)
    assert_equal(field1.field[0].x, 2.0)
    assert_equal(field1.field[0].y, 2.0)
    assert_equal(field1.field[0].energy, 280.48)
    assert_equal(field1.field[0].particles, 81707.5)
    assert_equal(field1.field[0].focus, 7.5)
    assert_equal(field1.field[0].f2_max, 0.0)
    assert_equal(field1.field[len(field1.field) - 1].x, 12.0)
    assert_equal(field1.field[len(field1.field) - 1].y, 2.0)
    assert_equal(field1.field[len(field1.field) - 1].energy, 197.58)
    assert_equal(field1.field[len(field1.field) - 1].particles, 40335.9)
    assert_equal(field1.field[len(field1.field) - 1].focus, 6.6)

    assert_equal(field2.patient_id, "P101")
    assert_equal(field2.projectile.name, "C")
    assert_equal(field2.projectile.charge, 6)
    assert_equal(field2.projectile.mass, 12)
    assert_equal(field2.gantry, 325)
    assert_equal(field2.couch, 90)
    assert_equal(field2.bolus, 40)
    assert_equal(field2.ripplefilter, 3)
    assert_equal(len(field2.field), 9238)
    assert_equal(field2.field[0].x, 4.0)
    assert_equal(field2.field[0].y, 2.0)
    assert_equal(field2.field[0].energy, 303.14)
    assert_equal(field2.field[0].particles, 89073.9)
    assert_equal(field2.field[0].focus, 7.6)
    assert_equal(field2.field[0].f2_max, 0.0)
    assert_equal(field2.field[len(field2.field) - 1].x, 6.0)
    assert_equal(field2.field[len(field2.field) - 1].y, 4.0)
    assert_equal(field2.field[len(field2.field) - 1].energy, 232.2)
    assert_equal(field2.field[len(field2.field) - 1].particles, 12608.2)
    assert_equal(field2.field[len(field2.field) - 1].focus, 7.0)
