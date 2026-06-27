from std.testing import assert_equal, assert_true

from rst import RSTPlan, RasterSpot


def latest_reference_path() -> String:
    return "reference/trip4d_p101_cpu_20260511_140457"


def assert_close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    var diff = actual - expected
    if diff < 0.0:
        diff = -diff
    assert_true(diff <= tolerance)


def count_energy(spots: List[RasterSpot], energy: Float64) -> Int:
    var count = 0
    for i in range(len(spots)):
        if spots[i].energy == energy:
            count += 1
    return count


def sum_energy(spots: List[RasterSpot], energy: Float64) -> Float64:
    var total = 0.0
    for i in range(len(spots)):
        if spots[i].energy == energy:
            total += spots[i].particles
    return total


def sum_particles(spots: List[RasterSpot]) -> Float64:
    var total = 0.0
    for i in range(len(spots)):
        total += spots[i].particles
    return total


def count_runs(spots: List[RasterSpot]) -> Int:
    if len(spots) == 0:
        return 0
    var runs = 1
    var previous_energy = spots[0].energy
    var previous_focus = spots[0].focus
    for i in range(1, len(spots)):
        if spots[i].energy != previous_energy or spots[i].focus != previous_focus:
            runs += 1
            previous_energy = spots[i].energy
            previous_focus = spots[i].focus
    return runs


def run_start_index(spots: List[RasterSpot], run_number: Int) -> Int:
    if run_number == 0:
        return 0
    var run = 0
    var previous_energy = spots[0].energy
    var previous_focus = spots[0].focus
    for i in range(1, len(spots)):
        if spots[i].energy != previous_energy or spots[i].focus != previous_focus:
            run += 1
            if run == run_number:
                return i
            previous_energy = spots[i].energy
            previous_focus = spots[i].focus
    return -1


def min_x(spots: List[RasterSpot], energy: Float64) -> Float64:
    var value = 1.0e30
    for i in range(len(spots)):
        if spots[i].energy == energy and spots[i].x < value:
            value = spots[i].x
    return value


def max_x(spots: List[RasterSpot], energy: Float64) -> Float64:
    var value = -1.0e30
    for i in range(len(spots)):
        if spots[i].energy == energy and spots[i].x > value:
            value = spots[i].x
    return value


def min_y(spots: List[RasterSpot], energy: Float64) -> Float64:
    var value = 1.0e30
    for i in range(len(spots)):
        if spots[i].energy == energy and spots[i].y < value:
            value = spots[i].y
    return value


def max_y(spots: List[RasterSpot], energy: Float64) -> Float64:
    var value = -1.0e30
    for i in range(len(spots)):
        if spots[i].energy == energy and spots[i].y > value:
            value = spots[i].y
    return value


def main() raises:
    var ref_dir = latest_reference_path()
    var field1 = RSTPlan(ref_dir + "/StaticP101_2110_field1_iGTV_R.rst").field.copy()
    var field2 = RSTPlan(ref_dir + "/StaticP101_2110_field2_iGTV_R.rst").field.copy()

    assert_equal(len(field1), 9899)
    assert_equal(count_runs(field1), 24)
    var run1 = run_start_index(field1, 1)
    assert_equal(run1, 31)
    assert_close(field1[run1].energy, 277.19, 0.0)
    assert_close(field1[run1].focus, 7.5, 0.0)
    assert_close(sum_particles(field1), 372332137.9, 1.0e-4)
    assert_equal(count_energy(field1, 280.48), 31)
    assert_close(sum_energy(field1, 280.48), 3685892.0, 1.0e-4)
    assert_equal(count_energy(field1, 277.19), 134)
    assert_close(sum_energy(field1, 277.19), 10332206.3, 1.0e-4)
    assert_equal(count_energy(field1, 197.58), 4)
    assert_close(sum_energy(field1, 197.58), 143672.8, 1.0e-4)
    assert_close(min_x(field1, 280.48), -6.0, 0.0)
    assert_close(max_x(field1, 280.48), 6.0, 0.0)
    assert_close(min_y(field1, 280.48), -6.0, 0.0)
    assert_close(max_y(field1, 280.48), 24.0, 0.0)

    assert_equal(len(field2), 9238)
    assert_equal(count_runs(field2), 22)
    var field2_run1 = run_start_index(field2, 1)
    assert_equal(field2_run1, 15)
    assert_close(field2[field2_run1].energy, 299.94, 0.0)
    assert_close(field2[field2_run1].focus, 7.6, 0.0)
    assert_close(sum_particles(field2), 362081663.63, 1.0e-4)
    assert_equal(count_energy(field2, 303.14), 15)
    assert_close(sum_energy(field2, 303.14), 1479569.6, 1.0e-4)
    assert_equal(count_energy(field2, 299.94), 87)
    assert_close(sum_energy(field2, 299.94), 8650739.8, 1.0e-4)
    assert_equal(count_energy(field2, 232.2), 38)
    assert_close(sum_energy(field2, 232.2), 1021957.97, 1.0e-4)
    assert_close(min_x(field2, 303.14), -4.0, 0.0)
    assert_close(max_x(field2, 303.14), 4.0, 0.0)
    assert_close(min_y(field2, 303.14), -2.0, 0.0)
    assert_close(max_y(field2, 303.14), 2.0, 0.0)
