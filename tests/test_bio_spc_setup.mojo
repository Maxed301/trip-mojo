from phys_dose import (
    spc_interpolated_peak_cm_for_slice_energy,
    spc_spectrum_code_for_slice_energy,
)


def assert_close(got: Float64, expected: Float64, tol: Float64) raises:
    if abs(got - expected) > tol:
        raise Error("value mismatch")


def main() raises:
    if spc_spectrum_code_for_slice_energy(280.48) != 28000:
        raise Error("wrong SPC code for 280.48 MeV/u")
    if spc_spectrum_code_for_slice_energy(197.58) != 20000:
        raise Error("wrong SPC code for 197.58 MeV/u")

    assert_close(spc_interpolated_peak_cm_for_slice_energy(280.48), 15.12067607872252, 1e-12)
    assert_close(spc_interpolated_peak_cm_for_slice_energy(197.58), 8.209267876903123, 1e-12)
