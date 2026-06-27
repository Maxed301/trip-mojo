from std.testing import assert_equal

from dedx import read_dedx
from rbe_lowdose import read_rbe_lowdose
from spc import depth_bio_coefficients, read_spc_beam


def main() raises:
    var dedx = read_dedx("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/20040607.dedx")
    var beam = read_spc_beam("/home/max/Projects/TRIP_DATA/Basedata/GSI/carbon/SPC/12C.H2O.MeV28000.spc")
    var chordom = read_rbe_lowdose("/home/max/Projects/TRIP_DATA/Basedata/RBE/chordom02.rbe")
    var hirn = read_rbe_lowdose("/home/max/Projects/TRIP_DATA/Basedata/RBE/hirn02.rbe")
    var chordom_coeffs = depth_bio_coefficients(beam.depths[80], dedx, chordom)
    var hirn_coeffs = depth_bio_coefficients(beam.depths[80], dedx, hirn)
    assert_equal(hirn_coeffs.alpha0x1, chordom_coeffs.alpha0x1)
