from case_model import NativeField
from ddd import energy_for_peak_depth_cm, peak_depth_for_energy_cm
from field_geometry import Bounds3f
from sis import SISTable, focus_ceil_for_energy, nearest_sis_energy


@fieldwise_init
struct EnergyLayer(Copyable, Movable):
    var energy_mev_u: Float64
    var peak_range_mm_h2o: Float64
    var focus_mm: Float64


def build_energy_layers(field: NativeField, window: Bounds3f, sis: SISTable) raises -> List[EnergyLayer]:
    if window.min_z >= window.max_z:
        raise Error("bad water-range window")
    var requested = stepped_sis_energies_for_range(field, window, sis)
    var layers = List[EnergyLayer]()
    layers.reserve(len(requested))
    var requested_focus = field.raster_x_mm / focus_to_stepsize_factor()
    for i in range(len(requested)):
        var focus_energy = nearest_sis_energy(sis, requested[i], False, True)
        var focus = focus_ceil_for_energy(sis, focus_energy, requested_focus)
        layers.append(EnergyLayer(requested[i], peak_depth_for_energy_cm(requested[i]) * 10.0, focus))
    return layers^


def focus_to_stepsize_factor() -> Float64:
    # TRiP scancap default: sSccSlice.dFocus2StepsizeFactor = 1/3.
    return 0.3333333333333333


def contiguous_sis_energies_for_range(sis: SISTable, min_range_mm: Float64, max_range_mm: Float64) raises -> List[Float64]:
    var first = -1
    var last = -1
    for i in range(len(sis.energies)):
        var peak_mm = peak_depth_for_energy_cm(sis.energies[i].energy_mev_u) * 10.0
        if first < 0 and peak_mm >= min_range_mm:
            first = i
        if peak_mm <= max_range_mm:
            last = i
    if first < 0 or last < first:
        raise Error("no SIS energies cover requested water-range window")
    var values = List[Float64]()
    values.reserve(last - first + 1)
    for i in range(first, last + 1):
        values.append(sis.energies[i].energy_mev_u)
    return values^


def stepped_sis_energies_for_range(field: NativeField, window: Bounds3f, sis: SISTable) raises -> List[Float64]:
    var values = List[Float64]()
    var z = window.max_z
    var first = True
    while first or z >= window.min_z:
        var e_continuous = energy_for_peak_depth_cm(z * 0.1)
        var e = nearest_sis_energy(sis, e_continuous, first, False)
        append_unique(values, e)
        first = False
        z -= field.z_step_mm
    sort_values(values)
    return values^


def sis_energy_ceil_for_peak_range(sis: SISTable, range_mm_h2o: Float64) raises -> Float64:
    var best_energy = 0.0
    var best_peak = 1.0e300
    for i in range(len(sis.energies)):
        var peak = peak_depth_for_energy_cm(sis.energies[i].energy_mev_u) * 10.0
        if peak >= range_mm_h2o and peak < best_peak:
            best_peak = peak
            best_energy = sis.energies[i].energy_mev_u
    if best_peak == 1.0e300:
        raise Error("requested range is outside SIS table")
    return best_energy


def append_unique(mut values: List[Float64], value: Float64):
    for i in range(len(values)):
        if values[i] == value:
            return
    values.append(value)


def sort_values(mut values: List[Float64]):
    for i in range(1, len(values)):
        var v = values[i]
        var j = i
        while j > 0 and values[j - 1] > v:
            values[j] = values[j - 1]
            j -= 1
        values[j] = v
