from std.algorithm import parallelize
from std.math import sqrt

from case_model import NativeField
from ct_ray import CTVolume, ray_water_depth_scaled_mm
from fdcb_optimizer import FDCBScenarioSet
from f2max import annotate_spot_f2max
from geometry import Vec3, gantry_to_patient_point, patient_to_gantry_matrix, transform_point
from hlut import HLUT
from opt_voxels import OptVoxelSet
from optimization_model import OptimizationFieldState
from phys_dose import EnergyCurve, curve_index_for_energy, gaussian_support_fwhm_factor, load_curves, spot_voxel_contribution, spot_voxel_optimizer_contribution, trip_8ln2
from rst import RasterSpot
from sparse_optimizer import SparseDoseEntry, SparseDoseMatrix


def build_physical_sparse_dose_matrix(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    coefficient_epsilon: Float64 = 0.0,
) raises -> SparseDoseMatrix:
    if len(points) != len(h2o_depth_mm):
        raise Error("point/depth length mismatch")
    var gantry_points = List[Vec3]()
    gantry_points.reserve(len(points))
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    for i in range(len(points)):
        gantry_points.append(transform_point(matrix, points[i]))

    var unit_spots = unit_particle_spots(spots)
    var curves = load_curves(unit_spots, h2o_depth_mm, False)
    var spot_curve_indices = List[Int]()
    spot_curve_indices.reserve(len(unit_spots))
    for spot in range(len(unit_spots)):
        spot_curve_indices.append(curve_index_for_energy(curves, unit_spots[spot].energy))

    var entries = List[SparseDoseEntry]()
    var eight_ln2 = trip_8ln2()
    var half_over_pi = 0.15915494309189535
    var support2 = gaussian_support_fwhm_factor() * gaussian_support_fwhm_factor()
    for voxel in range(len(gantry_points)):
        for spot in range(len(unit_spots)):
            var coeff = spot_voxel_contribution(
                voxel,
                gantry_points[voxel],
                unit_spots[spot],
                curves[spot_curve_indices[spot]],
                eight_ln2,
                half_over_pi,
                support2,
                1.0,
                1.0,
                False,
            )
            if coeff > coefficient_epsilon:
                entries.append(SparseDoseEntry(voxel, spot, coeff))
    return SparseDoseMatrix(len(points), len(spots), entries^)


def build_native_physical_sparse_dose_matrix(
    field: NativeField,
    state: OptimizationFieldState,
    opt_voxels: OptVoxelSet,
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    d_geps: Float64 = 1.0e-4,
) raises -> SparseDoseMatrix:
    var points = opt_voxel_patient_centers(opt_voxels, ct)
    var h2o_depths = field_h2o_depths_for_points(field, points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm)
    var spots = optimization_state_to_raster_spots(state, True)
    annotate_spot_f2max(spots, ct, hlut, isocenter, field.gantry_degrees, field.couch_degrees, gaussian_support_fwhm_factor(), bolus_mm_h2o + off_h2o_mm)
    return build_trip_thresholded_physical_sparse_dose_matrix(points, spots, h2o_depths, field.gantry_degrees, isocenter, d_geps)


def native_field_max_fdc_per_opt_voxel(
    field: NativeField,
    state: OptimizationFieldState,
    opt_voxels: OptVoxelSet,
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
) raises -> List[Float64]:
    var points = opt_voxel_patient_centers(opt_voxels, ct)
    var h2o_depths = field_h2o_depths_for_points(field, points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm)
    var spots = optimization_state_to_raster_spots(state, True)
    annotate_spot_f2max(spots, ct, hlut, isocenter, field.gantry_degrees, field.couch_degrees, gaussian_support_fwhm_factor(), bolus_mm_h2o + off_h2o_mm)
    return max_fdc_per_voxel(points, spots, h2o_depths, field.gantry_degrees, isocenter)


def native_trip_thresholded_fdc_count(
    field: NativeField,
    state: OptimizationFieldState,
    opt_voxels: OptVoxelSet,
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    d_geps: Float64,
) raises -> Int:
    var points = opt_voxel_patient_centers(opt_voxels, ct)
    var h2o_depths = field_h2o_depths_for_points(field, points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm)
    var spots = optimization_state_to_raster_spots(state, True)
    annotate_spot_f2max(spots, ct, hlut, isocenter, field.gantry_degrees, field.couch_degrees, gaussian_support_fwhm_factor(), bolus_mm_h2o + off_h2o_mm)
    return count_trip_thresholded_physical_fdc(points, spots, h2o_depths, field.gantry_degrees, isocenter, d_geps)


def native_trip_basic_robust_fdc_counts(
    field: NativeField,
    state: OptimizationFieldState,
    opt_voxels: OptVoxelSet,
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    robust_range_mm_h2o: Float64,
    robust_position_mm: Float64,
    d_geps: Float64,
) raises -> List[Int]:
    var points = opt_voxel_patient_centers(opt_voxels, ct)
    var nominal_depths = field_h2o_depths_for_points(field, points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm)
    var spots = optimization_state_to_raster_spots(state, True)
    annotate_spot_f2max(spots, ct, hlut, isocenter, field.gantry_degrees, field.couch_degrees, gaussian_support_fwhm_factor(), bolus_mm_h2o + off_h2o_mm)
    var counts = List[Int]()
    counts.append(count_trip_thresholded_physical_fdc(points, spots, nominal_depths, field.gantry_degrees, isocenter, d_geps))
    if robust_range_mm_h2o > 0.0:
        counts.append(count_trip_thresholded_physical_fdc(points, spots, field_h2o_depths_for_points_scaled(field, points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm, 1.0 - robust_range_mm_h2o * 0.01), field.gantry_degrees, isocenter, d_geps))
        counts.append(count_trip_thresholded_physical_fdc(points, spots, field_h2o_depths_for_points_scaled(field, points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm, 1.0 + robust_range_mm_h2o * 0.01), field.gantry_degrees, isocenter, d_geps))
    if robust_position_mm > 0.0:
        counts.append(count_trip_thresholded_physical_fdc(points, spots, nominal_depths, field.gantry_degrees, Vec3(isocenter.x - robust_position_mm, isocenter.y, isocenter.z), d_geps))
        counts.append(count_trip_thresholded_physical_fdc(points, spots, nominal_depths, field.gantry_degrees, Vec3(isocenter.x + robust_position_mm, isocenter.y, isocenter.z), d_geps))
        counts.append(count_trip_thresholded_physical_fdc(points, spots, nominal_depths, field.gantry_degrees, Vec3(isocenter.x, isocenter.y - robust_position_mm, isocenter.z), d_geps))
        counts.append(count_trip_thresholded_physical_fdc(points, spots, nominal_depths, field.gantry_degrees, Vec3(isocenter.x, isocenter.y + robust_position_mm, isocenter.z), d_geps))
        counts.append(count_trip_thresholded_physical_fdc(points, spots, nominal_depths, field.gantry_degrees, Vec3(isocenter.x, isocenter.y, isocenter.z - robust_position_mm), d_geps))
        counts.append(count_trip_thresholded_physical_fdc(points, spots, nominal_depths, field.gantry_degrees, Vec3(isocenter.x, isocenter.y, isocenter.z + robust_position_mm), d_geps))
    return counts^


def build_native_basic_robust_physical_fdcb_scenarios(
    fields: List[NativeField],
    states: List[OptimizationFieldState],
    opt_voxels: OptVoxelSet,
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    robust_range_percent: Float64,
    robust_position_mm: Float64,
    d_geps: Float64,
) raises -> FDCBScenarioSet:
    if len(fields) != len(states):
        raise Error("field/state length mismatch")
    var points = opt_voxel_patient_centers(opt_voxels, ct)
    var scenario_matrices = List[SparseDoseMatrix]()
    var scenario_count = 1
    if robust_range_percent > 0.0:
        scenario_count += 2
    if robust_position_mm > 0.0:
        scenario_count += 6
    for scenario in range(scenario_count):
        scenario_matrices.append(build_empty_combined_matrix(len(points), states))

    var spot_offset = 0
    for field_index in range(len(fields)):
        var spots = optimization_state_to_raster_spots(states[field_index], True)
        annotate_spot_f2max(spots, ct, hlut, isocenter, fields[field_index].gantry_degrees, fields[field_index].couch_degrees, gaussian_support_fwhm_factor(), bolus_mm_h2o + off_h2o_mm)
        var nominal_depths = field_h2o_depths_for_points(fields[field_index], points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm)
        var scenario_index = 0
        append_offset_sparse_entries(
            scenario_matrices[scenario_index],
            build_trip_thresholded_physical_sparse_dose_matrix(points, spots, nominal_depths, fields[field_index].gantry_degrees, isocenter, d_geps),
            spot_offset,
        )
        scenario_index += 1
        if robust_range_percent > 0.0:
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(
                    points,
                    spots,
                    field_h2o_depths_for_points_scaled(fields[field_index], points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm, 1.0 - robust_range_percent * 0.01),
                    fields[field_index].gantry_degrees,
                    isocenter,
                    d_geps,
                ),
                spot_offset,
            )
            scenario_index += 1
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(
                    points,
                    spots,
                    field_h2o_depths_for_points_scaled(fields[field_index], points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm, 1.0 + robust_range_percent * 0.01),
                    fields[field_index].gantry_degrees,
                    isocenter,
                    d_geps,
                ),
                spot_offset,
            )
            scenario_index += 1
        if robust_position_mm > 0.0:
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(points, spots, nominal_depths, fields[field_index].gantry_degrees, Vec3(isocenter.x - robust_position_mm, isocenter.y, isocenter.z), d_geps),
                spot_offset,
            )
            scenario_index += 1
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(points, spots, nominal_depths, fields[field_index].gantry_degrees, Vec3(isocenter.x + robust_position_mm, isocenter.y, isocenter.z), d_geps),
                spot_offset,
            )
            scenario_index += 1
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(points, spots, nominal_depths, fields[field_index].gantry_degrees, Vec3(isocenter.x, isocenter.y - robust_position_mm, isocenter.z), d_geps),
                spot_offset,
            )
            scenario_index += 1
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(points, spots, nominal_depths, fields[field_index].gantry_degrees, Vec3(isocenter.x, isocenter.y + robust_position_mm, isocenter.z), d_geps),
                spot_offset,
            )
            scenario_index += 1
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(points, spots, nominal_depths, fields[field_index].gantry_degrees, Vec3(isocenter.x, isocenter.y, isocenter.z - robust_position_mm), d_geps),
                spot_offset,
            )
            scenario_index += 1
            append_offset_sparse_entries(
                scenario_matrices[scenario_index],
                build_trip_thresholded_physical_sparse_dose_matrix(points, spots, nominal_depths, fields[field_index].gantry_degrees, Vec3(isocenter.x, isocenter.y, isocenter.z + robust_position_mm), d_geps),
                spot_offset,
            )
        spot_offset += states[field_index].active_spots()
    return FDCBScenarioSet(scenario_matrices^)


def build_binned_physical_sparse_dose_matrix(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    coefficient_epsilon: Float64 = 0.0,
) raises -> SparseDoseMatrix:
    if len(points) != len(h2o_depth_mm):
        raise Error("point/depth length mismatch")
    var gantry_points = List[Vec3]()
    gantry_points.reserve(len(points))
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    for i in range(len(points)):
        gantry_points.append(transform_point(matrix, points[i]))

    var unit_spots = unit_particle_spots(spots)
    var curves = load_curves(unit_spots, h2o_depth_mm, False)
    var spot_curve_indices = List[Int]()
    spot_curve_indices.reserve(len(unit_spots))
    for spot in range(len(unit_spots)):
        spot_curve_indices.append(curve_index_for_energy(curves, unit_spots[spot].energy))

    var eight_ln2 = trip_8ln2()
    var half_over_pi = 0.15915494309189535
    var support = gaussian_support_fwhm_factor()
    var support2 = support * support
    var spot_bins = build_spot_bins(unit_spots, curves, spot_curve_indices, support)

    var entries = List[SparseDoseEntry]()
    for voxel in range(len(gantry_points)):
        var p = gantry_points[voxel].copy()
        var centre_bx = Int((p.x - spot_bins.min_x) / spot_bins.bin_size)
        var centre_by = Int((p.y - spot_bins.min_y) / spot_bins.bin_size)
        var bx0 = centre_bx - spot_bins.bin_radius
        var bx1 = centre_bx + spot_bins.bin_radius
        var by0 = centre_by - spot_bins.bin_radius
        var by1 = centre_by + spot_bins.bin_radius
        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= spot_bins.bins_x:
            bx1 = spot_bins.bins_x - 1
        if by1 >= spot_bins.bins_y:
            by1 = spot_bins.bins_y - 1
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                var bin_index = by * spot_bins.bins_x + bx
                for bin_item in range(len(spot_bins.spot_indices[bin_index])):
                    var spot_index = spot_bins.spot_indices[bin_index][bin_item]
                    var coeff = spot_voxel_optimizer_contribution(
                        voxel,
                        p,
                        unit_spots[spot_index],
                        curves[spot_curve_indices[spot_index]],
                        eight_ln2,
                        half_over_pi,
                        support2,
                        1.0,
                        1.0,
                        False,
                    )
                    if coeff > coefficient_epsilon:
                        entries.append(SparseDoseEntry(voxel, spot_index, coeff))
    return SparseDoseMatrix(len(points), len(spots), entries^)


def build_trip_thresholded_physical_sparse_dose_matrix(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    d_geps: Float64 = 1.0e-4,
) raises -> SparseDoseMatrix:
    if len(points) != len(h2o_depth_mm):
        raise Error("point/depth length mismatch")
    var gantry_points = List[Vec3]()
    gantry_points.reserve(len(points))
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    for i in range(len(points)):
        gantry_points.append(transform_point(matrix, points[i]))

    var unit_spots = unit_particle_spots(spots)
    var curves = load_curves(unit_spots, h2o_depth_mm, False)
    var spot_curve_indices = List[Int]()
    spot_curve_indices.reserve(len(unit_spots))
    for spot in range(len(unit_spots)):
        spot_curve_indices.append(curve_index_for_energy(curves, unit_spots[spot].energy))

    var eight_ln2 = trip_8ln2()
    var half_over_pi = 0.15915494309189535
    var support = gaussian_support_fwhm_factor()
    var support2 = support * support
    var spot_bins = build_spot_bins(unit_spots, curves, spot_curve_indices, support)
    var mev2gy = 1.602189e-8

    var voxel_entries = List[List[SparseDoseEntry]]()
    voxel_entries.resize(len(gantry_points), List[SparseDoseEntry]())
    var num_workers = 12

    @parameter
    def build_voxel_entries(voxel: Int):
        var p = gantry_points[voxel].copy()
        var max_fdc = 0.0
        var centre_bx = Int((p.x - spot_bins.min_x) / spot_bins.bin_size)
        var centre_by = Int((p.y - spot_bins.min_y) / spot_bins.bin_size)
        var bx0 = centre_bx - spot_bins.bin_radius
        var bx1 = centre_bx + spot_bins.bin_radius
        var by0 = centre_by - spot_bins.bin_radius
        var by1 = centre_by + spot_bins.bin_radius
        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= spot_bins.bins_x:
            bx1 = spot_bins.bins_x - 1
        if by1 >= spot_bins.bins_y:
            by1 = spot_bins.bins_y - 1
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                var bin_index = by * spot_bins.bins_x + bx
                for bin_item in range(len(spot_bins.spot_indices[bin_index])):
                    var spot_index = spot_bins.spot_indices[bin_index][bin_item]
                    var coeff = spot_voxel_optimizer_contribution(
                        voxel,
                        p,
                        unit_spots[spot_index],
                        curves[spot_curve_indices[spot_index]],
                        eight_ln2,
                        half_over_pi,
                        support2,
                        1.0,
                        1.0,
                        False,
                    )
                    if coeff > 0.0:
                        var fdc = coeff / mev2gy
                        if fdc > max_fdc:
                            max_fdc = fdc
        var threshold = max_fdc * d_geps
        var local_entries = List[SparseDoseEntry]()
        if threshold > 0.0:
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    var bin_index = by * spot_bins.bins_x + bx
                    for bin_item in range(len(spot_bins.spot_indices[bin_index])):
                        var spot_index = spot_bins.spot_indices[bin_index][bin_item]
                        var coeff = spot_voxel_optimizer_contribution(
                            voxel,
                            p,
                            unit_spots[spot_index],
                            curves[spot_curve_indices[spot_index]],
                            eight_ln2,
                            half_over_pi,
                            support2,
                            1.0,
                            1.0,
                            False,
                        )
                        if coeff / mev2gy > threshold:
                            local_entries.append(SparseDoseEntry(voxel, spot_index, coeff))
        voxel_entries[voxel] = local_entries^

    parallelize[build_voxel_entries](len(gantry_points), num_workers)

    var entries = List[SparseDoseEntry]()
    for voxel in range(len(voxel_entries)):
        for i in range(len(voxel_entries[voxel])):
            entries.append(voxel_entries[voxel][i].copy())
    return SparseDoseMatrix(len(points), len(spots), entries^)


def max_fdc_per_voxel(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
) raises -> List[Float64]:
    if len(points) != len(h2o_depth_mm):
        raise Error("point/depth length mismatch")
    var gantry_points = List[Vec3]()
    gantry_points.reserve(len(points))
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    for i in range(len(points)):
        gantry_points.append(transform_point(matrix, points[i]))

    var unit_spots = unit_particle_spots(spots)
    var curves = load_curves(unit_spots, h2o_depth_mm, False)
    var spot_curve_indices = List[Int]()
    spot_curve_indices.reserve(len(unit_spots))
    for spot in range(len(unit_spots)):
        spot_curve_indices.append(curve_index_for_energy(curves, unit_spots[spot].energy))

    var eight_ln2 = trip_8ln2()
    var half_over_pi = 0.15915494309189535
    var support = gaussian_support_fwhm_factor()
    var support2 = support * support
    var spot_bins = build_spot_bins(unit_spots, curves, spot_curve_indices, support)
    var mev2gy = 1.602189e-8
    var max_values = List[Float64]()
    max_values.resize(len(gantry_points), 0.0)
    var num_workers = 12

    @parameter
    def compute_voxel_max(voxel: Int):
        var p = gantry_points[voxel].copy()
        var max_fdc = 0.0
        var centre_bx = Int((p.x - spot_bins.min_x) / spot_bins.bin_size)
        var centre_by = Int((p.y - spot_bins.min_y) / spot_bins.bin_size)
        var bx0 = centre_bx - spot_bins.bin_radius
        var bx1 = centre_bx + spot_bins.bin_radius
        var by0 = centre_by - spot_bins.bin_radius
        var by1 = centre_by + spot_bins.bin_radius
        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= spot_bins.bins_x:
            bx1 = spot_bins.bins_x - 1
        if by1 >= spot_bins.bins_y:
            by1 = spot_bins.bins_y - 1
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                var bin_index = by * spot_bins.bins_x + bx
                for bin_item in range(len(spot_bins.spot_indices[bin_index])):
                    var spot_index = spot_bins.spot_indices[bin_index][bin_item]
                    var coeff = spot_voxel_optimizer_contribution(
                        voxel,
                        p,
                        unit_spots[spot_index],
                        curves[spot_curve_indices[spot_index]],
                        eight_ln2,
                        half_over_pi,
                        support2,
                        1.0,
                        1.0,
                        False,
                    )
                    if coeff > 0.0:
                        var fdc = coeff / mev2gy
                        if fdc > max_fdc:
                            max_fdc = fdc
        max_values[voxel] = max_fdc

    parallelize[compute_voxel_max](len(gantry_points), num_workers)
    return max_values^


def count_trip_thresholded_physical_fdc(
    points: List[Vec3],
    spots: List[RasterSpot],
    h2o_depth_mm: List[Float64],
    gantry: Float64,
    isocenter: Vec3,
    d_geps: Float64 = 1.0e-4,
) raises -> Int:
    if len(points) != len(h2o_depth_mm):
        raise Error("point/depth length mismatch")
    var gantry_points = List[Vec3]()
    gantry_points.reserve(len(points))
    var matrix = patient_to_gantry_matrix(isocenter, gantry, 90.0)
    for i in range(len(points)):
        gantry_points.append(transform_point(matrix, points[i]))

    var unit_spots = unit_particle_spots(spots)
    var curves = load_curves(unit_spots, h2o_depth_mm, False)
    var spot_curve_indices = List[Int]()
    spot_curve_indices.reserve(len(unit_spots))
    for spot in range(len(unit_spots)):
        spot_curve_indices.append(curve_index_for_energy(curves, unit_spots[spot].energy))

    var eight_ln2 = trip_8ln2()
    var half_over_pi = 0.15915494309189535
    var support = gaussian_support_fwhm_factor()
    var support2 = support * support
    var spot_bins = build_spot_bins(unit_spots, curves, spot_curve_indices, support)
    var mev2gy = 1.602189e-8
    var counts = List[Int]()
    counts.resize(len(gantry_points), 0)
    var num_workers = 12

    @parameter
    def count_voxel_entries(voxel: Int):
        var p = gantry_points[voxel].copy()
        var max_fdc = 0.0
        var centre_bx = Int((p.x - spot_bins.min_x) / spot_bins.bin_size)
        var centre_by = Int((p.y - spot_bins.min_y) / spot_bins.bin_size)
        var bx0 = centre_bx - spot_bins.bin_radius
        var bx1 = centre_bx + spot_bins.bin_radius
        var by0 = centre_by - spot_bins.bin_radius
        var by1 = centre_by + spot_bins.bin_radius
        if bx0 < 0:
            bx0 = 0
        if by0 < 0:
            by0 = 0
        if bx1 >= spot_bins.bins_x:
            bx1 = spot_bins.bins_x - 1
        if by1 >= spot_bins.bins_y:
            by1 = spot_bins.bins_y - 1
        for by in range(by0, by1 + 1):
            for bx in range(bx0, bx1 + 1):
                var bin_index = by * spot_bins.bins_x + bx
                for bin_item in range(len(spot_bins.spot_indices[bin_index])):
                    var spot_index = spot_bins.spot_indices[bin_index][bin_item]
                    var coeff = spot_voxel_optimizer_contribution(
                        voxel,
                        p,
                        unit_spots[spot_index],
                        curves[spot_curve_indices[spot_index]],
                        eight_ln2,
                        half_over_pi,
                        support2,
                        1.0,
                        1.0,
                        False,
                    )
                    if coeff > 0.0:
                        var fdc = coeff / mev2gy
                        if fdc > max_fdc:
                            max_fdc = fdc
        var threshold = max_fdc * d_geps
        var count = 0
        if threshold > 0.0:
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    var bin_index = by * spot_bins.bins_x + bx
                    for bin_item in range(len(spot_bins.spot_indices[bin_index])):
                        var spot_index = spot_bins.spot_indices[bin_index][bin_item]
                        var coeff = spot_voxel_optimizer_contribution(
                            voxel,
                            p,
                            unit_spots[spot_index],
                            curves[spot_curve_indices[spot_index]],
                            eight_ln2,
                            half_over_pi,
                            support2,
                            1.0,
                            1.0,
                            False,
                        )
                        if coeff / mev2gy > threshold:
                            count += 1
        counts[voxel] = count

    parallelize[count_voxel_entries](len(gantry_points), num_workers)
    var total = 0
    for i in range(len(counts)):
        total += counts[i]
    return total


def shifted_depths(depths: List[Float64], shift: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(depths))
    for i in range(len(depths)):
        out.append(depths[i] + shift)
    return out^


@fieldwise_init
struct SpotBins(Copyable, Movable):
    var min_x: Float64
    var min_y: Float64
    var bin_size: Float64
    var bins_x: Int
    var bins_y: Int
    var bin_radius: Int
    var spot_indices: List[List[Int]]


def build_spot_bins(
    spots: List[RasterSpot],
    curves: List[EnergyCurve],
    spot_curve_indices: List[Int],
    support: Float64,
) raises -> SpotBins:
    if len(spots) == 0:
        raise Error("cannot build sparse matrix without spots")
    var bin_size = 10.0
    var min_x = spots[0].x
    var max_x = spots[0].x
    var min_y = spots[0].y
    var max_y = spots[0].y
    var global_radius = 0.0
    for spot_index in range(len(spots)):
        if spots[spot_index].x < min_x:
            min_x = spots[spot_index].x
        if spots[spot_index].x > max_x:
            max_x = spots[spot_index].x
        if spots[spot_index].y < min_y:
            min_y = spots[spot_index].y
        if spots[spot_index].y > max_y:
            max_y = spots[spot_index].y
        var curve_index = spot_curve_indices[spot_index]
        var radius = sqrt(support * support * (spots[spot_index].focus * spots[spot_index].focus + curves[curve_index].max_fwhm2))
        if spots[spot_index].f2_max > 0.0:
            radius = support * sqrt(spots[spot_index].f2_max)
        if radius > global_radius:
            global_radius = radius
    global_radius += 10.0
    min_x -= global_radius
    min_y -= global_radius
    max_x += global_radius
    max_y += global_radius
    var bins_x = Int((max_x - min_x) / bin_size) + 1
    var bins_y = Int((max_y - min_y) / bin_size) + 1
    var spot_indices = List[List[Int]]()
    for _ in range(bins_x * bins_y):
        spot_indices.append(List[Int]())
    for spot_index in range(len(spots)):
        var bx = Int((spots[spot_index].x - min_x) / bin_size)
        var by = Int((spots[spot_index].y - min_y) / bin_size)
        spot_indices[by * bins_x + bx].append(spot_index)
    var bin_radius = Int(global_radius / bin_size) + 2
    return SpotBins(min_x, min_y, bin_size, bins_x, bins_y, bin_radius, spot_indices^)


def build_empty_combined_matrix(voxel_count: Int, states: List[OptimizationFieldState]) -> SparseDoseMatrix:
    var spot_count = 0
    for i in range(len(states)):
        spot_count += states[i].active_spots()
    return SparseDoseMatrix(voxel_count, spot_count, List[SparseDoseEntry]())


def append_offset_sparse_entries(mut destination: SparseDoseMatrix, source: SparseDoseMatrix, spot_offset: Int):
    for i in range(len(source.entries)):
        destination.entries.append(SparseDoseEntry(
            source.entries[i].voxel,
            source.entries[i].spot + spot_offset,
            source.entries[i].value,
        ))


def opt_voxel_patient_centers(opt_voxels: OptVoxelSet, ct: CTVolume) -> List[Vec3]:
    var points = List[Vec3]()
    points.reserve(len(opt_voxels.voxels))
    for i in range(len(opt_voxels.voxels)):
        points.append(ct_voxel_center(ct, opt_voxels.voxels[i].i, opt_voxels.voxels[i].j, opt_voxels.voxels[i].k))
    return points^


def field_h2o_depths_for_points(
    field: NativeField,
    points: List[Vec3],
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
) -> List[Float64]:
    return field_h2o_depths_for_points_scaled(field, points, ct, hlut, isocenter, bolus_mm_h2o, off_h2o_mm, 1.0)


def field_h2o_depths_for_points_scaled(
    field: NativeField,
    points: List[Vec3],
    ct: CTVolume,
    hlut: HLUT,
    isocenter: Vec3,
    bolus_mm_h2o: Float64,
    off_h2o_mm: Float64,
    hlut_scale: Float64,
) -> List[Float64]:
    var trip_ct = ct.copy()
    trip_ct.header.origin = Vec3(
        trip_cube_header_float(ct.header.origin.x),
        trip_cube_header_float(ct.header.origin.y),
        trip_cube_header_float(ct.header.origin.z),
    )
    for row in range(3):
        for col in range(3):
            trip_ct.header.directions[row, col] = trip_cube_header_float(ct.header.directions[row, col])
    var origin0 = gantry_to_patient_point(isocenter, field.gantry_degrees, field.couch_degrees, Vec3(0.0, 0.0, 0.0))
    var origin1 = gantry_to_patient_point(isocenter, field.gantry_degrees, field.couch_degrees, Vec3(0.0, 0.0, 1.0))
    var direction = Vec3(origin1.x - origin0.x, origin1.y - origin0.y, origin1.z - origin0.z)
    var depths = List[Float64]()
    depths.reserve(len(points))
    for i in range(len(points)):
        depths.append(ray_water_depth_scaled_mm(trip_ct, hlut, points[i], direction, hlut_scale) + bolus_mm_h2o + off_h2o_mm)
    return depths^


def ct_voxel_center(ct: CTVolume, i: Int, j: Int, k: Int) -> Vec3:
    return Vec3(
        trip_cube_header_float(ct.header.origin.x) + (Float64(i) + 0.5) * trip_cube_header_float(ct.header.directions[0, 0]),
        trip_cube_header_float(ct.header.origin.y) + (Float64(j) + 0.5) * trip_cube_header_float(ct.header.directions[1, 1]),
        trip_cube_header_float(ct.header.origin.z) + (Float64(k) + 0.5) * trip_cube_header_float(ct.header.directions[2, 2]),
    )


def trip_cube_header_float(value: Float64) -> Float64:
    return Float64(Float32(value))


def unit_particle_spots(spots: List[RasterSpot]) -> List[RasterSpot]:
    var out = List[RasterSpot]()
    out.reserve(len(spots))
    for i in range(len(spots)):
        out.append(RasterSpot(
            spots[i].x,
            spots[i].y,
            spots[i].energy,
            1.0,
            spots[i].focus,
            spots[i].range_mm_h2o,
            spots[i].f2_max,
            spots[i].active,
            spots[i].point_index,
        ))
    return out^


def optimization_state_to_raster_spots(state: OptimizationFieldState, unit_particles: Bool = False) -> List[RasterSpot]:
    var spots = List[RasterSpot]()
    spots.reserve(len(state.spots))
    for i in range(len(state.spots)):
        if not state.spots[i].inside:
            continue
        var particles = state.spots[i].particles
        if unit_particles:
            particles = 1.0
        spots.append(RasterSpot(
            state.spots[i].x,
            state.spots[i].y,
            state.spots[i].energy_mev_u,
            particles,
            state.spots[i].focus_mm,
            state.spots[i].range_mm_h2o,
            0.0,
            True,
            state.spots[i].raster_point_index,
        ))
    return spots^
