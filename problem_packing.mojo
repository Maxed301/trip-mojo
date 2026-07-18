"""Explicit adapters from convenient host models to OptimizationProblem."""

from robust_objective import (
    BiologicalScenarioSet,
    BioLQParams,
    PhysicalScenarioSet,
    VoxelObjective,
    validate_biological_inputs,
    validate_physical_inputs,
)
from optimization_problem import (
    FieldSlice,
    MinimumParticlePolicy,
    OptimizationProblem,
    ScenarioState,
    OptimizerSettings,
    DoseMatrixSlice,
    RobustScenario,
    OptimizationVoxel,
    OPTIMIZER_FLAG_BIOLOGICAL,
    MEV_TO_GY,
)


@fieldwise_init
struct NativeFieldSlice(Copyable, Movable):
    """Convenient host description of one field/energy point partition."""

    var field_index: Int
    var beam_index: Int
    var point_offset: Int
    var point_count: Int
    var raster_stride: Int
    var minimum_particles: Float64


@fieldwise_init
struct _PackedNativeFieldLayout(Copyable, Movable):
    var field_count: UInt32
    var field_slices: List[FieldSlice]
    var point_field_slice_indices: List[UInt32]
    var point_local_indices: List[UInt16]


def _single_native_field_slice(point_count: Int) -> List[NativeFieldSlice]:
    return [NativeFieldSlice(0, 0, 0, point_count, 0, 0.0)]


def _pack_native_field_layout(
    native_slices: List[NativeFieldSlice], point_count: Int
) raises -> _PackedNativeFieldLayout:
    if len(native_slices) == 0:
        raise Error("input layout requires at least one field slice")
    var packed = List[FieldSlice]()
    var point_slices = List[UInt32]()
    var local_points = List[UInt16]()
    point_slices.resize(point_count, UInt32(0))
    local_points.resize(point_count, UInt16(0))
    var expected_offset = 0
    var field_count = 0
    var previous_field = -1
    for slice_index in range(len(native_slices)):
        var native = native_slices[slice_index].copy()
        if (
            native.field_index < 0
            or native.beam_index < 0
            or native.point_count < 0
            or native.raster_stride < 0
            or native.minimum_particles < 0.0
        ):
            raise Error("input field-slice metadata is invalid")
        if native.point_offset != expected_offset:
            raise Error("input point ranges must be contiguous")
        if native.point_count > Int(UInt16.MAX) + 1:
            raise Error("native field slice exceeds UInt16 local indexing")
        if native.field_index != previous_field:
            if native.field_index != field_count:
                raise Error("input fields must be grouped and contiguous")
            field_count += 1
            previous_field = native.field_index
        for local_point in range(native.point_count):
            var point = native.point_offset + local_point
            if point >= point_count:
                raise Error("input field slices exceed the point arrays")
            point_slices[point] = UInt32(slice_index)
            local_points[point] = UInt16(local_point)
        packed.append(
            FieldSlice(
                UInt32(native.field_index),
                UInt32(native.beam_index),
                UInt64(native.point_offset),
                UInt32(native.point_count),
                UInt32(native.raster_stride),
                native.minimum_particles,
            )
        )
        expected_offset += native.point_count
    if expected_offset != point_count:
        raise Error("input field slices do not cover the point arrays")
    return _PackedNativeFieldLayout(
        UInt32(field_count), packed^, point_slices^, local_points^
    )


def pack_physical_problem(
    scenarios: PhysicalScenarioSet,
    objectives: List[VoxelObjective],
    particles: List[Float64],
    settings: OptimizerSettings,
    minimum_particle_policy: MinimumParticlePolicy,
) raises -> OptimizationProblem:
    """Pack the simple native sparse model as one field/energy slice."""
    return pack_physical_problem_with_fields(
        scenarios,
        objectives,
        _single_native_field_slice(len(particles)),
        particles,
        settings,
        minimum_particle_policy,
    )


def pack_physical_problem_with_fields(
    scenarios: PhysicalScenarioSet,
    objectives: List[VoxelObjective],
    native_field_slices: List[NativeFieldSlice],
    particles: List[Float64],
    settings: OptimizerSettings,
    minimum_particle_policy: MinimumParticlePolicy,
) raises -> OptimizationProblem:
    """Pack global sparse spot indices through explicit field partitions."""
    validate_physical_inputs(scenarios, objectives, particles)
    var layout = _pack_native_field_layout(native_field_slices, len(particles))
    for scenario_index in range(len(scenarios.matrices)):
        var matrix = scenarios.matrices[scenario_index].copy()
        for entry_index in range(len(matrix.entries)):
            var entry = matrix.entries[entry_index].copy()
            if (
                entry.voxel < 0
                or entry.voxel >= matrix.voxel_count
                or entry.spot < 0
                or entry.spot >= matrix.spot_count
            ):
                raise Error("native sparse optimizer entry is out of bounds")

    var direction = List[Float64]()
    var initial_gradient = List[Float64]()
    var point_active = List[UInt8]()
    direction.resize(len(particles), 0.0)
    initial_gradient.resize(len(particles), 0.0)
    point_active.resize(len(particles), UInt8(1))

    var voxels = List[OptimizationVoxel]()
    for voxel_index in range(len(objectives)):
        var objective = objectives[voxel_index].copy()
        voxels.append(
            OptimizationVoxel(
                objective.dose_p,
                objective.dose_w,
                objective.dose_d,
                objective.dose_w_max,
                0.0,
                0.0,
                0.0,
                objective.overdose_tolerance,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                Int32(0),
                Int32(0),
                UInt64(voxel_index * len(scenarios.matrices)),
            )
        )

    var voxel_scenarios = List[RobustScenario]()
    var scenario_states = List[ScenarioState]()
    var slices = List[DoseMatrixSlice]()
    var point_indices = List[UInt16]()
    var coefficients = List[Float64]()
    for voxel_index in range(len(objectives)):
        for scenario_index in range(len(scenarios.matrices)):
            var matrix = scenarios.matrices[scenario_index].copy()
            var slice_offset = UInt64(len(slices))
            for field_slice_index in range(len(layout.field_slices)):
                var coefficient_offset = UInt64(len(coefficients))
                for entry_index in range(len(matrix.entries)):
                    var entry = matrix.entries[entry_index].copy()
                    if (
                        entry.voxel == voxel_index
                        and layout.point_field_slice_indices[entry.spot]
                        == UInt32(field_slice_index)
                    ):
                        point_indices.append(
                            layout.point_local_indices[entry.spot]
                        )
                        coefficients.append(entry.value)
                slices.append(
                    DoseMatrixSlice(
                        UInt32(field_slice_index),
                        UInt32(0),
                        coefficient_offset,
                        UInt32(UInt64(len(coefficients)) - coefficient_offset),
                        1.0 / MEV_TO_GY,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                )
            voxel_scenarios.append(
                RobustScenario(slice_offset, UInt32(len(layout.field_slices)))
            )
            scenario_states.append(ScenarioState(0.0, 0.0, 0.0, 0.0))

    var rng_state = List[UInt32]()
    var problem = OptimizationProblem(
        settings.copy(),
        minimum_particle_policy.copy(),
        rng_state^,
        layout.field_count,
        UInt32(len(scenarios.matrices)),
        layout.field_slices.copy(),
        particles.copy(),
        direction^,
        initial_gradient^,
        point_active^,
        voxels^,
        voxel_scenarios^,
        scenario_states^,
        slices^,
        point_indices^,
        coefficients^,
    )
    problem.validate()
    return problem^


def pack_biological_problem(
    scenarios: BiologicalScenarioSet,
    objectives: List[VoxelObjective],
    lq_params: List[BioLQParams],
    particles: List[Float64],
    settings: OptimizerSettings,
    minimum_particle_policy: MinimumParticlePolicy,
) raises -> OptimizationProblem:
    """Pack the simple native biological model as one field/energy slice."""
    return pack_biological_problem_with_fields(
        scenarios,
        objectives,
        lq_params,
        _single_native_field_slice(len(particles)),
        particles,
        settings,
        minimum_particle_policy,
    )


def pack_biological_problem_with_fields(
    scenarios: BiologicalScenarioSet,
    objectives: List[VoxelObjective],
    lq_params: List[BioLQParams],
    native_field_slices: List[NativeFieldSlice],
    particles: List[Float64],
    settings: OptimizerSettings,
    minimum_particle_policy: MinimumParticlePolicy,
) raises -> OptimizationProblem:
    """Pack biological entries through explicit field/energy partitions."""
    validate_biological_inputs(scenarios, objectives, lq_params, particles)
    var layout = _pack_native_field_layout(native_field_slices, len(particles))
    for scenario_index in range(len(scenarios.matrices)):
        var matrix = scenarios.matrices[scenario_index].copy()
        for entry_index in range(len(matrix.entries)):
            var entry = matrix.entries[entry_index].copy()
            if (
                entry.voxel < 0
                or entry.voxel >= matrix.voxel_count
                or entry.spot < 0
                or entry.spot >= matrix.spot_count
            ):
                raise Error(
                    "native biological optimizer entry is out of bounds"
                )

    var spot_count = len(particles)
    var direction = List[Float64]()
    var initial_gradient = List[Float64]()
    var point_active = List[UInt8]()
    direction.resize(spot_count, 0.0)
    initial_gradient.resize(spot_count, 0.0)
    point_active.resize(spot_count, UInt8(1))
    var voxels = List[OptimizationVoxel]()
    for voxel_index in range(len(objectives)):
        var objective = objectives[voxel_index].copy()
        var lq = lq_params[voxel_index].copy()
        voxels.append(
            OptimizationVoxel(
                objective.dose_p,
                objective.dose_w,
                objective.dose_d,
                objective.dose_w_max,
                0.0,
                0.0,
                0.0,
                objective.overdose_tolerance,
                lq.cut_gy,
                lq.alpha,
                lq.beta,
                lq.slope_max(),
                lq.damage_cut(),
                Int32(0),
                Int32(0),
                UInt64(voxel_index * len(scenarios.matrices)),
            )
        )

    var voxel_scenarios = List[RobustScenario]()
    var states = List[ScenarioState]()
    var slices = List[DoseMatrixSlice]()
    var point_indices = List[UInt16]()
    var coefficients = List[Float64]()
    for voxel_index in range(len(objectives)):
        for scenario_index in range(len(scenarios.matrices)):
            var matrix = scenarios.matrices[scenario_index].copy()
            var slice_offset = UInt64(len(slices))
            for field_slice_index in range(len(layout.field_slices)):
                var coefficient_offset = UInt64(len(coefficients))
                var found = False
                var ddd = 0.0
                var alpha = 0.0
                var sqrt_beta = 0.0
                var let_mix = 0.0
                var let_bar = 0.0
                for entry_index in range(len(matrix.entries)):
                    var entry = matrix.entries[entry_index].copy()
                    if (
                        entry.voxel != voxel_index
                        or layout.point_field_slice_indices[entry.spot]
                        != UInt32(field_slice_index)
                    ):
                        continue
                    var entry_ddd = entry.ddd
                    var entry_alpha = entry.alpha
                    var entry_sqrt_beta = entry.sqrt_beta
                    var entry_let_mix = entry.let_mix
                    var entry_let_bar = entry.let_bar
                    if not found:
                        ddd = entry_ddd
                        alpha = entry_alpha
                        sqrt_beta = entry_sqrt_beta
                        let_mix = entry_let_mix
                        let_bar = entry_let_bar
                        found = True
                    elif (
                        entry_ddd != ddd
                        or entry_alpha != alpha
                        or entry_sqrt_beta != sqrt_beta
                        or entry_let_mix != let_mix
                        or entry_let_bar != let_bar
                    ):
                        raise Error(
                            "biological entries in one field slice have"
                            " different static coefficients"
                        )
                    point_indices.append(layout.point_local_indices[entry.spot])
                    coefficients.append(entry.g)
                slices.append(
                    DoseMatrixSlice(
                        UInt32(field_slice_index),
                        UInt32(0),
                        coefficient_offset,
                        UInt32(UInt64(len(coefficients)) - coefficient_offset),
                        ddd,
                        alpha,
                        sqrt_beta,
                        let_mix,
                        let_bar,
                    )
                )
            voxel_scenarios.append(
                RobustScenario(slice_offset, UInt32(len(layout.field_slices)))
            )
            states.append(ScenarioState(0.0, 0.0, 0.0, 0.0))
    var packed_settings = settings.copy()
    packed_settings.flags |= OPTIMIZER_FLAG_BIOLOGICAL
    var rng_state = List[UInt32]()
    var problem = OptimizationProblem(
        packed_settings^,
        minimum_particle_policy.copy(),
        rng_state^,
        layout.field_count,
        UInt32(len(scenarios.matrices)),
        layout.field_slices.copy(),
        particles.copy(),
        direction^,
        initial_gradient^,
        point_active^,
        voxels^,
        voxel_scenarios^,
        states^,
        slices^,
        point_indices^,
        coefficients^,
    )
    problem.validate()
    return problem^
