"""Backend-neutral, owned numeric boundary for robust fluence optimization.

All variable-length storage is held in flat contiguous Lists with fixed-width
indices. Native callers own those Lists directly. C callers may instead ask
Mojo to allocate the same storage, fill borrowed array pointers, and retain one
owner until the packed problem is destroyed.
"""

from std.algorithm import parallelize
from std.math import sqrt


comptime MINIMUM_PARTICLE_DISABLED = UInt32(0)
comptime MINIMUM_PARTICLE_SIMPLE = UInt32(1)
comptime MINIMUM_PARTICLE_COMPLEX_HOST_RNG = UInt32(2)
comptime MEV_TO_GY = 1.602189e-8
comptime FLOAT64_EPSILON = 2.220446049250313e-16
comptime PHYSICAL_CPU_THREADS = 12


@fieldwise_init
struct MinimumParticlePolicy(Copyable, Movable):
    """Host policy; numeric backends never advance minimum-particle RNG state.
    """

    var kind: UInt32
    var seed: UInt64
    var rng_front: UInt32
    var rng_rear: UInt32

    @staticmethod
    def disabled() -> MinimumParticlePolicy:
        return MinimumParticlePolicy(
            MINIMUM_PARTICLE_DISABLED, UInt64(0), UInt32(3), UInt32(0)
        )

    @staticmethod
    def complex_host_rng(seed: UInt64) -> MinimumParticlePolicy:
        return MinimumParticlePolicy(
            MINIMUM_PARTICLE_COMPLEX_HOST_RNG,
            seed,
            UInt32(3),
            UInt32(0),
        )


@fieldwise_init
struct OptimizerSettings(Copyable, Movable):
    var biological: Bool
    var include_dmax: Bool
    var max_iterations: UInt32
    var grace_iterations: UInt32
    var avoidance_voxel_count: UInt32
    var active_point_count: UInt32
    var total_iterations: UInt32
    var step_factor_iterations: UInt32
    var fractions: Float64
    var current_chi2: Float64
    var dose_p_weighted_avg2: Float64
    var epsilon: Float64
    var chi_square_limit: Float64
    var overdose_weight: Float64
    var prescribed_dose: Float64
    var configured_step_factor: Float64
    var current_step: Float64
    var update_elapsed: Float64

    @staticmethod
    def reference_defaults() -> OptimizerSettings:
        return OptimizerSettings(
            False,
            False,
            UInt32(0),
            UInt32(0),
            UInt32(0),
            UInt32(0),
            UInt32(0),
            UInt32(0),
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )


@fieldwise_init
struct FieldSlice(Copyable, Movable):
    var field_index: UInt32
    var beam_index: UInt32
    var point_offset: UInt64
    var point_count: UInt32
    var raster_stride: UInt32
    var minimum_particles: Float64


@fieldwise_init
struct OptimizationVoxel(Copyable, Movable):
    var prescribed_dose: Float64
    var dose_weight: Float64
    var dose_divisor: Float64
    var maximum_dose_weight: Float64
    var initial_dose: Float64
    var prescribed_let: Float64
    var let_weight: Float64
    var overdose_tolerance: Float64
    var rbe_cut: Float64
    var rbe_alpha: Float64
    var rbe_beta: Float64
    var rbe_slope_max: Float64
    var rbe_damage_cut: Float64
    var initial_min_scenario: Int32
    var initial_max_scenario: Int32
    var scenario_offset: UInt64


@fieldwise_init
struct RobustScenario(Copyable, Movable):
    var slice_offset: UInt64
    var slice_count: UInt32


@fieldwise_init
struct DoseMatrixSlice(Copyable, Movable):
    var field_slice_index: UInt32
    var matrix_slice_index: UInt32
    var coefficient_offset: UInt64
    var coefficient_count: UInt32
    var dose_coefficient: Float64
    var alpha_coefficient: Float64
    var sqrt_beta_coefficient: Float64
    var let_mix_coefficient: Float64
    var let_bar_coefficient: Float64


@fieldwise_init
struct ScenarioState(Copyable, Movable):
    var dose_minor: Float64
    var alpha_minor: Float64
    var sqrt_beta_minor: Float64
    var let_mix_minor: Float64


@fieldwise_init
struct OptimizationProblem(Copyable, Movable):
    """Owned packed problem. Backends borrow these arrays for one optimization.

    CT motion states are deliberately not a numeric axis here. TRiP appends
    every selected state's optimization voxels to `voxels`; each flattened
    voxel then owns the same robust-scenario range as a 3D voxel. This keeps
    4D setup/deformation in the control plane while CPU and accelerator
    backends run exactly the same sparse optimizer kernels.
    """

    var settings: OptimizerSettings
    var minimum_particle_policy: MinimumParticlePolicy
    var host_rng_state: List[UInt32]
    var field_count: UInt32
    var scenario_count: UInt32
    var field_slices: List[FieldSlice]
    var particles: List[Float64]
    var initial_direction: List[Float64]
    var initial_gradient: List[Float64]
    var point_active: List[UInt8]
    var voxels: List[OptimizationVoxel]
    var voxel_scenarios: List[RobustScenario]
    var scenario_states: List[ScenarioState]
    var slices: List[DoseMatrixSlice]
    var coefficient_point_indices: List[UInt16]
    var coefficients: List[Float64]

    def validate(
        self,
        external_coefficient_count: UInt64 = UInt64.MAX,
    ) raises:
        if (
            self.settings.fractions <= 0.0
            or self.settings.overdose_weight <= 0.0
        ):
            raise Error(
                "optimizer fractions and overdose weight must be positive"
            )
        if self.field_count == UInt32(0):
            raise Error("optimizer problem requires at least one field")
        if self.scenario_count == UInt32(0):
            raise Error("optimizer problem requires at least one scenario")
        if len(self.field_slices) == 0 or len(self.voxels) == 0:
            raise Error("optimizer problem requires field slices and voxels")
        if (
            len(self.particles) != len(self.initial_direction)
            or len(self.particles) != len(self.initial_gradient)
            or len(self.particles) != len(self.point_active)
        ):
            raise Error("optimizer point arrays have inconsistent lengths")
        if self.settings.active_point_count > UInt32(len(self.particles)):
            raise Error("optimizer active point count exceeds the point arrays")
        var coefficients_are_external = external_coefficient_count != UInt64.MAX
        if not coefficients_are_external and len(
            self.coefficient_point_indices
        ) != len(self.coefficients):
            raise Error(
                "optimizer coefficient arrays have inconsistent lengths"
            )
        var available_coefficients = UInt64(len(self.coefficients))
        if coefficients_are_external:
            available_coefficients = external_coefficient_count
        if len(self.voxel_scenarios) != len(self.voxels) * Int(
            self.scenario_count
        ):
            raise Error(
                "optimizer voxel/scenario table has inconsistent length"
            )
        if len(self.scenario_states) != len(self.voxel_scenarios):
            raise Error(
                "optimizer scenario state table has inconsistent length"
            )
        if (
            self.minimum_particle_policy.kind != MINIMUM_PARTICLE_DISABLED
            and self.minimum_particle_policy.kind != MINIMUM_PARTICLE_SIMPLE
            and self.minimum_particle_policy.kind
            != MINIMUM_PARTICLE_COMPLEX_HOST_RNG
        ):
            raise Error("unknown minimum-particle policy")
        if len(self.host_rng_state) != 0 and len(self.host_rng_state) != 31:
            raise Error("optimizer host RNG state must contain 31 words")
        if len(self.host_rng_state) == 31 and (
            self.minimum_particle_policy.rng_front >= UInt32(31)
            or self.minimum_particle_policy.rng_rear >= UInt32(31)
        ):
            raise Error("optimizer host RNG indices are out of bounds")

        var expected_point_offset = UInt64(0)
        var expected_field_index = UInt32(0)
        for i in range(len(self.field_slices)):
            var field_slice = self.field_slices[i].copy()
            if field_slice.field_index >= self.field_count:
                raise Error("field slice has invalid field index")
            if i == 0:
                if field_slice.field_index != UInt32(0):
                    raise Error("field slices must start with field zero")
            elif field_slice.field_index != expected_field_index:
                if field_slice.field_index != expected_field_index + UInt32(1):
                    raise Error("field slices must be grouped and contiguous")
                expected_field_index = field_slice.field_index
            if field_slice.point_count > UInt32(UInt16.MAX) + UInt32(1):
                raise Error("field slice exceeds UInt16 local point indexing")
            if (
                self.minimum_particle_policy.kind
                == MINIMUM_PARTICLE_COMPLEX_HOST_RNG
                and field_slice.minimum_particles > 0.0
                and field_slice.raster_stride == UInt32(0)
            ):
                raise Error("complexminp field slice requires a raster stride")
            if field_slice.point_offset != expected_point_offset:
                raise Error("field slice point ranges must be contiguous")
            expected_point_offset += UInt64(field_slice.point_count)
        if expected_point_offset != UInt64(len(self.particles)):
            raise Error("field slice point ranges do not cover point arrays")
        if expected_field_index + UInt32(1) != self.field_count:
            raise Error("field slices do not cover every field")

        var expected_slice_offset = UInt64(0)
        for voxel_index in range(len(self.voxels)):
            var voxel = self.voxels[voxel_index].copy()
            if voxel.dose_divisor == 0.0:
                raise Error("voxel dose divisor must be nonzero")
            if voxel.prescribed_dose == 0.0 and (
                voxel.initial_min_scenario < Int32(0)
                or voxel.initial_max_scenario < Int32(0)
                or UInt32(voxel.initial_min_scenario) >= self.scenario_count
                or UInt32(voxel.initial_max_scenario) >= self.scenario_count
            ):
                raise Error("voxel initial scenario index is out of bounds")
            if (
                self.settings.biological
                and voxel.prescribed_dose != 0.0
                and voxel.rbe_alpha == 0.0
                and voxel.rbe_beta == 0.0
            ):
                raise Error("biological voxel has zero RBE alpha and beta")
            if (
                self.settings.biological
                and voxel.prescribed_dose != 0.0
                and voxel.rbe_slope_max == 0.0
            ):
                raise Error("biological voxel has zero high-dose RBE slope")
            if voxel.scenario_offset != UInt64(voxel_index) * UInt64(
                self.scenario_count
            ):
                raise Error(
                    "voxel scenario ranges must be voxel-major and contiguous"
                )
            for scenario_index in range(Int(self.scenario_count)):
                var vs_index = Int(voxel.scenario_offset) + scenario_index
                var voxel_scenario = self.voxel_scenarios[vs_index].copy()
                if voxel_scenario.slice_offset + UInt64(
                    voxel_scenario.slice_count
                ) > UInt64(len(self.slices)):
                    raise Error("voxel scenario slice range is out of bounds")
                if voxel_scenario.slice_offset != expected_slice_offset:
                    raise Error(
                        "voxel scenario slice ranges must be contiguous"
                    )
                expected_slice_offset += UInt64(voxel_scenario.slice_count)
        if expected_slice_offset != UInt64(len(self.slices)):
            raise Error("voxel scenario ranges do not cover packed slices")

        for slice_index in range(len(self.slices)):
            var packed_slice = self.slices[slice_index].copy()
            if Int(packed_slice.field_slice_index) >= len(self.field_slices):
                raise Error("optimizer slice has invalid field-slice index")
            if (
                packed_slice.coefficient_offset
                + UInt64(packed_slice.coefficient_count)
                > available_coefficients
            ):
                raise Error(
                    "optimizer slice coefficient range is out of bounds"
                )
            var field_slice = self.field_slices[
                Int(packed_slice.field_slice_index)
            ].copy()
            if coefficients_are_external:
                continue
            for entry in range(Int(packed_slice.coefficient_count)):
                var coefficient_index = (
                    Int(packed_slice.coefficient_offset) + entry
                )
                if (
                    UInt32(self.coefficient_point_indices[coefficient_index])
                    >= field_slice.point_count
                ):
                    raise Error(
                        "coefficient point index is outside its field slice"
                    )


@fieldwise_init
struct ObjectiveEvaluation(Copyable, Movable):
    var dose_by_voxel_scenario: List[Float64]
    var dose_min: List[Float64]
    var dose_max: List[Float64]
    var min_scenario: List[Int32]
    var max_scenario: List[Int32]
    var gradient: List[Float64]
    var chi2: Float64
    var dose_p_weighted_avg2: Float64
    var gradient_norm: Float64


def compute_forward_dose(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> List[Float64]:
    problem.validate()
    return compute_forward_dose_validated(problem, particles)


def compute_forward_dose_validated(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> List[Float64]:
    comptime assert PHYSICAL_CPU_THREADS > 0
    if len(particles) != len(problem.particles):
        raise Error(
            "particle vector length does not match packed optimization problem"
        )
    var dose = List[Float64]()
    dose.resize(len(problem.voxel_scenarios), 0.0)
    var particle_ptr = Span(particles).unsafe_ptr()
    var voxel_ptr = Span(problem.voxels).unsafe_ptr()
    var scenario_ptr = Span(problem.voxel_scenarios).unsafe_ptr()
    var state_ptr = Span(problem.scenario_states).unsafe_ptr()
    var slice_ptr = Span(problem.slices).unsafe_ptr()
    var field_slice_ptr = Span(problem.field_slices).unsafe_ptr()
    var index_ptr = Span(problem.coefficient_point_indices).unsafe_ptr()
    var coefficient_ptr = Span(problem.coefficients).unsafe_ptr()

    @parameter
    def compute_dose(vs_index: Int):
        var total = state_ptr[vs_index].dose_minor
        var voxel_scenario = scenario_ptr[vs_index].copy()
        for local_slice in range(Int(voxel_scenario.slice_count)):
            var packed_slice = slice_ptr[
                Int(voxel_scenario.slice_offset) + local_slice
            ].copy()
            var field_slice = field_slice_ptr[
                Int(packed_slice.field_slice_index)
            ].copy()
            var slice_dot = 0.0
            var coefficient_base = Int(packed_slice.coefficient_offset)
            var point_base = Int(field_slice.point_offset)
            for local_entry in range(Int(packed_slice.coefficient_count)):
                var coefficient_index = coefficient_base + local_entry
                var point_index = point_base + Int(index_ptr[coefficient_index])
                slice_dot += particle_ptr[point_index] * Float64(
                    coefficient_ptr[coefficient_index]
                )
            total += slice_dot * Float64(packed_slice.dose_coefficient)
        var voxel_index = vs_index // Int(problem.scenario_count)
        dose[vs_index] = total * MEV_TO_GY + voxel_ptr[voxel_index].initial_dose

    parallelize[compute_dose](len(dose), PHYSICAL_CPU_THREADS)
    return dose^


def evaluate_physical_objective(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> ObjectiveEvaluation:
    problem.validate()
    return evaluate_physical_objective_validated(problem, particles)


def evaluate_physical_objective_validated(
    problem: OptimizationProblem, particles: List[Float64]
) raises -> ObjectiveEvaluation:
    var scenario_dose = compute_forward_dose_validated(problem, particles)
    var dose_min = List[Float64]()
    var dose_max = List[Float64]()
    var min_scenario = List[Int32]()
    var max_scenario = List[Int32]()
    var gradient = List[Float64]()
    dose_min.resize(len(problem.voxels), 0.0)
    dose_max.resize(len(problem.voxels), 0.0)
    min_scenario.resize(len(problem.voxels), Int32(0))
    max_scenario.resize(len(problem.voxels), Int32(0))
    gradient.resize(len(problem.particles), 0.0)
    var chi2 = 0.0
    var weighted = 0.0

    for voxel_index in range(len(problem.voxels)):
        var voxel = problem.voxels[voxel_index].copy()
        if voxel.prescribed_dose == 0.0:
            min_scenario[voxel_index] = voxel.initial_min_scenario
            max_scenario[voxel_index] = voxel.initial_max_scenario
            continue
        var base = Int(voxel.scenario_offset)
        var dmin = scenario_dose[base]
        var dmax = scenario_dose[base]
        var imin = 0
        var imax = 0
        for scenario_index in range(1, Int(problem.scenario_count)):
            var value = scenario_dose[base + scenario_index]
            if value < dmin:
                dmin = value
                imin = scenario_index
            if value > dmax:
                dmax = value
                imax = scenario_index
        dose_min[voxel_index] = dmin
        dose_max[voxel_index] = dmax
        min_scenario[voxel_index] = Int32(imin)
        max_scenario[voxel_index] = Int32(imax)

        var prescribed = abs_f64(voxel.prescribed_dose)
        var selected_dose = dmin
        if voxel.prescribed_dose < 0.0:
            selected_dose = dmax
        var weight = voxel.dose_weight / voxel.dose_divisor
        var chi_residual = prescribed - selected_dose
        if chi_residual < 0.0 or voxel.prescribed_dose > 0.0:
            chi2 += (chi_residual * weight) * (chi_residual * weight)
        weighted += (weight * voxel.prescribed_dose) * (
            weight * voxel.prescribed_dose
        )
        if problem.settings.include_dmax and voxel.prescribed_dose > 0.0:
            var max_weight = voxel.maximum_dose_weight / voxel.dose_divisor
            var max_residual = prescribed - dmax
            chi2 += (max_residual * max_weight) * (max_residual * max_weight)
            weighted += (max_weight * voxel.prescribed_dose) * (
                max_weight * voxel.prescribed_dose
            )
    var point_count = len(problem.particles)
    var thread_gradients = List[Float64]()
    thread_gradients.resize(PHYSICAL_CPU_THREADS * point_count, 0.0)

    @parameter
    def scatter_worker(worker: Int):
        var start = worker * len(problem.voxels) // PHYSICAL_CPU_THREADS
        var end = (worker + 1) * len(problem.voxels) // PHYSICAL_CPU_THREADS
        var output_base = worker * point_count
        for voxel_index in range(start, end):
            var voxel = problem.voxels[voxel_index].copy()
            if voxel.prescribed_dose == 0.0:
                continue
            var selected = Int(min_scenario[voxel_index])
            var selected_dose = dose_min[voxel_index]
            if voxel.prescribed_dose < 0.0:
                selected = Int(max_scenario[voxel_index])
                selected_dose = dose_max[voxel_index]
            scatter_packed_physical_gradient(
                problem,
                particles,
                voxel_index,
                selected,
                selected_dose,
                voxel.dose_weight,
                thread_gradients,
                False,
                output_base,
            )
            if problem.settings.include_dmax and voxel.prescribed_dose > 0.0:
                scatter_packed_physical_gradient(
                    problem,
                    particles,
                    voxel_index,
                    Int(max_scenario[voxel_index]),
                    dose_max[voxel_index],
                    voxel.maximum_dose_weight,
                    thread_gradients,
                    True,
                    output_base,
                )

    parallelize[scatter_worker](PHYSICAL_CPU_THREADS, PHYSICAL_CPU_THREADS)
    for worker in range(PHYSICAL_CPU_THREADS):
        var base = worker * point_count
        for point in range(point_count):
            gradient[point] += thread_gradients[base + point]

    var norm2 = 0.0
    for i in range(len(gradient)):
        norm2 += gradient[i] * gradient[i]
    return ObjectiveEvaluation(
        scenario_dose^,
        dose_min^,
        dose_max^,
        min_scenario^,
        max_scenario^,
        gradient^,
        chi2,
        weighted,
        sqrt(norm2),
    )


def scatter_packed_physical_gradient(
    problem: OptimizationProblem,
    particles: List[Float64],
    voxel_index: Int,
    scenario_index: Int,
    selected_dose: Float64,
    pass_weight: Float64,
    mut gradient: List[Float64],
    maximum_pass: Bool = False,
    output_base: Int = 0,
):
    var voxel = problem.voxels[voxel_index].copy()
    var residual = abs_f64(voxel.prescribed_dose) - selected_dose
    var divisor = voxel.dose_divisor
    if voxel.prescribed_dose < 0.0 or maximum_pass:
        if residual > 0.0:
            residual = 0.0
    elif residual < 0.0:
        if residual > voxel.prescribed_dose * -voxel.overdose_tolerance:
            residual = 0.0
        else:
            residual += voxel.prescribed_dose * voxel.overdose_tolerance
            divisor /= problem.settings.overdose_weight
    if residual == 0.0:
        return
    var scaled_weight = pass_weight / divisor
    var factor = residual * scaled_weight * (2.0 * scaled_weight)
    var vs_index = Int(voxel.scenario_offset) + scenario_index
    var voxel_scenario = problem.voxel_scenarios[vs_index].copy()
    var particle_ptr = Span(particles).unsafe_ptr()
    var active_ptr = Span(problem.point_active).unsafe_ptr()
    var gradient_ptr = Span(gradient).unsafe_ptr()
    var slice_ptr = Span(problem.slices).unsafe_ptr()
    var field_slice_ptr = Span(problem.field_slices).unsafe_ptr()
    var index_ptr = Span(problem.coefficient_point_indices).unsafe_ptr()
    var coefficient_ptr = Span(problem.coefficients).unsafe_ptr()
    for local_slice in range(Int(voxel_scenario.slice_count)):
        var packed_slice = slice_ptr[
            Int(voxel_scenario.slice_offset) + local_slice
        ].copy()
        var field_slice = field_slice_ptr[
            Int(packed_slice.field_slice_index)
        ].copy()
        var slice_factor = (
            factor * Float64(packed_slice.dose_coefficient) * MEV_TO_GY
        )
        for local_entry in range(Int(packed_slice.coefficient_count)):
            var coefficient_index = (
                Int(packed_slice.coefficient_offset) + local_entry
            )
            var point_index = Int(field_slice.point_offset) + Int(
                index_ptr[coefficient_index]
            )
            if (
                active_ptr[point_index] != UInt8(0)
                and particle_ptr[point_index] != 0.0
            ):
                gradient_ptr[
                    output_base + point_index
                ] += slice_factor * Float64(coefficient_ptr[coefficient_index])


def abs_f64(value: Float64) -> Float64:
    if value < 0.0:
        return -value
    return value
