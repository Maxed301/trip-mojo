#ifndef TRIP_MOJO_OPTIMIZER_ABI_H
#define TRIP_MOJO_OPTIMIZER_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MOJO_OPTIMIZER_FLAG_ROBUST_INCLUDE_DMAX = 1u << 0,
    MOJO_OPTIMIZER_FLAG_BIOLOGICAL = 1u << 1,
    MOJO_OPTIMIZER_FLAG_INITIALIZE = 1u << 2,
    MOJO_OPTIMIZER_FLAG_DEVICE_BOOTSTRAP = 1u << 3,
    MOJO_MINIMUM_PARTICLE_DISABLED = 0,
    MOJO_MINIMUM_PARTICLE_SIMPLE = 1,
    MOJO_MINIMUM_PARTICLE_COMPLEX_HOST_RNG = 2,
    MOJO_DOSE_ALGORITHM_MS = 1,
    MOJO_DOSE_ALGORITHM_MSDB = 2,
    MOJO_OPTIMIZATION_RESULT_FINAL_MIN_PARTICLES_APPLIED = 1u << 0
};

/* Field slices are grouped by contiguous field_index values starting at zero.
 * Their point ranges partition every point exactly once. MojoDoseMatrixSlice indices
 * select this table; coefficient_point_indices are UInt16-local to that range.
 */
typedef struct {
    uint32_t field_index;
    uint32_t beam_index;
    uint64_t point_offset;
    uint32_t point_count;
    uint32_t raster_stride; /* Raster row width for host complexminp neighbors. */
    double minimum_particles;
} MojoFieldSlice;

typedef struct {
    double prescribed_dose;
    double dose_weight;
    double dose_divisor;
    double maximum_dose_weight;
    double initial_dose;
    double prescribed_let;
    double let_weight;
    double overdose_tolerance;
    double rbe_cut;
    double rbe_alpha;
    double rbe_beta;
    double rbe_slope_max;
    double rbe_damage_cut;
    int32_t initial_min_scenario;
    int32_t initial_max_scenario;
    uint64_t scenario_offset;
} MojoOptimizationVoxel;

typedef struct {
    uint64_t slice_offset;
    uint32_t slice_count;
    uint32_t reserved;
} MojoRobustScenario;

typedef struct {
    uint32_t field_slice_index;
    uint32_t matrix_slice_index; /* Used only by procedural external matrices. */
    uint64_t coefficient_offset;
    uint32_t coefficient_count;
    double dose_coefficient;
    double alpha_coefficient;
    double sqrt_beta_coefficient;
    double let_mix_coefficient;
    double let_bar_coefficient;
} MojoDoseMatrixSlice;

typedef struct {
    double dose_minor;
    double alpha_minor;
    double sqrt_beta_minor;
    double let_mix_minor;
} MojoScenarioState;

typedef struct {
    uint32_t flags;
    uint32_t minimum_particle_policy;
    uint32_t dose_algorithm;
    uint64_t rng_seed;
    uint32_t rng_front;
    uint32_t rng_rear;
    uint64_t rng_state_count;
    uint32_t field_count;
    uint32_t scenario_count;
    uint32_t max_iterations;
    uint32_t grace_iterations;
    uint32_t avoidance_voxel_count;
    uint32_t active_point_count;
    uint32_t total_iterations;
    uint32_t step_factor_iterations;
    double fractions;
    double current_chi2;
    double dose_p_weighted_avg2;
    double epsilon;
    double chi_square_limit;
    double overdose_weight;
    double prescribed_dose;
    double configured_step_factor;
    double current_step;
    double update_elapsed;
    uint64_t field_slice_count;
    uint64_t point_count;
    uint64_t voxel_count;
    uint64_t voxel_scenario_count;
    uint64_t slice_count;
    uint64_t coefficient_count;
    const MojoFieldSlice *field_slices;
    const uint32_t *rng_state;
    const double *particles;
    const double *initial_direction;
    const double *initial_gradient;
    const uint8_t *point_active;
    const MojoOptimizationVoxel *voxels;
    const MojoRobustScenario *voxel_scenarios;
    const MojoScenarioState *scenario_states;
    const MojoDoseMatrixSlice *slices;
    const uint16_t *coefficient_point_indices;
    const double *coefficients;
} MojoOptimizationProblem;

/* Mojo owns these pre-sized contiguous arrays. C fills every element exactly
 * once before invoking a storage optimizer entry point. */
typedef struct {
    MojoFieldSlice *field_slices;
    uint32_t *rng_state;
    double *particles;
    double *initial_direction;
    double *initial_gradient;
    uint8_t *point_active;
    MojoOptimizationVoxel *voxels;
    MojoRobustScenario *voxel_scenarios;
    MojoScenarioState *scenario_states;
    MojoDoseMatrixSlice *slices;
    uint16_t *coefficient_point_indices;
    double *coefficients;
} MojoProblemArrays;

typedef struct MojoProblemHandle MojoProblemHandle;
struct MojoDeviceMatrix;

typedef struct {
    double chi2;
    double residual_percent;
    double relative_chi2_change;
    double step_factor;
    double exact_step;
    /* Accepted regular iterations, excluding the initialization update. */
    uint64_t iterations;
    uint64_t backtracks;
    uint64_t minimum_particle_deleted;
    uint64_t random_draws;
    uint32_t stop_reason;
    uint32_t flags;
} MojoOptimizationResult;

typedef struct {
    double chi2;
    double dose_p_weighted_avg2;
    double gradient_norm;
    double residual_percent;
} MojoEvaluationResult;

int32_t trip_optimizer_create_problem(
    const MojoOptimizationProblem *problem_template,
    MojoProblemHandle **storage_out,
    MojoProblemArrays *arrays_out
);

int32_t trip_optimizer_create_problem_with_matrix(
    const MojoOptimizationProblem *problem_template,
    struct MojoDeviceMatrix *matrix_storage,
    MojoProblemHandle **storage_out,
    MojoProblemArrays *arrays_out
);

int32_t trip_optimizer_create_problem_with_matrix_shards(
    const MojoOptimizationProblem *problem_template,
    struct MojoDeviceMatrix *const *matrix_storages,
    const uint64_t *matrix_entry_counts,
    const uint64_t *voxel_offsets,
    const uint64_t *voxel_counts,
    uint32_t device_count,
    MojoProblemHandle **storage_out,
    MojoProblemArrays *arrays_out
);

int32_t trip_optimizer_destroy_problem(MojoProblemHandle *storage);

int32_t trip_optimizer_destroy_problem_with_matrix(
    MojoProblemHandle *storage
);

int32_t trip_optimizer_destroy_problem_with_matrix_shards(
    MojoProblemHandle *storage
);

int32_t trip_optimizer_evaluate_problem(
    MojoProblemHandle *storage,
    double *gradient_out,
    uint64_t gradient_out_count,
    MojoEvaluationResult *result_out
);

int32_t trip_optimizer_optimize_problem(
    MojoProblemHandle *storage,
    double *particles_out,
    uint64_t particles_out_count,
    MojoOptimizationResult *result_out
);

/* Returns -3 when the library was built without an available accelerator. */
int32_t trip_optimizer_optimize_problem_on_device(
    MojoProblemHandle *storage,
    double *particles_out,
    uint64_t particles_out_count,
    MojoOptimizationResult *result_out
);

/*  accepts two or three devices and shards packed voxels. */
int32_t trip_optimizer_optimize_problem_on_devices(
    MojoProblemHandle *storage,
    double *particles_out,
    uint64_t particles_out_count,
    uint32_t device_count,
    MojoOptimizationResult *result_out
);

int32_t trip_optimizer_optimize_matrix_problem(
    MojoProblemHandle *storage,
    double *particles_out,
    uint64_t particles_out_count,
    MojoOptimizationResult *result_out
);

int32_t trip_optimizer_optimize_matrix_problem_shards(
    MojoProblemHandle *storage,
    double *particles_out,
    uint64_t particles_out_count,
    MojoOptimizationResult *result_out
);

int32_t trip_optimizer_evaluate_view(
    const MojoOptimizationProblem *problem,
    double *gradient_out,
    uint64_t gradient_out_count,
    MojoEvaluationResult *result_out
);

int32_t trip_optimizer_optimize_view(
    const MojoOptimizationProblem *problem,
    double *particles_out,
    uint64_t particles_out_count,
    MojoOptimizationResult *result_out
);

/* Returns -3 when the library was built without an available accelerator. */
int32_t trip_optimizer_optimize_view_on_device(
    const MojoOptimizationProblem *problem,
    double *particles_out,
    uint64_t particles_out_count,
    MojoOptimizationResult *result_out
);

#ifdef __cplusplus
}
#endif

#endif
