#ifndef TRIP_MOJO_FDCB_ABI_V1_H
#define TRIP_MOJO_FDCB_ABI_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    FDCB_ABI_VERSION_V1 = 1,
    FDCB_FLAG_ROBUST_INCLUDE_DMAX = 1u << 0,
    FDCB_FLAG_BIOLOGICAL = 1u << 1,
    FDCB_FLAG_INITIALIZE = 1u << 2,
    FDCB_FLAG_DEVICE_BOOTSTRAP = 1u << 3,
    FDCB_PRECISION_REFERENCE = 1,
    FDCB_PRECISION_MIXED32 = 2,
    FDCB_MIN_PARTICLE_DISABLED = 0,
    FDCB_MIN_PARTICLE_SIMPLE = 1,
    FDCB_MIN_PARTICLE_COMPLEX_HOST_RNG = 2,
    FDCB_OPTIMIZER_FDCB = 1,
    FDCB_DOSE_MS = 1,
    FDCB_DOSE_MSDB = 2,
    FDCB_BIOLOGY_NONE = 0,
    FDCB_BIOLOGY_LOW_DOSE = 1,
    FDCB_RESULT_FINAL_MIN_PARTICLES_APPLIED = 1u << 0
};

/* CPU entry points accept REFERENCE only. Accelerator entry points accept the
 * mode selected when their library was built; mixed32 is explicit and
 * experimental, never an implicit downgrade of a reference problem.
 */

/* Field slices are grouped by contiguous field_index values starting at zero.
 * Their point ranges partition every point exactly once. FDCBSliceV1 indices
 * select this table; coefficient_point_indices are UInt16-local to that range.
 */
typedef struct {
    uint32_t field_index;
    uint32_t beam_index;
    uint64_t point_offset;
    uint32_t point_count;
    uint32_t raster_stride; /* Raster row width for host complexminp neighbors. */
    double minimum_particles;
} FDCBFieldSliceV1;

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
} FDCBVoxelV1;

typedef struct {
    uint64_t slice_offset;
    uint32_t slice_count;
    uint32_t reserved;
} FDCBVoxelScenarioV1;

typedef struct {
    uint32_t field_slice_index;
    uint32_t reserved;
    uint64_t coefficient_offset;
    uint32_t coefficient_count;
    double dose_coefficient;
    double alpha_coefficient;
    double sqrt_beta_coefficient;
    double let_mix_coefficient;
    double let_bar_coefficient;
} FDCBSliceV1;

typedef struct {
    double dose_minor;
    double alpha_minor;
    double sqrt_beta_minor;
    double let_mix_minor;
} FDCBScenarioStateV1;

typedef struct {
    uint32_t version;
    uint32_t flags;
    uint32_t precision_mode;
    uint32_t minimum_particle_policy;
    uint32_t optimizer_algorithm;
    uint32_t dose_algorithm;
    uint32_t biology_model;
    uint32_t max_threads;
    uint32_t reserved;
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
    const FDCBFieldSliceV1 *field_slices;
    const uint32_t *rng_state;
    const double *particles;
    const double *initial_direction;
    const double *initial_gradient;
    const uint8_t *point_active;
    const FDCBVoxelV1 *voxels;
    const FDCBVoxelScenarioV1 *voxel_scenarios;
    const FDCBScenarioStateV1 *scenario_states;
    const FDCBSliceV1 *slices;
    const uint16_t *coefficient_point_indices;
    const double *coefficients;
} FDCBProblemViewV1;

/* Mojo owns these pre-sized contiguous arrays. C fills every element exactly
 * once before invoking a storage optimizer entry point. */
typedef struct {
    FDCBFieldSliceV1 *field_slices;
    uint32_t *rng_state;
    double *particles;
    double *initial_direction;
    double *initial_gradient;
    uint8_t *point_active;
    FDCBVoxelV1 *voxels;
    FDCBVoxelScenarioV1 *voxel_scenarios;
    FDCBScenarioStateV1 *scenario_states;
    FDCBSliceV1 *slices;
    uint16_t *coefficient_point_indices;
    double *coefficients;
} FDCBWritableArraysV1;

typedef struct FDCBProblemStorageV1 FDCBProblemStorageV1;
struct FDCBMatrixStorageV1;

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
} FDCBResultV1;

typedef struct {
    double chi2;
    double dose_p_weighted_avg2;
    double gradient_norm;
    double residual_percent;
} FDCBEvaluationResultV1;

int32_t trip_fdcb_storage_create_v1(
    const FDCBProblemViewV1 *problem_template,
    FDCBProblemStorageV1 **storage_out,
    FDCBWritableArraysV1 *arrays_out
);

int32_t trip_fdcb_matrix_problem_storage_create_v1(
    const FDCBProblemViewV1 *problem_template,
    struct FDCBMatrixStorageV1 *matrix_storage,
    FDCBProblemStorageV1 **storage_out,
    FDCBWritableArraysV1 *arrays_out
);

int32_t trip_fdcb_storage_destroy_v1(FDCBProblemStorageV1 *storage);

int32_t trip_fdcb_matrix_problem_storage_destroy_v1(
    FDCBProblemStorageV1 *storage
);

int32_t trip_fdcb_storage_evaluate_v1(
    FDCBProblemStorageV1 *storage,
    double *gradient_out,
    uint64_t gradient_out_count,
    FDCBEvaluationResultV1 *result_out
);

int32_t trip_fdcb_storage_optimize_v1(
    FDCBProblemStorageV1 *storage,
    double *particles_out,
    uint64_t particles_out_count,
    FDCBResultV1 *result_out
);

/* Returns -3 unless built with FDCB_ABI_ACCELERATOR=true and a GPU target. */
int32_t trip_fdcb_storage_optimize_accelerator_v1(
    FDCBProblemStorageV1 *storage,
    double *particles_out,
    uint64_t particles_out_count,
    FDCBResultV1 *result_out
);

int32_t trip_fdcb_matrix_problem_optimize_accelerator_v1(
    FDCBProblemStorageV1 *storage,
    double *particles_out,
    uint64_t particles_out_count,
    FDCBResultV1 *result_out
);

int32_t trip_fdcb_evaluate_v1(
    const FDCBProblemViewV1 *problem,
    double *gradient_out,
    uint64_t gradient_out_count,
    FDCBEvaluationResultV1 *result_out
);

int32_t trip_fdcb_optimize_v1(
    const FDCBProblemViewV1 *problem,
    double *particles_out,
    uint64_t particles_out_count,
    FDCBResultV1 *result_out
);

/* Returns -3 unless built with FDCB_ABI_ACCELERATOR=true and a GPU target. */
int32_t trip_fdcb_optimize_accelerator_v1(
    const FDCBProblemViewV1 *problem,
    double *particles_out,
    uint64_t particles_out_count,
    FDCBResultV1 *result_out
);

#ifdef __cplusplus
}
#endif

#endif
