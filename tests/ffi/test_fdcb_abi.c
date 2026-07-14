#include "fdcb_abi_v1.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

_Static_assert(sizeof(FDCBFieldSliceV1) == 32, "field slice ABI");
_Static_assert(sizeof(FDCBVoxelV1) == 120, "voxel ABI");
_Static_assert(sizeof(FDCBVoxelScenarioV1) == 16, "scenario ABI");
_Static_assert(sizeof(FDCBSliceV1) == 64, "slice ABI");
_Static_assert(sizeof(FDCBScenarioStateV1) == 32, "state ABI");
_Static_assert(offsetof(FDCBProblemViewV1, field_slices) % 8 == 0, "pointer alignment");

int main(void) {
    const double mev_to_gy = 1.602189e-8;
    FDCBFieldSliceV1 field_slice = {0, 0, 0, 2, 0, 0.0};
    double particles[2] = {100.0, 50.0};
    double zeros[2] = {0.0, 0.0};
    uint8_t active[2] = {1, 1};
    FDCBVoxelV1 voxels[2] = {
        {.prescribed_dose = 3.0, .dose_weight = 1.0, .dose_divisor = 1.0,
         .overdose_tolerance = 0.05, .scenario_offset = 0},
        {.prescribed_dose = -2.0, .dose_weight = 0.3, .dose_divisor = 1.0,
         .overdose_tolerance = 0.05, .scenario_offset = 2},
    };
    FDCBVoxelScenarioV1 scenarios[4] = {
        {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0},
    };
    FDCBScenarioStateV1 states[4] = {{0}};
    FDCBSliceV1 slices[4] = {
        {.coefficient_offset = 0, .coefficient_count = 2, .dose_coefficient = 1.0 / mev_to_gy},
        {.coefficient_offset = 2, .coefficient_count = 2, .dose_coefficient = 1.0 / mev_to_gy},
        {.coefficient_offset = 4, .coefficient_count = 2, .dose_coefficient = 1.0 / mev_to_gy},
        {.coefficient_offset = 6, .coefficient_count = 2, .dose_coefficient = 1.0 / mev_to_gy},
    };
    uint16_t indices[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    double coefficients[8] = {0.008, 0.015, 0.012, 0.025, 0.025, 0.008, 0.035, 0.012};
    FDCBProblemViewV1 problem;
    memset(&problem, 0, sizeof(problem));
    problem.version = FDCB_ABI_VERSION_V1;
    problem.precision_mode = FDCB_PRECISION_REFERENCE;
    problem.minimum_particle_policy = FDCB_MIN_PARTICLE_DISABLED;
    problem.optimizer_algorithm = FDCB_OPTIMIZER_FDCB;
    problem.dose_algorithm = FDCB_DOSE_MS;
    problem.biology_model = FDCB_BIOLOGY_NONE;
    problem.max_threads = 12;
    problem.field_count = 1;
    problem.scenario_count = 2;
    problem.max_iterations = 7;
    problem.grace_iterations = 100;
    problem.fractions = 1.0;
    problem.overdose_weight = 1.0;
    problem.field_slice_count = 1;
    problem.point_count = 2;
    problem.voxel_count = 2;
    problem.voxel_scenario_count = 4;
    problem.slice_count = 4;
    problem.coefficient_count = 8;
    problem.field_slices = &field_slice;
    problem.particles = particles;
    problem.initial_direction = zeros;
    problem.initial_gradient = zeros;
    problem.point_active = active;
    problem.voxels = voxels;
    problem.voxel_scenarios = scenarios;
    problem.scenario_states = states;
    problem.slices = slices;
    problem.coefficient_point_indices = indices;
    problem.coefficients = coefficients;

    double output[2];
    double gradient[2];
    FDCBEvaluationResultV1 evaluation;
    assert(trip_fdcb_evaluate_v1(&problem, gradient, 1, &evaluation) == -2);
    assert(trip_fdcb_evaluate_v1(&problem, gradient, 2, &evaluation) == 0);
    assert(isfinite(evaluation.chi2));
    assert(evaluation.gradient_norm > 0.0);
    problem.version++;
    assert(trip_fdcb_evaluate_v1(&problem, gradient, 2, &evaluation) == -1);
    problem.version = FDCB_ABI_VERSION_V1;
    problem.optimizer_algorithm = 99;
    assert(trip_fdcb_evaluate_v1(&problem, gradient, 2, &evaluation) == -1);
    problem.optimizer_algorithm = FDCB_OPTIMIZER_FDCB;
    FDCBResultV1 result;
    assert(trip_fdcb_optimize_v1(&problem, output, 2, &result) == 0);
    assert(result.iterations == 7);
    assert(isfinite(result.chi2));
    assert(isfinite(result.exact_step));
    assert(output[0] != particles[0] || output[1] != particles[1]);
    FDCBResultV1 initialized_result;
    problem.flags = FDCB_FLAG_INITIALIZE;
    problem.max_iterations = 1;
    assert(trip_fdcb_optimize_v1(&problem, output, 2, &initialized_result) == 0);
    assert(initialized_result.iterations == 1);
    problem.flags = 0;
    problem.max_iterations = 7;
    double accelerator_output[2];
    FDCBResultV1 accelerator_result;
    int32_t accelerator_status = trip_fdcb_optimize_accelerator_v1(
        &problem, accelerator_output, 2, &accelerator_result);
#ifdef FDCB_TEST_MIXED32
    assert(accelerator_status == -1);
    problem.precision_mode = FDCB_PRECISION_MIXED32;
    assert(trip_fdcb_evaluate_v1(&problem, gradient, 2, &evaluation) == -1);
    assert(trip_fdcb_optimize_v1(&problem, output, 2, &result) == -1);
    accelerator_status = trip_fdcb_optimize_accelerator_v1(
        &problem, accelerator_output, 2, &accelerator_result);
    assert(accelerator_status == 0);
    assert(accelerator_result.iterations == 7);
    assert(isfinite(accelerator_result.chi2));
#else
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    assert(accelerator_status == 0);
#else
    assert(accelerator_status == 0 || accelerator_status == -3);
#endif
    if (accelerator_status == 0) {
        double scale = fabs(result.chi2) > 1.0 ? fabs(result.chi2) : 1.0;
        assert(accelerator_result.iterations == result.iterations);
        assert(fabs(accelerator_result.chi2 - result.chi2) <=
               scale * 8.0 * DBL_EPSILON);
        for (size_t i = 0; i < 2; ++i) {
            assert(isfinite(accelerator_output[i]));
        }
        assert(accelerator_output[0] != particles[0] ||
               accelerator_output[1] != particles[1]);
    }
#endif
    return 0;
}
