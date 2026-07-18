#include "fdcb_abi_v1.h"
#include "fdcb_matrix_abi_v1.h"

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
_Static_assert(sizeof(FDCBMatrixEnergySliceV1) == 16, "matrix energy ABI");
_Static_assert(sizeof(FDCBMatrixPointV1) == 24, "matrix point ABI");
_Static_assert(sizeof(FDCBMatrixGroupV1) == 56, "matrix group ABI");
_Static_assert(sizeof(FDCBMatrixRawEnergyV1) == 40, "matrix raw energy ABI");
_Static_assert(sizeof(FDCBMatrixRawGroupV1) == 32, "matrix raw group ABI");
_Static_assert(sizeof(FDCBMatrixDDDTableV1) == 8, "matrix DDD table ABI");
_Static_assert(sizeof(FDCBMatrixDDDEntryV1) == 40, "matrix DDD entry ABI");
_Static_assert(sizeof(FDCBMatrixProblemViewV1) == 104, "matrix view ABI");
_Static_assert(sizeof(FDCBMatrixResultV1) == 72, "matrix result ABI");

static double matrix_reference_output[260];
static FDCBResultV1 matrix_reference_result;

static void test_matrix_accelerator(int procedural) {
    FDCBMatrixEnergySliceV1 energies[2] = {
        {0, 260, 0}, {0, 260, 0},
    };
    FDCBMatrixPointV1 points[260];
    FDCBMatrixGroupV1 group = {
        .slice_offset = 0,
        .slice_count = 2,
        .bev_x = 0.0,
        .bev_y = 0.0,
        .relative_cutoff = 0.1,
    };
    FDCBMatrixRawEnergyV1 raw_energies[2] = {
        {0, 0, 0.0, 0.0, 100.0, 100.0},
        {0, 0, 0.0, 4.0, 100.0, 100.0},
    };
    FDCBMatrixRawGroupV1 raw_group = {0, 2, 5.0, 0.0, 0.0};
    FDCBMatrixDDDTableV1 table = {0, 2};
    FDCBMatrixDDDEntryV1 entries[2] = {
        {0.0, 1.0, 1.0, 0.0, 0.0},
        {1.0, 3.0, 1.0, 0.0, 0.0},
    };
    FDCBMatrixProblemViewV1 problem = {
        .version = FDCB_MATRIX_ABI_VERSION_V1,
        .group_count = 1,
        .energy_slice_count = 2,
        .maximum_group_slices = 2,
        .slice_count = 2,
        .point_count = 260,
        .ddd_table_count = 1,
        .ddd_entry_count = 2,
        .energy_slices = energies,
        .points = points,
        .groups = &group,
        .raw_energies = raw_energies,
        .raw_groups = &raw_group,
        .ddd_tables = &table,
        .ddd_entries = entries,
    };
    if (procedural)
        problem.flags = FDCB_MATRIX_FLAG_DEVICE_ONLY |
                        FDCB_MATRIX_FLAG_FORCE_PROCEDURAL;
    FDCBMatrixStorageV1 *storage = NULL;
    FDCBMatrixResultV1 result;
    uint32_t expected_count = 0;
    for (uint32_t point = 0; point < 260; ++point) {
        points[point].x = point % 7 == 0 ? 5.0 : 0.0;
        points[point].y = 0.0;
        points[point].f2_max = 100.0;
        if (point % 7 != 0) ++expected_count;
    }
    int32_t status = trip_fdcb_matrix_build_accelerator_v1(
        &problem, &storage, &result);
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    assert(status == 0);
#else
    assert(status == 0 || status == -3);
#endif
    if (status != 0) return;
    assert(storage != NULL);
    assert(result.entry_count == 2 * expected_count);
    assert(result.slice_count == 2);
    assert(result.group_count == 1);
    assert(!!(result.flags & FDCB_MATRIX_RESULT_PROCEDURAL) == !!procedural);
    assert(result.group_entry_counts[0] == 2 * expected_count);
    assert(result.slice_entry_counts[0] == expected_count);
    assert(result.slice_entry_counts[1] == expected_count);
    assert(result.slice_dose[0] == 2.0);
    assert(result.slice_dose[1] == 2.0);
    if (!procedural) {
        for (uint32_t matrix_slice = 0; matrix_slice < 2; ++matrix_slice) {
            uint32_t output = matrix_slice * expected_count;
            for (uint32_t point = 0; point < 260; ++point) {
                if (point % 7 == 0) continue;
                assert(result.point_indices[output] == point);
                assert(isfinite(result.coefficients[output]));
                assert(result.coefficients[output] > 0.0);
                ++output;
            }
        }
    }
    {
        const double mev_to_gy = 1.602189e-8;
        FDCBFieldSliceV1 field_slice = {0, 0, 0, 260, 0, 0.0};
        double particles[260];
        double zeros[260] = {0};
        uint8_t active[260];
        FDCBVoxelV1 voxel = {
            .prescribed_dose = 1.0,
            .dose_weight = 1.0,
            .dose_divisor = 1.0,
            .overdose_tolerance = 0.05,
            .scenario_offset = 0,
        };
        FDCBVoxelScenarioV1 scenario = {0, 2, 0};
        FDCBScenarioStateV1 state = {0};
        FDCBSliceV1 slices[2] = {
            {
                .field_slice_index = 0,
                .matrix_slice_index = 1,
                .coefficient_offset = expected_count,
                .coefficient_count = expected_count,
                .dose_coefficient = 1.0 / mev_to_gy,
            },
            {
                .field_slice_index = 0,
                .matrix_slice_index = 0,
                .coefficient_offset = 0,
                .coefficient_count = expected_count,
                .dose_coefficient = 1.0 / mev_to_gy,
            },
        };
        FDCBProblemViewV1 optimizer = {0};
        for (uint32_t point = 0; point < 260; ++point) {
            particles[point] = 1.0;
            active[point] = 1;
        }
        optimizer.version = FDCB_ABI_VERSION_V1;
        optimizer.flags = FDCB_FLAG_INITIALIZE | FDCB_FLAG_DEVICE_BOOTSTRAP;
        optimizer.precision_mode = FDCB_PRECISION_REFERENCE;
        optimizer.minimum_particle_policy = FDCB_MIN_PARTICLE_DISABLED;
        optimizer.optimizer_algorithm = FDCB_OPTIMIZER_FDCB;
        optimizer.dose_algorithm = FDCB_DOSE_MS;
        optimizer.biology_model = FDCB_BIOLOGY_NONE;
        optimizer.max_threads = 12;
        optimizer.field_count = 1;
        optimizer.scenario_count = 1;
        optimizer.max_iterations = 1;
        optimizer.grace_iterations = 100;
        optimizer.active_point_count = 260;
        optimizer.fractions = 1.0;
        optimizer.overdose_weight = 1.0;
        optimizer.field_slice_count = 1;
        optimizer.point_count = 260;
        optimizer.voxel_count = 1;
        optimizer.voxel_scenario_count = 1;
        optimizer.slice_count = 2;
        optimizer.coefficient_count = result.entry_count;

        FDCBProblemStorageV1 *problem_storage = NULL;
        FDCBWritableArraysV1 arrays;
        assert(trip_fdcb_matrix_problem_storage_create_v1(
                   &optimizer, storage, &problem_storage, &arrays) == 0);
        assert(problem_storage != NULL);
        memcpy(arrays.field_slices, &field_slice, sizeof(field_slice));
        memcpy(arrays.particles, particles, sizeof(particles));
        memcpy(arrays.initial_direction, zeros, sizeof(zeros));
        memcpy(arrays.initial_gradient, zeros, sizeof(zeros));
        memcpy(arrays.point_active, active, sizeof(active));
        memcpy(arrays.voxels, &voxel, sizeof(voxel));
        memcpy(arrays.voxel_scenarios, &scenario, sizeof(scenario));
        memcpy(arrays.scenario_states, &state, sizeof(state));
        memcpy(arrays.slices, slices, sizeof(slices));

        double output[260];
        FDCBResultV1 optimizer_result;
        assert(trip_fdcb_matrix_problem_optimize_accelerator_v1(
                   problem_storage, output, 260, &optimizer_result) == 0);
        assert(optimizer_result.iterations == 1);
        assert(isfinite(optimizer_result.chi2));
        for (uint32_t point = 0; point < 260; ++point) {
            assert(isfinite(output[point]));
        }
        if (!procedural) {
            memcpy(matrix_reference_output, output, sizeof(output));
            matrix_reference_result = optimizer_result;
        } else {
            assert(memcmp(matrix_reference_output, output, sizeof(output)) == 0);
            assert(matrix_reference_result.iterations == optimizer_result.iterations);
            assert(matrix_reference_result.chi2 == optimizer_result.chi2);
            assert(matrix_reference_result.residual_percent ==
                   optimizer_result.residual_percent);
            assert(matrix_reference_result.stop_reason ==
                   optimizer_result.stop_reason);
        }
        assert(trip_fdcb_matrix_problem_storage_destroy_v1(problem_storage) ==
               0);
    }
}

int main(void) {
    test_matrix_accelerator(0);
    test_matrix_accelerator(1);
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
    assert(result.flags & FDCB_RESULT_FINAL_MIN_PARTICLES_APPLIED);
    assert(isfinite(result.chi2));
    assert(isfinite(result.exact_step));
    assert(output[0] != particles[0] || output[1] != particles[1]);
    FDCBResultV1 initialized_result;
    problem.flags = FDCB_FLAG_INITIALIZE;
    problem.max_iterations = 1;
    assert(trip_fdcb_optimize_v1(&problem, output, 2, &initialized_result) == 0);
    assert(initialized_result.iterations == 1);
    assert(initialized_result.flags & FDCB_RESULT_FINAL_MIN_PARTICLES_APPLIED);
    problem.flags = 0;
    problem.max_iterations = 7;
    assert(trip_fdcb_optimize_v1(&problem, output, 2, &result) == 0);

    FDCBProblemViewV1 storage_template = problem;
    storage_template.field_slices = NULL;
    storage_template.rng_state = NULL;
    storage_template.particles = NULL;
    storage_template.initial_direction = NULL;
    storage_template.initial_gradient = NULL;
    storage_template.point_active = NULL;
    storage_template.voxels = NULL;
    storage_template.voxel_scenarios = NULL;
    storage_template.scenario_states = NULL;
    storage_template.slices = NULL;
    storage_template.coefficient_point_indices = NULL;
    storage_template.coefficients = NULL;
    FDCBProblemStorageV1 *storage = NULL;
    FDCBWritableArraysV1 arrays;
    assert(trip_fdcb_storage_create_v1(
               &storage_template, &storage, &arrays) == 0);
    assert(storage != NULL);
    memcpy(arrays.field_slices, &field_slice, sizeof(field_slice));
    memcpy(arrays.particles, particles, sizeof(particles));
    memcpy(arrays.initial_direction, zeros, sizeof(zeros));
    memcpy(arrays.initial_gradient, zeros, sizeof(zeros));
    memcpy(arrays.point_active, active, sizeof(active));
    memcpy(arrays.voxels, voxels, sizeof(voxels));
    memcpy(arrays.voxel_scenarios, scenarios, sizeof(scenarios));
    memcpy(arrays.scenario_states, states, sizeof(states));
    memcpy(arrays.slices, slices, sizeof(slices));
    memcpy(arrays.coefficient_point_indices, indices, sizeof(indices));
    memcpy(arrays.coefficients, coefficients, sizeof(coefficients));
    double storage_gradient[2];
    FDCBEvaluationResultV1 storage_evaluation;
    assert(trip_fdcb_storage_evaluate_v1(
               storage, storage_gradient, 2, &storage_evaluation) == 0);
    assert(storage_evaluation.chi2 == evaluation.chi2);
    assert(memcmp(storage_gradient, gradient, sizeof(gradient)) == 0);
    double storage_output[2];
    FDCBResultV1 storage_result;
    assert(trip_fdcb_storage_optimize_v1(
               storage, storage_output, 1, &storage_result) == -2);
    assert(trip_fdcb_storage_optimize_v1(
               storage, storage_output, 2, &storage_result) == 0);
    assert(storage_result.iterations == result.iterations);
    assert(storage_result.chi2 == result.chi2);
    assert(memcmp(storage_output, output, sizeof(output)) == 0);

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
    double storage_accelerator_output[2];
    FDCBResultV1 storage_accelerator_result;
    int32_t storage_accelerator_status =
        trip_fdcb_storage_optimize_accelerator_v1(
            storage, storage_accelerator_output, 2,
            &storage_accelerator_result);
    assert(storage_accelerator_status == accelerator_status);
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
        assert(storage_accelerator_result.iterations ==
               accelerator_result.iterations);
        assert(storage_accelerator_result.chi2 == accelerator_result.chi2);
        assert(memcmp(storage_accelerator_output, accelerator_output,
                      sizeof(accelerator_output)) == 0);
    }
#endif
    assert(trip_fdcb_storage_destroy_v1(storage) == 0);
    return 0;
}
