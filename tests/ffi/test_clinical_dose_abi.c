#include "clinical_dose_abi_v1.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

_Static_assert(sizeof(ClinicalDosePointV1) == 40, "point ABI layout");
_Static_assert(offsetof(ClinicalDosePointV1, particles) == 32,
               "point particle offset");
_Static_assert(sizeof(ClinicalDoseDDDEntryV1) == 40, "DDD entry ABI layout");
_Static_assert(sizeof(ClinicalDoseDDDTableV1) == 12, "DDD table ABI layout");
_Static_assert(sizeof(ClinicalDoseBioEntryV1) == 48, "bio entry ABI layout");
_Static_assert(sizeof(ClinicalDoseBioTableV1) == 16, "bio table ABI layout");
_Static_assert(sizeof(ClinicalDoseEnergyV1) == 64, "energy ABI layout");
_Static_assert(sizeof(ClinicalDoseFieldV1) == 32, "field ABI layout");
_Static_assert(sizeof(ClinicalDoseGridV1) == 88, "grid ABI layout");
_Static_assert(sizeof(ClinicalDoseCTStateV1) == 120, "CT state ABI layout");
_Static_assert(sizeof(ClinicalDoseStateV1) == 8, "state ABI layout");
_Static_assert(sizeof(ClinicalDosePositionV1) == 24, "position ABI layout");
_Static_assert(sizeof(ClinicalDoseGridFieldV1) == 184,
               "grid field ABI layout");
_Static_assert(sizeof(ClinicalDoseOutputV1) == 48, "output ABI layout");
_Static_assert(sizeof(ClinicalDoseProblemViewV1) == 296,
               "problem view ABI layout");

int main(void) {
    const ClinicalDoseGridV1 grid = {
        1, 1, 1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    };
    const int32_t voxel_voi[1] = {0};
    const int16_t ct_data[1] = {0};
    const double hlut_x[2] = {-32768.0, 32767.0};
    const double hlut_y[2] = {1.0, 1.0};
    const ClinicalDoseCTStateV1 ct_state = {grid, 0, 0.0, 0.0, 0};
    ClinicalDoseGridFieldV1 grid_field;
    memset(&grid_field, 0, sizeof(grid_field));
    grid_field.patient_to_gantry[0] = 1.0f;
    grid_field.patient_to_gantry[5] = 1.0f;
    grid_field.patient_to_gantry[10] = 1.0f;
    grid_field.direction[2] = 1.0f;
    grid_field.window_x0 = grid_field.window_y0 = -10.0f;
    grid_field.window_x1 = grid_field.window_y1 = 10.0f;
    const ClinicalDoseFieldV1 field = {0, 1, 100.0f, 8832.0f, 7806.0f};
    ClinicalDoseEnergyV1 energy = {
        0, 3, 0, 0, 1.0f, 0.0f, -10.0f, 10.0f, -10.0f, 10.0f
    };
    ClinicalDosePointV1 points[3] = {
        {0.5f, 0.5f, 0.0f, 100.0f, 10.0},
        {0.5f, 0.5f, 0.0f, 100.0f, 5.0},
        {0.5f, 20.0f, 0.0f, 1.0f, 1.0e9}
    };
    const ClinicalDoseDDDTableV1 ddd_table = {0, 2, 5};
    const ClinicalDoseDDDEntryV1 ddd_entries[2] = {
        {0.0f, 2.0f, 1.0f, 0.0f, 1.0f},
        {1.0f, 2.0f, 1.0f, 0.0f, 1.0f}
    };
    const ClinicalDoseBioTableV1 bio_table = {0, 2, 1.0f};
    ClinicalDoseBioEntryV1 bio_entries[2] = {
        {0.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
        {1.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}
    };
    ClinicalDoseProblemViewV1 problem;
    memset(&problem, 0, sizeof(problem));
    problem.version = CLINICAL_DOSE_VERSION_V1;
    problem.grid_voxel_count = problem.state_count = problem.field_count = 1;
    problem.energy_count = problem.ddd_table_count = 1;
    problem.point_count = 3;
    problem.ddd_entry_count = 2;
    problem.bio_table_count = problem.voi_count = 1;
    problem.bio_entry_count = problem.hlut_count = 2;
    problem.algorithm = CLINICAL_DOSE_ALGORITHM_MS;
    problem.biology_model = CLINICAL_DOSE_BIOLOGY_NONE;
    problem.max_threads = 1;
    problem.ct_value_count = 1;
    problem.dose_grid = grid;
    problem.voxel_voi = voxel_voi;
    problem.ct_states = &ct_state;
    problem.ct_data = ct_data;
    problem.hlut_x = hlut_x;
    problem.hlut_y = hlut_y;
    problem.grid_fields = &grid_field;
    problem.fields = &field;
    problem.energies = &energy;
    problem.points = points;
    problem.ddd_tables = &ddd_table;
    problem.ddd_entries = ddd_entries;
    problem.bio_tables = &bio_table;
    problem.bio_entries = bio_entries;

    ClinicalDoseOutputV1 output;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 0) == -2);
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == 0);
    const double lateral = ((0.6932 * 8.0) / 2.0 * 0.5) / acos(-1.0);
    const double fluence = lateral * 15.0;
    assert(fabs(output.absorbed_dose - fluence * 2.0) < 1.0e-12);
    assert(output.alpha == 0.0);
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    ClinicalDoseOutputV1 accelerator_output;
    problem.voxel_voi = NULL;
    assert(trip_clinical_dose_compute_accelerator_v1(
        &problem, &accelerator_output, 1) == 0);
    assert(fabs(accelerator_output.absorbed_dose - output.absorbed_dose) < 1.0e-12);
    assert(accelerator_output.alpha == 0.0);
    problem.voxel_voi = voxel_voi;
#endif

    problem.flags = CLINICAL_DOSE_BIOLOGICAL;
    problem.biology_model = CLINICAL_DOSE_BIOLOGY_LOW_DOSE;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == 0);
    assert(fabs(output.absorbed_dose - fluence * 2.0) < 1.0e-12);
    assert(fabs(output.alpha - fluence * 3.0) < 1.0e-12);
    assert(fabs(output.sqrt_beta - fluence * 4.0) < 1.0e-12);
    assert(fabs(output.let_mix - fluence * 5.0) < 1.0e-12);
    assert(fabs(output.let_bar - fluence * 6.0) < 1.0e-12);
    assert(fabs(output.let_dm_sum - fluence * 7.0) < 1.0e-12);
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    assert(trip_clinical_dose_compute_accelerator_v1(
        &problem, &accelerator_output, 1) == 0);
    assert(fabs(accelerator_output.absorbed_dose - output.absorbed_dose) < 1.0e-12);
    assert(fabs(accelerator_output.alpha - output.alpha) < 1.0e-12);
    assert(fabs(accelerator_output.sqrt_beta - output.sqrt_beta) < 1.0e-12);
    assert(fabs(accelerator_output.let_mix - output.let_mix) < 1.0e-12);
    assert(fabs(accelerator_output.let_bar - output.let_bar) < 1.0e-12);
    assert(fabs(accelerator_output.let_dm_sum - output.let_dm_sum) < 1.0e-12);
#else
    assert(trip_clinical_dose_compute_accelerator_v1(&problem, &output, 1) == -3);
#endif
    energy.range_shifter = 0.25f;
    points[0].delta_z = points[1].delta_z = 0.25f;
    bio_entries[1].alpha = 9.0f;
    bio_entries[1].sqrt_beta = 10.0f;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == 0);
    assert(fabs(output.alpha - fluence * 9.0) < 1.0e-12);
    assert(fabs(output.sqrt_beta - fluence * 10.0) < 1.0e-12);
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    assert(trip_clinical_dose_compute_accelerator_v1(
        &problem, &accelerator_output, 1) == 0);
    assert(fabs(accelerator_output.alpha - output.alpha) < 1.0e-12);
    assert(fabs(accelerator_output.sqrt_beta - output.sqrt_beta) < 1.0e-12);
#endif
    energy.range_shifter = 0.0f;
    points[0].delta_z = points[1].delta_z = 0.0f;
    bio_entries[1].alpha = 3.0f;
    bio_entries[1].sqrt_beta = 4.0f;
    const ClinicalDoseCTStateV1 ct_states[2] = {ct_state, ct_state};
    const ClinicalDoseStateV1 states[2] = {{0, 1}, {1, 1}};
    ClinicalDoseGridFieldV1 state_grid_fields[2] = {grid_field, grid_field};
    state_grid_fields[1].field_index = 1;
    ClinicalDoseFieldV1 state_fields[2] = {field, field};
    ClinicalDosePositionV1 transformed[2] = {
        {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}
    };
    problem.state_count = 2;
    problem.ct_states = ct_states;
    problem.states = states;
    problem.field_count = 2;
    problem.grid_fields = state_grid_fields;
    problem.fields = state_fields;
    problem.transformed_voxels = transformed;
    problem.transformed_voxel_count = 2;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == 0);
    assert(fabs(output.absorbed_dose - 2.0 * fluence * 2.0) < 1.0e-12);
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    assert(trip_clinical_dose_compute_accelerator_v1(
        &problem, &accelerator_output, 1) == 0);
    assert(fabs(accelerator_output.absorbed_dose - output.absorbed_dose) < 1.0e-12);
#endif
    transformed[1].x = 20.0;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == 0);
    assert(fabs(output.absorbed_dose - fluence * 2.0) < 1.0e-12);
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    assert(trip_clinical_dose_compute_accelerator_v1(
        &problem, &accelerator_output, 1) == 0);
    assert(fabs(accelerator_output.absorbed_dose - output.absorbed_dose) < 1.0e-12);
#endif
    transformed[1].x = 0.5;
    state_fields[1].energy_offset = 1;
    state_fields[1].energy_count = 0;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == 0);
    assert(fabs(output.absorbed_dose - fluence * 2.0) < 1.0e-12);
#ifdef FDCB_TEST_REQUIRE_ACCELERATOR
    assert(trip_clinical_dose_compute_accelerator_v1(
        &problem, &accelerator_output, 1) == 0);
    assert(fabs(accelerator_output.absorbed_dose - output.absorbed_dose) < 1.0e-12);
#endif
    problem.transformed_voxel_count = 1;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == -1);
    problem.state_count = 1;
    problem.ct_states = &ct_state;
    problem.states = NULL;
    problem.transformed_voxels = NULL;
    problem.transformed_voxel_count = 0;
    problem.field_count = 1;
    problem.grid_fields = &grid_field;
    problem.fields = &field;
    problem.energy_count = 0;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == -1);
    problem.energy_count = 1;
    problem.algorithm = 99;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == -1);
    return 0;
}
