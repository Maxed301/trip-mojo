#include "clinical_dose_abi_v1.h"

#include <assert.h>
#include <math.h>
#include <string.h>

int main(void) {
    const ClinicalDoseGridV1 grid = {
        1, 1, 1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0
    };
    const int32_t voxel_voi[1] = {0};
    const int16_t ct_data[1] = {0};
    const float hlut_x[2] = {-32768.0f, 32767.0f};
    const float hlut_y[2] = {1.0f, 1.0f};
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
    const ClinicalDoseEnergyV1 energy = {
        0, 3, 0, 0, 1.0f, 0.0f, -10.0f, 10.0f, -10.0f, 10.0f
    };
    const ClinicalDosePointV1 points[3] = {
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
    const ClinicalDoseBioEntryV1 bio_entries[2] = {
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

    problem.flags = CLINICAL_DOSE_BIOLOGICAL;
    problem.biology_model = CLINICAL_DOSE_BIOLOGY_LOW_DOSE;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == 0);
    assert(fabs(output.absorbed_dose - fluence * 2.0) < 1.0e-12);
    assert(fabs(output.alpha - fluence * 3.0) < 1.0e-12);
    assert(fabs(output.sqrt_beta - fluence * 4.0) < 1.0e-12);
    assert(fabs(output.let_mix - fluence * 5.0) < 1.0e-12);
    assert(fabs(output.let_bar - fluence * 6.0) < 1.0e-12);
    assert(fabs(output.let_dm_sum - fluence * 7.0) < 1.0e-12);
    problem.state_count = 2;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == -1);
    problem.state_count = 1;
    problem.energy_count = 0;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == -1);
    problem.energy_count = 1;
    problem.algorithm = 99;
    assert(trip_clinical_dose_compute_v1(&problem, &output, 1) == -1);
    return 0;
}
