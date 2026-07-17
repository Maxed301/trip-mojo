#ifndef TRIP_MOJO_CLINICAL_DOSE_ABI_V1_H
#define TRIP_MOJO_CLINICAL_DOSE_ABI_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CLINICAL_DOSE_VERSION_V1 = 1,
    CLINICAL_DOSE_BIOLOGICAL = 1u << 0,
    CLINICAL_DOSE_DIVERGENT = 1u << 1,
    CLINICAL_DOSE_ALGORITHM_MS = 1,
    CLINICAL_DOSE_ALGORITHM_MSDB = 2,
    CLINICAL_DOSE_BIOLOGY_NONE = 0,
    CLINICAL_DOSE_BIOLOGY_LOW_DOSE = 1
};

typedef struct {
    double x, y, delta_z, f2_max, particles;
} ClinicalDosePointV1;

typedef struct {
    double z_cm, dose, fwhm1, mix, fwhm2;
} ClinicalDoseDDDEntryV1;

typedef struct {
    uint32_t entry_offset, entry_count, column_count;
} ClinicalDoseDDDTableV1;

typedef struct {
    double z_cm, alpha, sqrt_beta, let_mix, let_bar, let_dm_sum;
} ClinicalDoseBioEntryV1;

typedef struct {
    uint32_t entry_offset, entry_count;
    double z_scale;
} ClinicalDoseBioTableV1;

typedef struct {
    uint32_t point_offset, point_count, ddd_table, bio_table_offset;
    double focus, range_shifter;
    double window_x0, window_x1, window_y0, window_y1;
} ClinicalDoseEnergyV1;

typedef struct {
    uint32_t energy_offset, energy_count;
    double dose_extension2, scanner_x, scanner_y;
} ClinicalDoseFieldV1;

typedef struct {
    int32_t nx, ny, nz;
    double x0, y0, z0;
    double dx, dy, dz;
    double boundary_x0, boundary_y0, boundary_z0;
} ClinicalDoseGridV1;

typedef struct {
    ClinicalDoseGridV1 grid;
    uint64_t data_offset, x_boundary_offset, y_boundary_offset, z_boundary_offset;
    double pmod, pore_size;
    int32_t byte_swap;
} ClinicalDoseCTStateV1;

typedef struct {
    uint32_t field_offset, field_count;
} ClinicalDoseStateV1;

typedef struct {
    double x, y, z;
} ClinicalDosePositionV1;

typedef struct {
    uint32_t field_index;
    double patient_to_gantry[12];
    double direction[3];
    double off_h2o, bolus;
    double window_x0, window_x1, window_y0, window_y1;
    int32_t gated;
} ClinicalDoseGridFieldV1;

typedef struct {
    double absorbed_dose, alpha, sqrt_beta, let_mix, let_bar, let_dm_sum;
} ClinicalDoseOutputV1;

typedef struct {
    uint32_t version, flags;
    uint32_t grid_voxel_count, state_count, field_count;
    uint32_t energy_count, point_count;
    uint32_t ddd_table_count, ddd_entry_count;
    uint32_t bio_table_count, bio_entry_count, voi_count;
    uint32_t hlut_count, algorithm, biology_model, max_threads, struct_size;
    uint64_t ct_value_count, ct_boundary_count, dose_axis_count;
    uint64_t dose_x_offset, dose_y_offset, dose_z_offset;
    ClinicalDoseGridV1 dose_grid;
    const int32_t *voxel_voi;
    const ClinicalDoseCTStateV1 *ct_states;
    const int16_t *ct_data;
    const double *ct_boundaries;
    const double *dose_axis_centers;
    const double *hlut_x, *hlut_y;
    const ClinicalDoseGridFieldV1 *grid_fields;
    const ClinicalDoseFieldV1 *fields;
    const ClinicalDoseEnergyV1 *energies;
    const ClinicalDosePointV1 *points;
    const ClinicalDoseDDDTableV1 *ddd_tables;
    const ClinicalDoseDDDEntryV1 *ddd_entries;
    const ClinicalDoseBioTableV1 *bio_tables;
    const ClinicalDoseBioEntryV1 *bio_entries;
    /* Empty for static dose. For 4D, state-major positions transformed from
       the reference dose grid and contiguous field ranges are required. */
    uint64_t transformed_voxel_count;
    const ClinicalDoseStateV1 *states;
    const ClinicalDosePositionV1 *transformed_voxels;
} ClinicalDoseProblemViewV1;

/* Returns 0 on success, -1 for invalid input, and -2 for output-size mismatch. */
int32_t trip_clinical_dose_compute_v1(
    const ClinicalDoseProblemViewV1 *problem,
    ClinicalDoseOutputV1 *output,
    uint64_t output_count
);

/* Same contract on the shared accelerator backend; -3 if not compiled in. */
int32_t trip_clinical_dose_compute_accelerator_v1(
    const ClinicalDoseProblemViewV1 *problem,
    ClinicalDoseOutputV1 *output,
    uint64_t output_count
);

#ifdef __cplusplus
}
#endif

#endif
