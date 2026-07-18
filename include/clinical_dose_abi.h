#ifndef TRIP_MOJO_CLINICAL_DOSE_ABI_H
#define TRIP_MOJO_CLINICAL_DOSE_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CLINICAL_DOSE_ALGORITHM_MS = 1,
    CLINICAL_DOSE_ALGORITHM_MSDB = 2,
    CLINICAL_DOSE_BIOLOGY_NONE = 0,
    CLINICAL_DOSE_BIOLOGY_LOW_DOSE = 1
};

typedef struct {
    double x, y, delta_z, f2_max, particles;
} ClinicalDosePoint;

typedef struct {
    double z_cm, dose, fwhm1, mix, fwhm2;
} ClinicalDoseDDDEntry;

typedef struct {
    uint32_t entry_offset, entry_count, column_count;
} ClinicalDoseDDDTable;

typedef struct {
    double z_cm, alpha, sqrt_beta, let_mix, let_bar, let_dm_sum;
} ClinicalDoseBioEntry;

typedef struct {
    uint32_t entry_offset, entry_count;
    double z_scale;
} ClinicalDoseBioTable;

typedef struct {
    uint32_t point_offset, point_count, ddd_table, bio_table_offset;
    double focus, range_shifter;
    double window_x0, window_x1, window_y0, window_y1;
} ClinicalDoseEnergy;

typedef struct {
    uint32_t energy_offset, energy_count;
    double dose_extension2, scanner_x, scanner_y;
} ClinicalDoseField;

typedef struct {
    int32_t nx, ny, nz;
    double x0, y0, z0;
    double dx, dy, dz;
    double boundary_x0, boundary_y0, boundary_z0;
} ClinicalDoseGrid;

typedef struct {
    ClinicalDoseGrid grid;
    uint64_t data_offset, x_boundary_offset, y_boundary_offset, z_boundary_offset;
    double pmod, pore_size;
    int32_t byte_swap;
} ClinicalDoseCTState;

typedef struct {
    uint32_t field_offset, field_count;
} ClinicalDoseState;

typedef struct {
    double x, y, z;
} ClinicalDosePosition;

typedef struct {
    uint32_t field_index;
    double patient_to_gantry[12];
    double direction[3];
    double off_h2o, bolus;
    double window_x0, window_x1, window_y0, window_y1;
    int32_t gated;
} ClinicalDoseGridField;

typedef struct {
    double absorbed_dose, alpha, sqrt_beta, let_mix, let_bar, let_dm_sum;
} ClinicalDoseOutput;

typedef struct {
    uint32_t grid_voxel_count, state_count, field_count;
    uint32_t energy_count, point_count;
    uint32_t ddd_table_count, ddd_entry_count;
    uint32_t bio_table_count, bio_entry_count, voi_count;
    uint32_t hlut_count, algorithm, biology_model, max_threads, struct_size;
    uint64_t ct_value_count, ct_boundary_count, dose_axis_count;
    uint64_t dose_x_offset, dose_y_offset, dose_z_offset;
    ClinicalDoseGrid dose_grid;
    const int32_t *voxel_voi;
    const ClinicalDoseCTState *ct_states;
    const int16_t *ct_data;
    const double *ct_boundaries;
    const double *dose_axis_centers;
    const double *hlut_x, *hlut_y;
    const ClinicalDoseGridField *grid_fields;
    const ClinicalDoseField *fields;
    const ClinicalDoseEnergy *energies;
    const ClinicalDosePoint *points;
    const ClinicalDoseDDDTable *ddd_tables;
    const ClinicalDoseDDDEntry *ddd_entries;
    const ClinicalDoseBioTable *bio_tables;
    const ClinicalDoseBioEntry *bio_entries;
    /* Empty for static dose. For 4D, state-major positions transformed from
       the reference dose grid and contiguous field ranges are required. */
    uint64_t transformed_voxel_count;
    const ClinicalDoseState *states;
    const ClinicalDosePosition *transformed_voxels;
} ClinicalDoseProblem;

/* Returns 0 on success, -1 for invalid input, and -2 for output-size mismatch. */
int32_t trip_compute_clinical_dose(
    const ClinicalDoseProblem *problem,
    ClinicalDoseOutput *output,
    uint64_t output_count
);

/* Same contract on the shared accelerator backend; -3 if not compiled in. */
int32_t trip_compute_clinical_dose_on_device(
    const ClinicalDoseProblem *problem,
    ClinicalDoseOutput *output,
    uint64_t output_count
);

#ifdef __cplusplus
}
#endif

#endif
