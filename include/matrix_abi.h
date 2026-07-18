#ifndef TRIP_MOJO_MATRIX_ABI_H
#define TRIP_MOJO_MATRIX_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MOJO_MATRIX_FLAG_DEVICE_ONLY = 1u << 0,
    MOJO_MATRIX_FLAG_FORCE_PROCEDURAL = 1u << 1,
    MOJO_MATRIX_DEVICE_ID_SHIFT = 8,
    MOJO_MATRIX_DEVICE_ID_MASK = 0xffu << MOJO_MATRIX_DEVICE_ID_SHIFT,
};

enum {
    MOJO_MATRIX_RESULT_PROCEDURAL = 1u << 0,
};

typedef struct {
    uint64_t point_offset;
    uint32_t point_count;
    uint32_t reserved;
} MojoMatrixEnergySlice;

typedef struct {
    double x;
    double y;
    double f2_max;
} MojoMatrixPoint;

typedef struct {
    uint64_t slice_offset;
    uint32_t slice_count;
    uint32_t reserved;
    double bev_x;
    double bev_y;
    double relative_cutoff;
    double point_shift_x;
    double point_shift_y;
} MojoMatrixGroup;

typedef struct {
    uint32_t energy_slice;
    uint32_t ddd_table;
    double depth_shift_mm;
    double focus_squared;
    double lateral_limit_scale;
    double fallback_scale;
} MojoRawMatrixEnergy;

typedef struct {
    uint32_t energy_offset;
    uint32_t energy_count;
    double depth_mm;
    double divergence_x;
    double divergence_y;
} MojoRawMatrixGroup;

typedef struct {
    uint32_t entry_offset;
    uint32_t entry_count;
} MojoMatrixDepthDoseTable;

typedef struct {
    double depth_cm;
    double dose;
    double fwhm1;
    double mixture;
    double fwhm2;
} MojoMatrixDepthDoseEntry;

typedef struct {
    uint32_t group_count;
    uint32_t energy_slice_count;
    uint32_t maximum_group_slices;
    uint64_t slice_count;
    uint64_t point_count;
    uint32_t ddd_table_count;
    uint32_t flags;
    uint64_t ddd_entry_count;
    const MojoMatrixEnergySlice *energy_slices;
    const MojoMatrixPoint *points;
    const MojoMatrixGroup *groups;
    const MojoRawMatrixEnergy *raw_energies;
    const MojoRawMatrixGroup *raw_groups;
    const MojoMatrixDepthDoseTable *ddd_tables;
    const MojoMatrixDepthDoseEntry *ddd_entries;
} MojoMatrixBuildInput;

typedef struct {
    uint64_t entry_count;
    uint64_t slice_count;
    uint32_t group_count;
    uint32_t flags;
    const double *group_maximum;
    const uint32_t *group_entry_counts;
    const uint32_t *slice_entry_counts;
    const double *slice_dose;
    const uint16_t *point_indices;
    const double *coefficients;
} MojoMatrixBuildResult;

typedef struct MojoDeviceMatrix MojoDeviceMatrix;

int32_t trip_optimizer_build_device_matrix(
    const MojoMatrixBuildInput *problem,
    MojoDeviceMatrix **storage,
    MojoMatrixBuildResult *result);

int32_t trip_optimizer_destroy_device_matrix(MojoDeviceMatrix *storage);

int32_t trip_optimizer_release_matrix_build_buffers(
    MojoDeviceMatrix *storage);

#ifdef __cplusplus
}
#endif

#endif
