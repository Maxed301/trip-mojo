#ifndef TRIP_MOJO_FDCB_MATRIX_ABI_V1_H
#define TRIP_MOJO_FDCB_MATRIX_ABI_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FDCB_MATRIX_ABI_VERSION_V1 1u

enum {
    FDCB_MATRIX_FLAG_DEVICE_ONLY = 1u << 0,
    FDCB_MATRIX_DEVICE_ID_SHIFT = 8,
    FDCB_MATRIX_DEVICE_ID_MASK = 0xffu << FDCB_MATRIX_DEVICE_ID_SHIFT,
};

typedef struct {
    uint64_t point_offset;
    uint32_t point_count;
    uint32_t reserved;
} FDCBMatrixEnergySliceV1;

typedef struct {
    double x;
    double y;
    double f2_max;
} FDCBMatrixPointV1;

typedef struct {
    uint64_t slice_offset;
    uint32_t slice_count;
    uint32_t reserved;
    double bev_x;
    double bev_y;
    double relative_cutoff;
    double point_shift_x;
    double point_shift_y;
} FDCBMatrixGroupV1;

typedef struct {
    uint32_t energy_slice;
    uint32_t ddd_table;
    double depth_shift_mm;
    double focus_squared;
    double lateral_limit_scale;
    double fallback_scale;
} FDCBMatrixRawEnergyV1;

typedef struct {
    uint32_t energy_offset;
    uint32_t energy_count;
    double depth_mm;
    double divergence_x;
    double divergence_y;
} FDCBMatrixRawGroupV1;

typedef struct {
    uint32_t entry_offset;
    uint32_t entry_count;
} FDCBMatrixDDDTableV1;

typedef struct {
    double depth_cm;
    double dose;
    double fwhm1;
    double mixture;
    double fwhm2;
} FDCBMatrixDDDEntryV1;

typedef struct {
    uint32_t version;
    uint32_t group_count;
    uint32_t energy_slice_count;
    uint32_t maximum_group_slices;
    uint64_t slice_count;
    uint64_t point_count;
    uint32_t ddd_table_count;
    uint32_t flags;
    uint64_t ddd_entry_count;
    const FDCBMatrixEnergySliceV1 *energy_slices;
    const FDCBMatrixPointV1 *points;
    const FDCBMatrixGroupV1 *groups;
    const FDCBMatrixRawEnergyV1 *raw_energies;
    const FDCBMatrixRawGroupV1 *raw_groups;
    const FDCBMatrixDDDTableV1 *ddd_tables;
    const FDCBMatrixDDDEntryV1 *ddd_entries;
} FDCBMatrixProblemViewV1;

typedef struct {
    uint64_t entry_count;
    uint64_t slice_count;
    uint32_t group_count;
    uint32_t reserved;
    const double *group_maximum;
    const uint32_t *group_entry_counts;
    const uint32_t *slice_entry_counts;
    const double *slice_dose;
    const uint16_t *point_indices;
    const double *coefficients;
} FDCBMatrixResultV1;

typedef struct FDCBMatrixStorageV1 FDCBMatrixStorageV1;

int32_t trip_fdcb_matrix_build_accelerator_v1(
    const FDCBMatrixProblemViewV1 *problem,
    FDCBMatrixStorageV1 **storage,
    FDCBMatrixResultV1 *result);

int32_t trip_fdcb_matrix_storage_destroy_v1(FDCBMatrixStorageV1 *storage);

#ifdef __cplusplus
}
#endif

#endif
