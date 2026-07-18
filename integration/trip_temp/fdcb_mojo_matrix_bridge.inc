#include "fdcb_matrix_abi_v1.h"

struct FDCBMojoMatrixGroupHost {
   struct STRPFLDDOSE *dose;
   uint32_t beam_count;
   uint32_t field_index;
};

struct FDCBMojoMatrixArena {
   double *coefficients;
   uint16_t *point_indices;
   uint64_t entry_count;
   uint64_t references;
   uint32_t device_id;
   int procedural;
   uint64_t voxel_offset;
   uint64_t voxel_count;
   uint32_t group_count;
   uint64_t *group_offsets;
   uint32_t *group_slice_offsets;
   uint32_t *group_beam_counts;
   FDCBMatrixStorageV1 *storage;
   struct FDCBMojoMatrixArena *next;
};

static struct FDCBMojoMatrixArena *gFDCBMojoMatrixArenas = NULL;

static void FDCBMojoMatrixFreeArena(struct FDCBMojoMatrixArena *arena) {
   if ( arena == NULL ) return;
   if ( arena->storage != NULL )
      trip_fdcb_matrix_storage_destroy_v1(arena->storage);
   free(arena->group_offsets);
   free(arena->group_slice_offsets);
   free(arena->group_beam_counts);
   free(arena);
}

static int FDCBMojoMatrixDirectView(
   FDCBMatrixStorageV1 **storage,
   double **coefficients,
   uint16_t **point_indices,
   uint64_t *entry_count
) {
   struct FDCBMojoMatrixArena *arena = gFDCBMojoMatrixArenas;
   if ( arena == NULL || arena->next != NULL || arena->storage == NULL )
      return -1;
   *storage = arena->storage;
   *coefficients = arena->coefficients;
   *point_indices = arena->point_indices;
   *entry_count = arena->entry_count;
   return 0;
}

static int FDCBMojoMatrixDirectViews(
   uint32_t device_count,
   FDCBMatrixStorageV1 **storages,
   double **coefficients,
   uint16_t **point_indices,
   uint64_t *entry_counts,
   uint64_t *voxel_offsets,
   uint64_t *voxel_counts
) {
   uint32_t found = 0;
   memset(storages, 0, device_count * sizeof(*storages));
   for ( struct FDCBMojoMatrixArena *arena = gFDCBMojoMatrixArenas;
         arena != NULL; arena = arena->next ) {
      uint32_t device = arena->device_id;
      if ( device >= device_count || storages[device] != NULL ||
           arena->storage == NULL || arena->voxel_count == 0 )
         return -1;
      storages[device] = arena->storage;
      coefficients[device] = arena->coefficients;
      point_indices[device] = arena->point_indices;
      entry_counts[device] = arena->entry_count;
      voxel_offsets[device] = arena->voxel_offset;
      voxel_counts[device] = arena->voxel_count;
      ++found;
   }
   return found == device_count ? 0 : -1;
}

static void FDCBMojoMatrixTransferDirectStorage(
   FDCBMatrixStorageV1 *storage
) {
   for ( struct FDCBMojoMatrixArena *arena = gFDCBMojoMatrixArenas;
         arena != NULL; arena = arena->next )
      if ( arena->storage == storage ) {
         arena->storage = NULL;
         return;
      }
}

static void FDCBMojoMatrixReleaseArenas(void) {
   while ( gFDCBMojoMatrixArenas != NULL ) {
      struct FDCBMojoMatrixArena *arena = gFDCBMojoMatrixArenas;
      gFDCBMojoMatrixArenas = arena->next;
      FDCBMojoMatrixFreeArena(arena);
   }
}

static struct FDCBMojoMatrixArena *FDCBMojoMatrixDoseArena(
   const struct STRPFLDDOSE *dose
) {
   uintptr_t address = (uintptr_t) dose->sFDCPstore.piPnt;
   for ( struct FDCBMojoMatrixArena *arena = gFDCBMojoMatrixArenas;
         arena != NULL; arena = arena->next ) {
      uintptr_t begin = (uintptr_t) arena->point_indices;
      uintptr_t end = begin + arena->entry_count * sizeof(*arena->point_indices);
      if ( address >= begin && address < end ) return arena;
   }
   return NULL;
}

static int FDCBMojoMatrixDirectSlice(
   const struct STRPFLDDOSE *dose,
   int voxel,
   const struct STRPFLDDOSECORRSLICE *slice,
   uint64_t *offset,
   uint32_t *matrix_slice_index
) {
   static __thread struct FDCBMojoMatrixArena *cached_arena = NULL;
   static __thread const struct STRPFLDDOSECORRSLICE *cached_slices = NULL;
   static __thread uint64_t cached_group_offset = UINT64_MAX;
   static __thread uint32_t cached_group = 0;
   struct FDCBMojoMatrixArena *arena = FDCBMojoMatrixDoseArena(dose);
   uint64_t direct_offset;
   (void) voxel;
   if ( arena == NULL ) return -1;
   direct_offset = (uint64_t) (slice->sFDCP.piPnt - arena->point_indices);
   if ( slice->sFDCP.nPnt > arena->entry_count ||
        direct_offset > arena->entry_count - slice->sFDCP.nPnt ||
        (!arena->procedural &&
         slice->sFDCP.pdG != arena->coefficients + direct_offset) )
      return -1;
   *matrix_slice_index = 0;
   if ( arena->procedural ) {
      uint64_t group_offset = (uint64_t) (
         dose->sFDCPstore.piPnt - arena->point_indices);
      uint32_t group = cached_group;
      if ( cached_arena != arena || cached_slices != dose->psFDCS ||
           cached_group_offset != group_offset ) {
         uint32_t lo = 0;
         uint32_t hi = arena->group_count + 1;
         while ( lo < hi ) {
            uint32_t middle = lo + (hi - lo) / 2;
            if ( arena->group_offsets[middle] <= group_offset )
               lo = middle + 1;
            else
               hi = middle;
         }
         if ( lo == 0 ) return -1;
         group = lo - 1;
         if ( group >= arena->group_count ||
              arena->group_offsets[group] != group_offset ||
              arena->group_offsets[group + 1] - group_offset !=
                 dose->sFDCPstore.nPnt )
            return -1;
         cached_arena = arena;
         cached_slices = dose->psFDCS;
         cached_group_offset = group_offset;
         cached_group = group;
      }
      {
         ptrdiff_t beam = slice - dose->psFDCS;
         uint32_t beam_count = arena->group_beam_counts[group];
         uint64_t source_slice;
         if ( beam < 0 || (uint64_t) beam >= beam_count ) return -1;
         source_slice = arena->group_slice_offsets[group] + beam_count - 1u -
                        (uint64_t) beam;
         if ( source_slice > UINT32_MAX ) return -1;
         *matrix_slice_index = (uint32_t) source_slice;
      }
   }
   *offset = direct_offset;
   return 0;
}

static int FDCBMojoMatrixDoseDevice(const struct STRPFLDDOSE *dose) {
   struct FDCBMojoMatrixArena *arena = FDCBMojoMatrixDoseArena(dose);
   return arena == NULL ? -1 : (int) arena->device_id;
}

static void FDCBMojoMatrixResetArenaReferences(void) {
   for ( struct FDCBMojoMatrixArena *arena = gFDCBMojoMatrixArenas;
         arena != NULL; arena = arena->next )
      arena->references = 0;
}

static void FDCBMojoMatrixRetainDose(const struct STRPFLDDOSE *dose) {
   struct FDCBMojoMatrixArena *arena = FDCBMojoMatrixDoseArena(dose);
   if ( arena != NULL ) ++arena->references;
}

static void FDCBMojoMatrixReleaseDose(const struct STRPFLDDOSE *dose) {
   struct FDCBMojoMatrixArena *arena = FDCBMojoMatrixDoseArena(dose);
   if ( arena == NULL || arena->references == 0 ) return;
   __atomic_fetch_sub(&arena->references, 1, __ATOMIC_RELAXED);
}

static void FDCBMojoMatrixDiscardUnreferencedArenas(void) {
   struct FDCBMojoMatrixArena **link = &gFDCBMojoMatrixArenas;
   while ( *link != NULL ) {
      struct FDCBMojoMatrixArena *arena = *link;
      if ( arena->references != 0 ) {
         link = &arena->next;
         continue;
      }
      *link = arena->next;
      FDCBMojoMatrixFreeArena(arena);
   }
}

struct FDCBMojoMatrixBuild {
   FDCBMatrixEnergySliceV1 *energies;
   FDCBMatrixPointV1 *points;
   FDCBMatrixRawEnergyV1 *raw_energies;
   FDCBMatrixDDDTableV1 *ddd_tables;
   FDCBMatrixDDDEntryV1 *ddd_entries;
   uint8_t *fallback_initialized;
   uint32_t *field_energy_offsets;
   uint32_t *field_beam_counts;
   uint8_t *field_nonempty;
   FDCBMatrixGroupV1 *groups;
   FDCBMatrixRawGroupV1 *raw_groups;
   struct FDCBMojoMatrixGroupHost *group_hosts;
   uint32_t energy_count;
   uint64_t point_count;
   uint64_t ddd_entry_count;
   uint32_t group_count;
   uint64_t slice_count;
   uint32_t group_capacity;
   uint32_t maximum_group_slices;
   uint32_t flush_count;
   uint32_t device_id;
   uint64_t voxel_offset;
   uint64_t voxel_count;
   double accelerator_seconds;
   double import_seconds;
};

static void FDCBMojoMatrixBuildFree(struct FDCBMojoMatrixBuild *build) {
   if ( build == NULL ) return;
   free(build->energies);
   free(build->points);
   free(build->raw_energies);
   free(build->ddd_tables);
   free(build->ddd_entries);
   free(build->fallback_initialized);
   free(build->field_energy_offsets);
   free(build->field_beam_counts);
   free(build->field_nonempty);
   free(build->groups);
   free(build->raw_groups);
   free(build->group_hosts);
   memset(build, 0, sizeof(*build));
}

static int FDCBMojoMatrixStaticInputInit(
   struct FDCBMojoMatrixBuild *build,
   struct STRPOPT *opt,
   struct STRPFLDBASE *fields
) {
   struct Node *node;
   uint32_t field = 0;
   uint64_t point_count = 0;
   uint64_t ddd_entry_count = 0;
   uint32_t energy_count = 0;

   for ( node = fields->sLH.lh_Head; node->ln_Succ;
         node = node->ln_Succ, ++field ) {
      struct STRPFLD *trip_field = (struct STRPFLD *) node;
      if ( trip_field->sRst.nStates != 0 ) {
         fprintf(stderr, "<E> Mojo matrix builder does not yet support per-state RSTs\n");
         return CMDERR_ERROR | ERRTRPOPT_BADOPTALGORITHM;
      }
      if ( trip_field->psRamodFld &&
           (trip_field->psRamodFld->ulFlag & TRPRAMODFLD_STATICPOST) ) {
         fprintf(stderr, "<E> Mojo matrix builder does not support static post range modulators\n");
         return CMDERR_ERROR | ERRTRPOPT_BADOPTALGORITHM;
      }
      energy_count += (uint32_t) trip_field->sRst.nB;
      for ( int beam = 0; beam < trip_field->sRst.nB; ++beam ) {
         point_count += (uint64_t) trip_field->sRst.psRS[beam].nPnt;
         ddd_entry_count += (uint64_t) trip_field->sRst.psRS[beam].sDDD.nz;
      }
   }
   if ( field != (uint32_t) opt->nF || energy_count == 0 ||
        point_count > SIZE_MAX / sizeof(*build->points) ||
        ddd_entry_count > UINT32_MAX ) {
      return CMDERR_ERROR | ERRTRPOPT_SYS;
   }

   build->energies = calloc(energy_count, sizeof(*build->energies));
   build->points = calloc((size_t) point_count, sizeof(*build->points));
   build->raw_energies = calloc(energy_count, sizeof(*build->raw_energies));
   build->ddd_tables = calloc(energy_count, sizeof(*build->ddd_tables));
   build->ddd_entries = calloc((size_t) ddd_entry_count,
                               sizeof(*build->ddd_entries));
   build->fallback_initialized = calloc(energy_count,
                                        sizeof(*build->fallback_initialized));
   build->field_energy_offsets = calloc((size_t) opt->nF,
                                        sizeof(*build->field_energy_offsets));
   build->field_beam_counts = calloc((size_t) opt->nF,
                                     sizeof(*build->field_beam_counts));
   build->field_nonempty = calloc((size_t) opt->nF,
                                  sizeof(*build->field_nonempty));
   if ( build->energies == NULL || build->points == NULL ||
        build->raw_energies == NULL || build->ddd_tables == NULL ||
        build->ddd_entries == NULL || build->fallback_initialized == NULL ||
        build->field_energy_offsets == NULL ||
        build->field_beam_counts == NULL || build->field_nonempty == NULL ) {
      return CMDERR_ERROR | ERRTRPOPT_SYS;
   }

   energy_count = 0;
   point_count = 0;
   ddd_entry_count = 0;
   field = 0;
   for ( node = fields->sLH.lh_Head; node->ln_Succ;
         node = node->ln_Succ, ++field ) {
      struct STRPFLD *trip_field = (struct STRPFLD *) node;
      struct STRPRST *rst = &trip_field->sRst;
      build->field_energy_offsets[field] = energy_count;
      build->field_beam_counts[field] = (uint32_t) rst->nB;
      if ( (uint32_t) rst->nB > build->maximum_group_slices )
         build->maximum_group_slices = (uint32_t) rst->nB;
      for ( int beam = 0; beam < rst->nB; ++beam, ++energy_count ) {
         struct STRPRSTSLICE *slice = rst->psRS + beam;
         struct STRPDDD *ddd = &slice->sDDD;
         FDCBMatrixEnergySliceV1 *energy = build->energies + energy_count;
         FDCBMatrixRawEnergyV1 *raw = build->raw_energies + energy_count;
         FDCBMatrixDDDTableV1 *table = build->ddd_tables + energy_count;
         energy->point_offset = point_count;
         energy->point_count = (uint32_t) slice->nPnt;
         for ( int point = 0; point < slice->nPnt; ++point, ++point_count ) {
            FDCBMatrixPointV1 *out = build->points + point_count;
            out->x = slice->psPnt[point].dv[XX];
            out->y = slice->psPnt[point].dv[YY];
            out->f2_max = (slice->psPnt[point].ulFlag & TRPRSTPNT_INSIDE) ?
                          slice->psPnt[point].dF2Max : -1.0;
         }
         raw->energy_slice = energy_count;
         raw->ddd_table = energy_count;
         raw->depth_shift_mm = slice->sRSh.dThick;
         raw->focus_squared = slice->dF * slice->dF;
         raw->lateral_limit_scale = rst->dDoseExtension * rst->dDoseExtension;
         raw->fallback_scale = slice->dDoseExtension * slice->dDoseExtension;
         table->entry_offset = (uint32_t) ddd_entry_count;
         table->entry_count = (uint32_t) ddd->nz;
         for ( int depth = 0; depth < ddd->nz; ++depth, ++ddd_entry_count ) {
            FDCBMatrixDDDEntryV1 *out = build->ddd_entries + ddd_entry_count;
            out->depth_cm = ddd->pdTab[TRPDDD_X][depth];
            out->dose = ddd->pdTab[TRPDDD_D][depth];
            if ( ddd->nCol > 2 ) {
               out->fwhm1 = ddd->pdTab[TRPDDD_FWHM1][depth];
               out->mixture = ddd->pdTab[TRPDDD_G][depth];
               out->fwhm2 = ddd->pdTab[TRPDDD_FWHM2][depth];
            }
         }
      }
   }
   build->energy_count = energy_count;
   build->point_count = point_count;
   build->ddd_entry_count = ddd_entry_count;
   return 0;
}

static void FDCBMojoMatrixInitializeFallbacks(
   struct FDCBMojoMatrixBuild *build,
   uint32_t field,
   struct STRPRST *rst,
   struct STRPFLDDOSE *dose
) {
   uint32_t energy_offset = build->field_energy_offsets[field];
   for ( int beam = rst->nB - 1; beam >= 0; --beam ) {
      uint32_t energy_index = energy_offset + (uint32_t) beam;
      struct STRPRSTSLICE *slice;
      double values[TRPDDD_MAXTAB] = {0};
      double maximum_fwhm;
      double fallback;
      FDCBMatrixEnergySliceV1 *energy;
      if ( build->fallback_initialized[energy_index] ) continue;
      slice = rst->psRS + beam;
      if ( TRPDDDTabInterpolate(&slice->sDDD,
                                TRPMM2CM(dose->dzH2O +
                                         slice->sRSh.dThick),
                                values, TRPDDD_QUIET) != 0 ||
           values[TRPDDD_D] == 0.0 )
         continue;
      maximum_fwhm = values[TRPDDD_FWHM1];
      if ( values[TRPDDD_FWHM2] > maximum_fwhm )
         maximum_fwhm = values[TRPDDD_FWHM2];
      fallback = (slice->dF * slice->dF +
                  maximum_fwhm * maximum_fwhm) *
                 slice->dDoseExtension * slice->dDoseExtension;
      energy = build->energies + energy_index;
      for ( uint32_t point = 0; point < energy->point_count; ++point ) {
         struct STRPRSTPNT *source = slice->psPnt + point;
         if ( (source->ulFlag & TRPRSTPNT_INSIDE) &&
              source->dF2Max == 0.0 )
            source->dF2Max = fallback;
         build->points[energy->point_offset + point].f2_max =
            (source->ulFlag & TRPRSTPNT_INSIDE) ? source->dF2Max : -1.0;
      }
      build->fallback_initialized[energy_index] = 1;
   }
}

static int FDCBMojoMatrixImport(
   struct FDCBMojoMatrixBuild *build,
   FDCBMatrixStorageV1 *storage,
   const FDCBMatrixResultV1 *result
) {
   struct FDCBMojoMatrixArena *arena;
   uint64_t *group_offsets = NULL;
   int import_error = 0;
   int pack_threads = FDCBMojoPackThreads();
   if ( storage == NULL ) return CMDERR_ERROR | ERRTRPOPT_BADADDR;
   if ( result->group_count != build->group_count ||
        result->slice_count != build->slice_count )
      return CMDERR_ERROR | ERRTRPOPT_BADADDR;
   arena = calloc(1, sizeof(*arena));
   if ( arena == NULL ) return CMDERR_ERROR | ERRTRPOPT_SYS;
   group_offsets = calloc((size_t) build->group_count + 1,
                          sizeof(*group_offsets));
   if ( group_offsets == NULL ) {
      free(arena);
      return CMDERR_ERROR | ERRTRPOPT_SYS;
   }
   arena->coefficients = (double *) result->coefficients;
   arena->point_indices = (uint16_t *) result->point_indices;
   arena->storage = storage;
   arena->procedural =
      (result->flags & FDCB_MATRIX_RESULT_PROCEDURAL) != 0;
   arena->device_id = build->device_id;
   arena->voxel_offset = build->voxel_offset;
   arena->voxel_count = build->voxel_count;
   arena->group_count = build->group_count;
   if ( arena->procedural ) {
      arena->group_slice_offsets = calloc(
         build->group_count, sizeof(*arena->group_slice_offsets));
      arena->group_beam_counts = calloc(
         build->group_count, sizeof(*arena->group_beam_counts));
      if ( arena->group_slice_offsets == NULL ||
           arena->group_beam_counts == NULL ) {
         free(group_offsets);
         free(arena->group_slice_offsets);
         free(arena->group_beam_counts);
         free(arena);
         return CMDERR_ERROR | ERRTRPOPT_SYS;
      }
   }
   for ( uint32_t group = 0; group < build->group_count; ++group ) {
      struct STRPFLDDOSE *dose = build->group_hosts[group].dose;
      uint32_t beam_count = build->group_hosts[group].beam_count;
      if ( dose == NULL || dose->psFDCS == NULL ||
           dose->nFDC < (int) beam_count ) {
         free(group_offsets);
         free(arena->group_slice_offsets);
         free(arena->group_beam_counts);
         free(arena);
         return CMDERR_ERROR | ERRTRPOPT_BADADDR;
      }
      if ( arena->procedural ) {
         if ( build->groups[group].slice_offset > UINT32_MAX ) {
            free(group_offsets);
            free(arena->group_slice_offsets);
            free(arena->group_beam_counts);
            free(arena);
            return CMDERR_ERROR | ERRTRPOPT_BADADDR;
         }
         arena->group_slice_offsets[group] =
            (uint32_t) build->groups[group].slice_offset;
         arena->group_beam_counts[group] = beam_count;
      }
      group_offsets[group + 1] = group_offsets[group] +
                                 result->group_entry_counts[group];
   }
   if ( group_offsets[build->group_count] != result->entry_count ) {
      free(group_offsets);
      free(arena->group_slice_offsets);
      free(arena->group_beam_counts);
      free(arena);
      return CMDERR_ERROR | ERRTRPOPT_BADADDR;
   }

#pragma omp parallel for schedule(static) num_threads(pack_threads)
   for ( int64_t group_index = 0;
         group_index < (int64_t) build->group_count; ++group_index ) {
      struct STRPFLDDOSE *dose = build->group_hosts[group_index].dose;
      uint32_t beam_count = build->group_hosts[group_index].beam_count;
      uint32_t group_entries = result->group_entry_counts[group_index];
      uint32_t field_index = build->group_hosts[group_index].field_index;
      uint64_t entry_offset = group_offsets[group_index];
      uint32_t local_offset = 0;
      if ( dose->sFDCPstore.mPnt ) TRPFDCPfree(&dose->sFDCPstore);
      dose->sFDCPstore.pdG = arena->procedural ?
         arena->coefficients : arena->coefficients + entry_offset;
      dose->sFDCPstore.piPnt = arena->point_indices + entry_offset;
      dose->sFDCPstore.nPnt = group_entries;
      dose->sFDCPstore.mPnt = 0;
      dose->dMaxFDC = result->group_maximum[group_index];
      for ( uint32_t local_slice = 0; local_slice < beam_count;
            ++local_slice ) {
         uint64_t slice_index = build->groups[group_index].slice_offset +
                                (uint64_t) local_slice;
         uint32_t beam = beam_count - 1u - local_slice;
         uint32_t count = result->slice_entry_counts[slice_index];
         struct STRPFLDDOSECORRSLICE *destination = dose->psFDCS + beam;
         destination->dDDD = result->slice_dose[slice_index];
         if ( destination->sFDCP.mPnt ) TRPFDCPfree(&destination->sFDCP);
         destination->sFDCP.nPnt = count;
         destination->sFDCP.mPnt = 0;
         destination->sFDCP.piPnt = arena->point_indices + entry_offset;
         destination->sFDCP.pdG = arena->procedural ?
            arena->coefficients : arena->coefficients + entry_offset;
         local_offset += count;
         entry_offset += count;
      }
      dose->sFDCPstore.nPnt = local_offset;
      if ( local_offset != group_entries ) {
#pragma omp atomic write
         import_error = 1;
      }
      if ( group_entries != 0 )
         __atomic_store_n(build->field_nonempty + field_index, 1,
                          __ATOMIC_RELAXED);
   }
   if ( import_error ) {
      free(group_offsets);
      free(arena->group_slice_offsets);
      free(arena->group_beam_counts);
      free(arena);
      return CMDERR_ERROR | ERRTRPOPT_BADADDR;
   }
   if ( arena->procedural )
      arena->group_offsets = group_offsets;
   else
      free(group_offsets);
   arena->next = gFDCBMojoMatrixArenas;
   arena->entry_count = result->entry_count;
   gFDCBMojoMatrixArenas = arena;
   return 0;
}

static int FDCBMojoMatrixFlush(struct FDCBMojoMatrixBuild *build) {
   FDCBMatrixProblemViewV1 problem;
   FDCBMatrixResultV1 result;
   FDCBMatrixStorageV1 *storage = NULL;
   double phase;
   int rc;
   if ( build->group_count == 0 ) return 0;
   memset(&problem, 0, sizeof(problem));
   memset(&result, 0, sizeof(result));
   problem.version = FDCB_MATRIX_ABI_VERSION_V1;
   if ( FDCBMojoUsesDeviceMatrix() )
      problem.flags |= FDCB_MATRIX_FLAG_DEVICE_ONLY |
         (build->device_id << FDCB_MATRIX_DEVICE_ID_SHIFT);
   problem.group_count = build->group_count;
   problem.energy_slice_count = build->energy_count;
   problem.maximum_group_slices = build->maximum_group_slices;
   problem.slice_count = build->slice_count;
   problem.point_count = build->point_count;
   problem.ddd_table_count = build->energy_count;
   problem.ddd_entry_count = build->ddd_entry_count;
   problem.energy_slices = build->energies;
   problem.points = build->points;
   problem.groups = build->groups;
   problem.raw_energies = build->raw_energies;
   problem.raw_groups = build->raw_groups;
   problem.ddd_tables = build->ddd_tables;
   problem.ddd_entries = build->ddd_entries;
   phase = omp_get_wtime();
   rc = trip_fdcb_matrix_build_accelerator_v1(&problem, &storage, &result);
   build->accelerator_seconds += omp_get_wtime() - phase;
   if ( rc == 0 ) {
      phase = omp_get_wtime();
      rc = FDCBMojoMatrixImport(build, storage, &result);
      build->import_seconds += omp_get_wtime() - phase;
      if ( rc == 0 ) storage = NULL;
   }
   if ( storage != NULL ) trip_fdcb_matrix_storage_destroy_v1(storage);
   build->group_count = 0;
   build->slice_count = 0;
   ++build->flush_count;
   return rc == 0 ? 0 : (CMDERR_ERROR | ERRTRPOPT_SYS);
}

static int TRPOptDoseMatrixPhysMojoBuild(
   struct STRPOPT *opt,
   struct STRPFLDBASE *fields,
   struct STRPOPTVOX *voxels,
   int voxel_count,
   int first_voxel,
   int dose_algorithm
) {
   struct FDCBMojoMatrixBuild build;
   struct Node *node;
   uint32_t field = 0;
   int rc;
   int direct = FDCBMojoUsesDeviceMatrix();
   int device_count = direct ? FDCBMojoDeviceCount() : 1;
   int matrix_voxel_count = voxel_count - first_voxel;
   double start = omp_get_wtime();
   memset(&build, 0, sizeof(build));
   FDCBMojoMatrixReleaseArenas();
   if ( opt == NULL || fields == NULL || voxels == NULL ||
        first_voxel < 0 || first_voxel > voxel_count )
      return CMDERR_ERROR | ERRTRPOPT_BADADDR;
   if ( opt->ulFlag & (TRPOPT_LUNGMOD | TRPOPT_LATTRACKING |
                       TRPOPT_4DMG_BY_VOI | TRPOPT_CALCOPTDVH) ) {
      fprintf(stderr,
              "<E> Mojo matrix builder does not support lung modulation, "
              "lateral tracking, 4D-MG-by-VOI, or optimization DVH\n");
      return CMDERR_ERROR | ERRTRPOPT_BADOPTALGORITHM;
   }
   rc = FDCBMojoMatrixStaticInputInit(&build, opt, fields);
   if ( rc ) goto done;
   if ( direct && (matrix_voxel_count < device_count || device_count > 3) ) {
      rc = CMDERR_ERROR | ERRTRPOPT_SYS;
      goto done;
   }
   build.device_id = 0;
   build.voxel_offset = (uint64_t) first_voxel;
   build.group_capacity = 131072u;
   if ( direct ) {
      uint64_t scenario_groups = (opt->ulFlag & TRPOPT_ROBUST) ?
         (uint64_t) opt->nRobustScenario + 1u : 1u;
      uint64_t total_groups = (uint64_t) (voxel_count - first_voxel) *
                              scenario_groups * (uint64_t) opt->nF;
      if ( total_groups == 0 || total_groups > UINT32_MAX ) {
         rc = CMDERR_ERROR | ERRTRPOPT_SYS;
         goto done;
      }
      build.group_capacity = (uint32_t) total_groups;
   }
   build.groups = calloc(build.group_capacity, sizeof(*build.groups));
   build.raw_groups = calloc(build.group_capacity, sizeof(*build.raw_groups));
   build.group_hosts = calloc(build.group_capacity, sizeof(*build.group_hosts));
   if ( build.groups == NULL || build.raw_groups == NULL ||
        build.group_hosts == NULL ) {
      rc = CMDERR_ERROR | ERRTRPOPT_SYS;
      goto done;
   }

   for ( int voxel = first_voxel; voxel < voxel_count; ++voxel ) {
      int matrix_device = direct ?
         (int) (((int64_t) (voxel - first_voxel) * device_count) /
                matrix_voxel_count) : 0;
      if ( direct && matrix_device != (int) build.device_id ) {
         build.voxel_count = (uint64_t) voxel - build.voxel_offset;
         rc = FDCBMojoMatrixFlush(&build);
         if ( rc ) goto done;
         build.device_id = (uint32_t) matrix_device;
         build.voxel_offset = (uint64_t) voxel;
      }
      int scenario_end = (opt->ulFlag & TRPOPT_ROBUST) ?
                         opt->nRobustScenario : 0;
      for ( int scenario = -1; scenario < scenario_end; ++scenario ) {
         field = 0;
         for ( node = fields->sLH.lh_Head; node->ln_Succ;
               node = node->ln_Succ, ++field ) {
            struct STRPFLD *trip_field = (struct STRPFLD *) node;
            struct STRPRST *rst = &trip_field->sRst;
            uint32_t beam_count = build.field_beam_counts[field];
            struct STRPOPTVOX *active = scenario < 0 ?
               voxels + voxel : voxels[voxel].psRobustOV + scenario;
            struct STRPFLDDOSE *dose;
            FDCBMatrixGroupV1 *group;
            FDCBMatrixRawGroupV1 *raw_group;
            if ( active->psFD == NULL ) {
               rc = CMDERR_ERROR | ERRTRPOPT_BADFIELD;
               goto done;
            }
            if ( build.group_count == build.group_capacity ) {
               rc = FDCBMojoMatrixFlush(&build);
               if ( rc ) goto done;
            }
            dose = active->psFD + field;
#if defined TRPFLD_1504
            if ( dose->dMGEps != dose->dGEps ) {
               fprintf(stderr,
                       "<E> Mojo matrix arena requires identical GEps/MGEps\n");
               rc = CMDERR_ERROR | ERRTRPOPT_BADOPTALGORITHM;
               goto done;
            }
#endif
            group = build.groups + build.group_count;
            raw_group = build.raw_groups + build.group_count;
            group->slice_offset = build.slice_count;
            group->slice_count = beam_count;
            group->bev_x = dose->dvr[XX];
            group->bev_y = dose->dvr[YY];
#if defined TRPFLD_1504
            group->relative_cutoff = dose->dMGEps;
#else
            group->relative_cutoff = dose->dGEps;
#endif
            raw_group->energy_offset = build.field_energy_offsets[field];
            raw_group->energy_count = beam_count;
            FDCBMojoMatrixInitializeFallbacks(&build, field, rst, dose);
            raw_group->depth_mm = dose->dzH2O;
            if ( dose_algorithm == TRPFLDDOSE_MULTIPLESCATTERDIVERGENT ) {
               raw_group->divergence_x =
                  dose->dvr[ZZ] / rst->sScC.dvScannerDistancexy[XX];
               raw_group->divergence_y =
                  dose->dvr[ZZ] / rst->sScC.dvScannerDistancexy[YY];
            }
            build.group_hosts[build.group_count].dose = dose;
            build.group_hosts[build.group_count].beam_count = beam_count;
            build.group_hosts[build.group_count].field_index = field;
            ++build.group_count;
            build.slice_count += beam_count;
         }
      }
   }
   build.voxel_count = (uint64_t) voxel_count - build.voxel_offset;
   rc = FDCBMojoMatrixFlush(&build);
   if ( rc == 0 )
      for ( field = 0; field < (uint32_t) opt->nF; ++field )
         if ( !build.field_nonempty[field] ) {
            fprintf(stderr, "No FDC for field %u\n", field + 1u);
            rc = CMDERR_ERROR | ERRTRPOPT_BADFIELD;
            break;
         }
   if ( rc == 0 )
      fprintf(stderr,
              "<TIME> FDCB Mojo physical matrix: %.6f sec "
              "(accelerator %.6f import %.6f flushes %u)\n",
              omp_get_wtime() - start, build.accelerator_seconds,
              build.import_seconds, build.flush_count);

done:
   if ( rc ) FDCBMojoMatrixReleaseArenas();
   FDCBMojoMatrixBuildFree(&build);
   return rc;
}
