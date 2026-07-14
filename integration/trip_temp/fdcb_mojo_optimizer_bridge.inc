#include "fdcb_abi_v1.h"
#include "fdcb_matrix_abi_v1.h"
#include <malloc.h>
#include <pthread.h>

static int gFDCBMojoFinalMinimumApplied = 0;

static int FDCBMojoPackThreads(void) {
   const char *value = getenv("TRIP_FDCB_MOJO_PACK_THREADS");
   char *end = NULL;
   long threads;
   if ( value == NULL || *value == '\0' ) return 12;
   threads = strtol(value, &end, 10);
   if ( *end != '\0' || threads < 1 || threads > 256 ) return 12;
   return (int) threads;
}

static void FDCBMojoMatrixReleaseArenas(void);
static void FDCBMojoMatrixResetArenaReferences(void);
static void FDCBMojoMatrixRetainDose(const struct STRPFLDDOSE *dose);
static void FDCBMojoMatrixReleaseDose(const struct STRPFLDDOSE *dose);
static void FDCBMojoMatrixDiscardUnreferencedArenas(void);
static int FDCBMojoMatrixDirectView(
   FDCBMatrixStorageV1 **storage, double **coefficients,
   uint16_t **point_indices, uint64_t *entry_count);
static void FDCBMojoMatrixTransferDirectStorage(
   FDCBMatrixStorageV1 *storage);

static struct STRPOPTVOX *FDCBMojoScenarioVoxel(
   struct STRPOPT *opt, int voxel, uint32_t scenario
) {
   if ( opt->ulFlag & TRPOPT_ROBUST )
      return opt->psOV[voxel].psRobustOV + scenario;
   return opt->psOV + voxel;
}

static void FDCBMojoReleaseMergedDose(struct STRPFLDDOSE *dose) {
   FDCBMojoMatrixReleaseDose(dose);
   free(dose->psFDC);
   free(dose->psMFDC);
   free(dose->psFDCS);
   TRPFDCPfree(&dose->sFDCPstore);
   TRPFDCPfree(&dose->sMFDCPstore);
   memset(dose, 0, sizeof(*dose));
}

static void *FDCBMojoTrimHeap(void *duration) {
   double start = omp_get_wtime();
   malloc_trim(0);
   *(double *) duration = omp_get_wtime() - start;
   return NULL;
}

static int FDCBMojoCaptureRNG(FDCBProblemViewV1 *problem,
                              uint32_t state[31],
                              int32_t saved[32]) {
   char temporary[256] = {0};
   char *previous = initstate(1U, temporary, sizeof(temporary));
   int32_t *raw;
   int type, rear;
   if ( previous == NULL ) return -1;
   raw = (int32_t *) previous;
   type = raw[0] % 5;
   rear = raw[0] / 5;
   if ( setstate(previous) == NULL || type != 3 || rear < 0 || rear >= 31 )
      return -1;
   for ( int i = 0; i < 31; ++i ) state[i] = (uint32_t) raw[i + 1];
   memcpy(saved, raw, 32 * sizeof(*saved));
   problem->rng_state_count = 31;
   problem->rng_rear = (uint32_t) rear;
   problem->rng_front = (uint32_t) ((rear + 3) % 31);
   return 0;
}

static int TRPOptFDCBMojoRun(struct STRPOPT *opt,
                             struct STRPDOSECALC *dose_calc) {
   double start = omp_get_wtime();
   double phase;
   FDCBProblemViewV1 problem;
   FDCBProblemStorageV1 *storage = NULL;
   FDCBWritableArraysV1 arrays;
   FDCBResultV1 result;
   uint32_t rng_state[31];
   int32_t saved_rng[32];
   uint32_t *field_base = NULL;
   uint64_t *slice_offsets = NULL;
   uint64_t *coefficient_offsets = NULL;
   double *output_particles = NULL;
   uint64_t point_count = 0;
   uint64_t voxel_scenario_count;
   uint64_t slice_count = 0;
   uint64_t coefficient_count = 0;
   uint64_t unmerged_slices = 0;
   FDCBMatrixStorageV1 *matrix_storage = NULL;
   double *matrix_coefficients = NULL;
   uint16_t *matrix_point_indices = NULL;
   uint64_t matrix_entry_count = 0;
   pthread_t trim_thread;
   double trim_seconds = 0.0;
   int trim_started = 0;
   int direct_matrix = 0;
   int device_bootstrap = 0;
   int direct_pack_error = 0;
   int pack_threads = FDCBMojoPackThreads();
   int rc = -1;

   memset(&problem, 0, sizeof(problem));
   memset(&arrays, 0, sizeof(arrays));
   memset(&result, 0, sizeof(result));
   gFDCBMojoFinalMinimumApplied = 0;
   problem.version = FDCB_ABI_VERSION_V1;
   problem.flags = FDCB_FLAG_INITIALIZE;
   device_bootstrap =
      getenv("TRIP_FDCB_MOJO_DEVICE_BOOTSTRAP") != NULL;
   if ( device_bootstrap ) problem.flags |= FDCB_FLAG_DEVICE_BOOTSTRAP;
   if ( opt->ulFlag & TRPOPT_BIO ) problem.flags |= FDCB_FLAG_BIOLOGICAL;
   if ( opt->ulFlag & TRPOPT_ROBUST_INCLUDE_DMAX )
      problem.flags |= FDCB_FLAG_ROBUST_INCLUDE_DMAX;
   problem.precision_mode = FDCB_PRECISION_REFERENCE;
   problem.minimum_particle_policy = (opt->ulFlag & TRPOPT_COMPLEXMINPART) ?
      FDCB_MIN_PARTICLE_COMPLEX_HOST_RNG : FDCB_MIN_PARTICLE_SIMPLE;
   problem.optimizer_algorithm = FDCB_OPTIMIZER_FDCB;
   problem.dose_algorithm = FDCB_DOSE_MSDB;
   problem.biology_model = (opt->ulFlag & TRPOPT_BIO) ?
      FDCB_BIOLOGY_LOW_DOSE : FDCB_BIOLOGY_NONE;
   problem.max_threads = 12;
   problem.rng_seed = opt->sVXR.iSeed;
   if ( FDCBMojoCaptureRNG(&problem, rng_state, saved_rng) ) goto done;
   problem.field_count = (uint32_t) opt->nF;
   problem.scenario_count = (opt->ulFlag & TRPOPT_ROBUST) ?
      (uint32_t) opt->nRobustScenario : 1U;
   problem.max_iterations = (uint32_t) opt->mIter;
   problem.grace_iterations = (uint32_t) opt->mGraceIter;
   problem.avoidance_voxel_count = (uint32_t) opt->nVOAR;
   problem.active_point_count = (uint32_t) opt->mPnt;
   problem.total_iterations = (uint32_t) opt->nTotalIter;
   problem.step_factor_iterations = (uint32_t) opt->mmyFacIter;
   problem.fractions = (dose_calc->psDI &&
                        dose_calc->psDI->sDF.nFractions > 0) ?
      (double) dose_calc->psDI->sDF.nFractions : 1.0;
   problem.current_chi2 = opt->dChi2;
   problem.dose_p_weighted_avg2 = opt->dDosePWeightedAvg2;
   problem.epsilon = opt->dEps;
   problem.chi_square_limit = opt->dChiSquareLimit;
   problem.overdose_weight = opt->dOverDoseWeight;
   problem.prescribed_dose = opt->dDosePrescribed;
   problem.configured_step_factor = opt->dmyFac;
   problem.current_step = opt->dmy;

   if ( !device_bootstrap ) {
      rc = TRPOptZeroGradientFDCS(opt, dose_calc, 0);
      if ( rc ) goto done;
   }
   field_base = calloc((size_t) opt->nF + 1, sizeof(*field_base));
   if ( field_base == NULL ) goto done;
   for ( int field = 0; field < opt->nF; ++field ) {
      field_base[field] = (uint32_t) problem.field_slice_count;
      problem.field_slice_count += (uint64_t) opt->psOF[field].nB;
      for ( int beam = 0; beam < opt->psOF[field].nB; ++beam )
         point_count += (uint64_t) opt->psOF[field].psORS[beam].nPnt;
   }
   field_base[opt->nF] = (uint32_t) problem.field_slice_count;
   problem.point_count = point_count;
   problem.voxel_count = (uint64_t) opt->nV;
   voxel_scenario_count = problem.voxel_count * problem.scenario_count;
   problem.voxel_scenario_count = voxel_scenario_count;
   slice_offsets = calloc((size_t) voxel_scenario_count + 1,
                          sizeof(*slice_offsets));
   coefficient_offsets = calloc((size_t) voxel_scenario_count + 1,
                                sizeof(*coefficient_offsets));
   if ( slice_offsets == NULL || coefficient_offsets == NULL ) goto done;

   phase = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(pack_threads) reduction(+:unmerged_slices)
   for ( int voxel = 0; voxel < opt->nV; ++voxel )
      for ( uint32_t scenario = 0; scenario < problem.scenario_count;
            ++scenario ) {
         uint64_t index = (uint64_t) voxel * problem.scenario_count + scenario;
         uint64_t local_slices = 0;
         uint64_t local_coefficients = 0;
         struct STRPOPTVOX *source = FDCBMojoScenarioVoxel(
            opt, voxel, scenario);
         for ( int field = 0; field < opt->nF; ++field )
            for ( int beam = 0; beam < source->psFD[field].nFDC; ++beam ) {
               struct STRPFLDDOSECORRSLICE *slice =
                  source->psFD[field].psFDCS + beam;
               if ( slice->sFDCP.mPnt || slice->sMFDCP.mPnt )
                  ++unmerged_slices;
               if ( slice->sFDCP.nPnt ) {
                  ++local_slices;
                  local_coefficients += slice->sFDCP.nPnt;
               }
            }
         slice_offsets[index] = local_slices;
         coefficient_offsets[index] = local_coefficients;
      }
   for ( uint64_t index = 0; index < voxel_scenario_count; ++index ) {
      uint64_t local_slices = slice_offsets[index];
      uint64_t local_coefficients = coefficient_offsets[index];
      slice_offsets[index] = slice_count;
      coefficient_offsets[index] = coefficient_count;
      slice_count += local_slices;
      coefficient_count += local_coefficients;
   }
   slice_offsets[voxel_scenario_count] = slice_count;
   coefficient_offsets[voxel_scenario_count] = coefficient_count;
   problem.slice_count = slice_count;
   problem.coefficient_count = coefficient_count;
   fprintf(stderr,
           "<TIME> FDCB Mojo pack count: %.6f sec; slices=%" PRIu64
           " coefficients=%" PRIu64 "\n",
           omp_get_wtime() - phase, slice_count, coefficient_count);

   FDCBMojoMatrixResetArenaReferences();
   for ( int voxel = 0; voxel < opt->nV; ++voxel )
      for ( uint32_t scenario = 0; scenario < problem.scenario_count;
            ++scenario ) {
         struct STRPOPTVOX *source = FDCBMojoScenarioVoxel(
            opt, voxel, scenario);
         for ( int field = 0; field < opt->nF; ++field )
            FDCBMojoMatrixRetainDose(source->psFD + field);
      }
   FDCBMojoMatrixDiscardUnreferencedArenas();

   if ( getenv("TRIP_FDCB_MOJO_DIRECT") != NULL ) {
      if ( FDCBMojoMatrixDirectView(&matrix_storage, &matrix_coefficients,
                                    &matrix_point_indices,
                                    &matrix_entry_count) != 0 ) {
         fprintf(stderr, "<E> direct Mojo optimizer requires one matrix arena\n");
         goto done;
      }
      problem.coefficient_count = matrix_entry_count;
      direct_matrix = 1;
   }

   phase = omp_get_wtime();
   if ( direct_matrix )
      rc = trip_fdcb_matrix_problem_storage_create_v1(
         &problem, matrix_storage, &storage, &arrays);
   else
      rc = trip_fdcb_storage_create_v1(&problem, &storage, &arrays);
   if ( rc ) goto done;
   if ( direct_matrix )
      FDCBMojoMatrixTransferDirectStorage(matrix_storage);
   fprintf(stderr, "<TIME> FDCB Mojo pack allocate: %.6f sec\n",
           omp_get_wtime() - phase);
   memcpy(arrays.rng_state, rng_state, sizeof(rng_state));

   {
      uint64_t point = 0;
      for ( int field = 0; field < opt->nF; ++field )
         for ( int beam = 0; beam < opt->psOF[field].nB; ++beam ) {
            uint32_t field_slice = field_base[field] + (uint32_t) beam;
            int count = opt->psOF[field].psORS[beam].nPnt;
            FDCBFieldSliceV1 *destination = arrays.field_slices + field_slice;
            destination->field_index = (uint32_t) field;
            destination->beam_index = (uint32_t) beam;
            destination->point_offset = point;
            destination->point_count = (uint32_t) count;
            destination->raster_stride =
               (uint32_t) opt->psOF[field].psRst->psRS[beam].nvxy[0];
            destination->minimum_particles =
               opt->psOF[field].psRst->sScC.pdMinParticles[
                  opt->psOF[field].psRst->psRS[beam].sScC.iMinParticles];
            for ( int local = 0; local < count; ++local, ++point ) {
               struct STRPOPTRSTPNT *source =
                  opt->psOF[field].psORS[beam].psORP + local;
               arrays.particles[point] =
                  opt->psOF[field].psRst->psRS[beam].psPnt[local].dN;
               arrays.initial_direction[point] =
                  device_bootstrap ? 0.0 : source->dh1;
               arrays.initial_gradient[point] =
                  device_bootstrap ? 0.0 : source->dg1;
               arrays.point_active[point] =
                  (opt->psOF[field].psRst->psRS[beam].psPnt[local].ulFlag &
                   TRPRSTPNT_INSIDE) != 0;
            }
         }
   }

   phase = omp_get_wtime();
   {
      const int chunk_size = 8192;
      for ( int base = 0; base < opt->nV; base += chunk_size ) {
         int end = base + chunk_size;
         if ( end > opt->nV ) end = opt->nV;
#pragma omp parallel for schedule(static) num_threads(pack_threads)
         for ( int voxel = base; voxel < end; ++voxel ) {
            struct STRPOPTVOX *nominal = opt->psOV + voxel;
            struct STRPRBE *rbe = (opt->ulFlag & TRPOPT_BIO) ?
               nominal->psVoi->psRN->svRBE + TRPRBE_MEAN : NULL;
            FDCBVoxelV1 *voxel_out = arrays.voxels + voxel;
            voxel_out->prescribed_dose = nominal->dDoseP;
            voxel_out->dose_weight = nominal->dDoseW;
            voxel_out->dose_divisor = nominal->dDoseD;
            voxel_out->maximum_dose_weight = nominal->dDoseWMax;
            voxel_out->initial_dose = nominal->dDoseI;
            voxel_out->prescribed_let = nominal->dLETbarP;
            voxel_out->let_weight = nominal->psVoi->dWeightLET;
            voxel_out->overdose_tolerance = nominal->psVoi->dOverdoseTolerance;
            if ( rbe ) {
               voxel_out->rbe_cut = rbe->dCut;
               voxel_out->rbe_alpha = rbe->dAlpha;
               voxel_out->rbe_beta = rbe->dBeta;
               voxel_out->rbe_slope_max = rbe->dSlopeMax;
               voxel_out->rbe_damage_cut = rbe->dDamageCut;
            }
            voxel_out->initial_min_scenario = nominal->nMinDoseScenario;
            voxel_out->initial_max_scenario = nominal->nMaxDoseScenario;
            voxel_out->scenario_offset =
               (uint64_t) voxel * problem.scenario_count;
            for ( uint32_t scenario = 0; scenario < problem.scenario_count;
                  ++scenario ) {
               uint64_t state_index = voxel_out->scenario_offset + scenario;
               struct STRPOPTVOX *source = FDCBMojoScenarioVoxel(
                  opt, voxel, scenario);
               FDCBVoxelScenarioV1 *scenario_out =
                  arrays.voxel_scenarios + state_index;
               FDCBScenarioStateV1 *state_out =
                  arrays.scenario_states + state_index;
               uint64_t slice = slice_offsets[state_index];
               uint64_t coefficient = coefficient_offsets[state_index];
               scenario_out->slice_offset = slice;
               scenario_out->slice_count =
                  (uint32_t) (slice_offsets[state_index + 1] - slice);
               scenario_out->reserved = 0;
               state_out->dose_minor = source->dDoseM;
               state_out->alpha_minor = source->dalpha;
               state_out->sqrt_beta_minor = source->dbeta;
               state_out->let_mix_minor = source->dLETMix;
               for ( int field = 0; field < opt->nF; ++field ) {
                  for ( int beam = 0; beam < source->psFD[field].nFDC;
                        ++beam ) {
                     struct STRPFLDDOSECORRSLICE *input =
                        source->psFD[field].psFDCS + beam;
                     if ( input->sFDCP.nPnt == 0 ) continue;
                     FDCBSliceV1 *output = arrays.slices + slice++;
                     output->field_slice_index =
                        field_base[field] + (uint32_t) beam;
                     output->reserved = 0;
                     if ( direct_matrix ) {
                        uint64_t direct_offset = (uint64_t)
                           (input->sFDCP.pdG - matrix_coefficients);
                        if ( input->sFDCP.nPnt > matrix_entry_count ||
                             direct_offset >
                                matrix_entry_count - input->sFDCP.nPnt ||
                             input->sFDCP.piPnt !=
                             matrix_point_indices + direct_offset ) {
#pragma omp atomic write
                           direct_pack_error = 1;
                           continue;
                        }
                        output->coefficient_offset = direct_offset;
                     } else {
                        output->coefficient_offset = coefficient;
                     }
                     output->coefficient_count = input->sFDCP.nPnt;
                     output->dose_coefficient = input->dDDD;
                     output->alpha_coefficient = input->dAlpha;
                     output->sqrt_beta_coefficient = input->dSqrtBeta;
                     output->let_mix_coefficient = input->dLETmix;
                     output->let_bar_coefficient = input->dLETbar;
                     if ( !direct_matrix ) {
                        memcpy(arrays.coefficient_point_indices + coefficient,
                               input->sFDCP.piPnt,
                               (size_t) input->sFDCP.nPnt * sizeof(uint16_t));
                        memcpy(arrays.coefficients + coefficient,
                               input->sFDCP.pdG,
                               (size_t) input->sFDCP.nPnt * sizeof(double));
                     }
                     coefficient += input->sFDCP.nPnt;
                  }
                  if ( unmerged_slices ) TRPFldDoseFree(source->psFD + field, 0);
                  else FDCBMojoReleaseMergedDose(source->psFD + field);
               }
            }
         }
         if ( direct_pack_error ) {
            fprintf(stderr, "<E> invalid direct Mojo matrix slice offset\n");
            rc = -1;
            goto done;
         }
         FDCBMojoMatrixDiscardUnreferencedArenas();
      }
   }
   fprintf(stderr, "<TIME> FDCB Mojo pack matrices: %.6f sec\n",
           omp_get_wtime() - phase);
   FDCBMojoMatrixReleaseArenas();
   if ( pthread_create(&trim_thread, NULL, FDCBMojoTrimHeap,
                       &trim_seconds) == 0 )
      trim_started = 1;
   output_particles = calloc((size_t) point_count, sizeof(*output_particles));
   if ( output_particles == NULL ) goto done;
   phase = omp_get_wtime();
   if ( direct_matrix )
      rc = trip_fdcb_matrix_problem_optimize_accelerator_v1(
         storage, output_particles, point_count, &result);
   else
      rc = trip_fdcb_storage_optimize_accelerator_v1(
         storage, output_particles, point_count, &result);
   fprintf(stderr, "<TIME> FDCB Mojo optimize: %.6f sec\n",
           omp_get_wtime() - phase);
   if ( rc ) goto done;
   {
      uint64_t point = 0;
      for ( int field = 0; field < opt->nF; ++field )
         for ( int beam = 0; beam < opt->psOF[field].nB; ++beam )
            for ( int local = 0; local < opt->psOF[field].psORS[beam].nPnt;
                  ++local, ++point ) {
               opt->psOF[field].psORS[beam].psORP[local].dN =
                  output_particles[point];
               opt->psOF[field].psRst->psRS[beam].psPnt[local].dN =
                  output_particles[point];
            }
   }
   opt->dChi2 = result.chi2;
   gFDCBMojoFinalMinimumApplied =
      (result.flags & FDCB_RESULT_FINAL_MIN_PARTICLES_APPLIED) != 0;
   fprintf(stderr,
           "<I> FDCB Mojo: iterations=%" PRIu64 " chi2=%.17g residual=%.17g "
           "deleted=%" PRIu64 " rng=%" PRIu64 "\n",
           result.iterations, result.chi2, result.residual_percent,
           result.minimum_particle_deleted, result.random_draws);
   if ( setstate((char *) saved_rng) == NULL ) rc = -1;

done:
   if ( trim_started ) pthread_join(trim_thread, NULL);
   if ( trim_started )
      fprintf(stderr, "<TIME> FDCB Mojo heap trim: %.6f sec (overlapped)\n",
              trim_seconds);
   free(field_base);
   free(slice_offsets);
   free(coefficient_offsets);
   free(output_particles);
   if ( storage ) {
      if ( direct_matrix )
         trip_fdcb_matrix_problem_storage_destroy_v1(storage);
      else
         trip_fdcb_storage_destroy_v1(storage);
   }
   fprintf(stderr, "<TIME> FDCB Mojo optimizer bridge: %.6f sec\n",
           omp_get_wtime() - start);
   return rc;
}
