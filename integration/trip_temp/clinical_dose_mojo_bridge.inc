#include "clinical_dose_abi.h"

struct TRPClinicalDoseMojoBuild {
   ClinicalDoseState *states;
   ClinicalDosePosition *positions;
   ClinicalDoseCTState *ct_states;
   int16_t *ct_data;
   double *ct_boundaries;
   double *dose_axis_centers;
   int32_t *voxel_voi;
   ClinicalDoseGridField *grid_fields;
   ClinicalDoseField *fields;
   ClinicalDoseEnergy *energies;
   ClinicalDosePoint *points;
   ClinicalDoseDDDTable *ddd_tables;
   ClinicalDoseDDDEntry *ddd_entries;
   ClinicalDoseBioTable *bio_tables;
   ClinicalDoseBioEntry *bio_entries;
   ClinicalDoseOutput *output;
};

static void TRPClinicalDoseMojoFree(struct TRPClinicalDoseMojoBuild *b) {
   if ( b == NULL ) return;
   free(b->states); free(b->positions); free(b->ct_states); free(b->ct_data);
   free(b->ct_boundaries);
   free(b->dose_axis_centers);
   free(b->voxel_voi); free(b->grid_fields); free(b->fields);
   free(b->energies); free(b->points); free(b->ddd_tables);
   free(b->ddd_entries); free(b->bio_tables); free(b->bio_entries);
   free(b->output); memset(b, 0, sizeof(*b));
}

static ClinicalDoseGrid TRPClinicalDoseMojoGrid(struct STRPCUBE *c) {
   ClinicalDoseGrid g;
   memset(&g, 0, sizeof(g));
   g.nx = c->nx; g.ny = c->ny; g.nz = c->nz;
   g.x0 = 0.5 * (c->pdx[0] + c->pdx[1]);
   g.y0 = 0.5 * (c->pdy[0] + c->pdy[1]);
   g.z0 = 0.5 * (c->pdz[0] + c->pdz[1]);
   g.dx = c->pdx[1] - c->pdx[0];
   g.dy = c->pdy[1] - c->pdy[0];
   g.dz = c->pdz[1] - c->pdz[0];
   g.boundary_x0 = c->pdx[0];
   g.boundary_y0 = c->pdy[0];
   g.boundary_z0 = c->pdz[0];
   return g;
}

static int TRPClinicalDoseMojoRegular(struct STRPCUBE *c) {
   const double step[3] = {c->pdx[1]-c->pdx[0], c->pdy[1]-c->pdy[0],
                           c->pdz[1]-c->pdz[0]};
   double *axis[3] = {c->pdx, c->pdy, c->pdz};
   int count[3] = {c->nx, c->ny, c->nz};
   for ( int a=0; a<3; ++a ) for ( int i=1; i<count[a]; ++i ) {
      double actual=axis[a][i+1]-axis[a][i];
      double scale=max(1.0,max(fabs(axis[a][i]),fabs(axis[a][i+1])));
      if ( fabs(actual-step[a]) > 32.0*DBL_EPSILON*scale ) return 0;
   }
   return step[0] > 0.0 && step[1] > 0.0 && step[2] > 0.0;
}

static struct STRPRST *TRPClinicalDoseMojoRst(struct STRPFLD *f, int state,
                                              int four_d) {
   return four_d ? f->sRst.psStateRst + state : &f->sRst;
}

static int TRPClinicalDoseMojoRun(
   struct STRPDOSECALC *dc, struct STRPFLDBASE *fb, struct STRPWDW *wdw,
   struct STRPTRAFO *trafo, uint32_t flag
) {
   struct TRPClinicalDoseMojoBuild b = {0};
   struct STRPCUBE *main_cube = NULL, dose_grid;
   struct STRPTRAFOCALC tc;
   struct STRPDEDX *dedx = NULL;
   struct STRPRBESPCBASE ***rs_base = NULL;
   struct STRPSPCBASE **spc_base = NULL;
   struct Node *node;
   ClinicalDoseProblem p;
   uint64_t ct_values=0, ct_boundaries=0, point_count=0, ddd_entries=0, bio_entries=0;
   uint32_t fields=0, energies=0, bio_tables=0;
   int rc=0, residual=-1, base_fields=0, dose_algorithm;
   const int four_d = (flag & TRPDOSECALC_4DCALC) != 0;
   const int biological = (dc->ulDoseObj & TRPDOSE_BITBIOL) != 0;
   const uint32_t allowed = TRPDOSE_BITPHYS | TRPDOSE_BITBIOL |
      TRPDOSE_BITSVV | TRPDOSE_BITRBE | TRPDOSE_BITDOSEMLET |
      TRPDOSE_BITALPHA | TRPDOSE_BITBETA;
   int states = four_d ? dc->psCT->nStates : 1;
   TRPCubeInit(&dose_grid, 0); memset(&tc, 0, sizeof(tc)); memset(&p, 0, sizeof(p));

   if ( (flag & (TRPDOSECALC_4DSMOOTHED | TRPDOSECALC_4DSTATE |
                 TRPDOSECALC_ARC)) || (dc->ulDoseObj & ~allowed) ||
        dc->psOxy != NULL || states < 1 ) {
      fprintf(stderr, "<E> Mojo clinical dose: unsupported 4D interpolation, state output, arc, oxygen, or dose object\n");
      rc = CMDERR_ERROR | ERRTRPDOSE_NOTYET; goto done;
   }
   dose_algorithm=TRPFldDoseAlgFromStr(NULL,dc->pcDoseAlgorithm,
                                      TRPFLDDOSEALGFROMSTR_RETURNALGONLY);
   if ( dose_algorithm!=TRPFLDDOSE_MULTIPLESCATTER &&
        dose_algorithm!=TRPFLDDOSE_MULTIPLESCATTERDIVERGENT ) {
      fprintf(stderr,"<E> Mojo clinical dose supports only ms and msdb\n");
      rc=CMDERR_ERROR|ERRTRPDOSE_BADALGORITHM; goto done;
   }
   rc = TRPDoseCalcSetupDirect(dc, &main_cube, fb, wdw, trafo, &tc, flag);
   if ( rc & CMDERR_MASK ) goto done;
   if ( biological ) {
      rc = TRPDoseCalcFldSetupDirect(dc, fb, &rs_base, &spc_base, flag);
      if ( rc & CMDERR_MASK ) goto done;
      residual = rc;
   }
   if ( dc->ulDoseObj & TRPDOSE_BITDOSEMLET ) {
      dedx = TRPdEdxBasedEdxFind(dc->psdEdxBase, "H2O", 0);
      if ( dedx == NULL ) { rc=CMDERR_ERROR|ERRTRPDOSE_BADDEDX; goto done; }
   }
   rc = TRPCubeResize2Wdw(main_cube, "", &dose_grid, wdw, 0.0,
                          TRPCUBE_NODATA, TRPCUBE_CLONENATIVE |
                          TRPCUBERESIZE2WDW_NODATA);
   if ( rc & CMDERR_MASK ) { rc = CMDERR_ERROR | ERRTRPDOSE_CLONEFAILED; goto done; }
   if ( !TRPClinicalDoseMojoRegular(&dose_grid) || dose_grid.nx <= 0 ||
        (uint64_t)dose_grid.nx*dose_grid.ny*dose_grid.nz > UINT32_MAX ) {
      fprintf(stderr, "<E> Mojo clinical dose requires a regular grid below 2^32 voxels\n");
      rc = CMDERR_ERROR | ERRTRPDOSE_BADPARAMETER; goto done;
   }
   for ( node=fb->sLH.lh_Head; node->ln_Succ; node=node->ln_Succ ) ++base_fields;
   fields = (uint32_t)(base_fields * states);
   for ( int s=0; s<states; ++s ) {
      struct STRPCT *ct = four_d ? dc->psCT->psCTState+s : dc->psCT;
      struct STRPCUBE *c = &ct->sCube;
      if ( c->lDataType != TRPCUBE_SHORT || !TRPClinicalDoseMojoRegular(c) ||
           ct->dPmod != 0.0 ) {
         fprintf(stderr, "<E> Mojo clinical dose requires regular short CT states without lung modulation\n");
         rc = CMDERR_ERROR | ERRTRPDOSE_NOTYET; goto done;
      }
      ct_values += (uint64_t)c->nx*c->ny*c->nz;
      ct_boundaries += (uint64_t)c->nx+c->ny+c->nz+3;
      int fi=0;
      for ( node=fb->sLH.lh_Head; node->ln_Succ; node=node->ln_Succ,++fi ) {
         struct STRPRST *rst = TRPClinicalDoseMojoRst((struct STRPFLD*)node,s,four_d);
         energies += (uint32_t)rst->nB;
         for ( int e=0; e<rst->nB; ++e ) {
            struct STRPRSTSLICE *slice=rst->psRS+e;
            point_count += (uint64_t)slice->nPnt;
            ddd_entries += (uint64_t)slice->sDDD.nz;
            if ( biological ) {
               struct STRPSPCB *spc=spc_base[fi]->psSB[slice->iBSpc];
               bio_tables += (uint32_t)dc->psVOIBase->nV;
               bio_entries += (uint64_t)spc->nz*dc->psVOIBase->nV;
            }
         }
      }
   }
   uint64_t voxels=(uint64_t)dose_grid.nx*dose_grid.ny*dose_grid.nz;
   if ( point_count > UINT32_MAX || ddd_entries > UINT32_MAX ||
        bio_entries > UINT32_MAX || ct_values > SIZE_MAX/sizeof(int16_t) ||
        ct_boundaries > SIZE_MAX/sizeof(double) ||
        (four_d && voxels*(uint64_t)states > SIZE_MAX/sizeof(*b.positions)) ) {
      rc = CMDERR_ERROR | ERRTRPDOSE_SYS; goto done;
   }
#define TRP_MOJO_ALLOC(member,count) do { \
   if ((count) && ((b.member=calloc((size_t)(count),sizeof(*b.member)))==NULL)) { \
      rc=CMDERR_ERROR|ERRTRPDOSE_SYS; goto done; } \
} while(0)
   TRP_MOJO_ALLOC(states,states); TRP_MOJO_ALLOC(ct_states,states);
   TRP_MOJO_ALLOC(ct_data,ct_values); TRP_MOJO_ALLOC(grid_fields,fields);
   TRP_MOJO_ALLOC(ct_boundaries,ct_boundaries);
   TRP_MOJO_ALLOC(dose_axis_centers,
                  (uint64_t)dose_grid.nx+dose_grid.ny+dose_grid.nz);
   TRP_MOJO_ALLOC(fields,fields); TRP_MOJO_ALLOC(energies,energies);
   TRP_MOJO_ALLOC(points,point_count); TRP_MOJO_ALLOC(ddd_tables,energies);
   TRP_MOJO_ALLOC(ddd_entries,ddd_entries); TRP_MOJO_ALLOC(bio_tables,bio_tables);
   TRP_MOJO_ALLOC(bio_entries,bio_entries); TRP_MOJO_ALLOC(output,voxels);
   if ( biological ) TRP_MOJO_ALLOC(voxel_voi,voxels);
   if ( four_d ) TRP_MOJO_ALLOC(positions,voxels*(uint64_t)states);
#undef TRP_MOJO_ALLOC

   p.dose_x_offset=0;
   p.dose_y_offset=(uint64_t)dose_grid.nx;
   p.dose_z_offset=p.dose_y_offset+(uint64_t)dose_grid.ny;
   p.dose_axis_count=p.dose_z_offset+(uint64_t)dose_grid.nz;
   for ( int x=0; x<dose_grid.nx; ++x )
      b.dose_axis_centers[p.dose_x_offset+x]=0.5*(dose_grid.pdx[x]+dose_grid.pdx[x+1]);
   for ( int y=0; y<dose_grid.ny; ++y )
      b.dose_axis_centers[p.dose_y_offset+y]=0.5*(dose_grid.pdy[y]+dose_grid.pdy[y+1]);
   for ( int z=0; z<dose_grid.nz; ++z )
      b.dose_axis_centers[p.dose_z_offset+z]=0.5*(dose_grid.pdz[z]+dose_grid.pdz[z+1]);
   uint64_t cto=0, cbo=0, po=0, ddo=0, beo=0; uint32_t fi=0, eo=0, bto=0;
   for ( int s=0; s<states; ++s ) {
      struct STRPCT *ct = four_d ? dc->psCT->psCTState+s : dc->psCT;
      struct STRPCUBE *c=&ct->sCube;
      b.states[s].field_offset=fi; b.states[s].field_count=(uint32_t)base_fields;
      b.ct_states[s].grid=TRPClinicalDoseMojoGrid(c);
      b.ct_states[s].data_offset=cto; b.ct_states[s].pmod=ct->dPmod;
      b.ct_states[s].pore_size=ct->dPoreSize; b.ct_states[s].byte_swap=0;
      b.ct_states[s].x_boundary_offset=cbo;
      memcpy(b.ct_boundaries+cbo,c->pdx,(size_t)(c->nx+1)*sizeof(*c->pdx)); cbo+=c->nx+1;
      b.ct_states[s].y_boundary_offset=cbo;
      memcpy(b.ct_boundaries+cbo,c->pdy,(size_t)(c->ny+1)*sizeof(*c->pdy)); cbo+=c->ny+1;
      b.ct_states[s].z_boundary_offset=cbo;
      memcpy(b.ct_boundaries+cbo,c->pdz,(size_t)(c->nz+1)*sizeof(*c->pdz)); cbo+=c->nz+1;
      b.ct_states[s].grid.dx=c->dvSteps[XX][2];
      b.ct_states[s].grid.dy=c->dvSteps[YY][2];
      b.ct_states[s].grid.dz=c->dvSteps[ZZ][2];
      for ( int z=0; z<c->nz; ++z ) for ( int y=0; y<c->ny; ++y ) {
         if ( c->lByteSwap )
            swapw((unsigned short*)c->piData[z][y],
                  (unsigned short*)(b.ct_data+cto),(unsigned long)c->nx);
         else
            memcpy(b.ct_data+cto,c->piData[z][y],(size_t)c->nx*sizeof(*b.ct_data));
         cto+=(uint64_t)c->nx;
      }
      int base=0;
      for ( node=fb->sLH.lh_Head; node->ln_Succ; node=node->ln_Succ,++base,++fi ) {
         struct STRPFLD *f=(struct STRPFLD*)node;
         struct STRPRST *rst=TRPClinicalDoseMojoRst(f,s,four_d);
         ClinicalDoseGridField *gf=b.grid_fields+fi;
         ClinicalDoseField *of=b.fields+fi;
         gf->field_index=fi;
         for ( int r=0; r<3; ++r ) for ( int col=0; col<4; ++col )
            gf->patient_to_gantry[r*4+col]=f->dvPatient2Gantry[r][col];
         memcpy(gf->direction,f->dvDir,sizeof(gf->direction));
         gf->off_h2o=f->sRst.sScC.dOffH2O; gf->bolus=f->sRst.sScC.dBolus;
         gf->window_x0=rst->sWdwLat.dvLim[XX][FR]; gf->window_x1=rst->sWdwLat.dvLim[XX][TO];
         gf->window_y0=rst->sWdwLat.dvLim[YY][FR]; gf->window_y1=rst->sWdwLat.dvLim[YY][TO];
         gf->gated=rst->bGating;
         of->energy_offset=eo; of->energy_count=(uint32_t)rst->nB;
         of->dose_extension2=f->sRst.dDoseExtension*f->sRst.dDoseExtension;
         of->scanner_x=rst->sScC.dvScannerDistancexy[XX];
         of->scanner_y=rst->sScC.dvScannerDistancexy[YY];
         for ( int e=0; e<rst->nB; ++e,++eo ) {
            struct STRPRSTSLICE *slice=rst->psRS+e; struct STRPDDD *ddd=&slice->sDDD;
            ClinicalDoseEnergy *oe=b.energies+eo;
            oe->point_offset=(uint32_t)po; oe->point_count=(uint32_t)slice->nPnt;
            oe->ddd_table=eo; oe->bio_table_offset=bto; oe->focus=slice->dF;
            oe->range_shifter=slice->sRSh.dThick;
            oe->window_x0=slice->sWdwLat.dvLim[XX][FR]; oe->window_x1=slice->sWdwLat.dvLim[XX][TO];
            oe->window_y0=slice->sWdwLat.dvLim[YY][FR]; oe->window_y1=slice->sWdwLat.dvLim[YY][TO];
            for ( int q=0; q<slice->nPnt; ++q,++po ) {
               struct STRPRSTPNT *rp=slice->psPnt+q; ClinicalDosePoint *op=b.points+po;
               op->x=rp->dv[XX]; op->y=rp->dv[YY]; op->delta_z=rp->dvDelta[ZZ];
               op->f2_max=rp->dF2Max;
               op->particles=(rp->ulFlag&TRPRSTPNT_INSIDE) && rp->dN>DBL_EPSILON ?
                  rp->dN : 0.0;
            }
            b.ddd_tables[eo].entry_offset=(uint32_t)ddo;
            b.ddd_tables[eo].entry_count=(uint32_t)ddd->nz;
            b.ddd_tables[eo].column_count=(uint32_t)ddd->nCol;
            for ( int z=0; z<ddd->nz; ++z,++ddo ) {
               ClinicalDoseDDDEntry *od=b.ddd_entries+ddo;
               od->z_cm=ddd->pdTab[TRPDDD_X][z]; od->dose=ddd->pdTab[TRPDDD_D][z];
               if ( ddd->nCol>2 ) { od->fwhm1=ddd->pdTab[TRPDDD_FWHM1][z];
                  od->mix=ddd->pdTab[TRPDDD_G][z]; od->fwhm2=ddd->pdTab[TRPDDD_FWHM2][z]; }
            }
            if ( biological ) for ( int v=0; v<dc->psVOIBase->nV; ++v,++bto ) {
               struct STRPSPCB *spc=spc_base[base]->psSB[slice->iBSpc];
               struct STRPRBESPCB *rbe=rs_base[base][v]->psRSB[slice->iBSpc];
               if ( spc->nz != rbe->nz || slice->dP==0.0 ) { rc=CMDERR_ERROR|ERRTRPDOSE_BADSPCBASE; goto done; }
               b.bio_tables[bto].entry_offset=(uint32_t)beo;
               b.bio_tables[bto].entry_count=(uint32_t)spc->nz;
               b.bio_tables[bto].z_scale=0.1*spc->dP/slice->dP;
               for ( int z=0; z<spc->nz; ++z,++beo ) {
                  ClinicalDoseBioEntry *ob=b.bio_entries+beo;
                  ob->z_cm=spc->psSD[z].dz; ob->alpha=rbe->psRSD[z].dalpha0x1;
                  ob->sqrt_beta=rbe->psRSD[z].dbeta0x1;
                  ob->let_mix=spc->psSD[z].dvMomLET[0]*spc->psSD[z].dvMomLET[1];
                  ob->let_bar=0.1*spc->psSD[z].dvMomLETDM[0]*spc->psSD[z].dvMomLETDM[1];
                  ob->let_dm_sum=spc->psSD[z].dvMomLETDM[0];
               }
            }
         }
      }
   }
   if ( biological ) {
#pragma omp parallel for schedule(static)
      for ( uint64_t i=0; i<voxels; ++i ) {
         int x=i%dose_grid.nx, y=(i/dose_grid.nx)%dose_grid.ny, z=i/(dose_grid.nx*dose_grid.ny);
         double pos[3]={0.5*(dose_grid.pdx[x]+dose_grid.pdx[x+1]),
                        0.5*(dose_grid.pdy[y]+dose_grid.pdy[y+1]),
                        0.5*(dose_grid.pdz[z]+dose_grid.pdz[z+1])};
         int v=TRPVOIBaseGetIndex(dc->psVOIBase,pos);
         b.voxel_voi[i]=((dc->ulFlag&TRPDOSECALC_VOILIST)&&v==residual)?-1:v;
      }
   }
   if ( four_d ) {
      int transform_error=0;
#pragma omp parallel for schedule(static)
      for ( uint64_t i=0; i<voxels*(uint64_t)states; ++i ) {
         int s=i/voxels; uint64_t q=i-s*voxels;
         int x=q%dose_grid.nx, y=(q/dose_grid.nx)%dose_grid.ny, z=q/(dose_grid.nx*dose_grid.ny);
         double from[4]={0.5*(dose_grid.pdx[x]+dose_grid.pdx[x+1]),
                         0.5*(dose_grid.pdy[y]+dose_grid.pdy[y+1]),
                         0.5*(dose_grid.pdz[z]+dose_grid.pdz[z+1]),1.0}, to[4];
         if ( TRPTrafoPointTransform(trafo,&tc,from,to,s,dc->psCT->iRefState,0)&CMDERR_MASK ) {
#pragma omp atomic write
            transform_error=1;
         }
         b.positions[i].x=to[0]; b.positions[i].y=to[1]; b.positions[i].z=to[2];
      }
      if ( transform_error ) { rc=CMDERR_ERROR|ERRTRPDOSE_PNTTRAFOFAILED; goto done; }
   }
   p.grid_voxel_count=(uint32_t)voxels; p.state_count=(uint32_t)states;
   p.field_count=fields; p.energy_count=energies; p.point_count=(uint32_t)point_count;
   p.ddd_table_count=energies; p.ddd_entry_count=(uint32_t)ddd_entries;
   p.bio_table_count=bio_tables; p.bio_entry_count=(uint32_t)bio_entries;
   p.voi_count=biological?(uint32_t)dc->psVOIBase->nV:0;
   p.hlut_count=(uint32_t)dc->psHLUT->nn;
   p.algorithm=dose_algorithm==TRPFLDDOSE_MULTIPLESCATTERDIVERGENT?
      CLINICAL_DOSE_ALGORITHM_MSDB:CLINICAL_DOSE_ALGORITHM_MS;
   p.biology_model=biological?CLINICAL_DOSE_BIOLOGY_LOW_DOSE:CLINICAL_DOSE_BIOLOGY_NONE;
   p.max_threads=(uint32_t)dc->nMaxThreads; p.struct_size=sizeof(p);
   p.ct_value_count=ct_values;
   p.ct_boundary_count=ct_boundaries;
   p.dose_grid=TRPClinicalDoseMojoGrid(&dose_grid); p.voxel_voi=b.voxel_voi;
   p.ct_states=b.ct_states; p.ct_data=b.ct_data;
   p.ct_boundaries=b.ct_boundaries;
   p.dose_axis_centers=b.dose_axis_centers;
   p.hlut_x=dc->psHLUT->pdCT2Path[TRPHLUT_X]; p.hlut_y=dc->psHLUT->pdCT2Path[TRPHLUT_Y];
   p.grid_fields=b.grid_fields; p.fields=b.fields; p.energies=b.energies; p.points=b.points;
   p.ddd_tables=b.ddd_tables; p.ddd_entries=b.ddd_entries;
   p.bio_tables=b.bio_tables; p.bio_entries=b.bio_entries;
   p.transformed_voxel_count=four_d?voxels*(uint64_t)states:0;
   p.states=four_d?b.states:NULL; p.transformed_voxels=b.positions;
   fprintf(stderr,"<I> Mojo clinical dose: backend=device states=%d voxels=%llu fields=%u\n",
           states,(unsigned long long)voxels,fields);
   rc=trip_compute_clinical_dose_on_device(&p,b.output,voxels);
   if ( rc ) { fprintf(stderr,"<E> Mojo clinical dose ABI failed: %d\n",rc);
      rc=CMDERR_ERROR|ERRTRPDOSE_BADFIELD; goto done; }
   double norm[TRPDOSE_MAX]; TRPDoseSetNorm(dc,norm);
   ssize_t offset[3]={0}; double corner[3]; TRPCubeIndex2Pos(&dose_grid,offset,corner);
   TRPCubePos2Index(main_cube,corner,offset);
   int store_error=0;
#pragma omp parallel for schedule(static)
   for ( uint64_t i=0; i<voxels; ++i ) if ( b.output[i].absorbed_dose>0.0 ) {
      int local[3]={(int)(i%dose_grid.nx),(int)((i/dose_grid.nx)%dose_grid.ny),(int)(i/(dose_grid.nx*dose_grid.ny))};
      int index[3]={local[0]+offset[0],local[1]+offset[1],local[2]+offset[2]};
      double value[TRPDOSE_MAX];
      int value_error=TRPDoseCalcStoreDose(dc,value,biological?b.voxel_voi[i]:-1,
            b.output[i].absorbed_dose,b.output[i].alpha,b.output[i].sqrt_beta,
            b.output[i].let_mix,b.output[i].let_bar,1.0,flag);
      if ( !value_error && (dc->ulDoseObj & TRPDOSE_BITDOSEMLET) ) {
         double dmlet[2]={b.output[i].let_dm_sum,b.output[i].let_bar};
         value_error=TRPDoseCalcStoreMeanLET(dc,value,b.output[i].let_mix,0.0,
                                             dmlet,dedx,flag);
      }
      if ( value_error ) {
#pragma omp atomic write
         store_error=1;
      } else for ( int d=0; d<TRPDOSE_MAX; ++d ) if ( dc->ulDoseObj&(1<<d) )
         if ( TRPDoseSetNormedDose(&dc->psD[d]->sCube,index,value[d],norm[d]) ) {
#pragma omp atomic write
            store_error=1;
         }
   }
   rc=store_error ? CMDERR_ERROR|ERRTRPDOSE_BADFIELD : 0;
done:
   if ( rs_base ) { for ( int f=0; f<base_fields; ++f ) free(rs_base[f]); free(rs_base); }
   free(spc_base); TRPCubeFree(&dose_grid,0); TRPClinicalDoseMojoFree(&b);
   return rc;
}
