#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "trprst.h"
#include "trpddd.h"
#include "trpdedx.h"

double tripffi_rst_pnt_dose_distance_dr2(
    double point_x,
    double point_y,
    double f2_max,
    double voxel_x,
    double voxel_y,
    double div_x,
    double div_y
) {
    struct STRPRSTPNT point;
    memset(&point, 0, sizeof(point));
    point.dv[0] = point_x;
    point.dv[1] = point_y;
    point.dF2Max = f2_max;

    double dvr[2] = {voxel_x, voxel_y};
    double div[2] = {div_x, div_y};
    double dr2 = 0.0;
    int rc = TRPRstPntDoseDistance(&point, dvr, &dr2, div);
    if (rc != 0) {
        return -1.0;
    }
    return dr2;
}

double tripffi_ddd2_interp_col(
    double z0,
    double d0,
    double fwhm10,
    double mix0,
    double fwhm20,
    double z1,
    double d1,
    double fwhm11,
    double mix1,
    double fwhm21,
    double depth_cm,
    int column
) {
    double z[2] = {z0, z1};
    double dose[2] = {d0, d1};
    double fwhm1[2] = {fwhm10, fwhm11};
    double mix[2] = {mix0, mix1};
    double fwhm2[2] = {fwhm20, fwhm21};
    double out[TRPDDD_MAXTAB];

    struct STRPDDD ddd;
    memset(&ddd, 0, sizeof(ddd));
    ddd.nz = 2;
    ddd.nCol = 5;
    ddd.pdTab[TRPDDD_X] = z;
    ddd.pdTab[TRPDDD_D] = dose;
    ddd.pdTab[TRPDDD_FWHM1] = fwhm1;
    ddd.pdTab[TRPDDD_G] = mix;
    ddd.pdTab[TRPDDD_FWHM2] = fwhm2;

    int rc = TRPDDDTabInterpolate(&ddd, depth_cm, out, TRPDDD_QUIET);
    if (rc != 0 || column < 0 || column >= TRPDDD_MAXTAB) {
        return -1.0;
    }
    return out[column];
}

double tripffi_dedx_eval(
    int z,
    int a,
    double energy
) {
    const char *path = getenv("TRIPFFI_DEDX_PATH");
    if (path == NULL) {
        return -1.0;
    }
    struct STRPDEDX dedx;
    TRPdEdxInit(&dedx, 0);
    int rc = TRPdEdxRead(&dedx, path, TRPDEDX_QUIET);
    if (rc != 0) {
        TRPdEdxFree(&dedx, 0);
        return -1.0;
    }
    struct STRPDEDXEVAL eval;
    memset(&eval, 0, sizeof(eval));
    eval.dE = energy;
    eval.sZA.dZ = (double)z;
    eval.sZA.lZ = z;
    eval.sZA.lA = a;
    eval.sZA.TRPZA_A = (double)a;
    rc = TRPdEdxEval(&dedx, &eval, 1, TRPDEDX_QUIET);
    TRPdEdxFree(&dedx, 0);
    if (rc != 0) {
        return -1.0;
    }
    return eval.ddE;
}
