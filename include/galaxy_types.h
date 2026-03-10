#ifndef GALAXY_TYPES_H
#define GALAXY_TYPES_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Units: 1 kpc, 1e10 Msun, 1 km/s
// G = 43007 kpc (km/s)^2 / (1e10 Msun)
// Time unit: 1 kpc / (1 km/s) = 0.978 Gyr
#define G_INTERNAL 43007.0f
#define TIME_UNIT 0.978f
#define THETA_BH 0.7f

#define PTYPE_DM    0
#define PTYPE_DISK  1
#define PTYPE_BULGE 2
#define PTYPE_GAS   3
#define PTYPE_STAR  4
#define PTYPE_BH    5

typedef struct {
    int N_total;
    int N_gas;
    int N_dm;
    int N_disk;
    int N_bulge;
    int N_star_new;
    int N_bh;
    float time;
    float dt;
    int snap_num;
} SnapHeader;

#endif
