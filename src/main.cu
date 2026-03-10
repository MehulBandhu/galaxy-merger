#include "../include/galaxy_types.h"
#include <sys/stat.h>

extern void gravity_compute(float *d_x, float *d_y, float *d_z,
                             float *d_ax, float *d_ay, float *d_az,
                             float *d_eps, float *d_mass,
                             float *h_x, float *h_y, float *h_z, float *h_mass,
                             int N, float box_half, float theta);
extern void gravity_cleanup();
extern void tree_init(int max_nodes);
extern void tree_free();

__global__ void kick(float *vx, float *vy, float *vz,
                     float *ax, float *ay, float *az,
                     int N, float hdt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    vx[i] += ax[i] * hdt;
    vy[i] += ay[i] * hdt;
    vz[i] += az[i] * hdt;
}

__global__ void drift(float *x, float *y, float *z,
                      float *vx, float *vy, float *vz,
                      int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
}

void write_snapshot(const char *dir, int num,
                    float *x, float *y, float *z,
                    float *vx, float *vy, float *vz,
                    float *mass, int *ptype, int *gid,
                    int N, int N_gas, float time, float dt) {
    char fn[256];
    snprintf(fn, sizeof(fn), "%s/snap_%04d.bin", dir, num);
    FILE *f = fopen(fn, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); return; }

    SnapHeader h;
    h.N_total = N; h.N_gas = N_gas; h.time = time; h.dt = dt; h.snap_num = num;
    h.N_dm = 0; h.N_disk = 0; h.N_bulge = 0; h.N_star_new = 0; h.N_bh = 0;
    for (int i = 0; i < N; i++) {
        switch (ptype[i]) {
            case PTYPE_DM: h.N_dm++; break;
            case PTYPE_DISK: h.N_disk++; break;
            case PTYPE_BULGE: h.N_bulge++; break;
            case PTYPE_GAS: h.N_gas++; break;
            case PTYPE_STAR: h.N_star_new++; break;
            case PTYPE_BH: h.N_bh++; break;
        }
    }
    fwrite(&h, sizeof(SnapHeader), 1, f);
    fwrite(x, 4, N, f); fwrite(y, 4, N, f); fwrite(z, 4, N, f);
    fwrite(vx, 4, N, f); fwrite(vy, 4, N, f); fwrite(vz, 4, N, f);
    fwrite(mass, 4, N, f);

    // pad: u and rho (zeros for gravity-only, keeps format compatible)
    float *zeros = (float*)calloc(N, sizeof(float));
    fwrite(zeros, 4, N, f); // u
    fwrite(zeros, 4, N, f); // rho
    free(zeros);

    fwrite(ptype, 4, N, f);
    fwrite(gid, 4, N, f);
    fclose(f);
    printf("  snap %04d  t=%.0f Myr\n", num, time * TIME_UNIT * 1000);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <ic.bin> [outdir] [t_end] [dt_snap] [dt]\n", argv[0]);
        return 1;
    }
    const char *ic_file = argv[1];
    const char *out_dir = argc > 2 ? argv[2] : "snapshots";
    float t_end   = argc > 3 ? atof(argv[3]) : 1.0f;
    float dt_snap  = argc > 4 ? atof(argv[4]) : 0.01f;
    float dt       = argc > 5 ? atof(argv[5]) : 0.0005f;

    mkdir(out_dir, 0755);

    // load IC
    FILE *f = fopen(ic_file, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", ic_file); return 1; }
    SnapHeader hdr;
    if (fread(&hdr, sizeof(SnapHeader), 1, f) != 1) return 1;
    int N = hdr.N_total;
    int N_gas = hdr.N_gas;
    printf("loaded %d particles\n", N);

    size_t fs = N * sizeof(float);
    size_t is = N * sizeof(int);
    float *h_x = (float*)malloc(fs), *h_y = (float*)malloc(fs), *h_z = (float*)malloc(fs);
    float *h_vx = (float*)malloc(fs), *h_vy = (float*)malloc(fs), *h_vz = (float*)malloc(fs);
    float *h_mass = (float*)malloc(fs);
    float *h_eps = (float*)malloc(fs);
    int *h_ptype = (int*)malloc(is), *h_gid = (int*)malloc(is);

    size_t nr = 0;
    nr += fread(h_x, 4, N, f);  nr += fread(h_y, 4, N, f);  nr += fread(h_z, 4, N, f);
    nr += fread(h_vx, 4, N, f); nr += fread(h_vy, 4, N, f); nr += fread(h_vz, 4, N, f);
    nr += fread(h_mass, 4, N, f);
    // skip u and rho
    float *tmp = (float*)malloc(fs);
    fread(tmp, 4, N, f); fread(tmp, 4, N, f);
    free(tmp);
    nr += fread(h_ptype, 4, N, f);
    nr += fread(h_gid, 4, N, f);
    fclose(f);
    printf("read %zu fields\n", nr / N);

    // softening: 100 pc for dark matter, 50 pc for baryons
    for (int i = 0; i < N; i++) {
        switch (h_ptype[i]) {
            case PTYPE_DM:    h_eps[i] = 0.1f;  break;
            case PTYPE_DISK:  h_eps[i] = 0.05f; break;
            case PTYPE_BULGE: h_eps[i] = 0.05f; break;
            case PTYPE_GAS:   h_eps[i] = 0.05f; break;
            case PTYPE_BH:    h_eps[i] = 0.05f; break;
            default:          h_eps[i] = 0.1f;
        }
    }

    // GPU allocation
    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_ax, *d_ay, *d_az, *d_mass, *d_eps;
    cudaMalloc(&d_x, fs);  cudaMalloc(&d_y, fs);  cudaMalloc(&d_z, fs);
    cudaMalloc(&d_vx, fs); cudaMalloc(&d_vy, fs); cudaMalloc(&d_vz, fs);
    cudaMalloc(&d_ax, fs); cudaMalloc(&d_ay, fs); cudaMalloc(&d_az, fs);
    cudaMalloc(&d_mass, fs); cudaMalloc(&d_eps, fs);

    cudaMemcpy(d_x, h_x, fs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, fs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, fs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, fs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, fs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, fs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, fs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_eps, h_eps, fs, cudaMemcpyHostToDevice);
    cudaMemset(d_ax, 0, fs); cudaMemset(d_ay, 0, fs); cudaMemset(d_az, 0, fs);

    float gpu_mb = (fs * 11) / (1024.0f * 1024.0f);
    printf("GPU memory: %.0f MB\n", gpu_mb);

    tree_init(N * 4);
    float box_half = 3000.0f;
    int blk = 256, grd = (N + blk - 1) / blk;

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);

    // initial snapshot
    write_snapshot(out_dir, 0, h_x, h_y, h_z, h_vx, h_vy, h_vz,
                   h_mass, h_ptype, h_gid, N, N_gas, 0, dt);

    // initial forces
    gravity_compute(d_x, d_y, d_z, d_ax, d_ay, d_az, d_eps, d_mass,
                    h_x, h_y, h_z, h_mass, N, box_half, THETA_BH);

    float t = 0, next_snap = dt_snap, total_ms = 0;
    int step = 0, snap = 1;

    printf("\nrunning %.0f Myr, dt=%.1f Myr, saving every %.1f Myr\n\n",
           t_end * TIME_UNIT * 1000, dt * TIME_UNIT * 1000, dt_snap * TIME_UNIT * 1000);

    while (t < t_end) {
        cudaEventRecord(ev0);

        kick<<<grd, blk>>>(d_vx, d_vy, d_vz, d_ax, d_ay, d_az, N, 0.5f*dt);
        drift<<<grd, blk>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, N, dt);

        cudaMemcpy(h_x, d_x, fs, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_y, d_y, fs, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_z, d_z, fs, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mass, d_mass, fs, cudaMemcpyDeviceToHost);

        cudaMemset(d_ax, 0, fs); cudaMemset(d_ay, 0, fs); cudaMemset(d_az, 0, fs);
        gravity_compute(d_x, d_y, d_z, d_ax, d_ay, d_az, d_eps, d_mass,
                        h_x, h_y, h_z, h_mass, N, box_half, THETA_BH);

        kick<<<grd, blk>>>(d_vx, d_vy, d_vz, d_ax, d_ay, d_az, N, 0.5f*dt);

        t += dt; step++;

        cudaEventRecord(ev1); cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
        total_ms += ms;

        if (t >= next_snap) {
            cudaMemcpy(h_vx, d_vx, fs, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vy, d_vy, fs, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vz, d_vz, fs, cudaMemcpyDeviceToHost);
            write_snapshot(out_dir, snap++, h_x, h_y, h_z, h_vx, h_vy, h_vz,
                           h_mass, h_ptype, h_gid, N, N_gas, t, dt);
            next_snap += dt_snap;
        }
        if (step % 200 == 0) {
            float avg = total_ms / step;
            float eta_h = avg * ((t_end - t) / dt) / 3.6e6f;
            printf("step %6d  t=%.0f Myr  %.1f ms/step  ETA %.1f h\n",
                   step, t * TIME_UNIT * 1000, avg, eta_h);
        }
    }

    printf("\nfinished: %d steps, %d snapshots, %.1f ms/step, %.1f min time\n",
           step, snap, total_ms/step, total_ms/60000);

    tree_free(); gravity_cleanup();
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    cudaFree(d_mass); cudaFree(d_eps);
    free(h_x); free(h_y); free(h_z);
    free(h_vx); free(h_vy); free(h_vz);
    free(h_mass); free(h_eps); free(h_ptype); free(h_gid);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
