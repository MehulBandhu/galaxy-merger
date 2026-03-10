#include "../include/galaxy_types.h"
#include <string.h>

// CPU octree

typedef struct {
    float cx, cy, cz;     // centre of mass
    float mass;
    float size;            // half-width
    float ox, oy, oz;     // geometric centre
    int children[8];
    int particle_start;
    int particle_count;
    int is_leaf;
} TreeNode;

static TreeNode *h_tree;
static int tree_count;
static int max_tree_nodes;

void tree_init(int max_nodes) {
    max_tree_nodes = max_nodes;
    h_tree = (TreeNode*)malloc(sizeof(TreeNode) * max_nodes);
}

void tree_free() { free(h_tree); }

static int tree_new_node(float ox, float oy, float oz, float size) {
    if (tree_count >= max_tree_nodes) {
        fprintf(stderr, "tree overflow: %d/%d\n", tree_count, max_tree_nodes);
        return -1;
    }
    int idx = tree_count++;
    TreeNode *n = &h_tree[idx];
    n->ox = ox; n->oy = oy; n->oz = oz;
    n->size = size;
    n->cx = 0; n->cy = 0; n->cz = 0;
    n->mass = 0;
    for (int i = 0; i < 8; i++) n->children[i] = -1;
    n->particle_start = -1;
    n->particle_count = 0;
    n->is_leaf = 1;
    return idx;
}

static int get_octant(float px, float py, float pz, float ox, float oy, float oz) {
    int oct = 0;
    if (px > ox) oct |= 1;
    if (py > oy) oct |= 2;
    if (pz > oz) oct |= 4;
    return oct;
}

static void child_origin(int octant, float pox, float poy, float poz,
                          float hs, float *cx, float *cy, float *cz) {
    *cx = pox + ((octant & 1) ? hs : -hs);
    *cy = poy + ((octant & 2) ? hs : -hs);
    *cz = poz + ((octant & 4) ? hs : -hs);
}

static void tree_insert(int node_idx, int pidx,
                         float px, float py, float pz, float pm,
                         float *ax, float *ay, float *az, float *am,
                         int depth) {
    if (depth > 40) return;
    TreeNode *node = &h_tree[node_idx];

    // empty leaf
    if (node->is_leaf && node->particle_count == 0) {
        node->particle_start = pidx;
        node->particle_count = 1;
        node->cx = px; node->cy = py; node->cz = pz;
        node->mass = pm;
        return;
    }

    // occupied leaf: split
    if (node->is_leaf && node->particle_count == 1) {
        node->is_leaf = 0;
        int oi = node->particle_start;
        float ox = ax[oi], oy = ay[oi], oz = az[oi], om = am[oi];
        node->particle_start = -1;
        node->particle_count = 0;

        float hs = node->size * 0.5f;
        int oct = get_octant(ox, oy, oz, node->ox, node->oy, node->oz);
        if (node->children[oct] < 0) {
            float cx, cy, cz;
            child_origin(oct, node->ox, node->oy, node->oz, hs, &cx, &cy, &cz);
            node->children[oct] = tree_new_node(cx, cy, cz, hs);
        }
        tree_insert(node->children[oct], oi, ox, oy, oz, om, ax, ay, az, am, depth+1);
    }

    // insert into child
    float hs = node->size * 0.5f;
    int oct = get_octant(px, py, pz, node->ox, node->oy, node->oz);
    if (node->children[oct] < 0) {
        float cx, cy, cz;
        child_origin(oct, node->ox, node->oy, node->oz, hs, &cx, &cy, &cz);
        node->children[oct] = tree_new_node(cx, cy, cz, hs);
    }
    tree_insert(node->children[oct], pidx, px, py, pz, pm, ax, ay, az, am, depth+1);

    // update centre of mass
    float nm = node->mass + pm;
    if (nm > 0) {
        node->cx = (node->cx * node->mass + px * pm) / nm;
        node->cy = (node->cy * node->mass + py * pm) / nm;
        node->cz = (node->cz * node->mass + pz * pm) / nm;
    }
    node->mass = nm;
}

int build_tree(float *x, float *y, float *z, float *mass, int N, float box_half) {
    tree_count = 0;
    tree_new_node(0, 0, 0, box_half);
    for (int i = 0; i < N; i++)
        tree_insert(0, i, x[i], y[i], z[i], mass[i], x, y, z, mass, 0);
    return tree_count;
}

// GPU force evaluation

typedef struct {
    float cx, cy, cz, mass, size;
    int children[8];
    int is_leaf;
    int particle_start;
} GPUTreeNode;

__device__ void tree_walk(float px, float py, float pz, float eps,
                          const GPUTreeNode *tree, int root, float theta2,
                          float *oax, float *oay, float *oaz) {
    int stack[64];
    int sp = 0;
    stack[sp++] = root;
    float ax = 0, ay = 0, az = 0;

    while (sp > 0) {
        int idx = stack[--sp];
        if (idx < 0) continue;
        GPUTreeNode nd = tree[idx];
        if (nd.mass == 0) continue;

        float dx = nd.cx - px;
        float dy = nd.cy - py;
        float dz = nd.cz - pz;
        float r2 = dx*dx + dy*dy + dz*dz + eps*eps;
        float s2 = 4.0f * nd.size * nd.size;

        if (!nd.is_leaf && s2 > theta2 * r2) {
            for (int c = 0; c < 8; c++)
                if (nd.children[c] >= 0 && sp < 63)
                    stack[sp++] = nd.children[c];
        } else {
            float ri = rsqrtf(r2);
            float r3i = ri * ri * ri;
            float mf = G_INTERNAL * nd.mass;
            ax += mf * dx * r3i;
            ay += mf * dy * r3i;
            az += mf * dz * r3i;
        }
    }
    *oax = ax; *oay = ay; *oaz = az;
}

__global__ void gravity_kernel(float *x, float *y, float *z,
                                float *ax, float *ay, float *az,
                                float *eps, GPUTreeNode *tree,
                                float theta2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float fax, fay, faz;
    tree_walk(x[i], y[i], z[i], eps[i], tree, 0, theta2, &fax, &fay, &faz);
    ax[i] = fax; ay[i] = fay; az[i] = faz;
}

// host interface 

static GPUTreeNode *d_tree = NULL;
static int d_tree_alloc = 0;

void gravity_compute(float *d_x, float *d_y, float *d_z,
                     float *d_ax, float *d_ay, float *d_az,
                     float *d_eps, float *d_mass,
                     float *h_x, float *h_y, float *h_z, float *h_mass,
                     int N, float box_half, float theta) {
    int nn = build_tree(h_x, h_y, h_z, h_mass, N, box_half);

    GPUTreeNode *h_gpu = (GPUTreeNode*)malloc(sizeof(GPUTreeNode) * nn);
    for (int i = 0; i < nn; i++) {
        h_gpu[i].cx = h_tree[i].cx; h_gpu[i].cy = h_tree[i].cy; h_gpu[i].cz = h_tree[i].cz;
        h_gpu[i].mass = h_tree[i].mass; h_gpu[i].size = h_tree[i].size;
        h_gpu[i].is_leaf = h_tree[i].is_leaf;
        h_gpu[i].particle_start = h_tree[i].particle_start;
        for (int c = 0; c < 8; c++) h_gpu[i].children[c] = h_tree[i].children[c];
    }

    if (nn > d_tree_alloc) {
        if (d_tree) cudaFree(d_tree);
        cudaMalloc(&d_tree, sizeof(GPUTreeNode) * nn * 2);
        d_tree_alloc = nn * 2;
    }
    cudaMemcpy(d_tree, h_gpu, sizeof(GPUTreeNode) * nn, cudaMemcpyHostToDevice);
    free(h_gpu);

    int blk = 256;
    int grd = (N + blk - 1) / blk;
    gravity_kernel<<<grd, blk>>>(d_x, d_y, d_z, d_ax, d_ay, d_az,
                                  d_eps, d_tree, theta * theta, N);
    cudaDeviceSynchronize();
}

void gravity_cleanup() {
    if (d_tree) cudaFree(d_tree);
    d_tree = NULL; d_tree_alloc = 0;
}
