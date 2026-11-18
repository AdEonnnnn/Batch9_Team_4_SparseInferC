/*
 * SparseInferC
 * Pointer-Based Sparse-Matrix Architecture for Energy-Efficient
 * Neural-Network Inference
 *
 * A fully connected network (784 -> 256 -> 128 -> 10, ReLU) is magnitude-pruned
 * to a chosen sparsity and run two ways:
 *   1. dense reference kernel   - fetches and multiplies every weight
 *   2. pointer-based CSR kernel - walks only the stored non-zero weights
 *
 * For every sparsity level the program checks correctness, counts MACs and
 * memory fetches, computes memory footprint and an analytic energy estimate,
 * and times both kernels.
 *
 * Build:  gcc -O2 -std=c11 -Wall -Wextra -o sparseinferc sparseinferc.c -lm
 * Run:    ./sparseinferc [results.csv]
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Configuration                                                       */
/* ------------------------------------------------------------------ */
#define NL 3
static const int SIZES[NL + 1] = {784, 256, 128, 10};
#define MAXW 256          /* widest hidden layer */
#define N_IN 200          /* inputs per timing / accuracy pass */
#define REPEATS 5         /* timing repeats, best one is reported */

/* Energy model, picojoules per operation. Approximate 45 nm figures
 * (M. Horowitz, ISSCC 2014): 32-bit float multiply 3.7, add 0.9,
 * 32-bit read from a small on-chip SRAM about 5, from DRAM about 640. */
#define E_MUL  3.7
#define E_ADD  0.9
#define E_SRAM 5.0
#define E_DRAM 640.0

/* ------------------------------------------------------------------ */
/* Data structures                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    int rows, cols, nnz;
    float *vals;      /* nnz non-zero weights, row after row            */
    int   *col_idx;   /* nnz column index of each stored weight         */
    int   *row_ptr;   /* rows+1 offsets: row r owns vals[row_ptr[r] ..
                         row_ptr[r+1]-1]                                */
} Csr;

typedef struct {
    uint64_t macs;    /* multiply-accumulates performed                 */
    uint64_t wfetch;  /* weight values fetched                          */
    uint64_t ifetch;  /* index words fetched (col_idx + row_ptr)        */
} Stats;

typedef struct {
    float *w[NL];     /* dense weights, pruned entries kept as zeros    */
    float *b[NL];     /* biases                                         */
    Csr   *s[NL];     /* same weights in CSR form                       */
} Net;

/* malloc wrapper: stop with a clear message instead of crashing on NULL */
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

/* ------------------------------------------------------------------ */
/* Sparse format                                                       */
/* ------------------------------------------------------------------ */
static Csr *csr_from_dense(const float *d, int rows, int cols) {
    Csr *m = xmalloc(sizeof *m);
    int nnz = 0;
    for (long i = 0; i < (long)rows * cols; i++) if (d[i] != 0.0f) nnz++;
    m->rows = rows; m->cols = cols; m->nnz = nnz;
    m->vals    = xmalloc(sizeof(float) * (size_t)(nnz ? nnz : 1));
    m->col_idx = xmalloc(sizeof(int)   * (size_t)(nnz ? nnz : 1));
    m->row_ptr = xmalloc(sizeof(int)   * (size_t)(rows + 1));
    int k = 0;
    for (int r = 0; r < rows; r++) {
        m->row_ptr[r] = k;
        for (int c = 0; c < cols; c++) {
            float v = d[(long)r * cols + c];
            if (v != 0.0f) { m->vals[k] = v; m->col_idx[k] = c; k++; }
        }
    }
    m->row_ptr[rows] = k;
    return m;
}

static void csr_free(Csr *m) {
    free(m->vals); free(m->col_idx); free(m->row_ptr); free(m);
}

static size_t csr_bytes(const Csr *m) {
    return sizeof(float) * (size_t)m->nnz + sizeof(int) * (size_t)m->nnz
         + sizeof(int) * (size_t)(m->rows + 1);
}

/* y = W x + b. The inner loop advances raw pointers through the stored
 * weights and their column indices; zero weights are never touched.   */
static void csr_matvec(const Csr *m, const float *x, const float *b,
                       float *y, Stats *st) {
    for (int r = 0; r < m->rows; r++) {
        float acc = b ? b[r] : 0.0f;
        const float *v   = m->vals    + m->row_ptr[r];
        const int   *c   = m->col_idx + m->row_ptr[r];
        const float *end = m->vals    + m->row_ptr[r + 1];
        if (st) {
            uint64_t n = (uint64_t)(end - v);
            st->macs += n; st->wfetch += n; st->ifetch += n;
        }
        for (; v < end; v++, c++) acc += *v * x[*c];
        y[r] = acc;
    }
    if (st) st->ifetch += (uint64_t)(m->rows + 1);
}

static void dense_matvec(const float *w, int rows, int cols, const float *x,
                         const float *b, float *y, Stats *st) {
    for (int r = 0; r < rows; r++) {
        float acc = b ? b[r] : 0.0f;
        const float *wr = w + (long)r * cols;
        if (st) { st->macs += (uint64_t)cols; st->wfetch += (uint64_t)cols; }
        for (int c = 0; c < cols; c++) acc += wr[c] * x[c];
        y[r] = acc;
    }
}

static void relu(float *v, int n) {
    for (int i = 0; i < n; i++) if (v[i] < 0.0f) v[i] = 0.0f;
}

/* ------------------------------------------------------------------ */
/* Network                                                             */
/* ------------------------------------------------------------------ */
static void infer(const Net *n, int sparse, const float *x, float *out,
                  Stats *st) {
    float a[MAXW], bb[MAXW];
    const float *in = x;
    for (int l = 0; l < NL; l++) {
        int last = (l == NL - 1);
        float *dst = last ? out : ((l % 2) ? bb : a);
        if (sparse) csr_matvec(n->s[l], in, n->b[l], dst, st);
        else dense_matvec(n->w[l], SIZES[l + 1], SIZES[l], in, n->b[l], dst, st);
        if (!last) relu(dst, SIZES[l + 1]);
        in = dst;
    }
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

/* Zero the smallest-magnitude fraction `sparsity` of the weights. */
static long prune_magnitude(float *w, long n, double sparsity) {
    long k = (long)floor(sparsity * (double)n + 0.5);
    if (k <= 0) return 0;
    if (k >= n) { memset(w, 0, sizeof(float) * (size_t)n); return n; }
    float *mag = xmalloc(sizeof(float) * (size_t)n);
    for (long i = 0; i < n; i++) mag[i] = fabsf(w[i]);
    qsort(mag, (size_t)n, sizeof(float), cmp_float);
    float thr = mag[k - 1];
    free(mag);
    long zeroed = 0;
    for (long i = 0; i < n && zeroed < k; i++)
        if (fabsf(w[i]) <= thr) { w[i] = 0.0f; zeroed++; }
    return zeroed;
}

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */
static uint32_t rng_state = 2463534242u;
static float frand(void) {                 /* uniform in (-1, 1), never 0 */
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    float v = ((rng_state >> 8) / 16777216.0f) * 2.0f - 1.0f;
    return v == 0.0f ? 1e-6f : v;
}

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static volatile double sink;

static double time_us_per_inference(const Net *n, int sparse, const float *X) {
    float out[10];
    double best = 1e30;
    for (int rep = 0; rep < REPEATS; rep++) {
        double t0 = now_ns();
        for (int i = 0; i < N_IN; i++) {
            infer(n, sparse, X + (long)i * SIZES[0], out, NULL);
            sink += out[0];
        }
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    return best / N_IN / 1000.0;
}

static int argmax10(const float *v) {
    int m = 0;
    for (int i = 1; i < 10; i++) if (v[i] > v[m]) m = i;
    return m;
}

/* ------------------------------------------------------------------ */
/* Unit tests                                                          */
/* ------------------------------------------------------------------ */
static int check(int cond, const char *name) {
    printf("  [%s] %s\n", cond ? "ok" : "FAIL", name);
    return cond ? 0 : 1;
}

static int unit_tests(void) {
    int fails = 0;
    printf("Unit tests\n");

    { /* known 2x3 example */
        float W[6] = {1, 0, 2,  0, 3, 0}, x[3] = {1, 2, 3}, b[2] = {0.5f, -1};
        float y[2];
        Csr *m = csr_from_dense(W, 2, 3);
        csr_matvec(m, x, b, y, NULL);
        fails += check(m->nnz == 3 && fabsf(y[0] - 7.5f) < 1e-6f
                       && fabsf(y[1] - 5.0f) < 1e-6f, "known 2x3 product");
        csr_free(m);
    }
    { /* identity */
        float I[9] = {1,0,0, 0,1,0, 0,0,1}, x[3] = {4, -5, 6}, y[3];
        Csr *m = csr_from_dense(I, 3, 3);
        csr_matvec(m, x, NULL, y, NULL);
        fails += check(y[0] == 4 && y[1] == -5 && y[2] == 6, "identity matrix");
        csr_free(m);
    }
    { /* all-zero matrix returns the bias */
        float Z[6] = {0}, x[3] = {1, 2, 3}, b[2] = {7, 8}, y[2];
        Csr *m = csr_from_dense(Z, 2, 3);
        csr_matvec(m, x, b, y, NULL);
        fails += check(m->nnz == 0 && y[0] == 7 && y[1] == 8, "all-zero matrix");
        csr_free(m);
    }
    { /* empty middle row */
        float W[9] = {1,2,3, 0,0,0, 4,5,6}, x[3] = {1, 1, 1}, y[3];
        Csr *m = csr_from_dense(W, 3, 3);
        csr_matvec(m, x, NULL, y, NULL);
        fails += check(y[0] == 6 && y[1] == 0 && y[2] == 15, "empty middle row");
        csr_free(m);
    }
    { /* 1x1 */
        float W[1] = {2.5f}, x[1] = {4}, y[1];
        Csr *m = csr_from_dense(W, 1, 1);
        csr_matvec(m, x, NULL, y, NULL);
        fails += check(y[0] == 10.0f, "1x1 matrix");
        csr_free(m);
    }
    { /* pruning hits the requested count exactly */
        float w[100];
        for (int i = 0; i < 100; i++) w[i] = frand();
        long z = prune_magnitude(w, 100, 0.75);
        int zeros = 0;
        for (int i = 0; i < 100; i++) if (w[i] == 0.0f) zeros++;
        fails += check(z == 75 && zeros == 75, "magnitude pruning count");
    }
    printf("  %s\n\n", fails ? "SOME UNIT TESTS FAILED" : "all unit tests passed");
    return fails;
}

/* ------------------------------------------------------------------ */
/* Benchmark                                                           */
/* ------------------------------------------------------------------ */
typedef struct {
    double sparsity;
    long nnz, total;
    uint64_t macs_d, macs_s;
    size_t mem_d, mem_s;
    double e_dram_d, e_dram_s, e_sram_d, e_sram_s;   /* microjoules */
    double t_d, t_s;                                 /* microseconds */
    double max_err;
    int match;
} Row;

int main(int argc, char **argv) {
    const char *csv_path = argc > 1 ? argv[1] : "results.csv";
    int fails = unit_tests();

    /* One fixed random network and one fixed set of inputs, reused for every
     * sparsity level so that only the pruning changes between rows.        */
    float *base_w[NL], *base_b[NL];
    for (int l = 0; l < NL; l++) {
        long n = (long)SIZES[l + 1] * SIZES[l];
        float scale = sqrtf(2.0f / (float)SIZES[l]);
        base_w[l] = xmalloc(sizeof(float) * (size_t)n);
        base_b[l] = xmalloc(sizeof(float) * (size_t)SIZES[l + 1]);
        for (long i = 0; i < n; i++) base_w[l][i] = frand() * scale;
        for (int i = 0; i < SIZES[l + 1]; i++) base_b[l][i] = frand() * 0.05f;
    }
    float *X = xmalloc(sizeof(float) * (size_t)N_IN * SIZES[0]);
    for (long i = 0; i < (long)N_IN * SIZES[0]; i++) X[i] = (frand() + 1.0f) * 0.5f;

    long total = 0;
    for (int l = 0; l < NL; l++) total += (long)SIZES[l + 1] * SIZES[l];

    const double levels[] = {0, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99};
    const int NLEV = (int)(sizeof levels / sizeof *levels);
    Row rows[16];

    printf("SparseInferC benchmark: 784 -> 256 -> 128 -> 10 MLP, %ld weights\n",
           total);
    printf("%d inputs per pass, best of %d passes\n\n", N_IN, REPEATS);
    printf("%5s %8s | %9s %9s | %8s %8s | %9s %9s | %8s %8s %7s | %8s %5s\n",
           "spars", "nnz", "MACs den", "MACs sp", "KB den", "KB sp",
           "uJ den*", "uJ sp*", "us den", "us sp", "speedup", "max err", "argm");

    for (int li = 0; li < NLEV; li++) {
        Net net;
        long nnz = 0;
        size_t mem_s = 0, mem_d = 0;
        for (int l = 0; l < NL; l++) {
            long n = (long)SIZES[l + 1] * SIZES[l];
            net.w[l] = xmalloc(sizeof(float) * (size_t)n);
            memcpy(net.w[l], base_w[l], sizeof(float) * (size_t)n);
            net.b[l] = base_b[l];
            prune_magnitude(net.w[l], n, levels[li]);
            net.s[l] = csr_from_dense(net.w[l], SIZES[l + 1], SIZES[l]);
            nnz   += net.s[l]->nnz;
            mem_s += csr_bytes(net.s[l]);
            mem_d += sizeof(float) * (size_t)n;
        }

        /* correctness: dense vs sparse on every input */
        double max_err = 0; int match = 0;
        for (int i = 0; i < N_IN; i++) {
            float yd[10], ys[10];
            const float *x = X + (long)i * SIZES[0];
            infer(&net, 0, x, yd, NULL);
            infer(&net, 1, x, ys, NULL);
            for (int k = 0; k < 10; k++) {
                double e = fabs((double)yd[k] - (double)ys[k]);
                if (e > max_err) max_err = e;
            }
            if (argmax10(yd) == argmax10(ys)) match++;
        }

        /* operation counts for ONE inference, measured inside the kernels */
        Stats sd = {0, 0, 0}, ss = {0, 0, 0};
        float tmp[10];
        infer(&net, 0, X, tmp, &sd);
        infer(&net, 1, X, tmp, &ss);

        double mac_pj = E_MUL + E_ADD;
        Row *r = &rows[li];
        r->sparsity = levels[li]; r->nnz = nnz; r->total = total;
        r->macs_d = sd.macs; r->macs_s = ss.macs;
        r->mem_d = mem_d; r->mem_s = mem_s;
        r->e_dram_d = ((double)sd.macs * mac_pj + (double)(sd.wfetch + sd.ifetch) * E_DRAM) * 1e-6;
        r->e_dram_s = ((double)ss.macs * mac_pj + (double)(ss.wfetch + ss.ifetch) * E_DRAM) * 1e-6;
        r->e_sram_d = ((double)sd.macs * mac_pj + (double)(sd.wfetch + sd.ifetch) * E_SRAM) * 1e-6;
        r->e_sram_s = ((double)ss.macs * mac_pj + (double)(ss.wfetch + ss.ifetch) * E_SRAM) * 1e-6;
        r->t_d = time_us_per_inference(&net, 0, X);
        r->t_s = time_us_per_inference(&net, 1, X);
        r->max_err = max_err; r->match = match;
        if (max_err > 1e-4 || match != N_IN) fails++;

        printf("%4.0f%% %8ld | %9llu %9llu | %8.1f %8.1f | %9.1f %9.1f | %8.1f %8.1f %6.2fx | %8.1e %3d/%d\n",
               levels[li] * 100, nnz,
               (unsigned long long)r->macs_d, (unsigned long long)r->macs_s,
               (double)mem_d / 1024.0, (double)mem_s / 1024.0,
               r->e_dram_d, r->e_dram_s, r->t_d, r->t_s, r->t_d / r->t_s,
               max_err, match, N_IN);

        for (int l = 0; l < NL; l++) { free(net.w[l]); csr_free(net.s[l]); }
    }
    printf("* energy per inference, weights streamed from DRAM (%.0f pJ per 32-bit word)\n",
           E_DRAM);

    /* analytic break-even sparsity: a stored weight costs one MAC, one value
     * fetch and one index fetch; a dense weight costs one MAC and one fetch */
    double mac = E_MUL + E_ADD;
    double be_dram = 1.0 - (mac + E_DRAM) / (mac + 2.0 * E_DRAM);
    double be_sram = 1.0 - (mac + E_SRAM) / (mac + 2.0 * E_SRAM);
    printf("\nAnalytic break-even sparsity (energy): DRAM %.1f%%, on-chip SRAM %.1f%%\n",
           be_dram * 100, be_sram * 100);
    printf("Analytic break-even sparsity (memory footprint): 50.0%% (value + 32-bit index per weight)\n");
    for (int li = 0; li < NLEV; li++)
        if (rows[li].t_d / rows[li].t_s > 1.0) {
            printf("Measured run-time break-even: first faster at %.0f%% sparsity\n",
                   rows[li].sparsity * 100);
            break;
        }

    FILE *f = fopen(csv_path, "w");
    if (f) {
        fprintf(f, "sparsity_pct,nnz,total_weights,macs_dense,macs_sparse,mem_dense_bytes,mem_sparse_bytes,"
                   "energy_dram_dense_uJ,energy_dram_sparse_uJ,energy_sram_dense_uJ,energy_sram_sparse_uJ,"
                   "time_dense_us,time_sparse_us,speedup,max_abs_err,argmax_match\n");
        for (int li = 0; li < NLEV; li++) {
            Row *r = &rows[li];
            fprintf(f, "%.0f,%ld,%ld,%llu,%llu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.3e,%d\n",
                    r->sparsity * 100, r->nnz, r->total,
                    (unsigned long long)r->macs_d, (unsigned long long)r->macs_s,
                    r->mem_d, r->mem_s, r->e_dram_d, r->e_dram_s, r->e_sram_d, r->e_sram_s,
                    r->t_d, r->t_s, r->t_d / r->t_s, r->max_err, r->match);
        }
        fclose(f);
        printf("\nResults written to %s\n", csv_path);
    }

    printf("\n%s\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    for (int l = 0; l < NL; l++) { free(base_w[l]); free(base_b[l]); }
    free(X);
    return fails ? 1 : 0;
}

