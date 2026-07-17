#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

enum { BUFFER_FLOATS = 1 << 20 };

static void fail(const char *message) {
    perror(message);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s reference.dos candidate.dos\n", argv[0]);
        return EXIT_FAILURE;
    }
    struct stat reference_stat, candidate_stat;
    if (stat(argv[1], &reference_stat) || stat(argv[2], &candidate_stat)) fail("stat");
    if (reference_stat.st_size != candidate_stat.st_size ||
        reference_stat.st_size % sizeof(float) != 0) {
        fprintf(stderr, "cube byte sizes differ or are not Float32\n");
        return EXIT_FAILURE;
    }
    FILE *reference = fopen(argv[1], "rb");
    FILE *candidate = fopen(argv[2], "rb");
    if (!reference || !candidate) fail("fopen");
    float *a = malloc(BUFFER_FLOATS * sizeof(*a));
    float *b = malloc(BUFFER_FLOATS * sizeof(*b));
    if (!a || !b) fail("malloc");

    uint64_t offset = 0, reference_nonzero = 0, candidate_nonzero = 0, union_nonzero = 0;
    uint64_t reference_only = 0, candidate_only = 0;
    uint64_t different = 0, over_millidose = 0, over_tenth = 0, over_one = 0;
    uint64_t max_index = 0;
    double error2 = 0.0, reference2 = 0.0, max_abs = 0.0;
    double reference_sum = 0.0, candidate_sum = 0.0, candidate_only_sum = 0.0;
    double overlap_error2 = 0.0;
    for (;;) {
        size_t count = fread(a, sizeof(*a), BUFFER_FLOATS, reference);
        if (fread(b, sizeof(*b), count, candidate) != count) fail("fread");
        if (count == 0) break;
        for (size_t i = 0; i < count; ++i) {
            double av = a[i], bv = b[i], delta = bv - av, absolute = fabs(delta);
            reference_nonzero += av != 0.0;
            candidate_nonzero += bv != 0.0;
            union_nonzero += av != 0.0 || bv != 0.0;
            reference_only += av != 0.0 && bv == 0.0;
            candidate_only += av == 0.0 && bv != 0.0;
            candidate_only_sum += av == 0.0 ? bv : 0.0;
            different += av != bv;
            over_millidose += absolute > 1.0e-3;
            over_tenth += absolute > 0.1;
            over_one += absolute > 1.0;
            overlap_error2 += av != 0.0 && bv != 0.0 ? delta * delta : 0.0;
            reference2 += av * av;
            error2 += delta * delta;
            reference_sum += av;
            candidate_sum += bv;
            if (absolute > max_abs) {
                max_abs = absolute;
                max_index = offset + i;
            }
        }
        offset += count;
    }
    if (ferror(reference) || ferror(candidate)) fail("fread");
    printf("voxels=%llu reference_nonzero=%llu candidate_nonzero=%llu union_nonzero=%llu "
           "relative_l2=%.17g max_abs=%.17g max_index=%llu "
           "reference_only=%llu candidate_only=%llu overlap_relative_l2=%.17g "
           "different=%llu abs_gt_1e-3=%llu abs_gt_0.1=%llu abs_gt_1=%llu "
           "reference_sum=%.17g candidate_sum=%.17g candidate_only_sum=%.17g "
           "sum_ratio=%.17g\n",
           (unsigned long long)offset,
           (unsigned long long)reference_nonzero,
           (unsigned long long)candidate_nonzero,
           (unsigned long long)union_nonzero,
           reference2 > 0.0 ? sqrt(error2 / reference2) : sqrt(error2),
           max_abs, (unsigned long long)max_index,
           (unsigned long long)reference_only,
           (unsigned long long)candidate_only,
           reference2 > 0.0 ? sqrt(overlap_error2 / reference2) : sqrt(overlap_error2),
           (unsigned long long)different,
           (unsigned long long)over_millidose,
           (unsigned long long)over_tenth,
           (unsigned long long)over_one,
           reference_sum, candidate_sum, candidate_only_sum,
           reference_sum != 0.0 ? candidate_sum / reference_sum : 0.0);
    free(a);
    free(b);
    fclose(reference);
    fclose(candidate);
    return EXIT_SUCCESS;
}
