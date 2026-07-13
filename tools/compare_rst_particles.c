#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t layer;
    double x, y, particles;
} Spot;

typedef struct {
    Spot *spots;
    size_t count, capacity;
} Plan;

static void fail(const char *message, const char *path) {
    fprintf(stderr, "%s: %s\n", message, path);
    exit(EXIT_FAILURE);
}

static void append(Plan *plan, Spot spot) {
    if (plan->count == plan->capacity) {
        size_t capacity = plan->capacity ? plan->capacity * 2 : 1024;
        Spot *spots = realloc(plan->spots, capacity * sizeof(*spots));
        if (!spots) fail("allocation failed", strerror(errno));
        plan->spots = spots;
        plan->capacity = capacity;
    }
    plan->spots[plan->count++] = spot;
}

static Plan read_plan(const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) fail("cannot open", path);
    Plan plan = {0};
    char line[512];
    uint32_t layer = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "submachine#", 11) == 0) {
            ++layer;
            continue;
        }
        unsigned count;
        if (sscanf(line, "#points %u", &count) != 1) continue;
        for (unsigned i = 0; i < count; ++i) {
            Spot spot = {.layer = layer};
            if (!fgets(line, sizeof(line), file) ||
                sscanf(line, "%lf %lf %lf", &spot.x, &spot.y,
                       &spot.particles) != 3) {
                fail("invalid point block", path);
            }
            append(&plan, spot);
        }
    }
    if (ferror(file)) fail("read failed", path);
    fclose(file);
    return plan;
}

static int compare_spot(const void *left, const void *right) {
    const Spot *a = left, *b = right;
    if (a->layer != b->layer) return a->layer < b->layer ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    if (a->x != b->x) return a->x < b->x ? -1 : 1;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s REFERENCE.rst CANDIDATE.rst\n", argv[0]);
        return EXIT_FAILURE;
    }
    Plan reference = read_plan(argv[1]);
    Plan candidate = read_plan(argv[2]);
    qsort(reference.spots, reference.count, sizeof(*reference.spots),
          compare_spot);
    qsort(candidate.spots, candidate.count, sizeof(*candidate.spots),
          compare_spot);

    size_t i = 0, j = 0, union_count = 0, shared = 0;
    double sum2 = 0.0, maximum = 0.0, reference_sum = 0.0;
    double candidate_sum = 0.0;
    while (i < reference.count || j < candidate.count) {
        double a = 0.0, b = 0.0;
        int order = i == reference.count ? 1 :
                    j == candidate.count ? -1 :
                    compare_spot(reference.spots + i, candidate.spots + j);
        if (order <= 0) a = reference.spots[i++].particles;
        if (order >= 0) b = candidate.spots[j++].particles;
        if (order == 0) ++shared;
        double difference = b - a;
        double magnitude = fabs(difference);
        sum2 += difference * difference;
        reference_sum += a;
        candidate_sum += b;
        if (magnitude > maximum) maximum = magnitude;
        ++union_count;
    }
    printf("reference_spots=%zu candidate_spots=%zu shared=%zu union=%zu "
           "rms=%.17g max_abs=%.17g reference_particles=%.17g "
           "candidate_particles=%.17g\n",
           reference.count, candidate.count, shared, union_count,
           union_count ? sqrt(sum2 / (double)union_count) : 0.0, maximum,
           reference_sum, candidate_sum);
    free(reference.spots);
    free(candidate.spots);
    return EXIT_SUCCESS;
}
