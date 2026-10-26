// iops.c FILE THREADS SECS [direct] [bytes] - random aligned preads of BYTES (4096) over FILE from THREADS threads for
// SECS seconds: reads/s and MB/s.  "direct" = O_DIRECT (the disk itself, no page cache).  larry, the Engram row reads.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static const char *path;
static int direct;
static size_t bs = 4096;
static uint64_t fsize;
static atomic_long nreads;
static atomic_int stop;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void *worker(void *arg) {
    uint64_t s = (uint64_t) (uintptr_t) arg * 0x9E3779B97F4A7C15ull + 1;
    int fd = open(path, O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) { perror("open"); return NULL; }
    void *buf;
    if (posix_memalign(&buf, 4096, bs)) return NULL;
    long n = 0;
    while (!atomic_load_explicit(&stop, memory_order_relaxed)) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        uint64_t off = (s % (fsize / bs)) * bs;
        if (pread(fd, buf, bs, (off_t) off) != (ssize_t) bs) { perror("pread"); break; }
        ++n;
    }
    atomic_fetch_add(&nreads, n);
    close(fd);
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "iops FILE THREADS SECS [direct] [bytes]\n"); return 1; }
    path = argv[1];
    int nt = atoi(argv[2]);
    double secs = atof(argv[3]);
    direct = argc > 4 && strcmp(argv[4], "direct") == 0;
    if (argc > 5) bs = (size_t) atol(argv[5]);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    fsize = (uint64_t) lseek(fd, 0, SEEK_END);
    close(fd);
    pthread_t *th = calloc((size_t) nt, sizeof *th);
    double t0 = now();
    for (int i = 0; i < nt; ++i) pthread_create(&th[i], NULL, worker, (void *) (uintptr_t) (i + 1));
    struct timespec ts = { (time_t) secs, (long) ((secs - (time_t) secs) * 1e9) };
    nanosleep(&ts, NULL);
    atomic_store(&stop, 1);
    for (int i = 0; i < nt; ++i) pthread_join(th[i], NULL);
    double dt = now() - t0;
    long n = atomic_load(&nreads);
    printf("%s: %d threads, %s %zu B: %.0f reads/s, %.0f MB/s\n", path, nt, direct ? "O_DIRECT" : "buffered", bs, n / dt,
           n * (double) bs / dt / 1e6);
    return 0;
}
